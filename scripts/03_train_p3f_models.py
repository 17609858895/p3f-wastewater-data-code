from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import rankdata
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "forecast_10min_features.parquet"
META_PATH = ROOT / "data" / "processed" / "forecast_metadata.json"
TABLE_DIR = ROOT / "tables"
MODEL_DIR = ROOT / "models" / "p3f"
PRED_DIR = ROOT / "data" / "predictions"

TARGETS = ["T1_PO4", "T1_NH4"]
HORIZON_ORDER = ["0.5h", "1h", "2h", "4h", "6h"]
SEED = 20260809


def suffix(horizon: str) -> str:
    return horizon.replace(".", "p")


def task_list(metadata: dict) -> list[dict]:
    return [
        {
            "target": target,
            "horizon": horizon,
            "steps": metadata["horizons"][horizon],
            "y": f"y_{target}_{suffix(horizon)}",
            "delta": f"delta_{target}_{suffix(horizon)}",
            "key": f"{target}_{suffix(horizon)}",
        }
        for target in TARGETS
        for horizon in HORIZON_ORDER
    ]


def select_feature_sets(features: list[str]) -> dict[str, list[str]]:
    def is_phase(name: str) -> bool:
        low = name.lower()
        return "processphase" in low or "phase_" in low or "transition_" in low

    def is_long(name: str) -> bool:
        if "_lag" in name:
            return int(name.rsplit("_lag", 1)[1]) > 36
        if "_roll" in name:
            return int(name.split("_roll", 1)[1].split("_", 1)[0]) > 36
        return False

    target_history = [
        c
        for c in features
        if c.startswith(("T1_PO4", "T1_NH4", "hour_", "doy_", "dow_"))
    ]
    return {
        "full": features,
        "no_phase": [c for c in features if not is_phase(c)],
        "short_history": [c for c in features if not is_long(c)],
        "target_history_only": target_history,
    }


def p3f_model(seed: int = SEED, objective: str = "huber") -> lgb.LGBMRegressor:
    params = dict(
        objective=objective,
        n_estimators=360,
        learning_rate=0.035,
        num_leaves=63,
        max_depth=10,
        min_child_samples=80,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.78,
        reg_alpha=0.15,
        reg_lambda=1.5,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )
    if objective == "huber":
        params["alpha"] = 0.9
    return lgb.LGBMRegressor(**params)


def direct_lgbm(seed: int = SEED) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="regression_l1",
        n_estimators=300,
        learning_rate=0.04,
        num_leaves=47,
        max_depth=9,
        min_child_samples=100,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.75,
        reg_alpha=0.2,
        reg_lambda=1.5,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )


def xgb_delta(seed: int = SEED) -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        objective="reg:pseudohubererror",
        n_estimators=320,
        learning_rate=0.04,
        max_depth=8,
        min_child_weight=20,
        subsample=0.80,
        colsample_bytree=0.75,
        reg_alpha=0.10,
        reg_lambda=2.0,
        tree_method="hist",
        device=os.environ.get("P3F_XGB_DEVICE", "cuda"),
        random_state=seed,
        n_jobs=8,
        verbosity=0,
    )


def metrics(y: np.ndarray, pred: np.ndarray, persistence: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    persistence = np.asarray(persistence, dtype=float)
    mae = mean_absolute_error(y, pred)
    rmse = math.sqrt(mean_squared_error(y, pred))
    p_mae = mean_absolute_error(y, persistence)
    p_rmse = math.sqrt(mean_squared_error(y, persistence))
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2_score(y, pred),
        "skill_mae": 1.0 - mae / p_mae if p_mae > 0 else np.nan,
        "skill_rmse": 1.0 - rmse / p_rmse if p_rmse > 0 else np.nan,
    }


def predict_absolute(model, X: pd.DataFrame, current: np.ndarray, delta: bool) -> np.ndarray:
    raw = model.predict(X)
    pred = current + raw if delta else raw
    return np.maximum(np.asarray(pred, dtype=float), 0.0)


def estimate_tree_parameters(model) -> int:
    if isinstance(model, lgb.LGBMRegressor):
        info = model.booster_.dump_model()
        return int(sum(tree["num_leaves"] for tree in info["tree_info"]))
    if isinstance(model, xgb.XGBRegressor):
        return int(len(model.get_booster().get_dump()))
    if hasattr(model, "named_steps") and "ridge" in model.named_steps:
        return int(np.size(model.named_steps["ridge"].coef_))
    return 0


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.clip(q, 0, 1)
    return out


def weekly_block_bootstrap(
    index: pd.DatetimeIndex,
    y: np.ndarray,
    pred: np.ndarray,
    persistence: np.ndarray,
    n_boot: int = 1000,
    seed: int = SEED,
) -> dict[str, float]:
    groups = ((index - index.min()) / pd.Timedelta(days=7)).astype(int)
    unique = np.unique(groups)
    model_sum = np.array([np.abs(y[groups == g] - pred[groups == g]).sum() for g in unique])
    pers_sum = np.array([np.abs(y[groups == g] - persistence[groups == g]).sum() for g in unique])
    counts = np.array([(groups == g).sum() for g in unique])
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(unique), size=(n_boot, len(unique)))
    mae_model = model_sum[draw].sum(axis=1) / counts[draw].sum(axis=1)
    mae_pers = pers_sum[draw].sum(axis=1) / counts[draw].sum(axis=1)
    skill = 1 - mae_model / mae_pers
    return {
        "median": float(np.median(skill)),
        "ci_low": float(np.quantile(skill, 0.025)),
        "ci_high": float(np.quantile(skill, 0.975)),
        "p_one_sided": float((np.sum(skill <= 0) + 1) / (len(skill) + 1)),
    }


def fit_cv_models(
    data: pd.DataFrame,
    metadata: dict,
    tasks: list[dict],
    features: list[str],
) -> pd.DataFrame:
    rows: list[dict] = []
    for fold in metadata["rolling_folds"]:
        train_mask = (data.index >= pd.Timestamp(fold["train_start"])) & (data.index <= pd.Timestamp(fold["train_end"]))
        valid_mask = (data.index >= pd.Timestamp(fold["valid_start"])) & (data.index <= pd.Timestamp(fold["valid_end"]))
        train = data.loc[train_mask]
        valid = data.loc[valid_mask]
        print(f"CV fold {fold['fold']}: train={len(train)} valid={len(valid)}", flush=True)

        delta_cols = [t["delta"] for t in tasks]
        ridge = make_pipeline(StandardScaler(), Ridge(alpha=20.0))
        ridge.fit(train[features], train[delta_cols])
        ridge_delta = ridge.predict(valid[features])

        for j, task in enumerate(tasks):
            target = task["target"]
            y = valid[task["y"]].to_numpy()
            current = valid[target].to_numpy()
            persistence = current
            seasonal = valid[f"{target}_lag144"].to_numpy()
            for model_name, pred in [
                ("Persistence", persistence),
                ("Seasonal-24h", seasonal),
                ("Ridge-delta", np.maximum(current + ridge_delta[:, j], 0)),
            ]:
                rows.append({"fold": fold["fold"], "model": model_name, **task, **metrics(y, pred, persistence)})

            direct = direct_lgbm(SEED + fold["fold"])
            direct.fit(train[features], train[task["y"]])
            pred_direct = predict_absolute(direct, valid[features], current, delta=False)
            rows.append({"fold": fold["fold"], "model": "LightGBM-direct", **task, **metrics(y, pred_direct, persistence)})

            xgb_model = xgb_delta(SEED + fold["fold"])
            xgb_model.fit(train[features], train[task["delta"]])
            pred_xgb = predict_absolute(xgb_model, valid[features], current, delta=True)
            rows.append({"fold": fold["fold"], "model": "XGBoost-delta", **task, **metrics(y, pred_xgb, persistence)})

            p3f = p3f_model(SEED + fold["fold"])
            p3f.fit(train[features], train[task["delta"]])
            pred_p3f = predict_absolute(p3f, valid[features], current, delta=True)
            rows.append({"fold": fold["fold"], "model": "P3F", **task, **metrics(y, pred_p3f, persistence)})
    return pd.DataFrame(rows)


def split_diagnostic(
    train_pool: pd.DataFrame,
    tasks: list[dict],
    features: list[str],
    cv_metrics: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []
    selected = [t for t in tasks if t["horizon"] == "4h"]
    for task in selected:
        target = task["target"]
        purged = cv_metrics.query("model == 'P3F' and target == @target and horizon == '4h'")
        for row in purged.itertuples():
            rows.append({"protocol": "Purged rolling", "repeat": int(row.fold), "target": target, "r2": row.r2, "skill_mae": row.skill_mae})

        for repeat, random_state in enumerate([17, 29, 41, 53], start=1):
            idx_train, idx_valid = train_test_split(np.arange(len(train_pool)), test_size=0.2, random_state=random_state, shuffle=True)
            tr = train_pool.iloc[idx_train]
            va = train_pool.iloc[idx_valid]
            model = p3f_model(SEED + random_state)
            model.fit(tr[features], tr[task["delta"]])
            pred = predict_absolute(model, va[features], va[target].to_numpy(), delta=True)
            m = metrics(va[task["y"]].to_numpy(), pred, va[target].to_numpy())
            rows.append({"protocol": "Random rows", "repeat": repeat, "target": target, "r2": m["r2"], "skill_mae": m["skill_mae"]})

        valid_starts = [pd.Timestamp(x) for x in ["2022-07-01 00:00:00+00:00", "2022-08-01 00:00:00+00:00", "2022-09-01 00:00:00+00:00", "2022-10-01 00:00:00+00:00"]]
        for repeat, valid_start in enumerate(valid_starts, start=1):
            valid_end = min(valid_start + pd.offsets.MonthEnd(1) + pd.Timedelta(hours=23, minutes=50), train_pool.index.max())
            tr = train_pool.loc[train_pool.index < valid_start]
            va = train_pool.loc[(train_pool.index >= valid_start) & (train_pool.index <= valid_end)]
            model = p3f_model(SEED + repeat + 100)
            model.fit(tr[features], tr[task["delta"]])
            pred = predict_absolute(model, va[features], va[target].to_numpy(), delta=True)
            m = metrics(va[task["y"]].to_numpy(), pred, va[target].to_numpy())
            rows.append({"protocol": "Chronological, no purge", "repeat": repeat, "target": target, "r2": m["r2"], "skill_mae": m["skill_mae"]})
    return pd.DataFrame(rows)


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(META_PATH.read_text(encoding="utf-8"))
    data = pd.read_parquet(DATA_PATH)
    features = metadata["features"]
    feature_sets = select_feature_sets(features)
    tasks = task_list(metadata)

    train = data.query("split == 'train'").copy()
    calibration = data.query("split == 'calibration'").copy()
    test = data.query("split == 'test'").copy()
    print(f"Loaded train={len(train)} calibration={len(calibration)} test={len(test)} features={len(features)}", flush=True)

    cv = fit_cv_models(data, metadata, tasks, features)
    cv.to_csv(TABLE_DIR / "TableS_cv_fold_metrics.csv", index=False, encoding="utf-8-sig")
    split_diag = split_diagnostic(train, tasks, features, cv)
    split_diag.to_csv(TABLE_DIR / "Fig03_split_protocol_diagnostic.csv", index=False, encoding="utf-8-sig")

    pred_index = calibration.index.append(test.index)
    pred = pd.DataFrame(index=pred_index)
    pred["split"] = data.loc[pred_index, "split"]
    pred["PROCESSPHASE_INLET"] = data.loc[pred_index, "PROCESSPHASE_INLET"]
    pred["transition_recent_1h"] = data.loc[pred_index, "transition_inlet_recent_1h"]
    pred["season"] = pd.Categorical.from_codes(((pred.index.month % 12) // 3), ["DJF", "MAM", "JJA", "SON"])
    for target in TARGETS:
        pred[f"current_{target}"] = data.loc[pred_index, target]

    delta_cols = [t["delta"] for t in tasks]
    ridge_start = time.perf_counter()
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=20.0))
    ridge.fit(train[features], train[delta_cols])
    ridge_fit_s = time.perf_counter() - ridge_start
    ridge_delta_pred = ridge.predict(data.loc[pred_index, features])
    ridge_delta_train = ridge.predict(train[features])
    joblib.dump(ridge, MODEL_DIR / "Ridge_delta_multioutput.joblib", compress=3)

    final_models: dict[str, dict[str, object]] = {"P3F": {}, "LightGBM-direct": {}, "XGBoost-delta": {}}
    compute_rows: list[dict] = []
    metric_records: list[dict] = []
    ablation_rows: list[dict] = []
    ablation_predictions: list[pd.DataFrame] = []

    ridge_latency_start = time.perf_counter()
    ridge.predict(test[features].iloc[:5000])
    ridge_latency = (time.perf_counter() - ridge_latency_start) / 5000 * 1000 * 1000
    compute_rows.append({"model": "Ridge-delta", "fit_seconds": ridge_fit_s, "latency_ms_per_1000": ridge_latency, "parameters_proxy": estimate_tree_parameters(ridge), "feature_count": len(features)})

    for j, task in enumerate(tasks):
        target = task["target"]
        current_all = data.loc[pred_index, target].to_numpy()
        current_train = train[target].to_numpy()
        observed_all = data.loc[pred_index, task["y"]].to_numpy()
        persistence_all = current_all
        seasonal_all = data.loc[pred_index, f"{target}_lag144"].to_numpy()
        pred[f"obs_{task['key']}"] = observed_all
        pred[f"pred_Persistence_{task['key']}"] = persistence_all
        pred[f"pred_Seasonal24h_{task['key']}"] = seasonal_all
        pred[f"pred_RidgeDelta_{task['key']}"] = np.maximum(current_all + ridge_delta_pred[:, j], 0)

        model_specs = [
            ("LightGBM-direct", direct_lgbm(), False, task["y"]),
            ("XGBoost-delta", xgb_delta(), True, task["delta"]),
            ("P3F", p3f_model(), True, task["delta"]),
        ]
        for model_name, model, is_delta, label_col in model_specs:
            start = time.perf_counter()
            model.fit(train[features], train[label_col])
            fit_s = time.perf_counter() - start
            model_pred_all = predict_absolute(model, data.loc[pred_index, features], current_all, is_delta)
            model_pred_train = predict_absolute(model, train[features], current_train, is_delta)
            key_name = model_name.replace("-", "").replace("3", "3")
            pred[f"pred_{key_name}_{task['key']}"] = model_pred_all
            final_models[model_name][task["key"]] = model
            if model_name == "P3F":
                joblib.dump(model, MODEL_DIR / f"P3F_{task['key']}.joblib", compress=3)

            latency_start = time.perf_counter()
            predict_absolute(model, test[features].iloc[:5000], test[target].iloc[:5000].to_numpy(), is_delta)
            latency_ms = (time.perf_counter() - latency_start) / 5000 * 1000 * 1000
            compute_rows.append({"model": model_name, "task": task["key"], "fit_seconds": fit_s, "latency_ms_per_1000": latency_ms, "parameters_proxy": estimate_tree_parameters(model), "feature_count": len(features)})

            train_m = metrics(train[task["y"]].to_numpy(), model_pred_train, current_train)
            test_mask_all = pred["split"].eq("test").to_numpy()
            test_m = metrics(observed_all[test_mask_all], model_pred_all[test_mask_all], persistence_all[test_mask_all])
            metric_records.append({"model": model_name, **task, **{f"train_{k}": v for k, v in train_m.items()}, **{f"test_{k}": v for k, v in test_m.items()}})

        ridge_pred_train = np.maximum(current_train + ridge_delta_train[:, j], 0)
        ridge_train_m = metrics(train[task["y"]], ridge_pred_train, current_train)
        ridge_test_pred = pred.loc[pred["split"].eq("test"), f"pred_RidgeDelta_{task['key']}"]
        ridge_test_m = metrics(test[task["y"]], ridge_test_pred, test[target])
        metric_records.append({"model": "Ridge-delta", **task, **{f"train_{k}": v for k, v in ridge_train_m.items()}, **{f"test_{k}": v for k, v in ridge_test_m.items()}})

        for baseline_name, train_pred_values, test_pred_values in [
            ("Persistence", current_train, test[target].to_numpy()),
            ("Seasonal-24h", train[f"{target}_lag144"].to_numpy(), test[f"{target}_lag144"].to_numpy()),
        ]:
            train_m = metrics(train[task["y"]], train_pred_values, current_train)
            test_m = metrics(test[task["y"]], test_pred_values, test[target])
            metric_records.append({"model": baseline_name, **task, **{f"train_{k}": v for k, v in train_m.items()}, **{f"test_{k}": v for k, v in test_m.items()}})

        full_p3f_cal = pred.loc[pred["split"].eq("calibration"), f"pred_P3F_{task['key']}"]
        full_p3f_test = pred.loc[pred["split"].eq("test"), f"pred_P3F_{task['key']}"]
        full_test_m = metrics(test[task["y"]], full_p3f_test, test[target])
        ablation_rows.append({"variant": "Full P3F", **task, **full_test_m, "fit_seconds": np.nan, "feature_count": len(features)})

        for variant, variant_features, is_delta in [
            ("No phase context", feature_sets["no_phase"], True),
            ("Short history only", feature_sets["short_history"], True),
            ("Target history only", feature_sets["target_history_only"], True),
            ("Direct target", feature_sets["full"], False),
        ]:
            label = task["delta"] if is_delta else task["y"]
            model = p3f_model(objective="huber")
            start = time.perf_counter()
            model.fit(train[variant_features], train[label])
            fit_s = time.perf_counter() - start
            p_cal = predict_absolute(model, calibration[variant_features], calibration[target].to_numpy(), is_delta)
            p_test = predict_absolute(model, test[variant_features], test[target].to_numpy(), is_delta)
            m = metrics(test[task["y"]], p_test, test[target])
            cal_abs = np.abs(calibration[task["y"]].to_numpy() - p_cal)
            q90 = np.quantile(cal_abs, 0.9, method="higher")
            coverage90 = np.mean((test[task["y"]].to_numpy() >= np.maximum(p_test - q90, 0)) & (test[task["y"]].to_numpy() <= p_test + q90))
            ablation_rows.append({"variant": variant, **task, **m, "fit_seconds": fit_s, "feature_count": len(variant_features), "coverage90": coverage90})
            ablation_predictions.append(pd.DataFrame({"date": test.index, "variant": variant, "target": target, "horizon": task["horizon"], "observed": test[task["y"]].to_numpy(), "prediction": p_test, "phase": test["PROCESSPHASE_INLET"].to_numpy(), "transition": test["transition_inlet_recent_1h"].to_numpy()}))

    metrics_df = pd.DataFrame(metric_records)
    cv_summary = cv.groupby(["model", "target", "horizon"], as_index=False).agg(cv_mean_r2=("r2", "mean"), cv_min_r2=("r2", "min"), cv_mean_skill=("skill_mae", "mean"), cv_min_skill=("skill_mae", "min"))
    metrics_df = metrics_df.merge(cv_summary, on=["model", "target", "horizon"], how="left")
    metrics_df["train_test_gap"] = metrics_df["train_r2"] - metrics_df["test_r2"]

    test_index = pred.index[pred["split"].eq("test")]
    bootstrap_rows: list[dict] = []
    model_column_map = {
        "Persistence": "Persistence",
        "Seasonal-24h": "Seasonal24h",
        "Ridge-delta": "RidgeDelta",
        "LightGBM-direct": "LightGBMdirect",
        "XGBoost-delta": "XGBoostdelta",
        "P3F": "P3F",
    }
    for task in tasks:
        y = pred.loc[test_index, f"obs_{task['key']}"].to_numpy()
        pers = pred.loc[test_index, f"pred_Persistence_{task['key']}"].to_numpy()
        for model_name, column_token in model_column_map.items():
            p = pred.loc[test_index, f"pred_{column_token}_{task['key']}"].to_numpy()
            b = weekly_block_bootstrap(test_index, y, p, pers, seed=SEED + len(bootstrap_rows))
            bootstrap_rows.append({"model": model_name, **task, **b})
    boot = pd.DataFrame(bootstrap_rows)
    p3f_mask = boot["model"].eq("P3F")
    boot.loc[p3f_mask, "q_bh"] = benjamini_hochberg(boot.loc[p3f_mask, "p_one_sided"].to_numpy())

    for task in tasks:
        key = task["key"]
        cal_mask = pred["split"].eq("calibration")
        test_mask = pred["split"].eq("test")
        cal_y = pred.loc[cal_mask, f"obs_{key}"].to_numpy()
        cal_p = pred.loc[cal_mask, f"pred_P3F_{key}"].to_numpy()
        test_p = pred.loc[test_mask, f"pred_P3F_{key}"].to_numpy()
        cal_residual = cal_y - cal_p
        cal_abs = np.abs(cal_residual)
        q50 = np.quantile(cal_abs, 0.50, method="higher")
        q90 = np.quantile(cal_abs, 0.90, method="higher")
        pred.loc[test_mask, f"lo50_{key}"] = np.maximum(test_p - q50, 0)
        pred.loc[test_mask, f"hi50_{key}"] = test_p + q50
        pred.loc[test_mask, f"lo90_{key}"] = np.maximum(test_p - q90, 0)
        pred.loc[test_mask, f"hi90_{key}"] = test_p + q90
        pred.loc[cal_mask, f"lo50_{key}"] = np.maximum(cal_p - q50, 0)
        pred.loc[cal_mask, f"hi50_{key}"] = cal_p + q50
        pred.loc[cal_mask, f"lo90_{key}"] = np.maximum(cal_p - q90, 0)
        pred.loc[cal_mask, f"hi90_{key}"] = cal_p + q90

        train_model = final_models["P3F"][key]
        train_p = predict_absolute(train_model, train[features], train[task["target"]].to_numpy(), True)
        q90_in = np.quantile(np.abs(train[task["y"]].to_numpy() - train_p), 0.90, method="higher")
        pred.loc[test_mask, f"lo90_in_sample_{key}"] = np.maximum(test_p - q90_in, 0)
        pred.loc[test_mask, f"hi90_in_sample_{key}"] = test_p + q90_in

        event_threshold = float(np.quantile(train[task["y"]].to_numpy(), 0.90))
        sorted_residuals = np.sort(cal_residual)
        probability = 1.0 - np.searchsorted(sorted_residuals, event_threshold - test_p, side="right") / len(sorted_residuals)
        pred.loc[test_mask, f"event_probability_{key}"] = probability
        pred.loc[test_mask, f"event_label_{key}"] = (pred.loc[test_mask, f"obs_{key}"].to_numpy() >= event_threshold).astype(int)
        pred.loc[cal_mask, f"event_probability_{key}"] = 1.0 - np.searchsorted(sorted_residuals, event_threshold - cal_p, side="right") / len(sorted_residuals)
        pred.loc[cal_mask, f"event_label_{key}"] = (cal_y >= event_threshold).astype(int)
        pred[f"event_threshold_{key}"] = event_threshold

    ad_features = [
        c
        for c in features
        if c in ["IN_METAL_Q", "T1_O2", "METAL_Q", "TEMPERATURE", "IN_Q", "MAX_CF", "PROCESSPHASE_INLET", "PROCESSPHASE_OUTLET", "T1_NH4", "T1_PO4"]
        or any(token in c for token in ["roll36_mean", "roll36_std", "phase_inlet_elapsed_h", "transition_inlet_recent_1h", "hour_sin", "hour_cos"])
    ]
    ad_mean = train[ad_features].mean()
    ad_std = train[ad_features].std(ddof=0).clip(lower=1e-6)
    for block_name, block in [("calibration", calibration), ("test", test)]:
        score = np.sqrt(np.mean(np.square((block[ad_features] - ad_mean) / ad_std), axis=1))
        pred.loc[block.index, "ad_score"] = score.to_numpy()
    ad_q90, ad_q975 = pred.loc[pred["split"].eq("calibration"), "ad_score"].quantile([0.90, 0.975]).to_numpy()
    pred["ad_class"] = pd.cut(pred["ad_score"], bins=[-np.inf, ad_q90, ad_q975, np.inf], labels=["in-domain", "borderline", "far-OOD"])

    uncertainty_rows: list[dict] = []
    for task in tasks:
        key = task["key"]
        sub = pred.loc[pred["split"].eq("test")].copy()
        y = sub[f"obs_{key}"].to_numpy()
        p = sub[f"pred_P3F_{key}"].to_numpy()
        for group_name, group_value in [
            ("all", "all"),
            ("phase", "phase 1"),
            ("phase", "phase 2"),
            ("transition", "stable"),
            ("transition", "recent transition"),
            ("concentration", "below training Q90"),
            ("concentration", "at/above training Q90"),
            ("AD", "in-domain"),
            ("AD", "borderline/far-OOD"),
        ]:
            if group_name == "all":
                mask = np.ones(len(sub), dtype=bool)
            elif group_name == "phase":
                phase_number = 1 if group_value.endswith("1") else 2
                mask = sub["PROCESSPHASE_INLET"].round().eq(phase_number).to_numpy()
            elif group_name == "transition":
                mask = sub["transition_recent_1h"].eq(0).to_numpy() if group_value == "stable" else sub["transition_recent_1h"].gt(0).to_numpy()
            elif group_name == "concentration":
                threshold = sub[f"event_threshold_{key}"].iloc[0]
                mask = y < threshold if group_value.startswith("below") else y >= threshold
            else:
                mask = sub["ad_class"].eq("in-domain").to_numpy() if group_value == "in-domain" else sub["ad_class"].ne("in-domain").to_numpy()
            if mask.sum() == 0:
                continue
            cover50 = np.mean((y[mask] >= sub.loc[mask, f"lo50_{key}"]) & (y[mask] <= sub.loc[mask, f"hi50_{key}"]))
            cover90 = np.mean((y[mask] >= sub.loc[mask, f"lo90_{key}"]) & (y[mask] <= sub.loc[mask, f"hi90_{key}"]))
            uncertainty_rows.append({"target": task["target"], "horizon": task["horizon"], "group_type": group_name, "group": group_value, "n": int(mask.sum()), "mae": mean_absolute_error(y[mask], p[mask]), "coverage50": cover50, "coverage90": cover90, "width50": np.mean(sub.loc[mask, f"hi50_{key}"] - sub.loc[mask, f"lo50_{key}"]), "width90": np.mean(sub.loc[mask, f"hi90_{key}"] - sub.loc[mask, f"lo90_{key}"])})

    volatility_rows: list[dict] = []
    test_pred = pred.loc[pred["split"].eq("test")]
    for target in TARGETS:
        task = next(t for t in tasks if t["target"] == target and t["horizon"] == "4h")
        vol_col = f"{target}_roll36_std"
        edges = np.unique(np.quantile(train[vol_col], np.linspace(0, 1, 6)))
        bins = pd.cut(test[vol_col], bins=edges, include_lowest=True, duplicates="drop")
        for phase in [1, 2]:
            for bin_value in bins.cat.categories:
                mask = bins.eq(bin_value).to_numpy() & test["PROCESSPHASE_INLET"].round().eq(phase).to_numpy()
                if mask.sum() < 50:
                    continue
                y = test[task["y"]].to_numpy()[mask]
                p = test_pred[f"pred_P3F_{task['key']}"] .to_numpy()[mask]
                persistence = test[target].to_numpy()[mask]
                volatility_rows.append({"target": target, "phase": phase, "volatility_bin": str(bin_value), "volatility_median": float(test.loc[mask, vol_col].median()), "n": int(mask.sum()), "skill_mae": metrics(y, p, persistence)["skill_mae"]})

    compute = pd.DataFrame(compute_rows)
    compute_summary = compute.groupby("model", as_index=False).agg(fit_seconds=("fit_seconds", "sum"), latency_ms_per_1000=("latency_ms_per_1000", "sum"), parameters_proxy=("parameters_proxy", "sum"), feature_count=("feature_count", "max"))
    compute_summary = pd.concat([pd.DataFrame([{"model": "Persistence", "fit_seconds": 0.0, "latency_ms_per_1000": 0.01, "parameters_proxy": 0, "feature_count": 1}, {"model": "Seasonal-24h", "fit_seconds": 0.0, "latency_ms_per_1000": 0.01, "parameters_proxy": 0, "feature_count": 1}]), compute_summary], ignore_index=True)

    ablation = pd.DataFrame(ablation_rows)
    full_calibration = []
    for task in tasks:
        key = task["key"]
        sub = pred.loc[pred["split"].eq("test")]
        cover90 = np.mean((sub[f"obs_{key}"] >= sub[f"lo90_{key}"]) & (sub[f"obs_{key}"] <= sub[f"hi90_{key}"]))
        full_calibration.append((task["target"], task["horizon"], cover90))
    for target, horizon, coverage in full_calibration:
        ablation.loc[(ablation["variant"] == "Full P3F") & (ablation["target"] == target) & (ablation["horizon"] == horizon), "coverage90"] = coverage

    predictions_path = PRED_DIR / "P3F_calibration_test_predictions.parquet"
    pred.to_parquet(predictions_path, compression="zstd")
    pd.concat(ablation_predictions, ignore_index=True).to_parquet(PRED_DIR / "P3F_ablation_test_predictions.parquet", compression="zstd")
    metrics_df.to_csv(TABLE_DIR / "Table3_model_metrics.csv", index=False, encoding="utf-8-sig")
    boot.to_csv(TABLE_DIR / "Table3_block_bootstrap_skill.csv", index=False, encoding="utf-8-sig")
    ablation.to_csv(TABLE_DIR / "Table4_ablation_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(uncertainty_rows).to_csv(TABLE_DIR / "Table5_uncertainty_subgroups.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(volatility_rows).to_csv(TABLE_DIR / "Fig03_volatility_skill.csv", index=False, encoding="utf-8-sig")
    compute_summary.to_csv(TABLE_DIR / "TableS_compute_profile.csv", index=False, encoding="utf-8-sig")
    (MODEL_DIR / "model_manifest.json").write_text(json.dumps({"model": "P3F", "expansion": "Phase-aware, Persistence-corrected Probabilistic Forecaster", "feature_count": len(features), "features": features, "ad_features": ad_features, "ad_q90": float(ad_q90), "ad_q975": float(ad_q975), "test_opened_once": True, "claim_boundary": "Application/method combination; not a language model and not claimed as globally novel algorithm."}, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ok", "predictions": str(predictions_path), "metrics_rows": len(metrics_df), "cv_rows": len(cv), "ablation_rows": len(ablation)}), flush=True)


if __name__ == "__main__":
    main()
