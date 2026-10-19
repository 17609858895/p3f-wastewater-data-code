"""Fit the selected P3F v2 configuration on the full training pool, calibrate the interval
layer on the calibration block, and evaluate once on the future-test block.

Requires tables/B_selected_configuration.json from script 52. v1 tables that are replaced are
first copied to tables/v1_huber_split_conformal/. Comparator predictions (persistence,
seasonal, ridge, LightGBM-direct, XGBoost-delta) are reused unchanged from the v1 prediction
file; v1 P3F columns are kept with the suffix P3Fv1.
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p3f_b_common as C

V1_DIR = C.TABLE_DIR / "v1_huber_split_conformal"
REPLACED = ["Table3_model_metrics.csv", "Table3_block_bootstrap_skill.csv", "Table5_uncertainty_subgroups.csv",
            "Fig03_volatility_skill.csv", "TableS_compute_profile.csv", "TableS11_cross_strata.csv",
            "TableS14_rolling_recalibration.csv", "Fig06f_weekly_coverage_4h.csv"]
MODEL_TOKENS = {"Persistence": "Persistence", "Seasonal-24h": "Seasonal24h", "Ridge-delta": "RidgeDelta",
                "LightGBM-direct": "LightGBMdirect", "XGBoost-delta": "XGBoostdelta", "P3F": "P3F"}


def metrics(y, p, pers) -> dict:
    y, p, pers = (np.asarray(v, float) for v in (y, p, pers))
    mae, rmse = np.mean(np.abs(y - p)), np.sqrt(np.mean((y - p) ** 2))
    pm, pr = np.mean(np.abs(y - pers)), np.sqrt(np.mean((y - pers) ** 2))
    return {"mae": mae, "rmse": rmse, "r2": 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2),
            "skill_mae": 1 - mae / pm, "skill_rmse": 1 - rmse / pr}


def weekly_block_bootstrap(index, y, pred, pers, n_boot=1000, seed=C.SEED) -> dict:
    groups = ((index - index.min()) / pd.Timedelta(days=7)).astype(int)
    unique = np.unique(groups)
    ms = np.array([np.abs(y[groups == g] - pred[groups == g]).sum() for g in unique])
    ps = np.array([np.abs(y[groups == g] - pers[groups == g]).sum() for g in unique])
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(unique), size=(n_boot, len(unique)))
    skill = 1 - ms[draw].sum(1) / ps[draw].sum(1)
    return {"median": float(np.median(skill)), "ci_low": float(np.quantile(skill, 0.025)),
            "ci_high": float(np.quantile(skill, 0.975)), "p_one_sided": float((np.sum(skill <= 0) + 1) / (n_boot + 1))}


def benjamini_hochberg(p):
    p = np.asarray(p, float); order = np.argsort(p); ranked = p[order]
    q = np.minimum.accumulate((ranked * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    out = np.empty_like(q); out[order] = np.clip(q, 0, 1); return out


def main() -> None:
    sel = json.loads(C.SELECTED_PATH.read_text(encoding="utf-8"))
    objective, method = sel["point_objective"], sel["interval_method"]
    print("selected", objective, method, flush=True)
    V1_DIR.mkdir(exist_ok=True)
    for name in REPLACED:
        src = C.TABLE_DIR / name
        if src.exists() and not (V1_DIR / name).exists():
            shutil.copy2(src, V1_DIR / name)
    C.MODEL_DIR_B.mkdir(parents=True, exist_ok=True)

    meta = C.load_meta(); features = meta["features"]; tasks = C.task_list(meta)
    data = pd.read_parquet(C.DATA_PATH)
    train, cal, test = (data.loc[data["split"].eq(s)] for s in ("train", "calibration", "test"))
    old = pd.read_parquet(C.OLD_PRED_PATH)
    pred = old.copy()
    rename = {}
    for c in old.columns:
        for stem in ("pred_P3F_", "lo90_in_sample_", "hi90_in_sample_", "lo50_", "hi50_", "lo90_", "hi90_", "event_probability_"):
            if c.startswith(stem):
                rename[c] = c.replace(stem, stem.replace("P3F", "P3Fv1") if stem == "pred_P3F_" else stem + "v1_", 1)
                break
    pred = pred.rename(columns=rename)
    cal_mask, test_mask = pred["split"].eq("calibration").to_numpy(), pred["split"].eq("test").to_numpy()
    assert pred.index[cal_mask].equals(cal.index) and pred.index[test_mask].equals(test.index)

    compute_rows, method_rows, cross_rows, info_rows = [], [], [], []
    train_metrics = {}
    for task in tasks:
        k = task["key"]
        t0 = time.perf_counter()
        models = {"point": C.booster(objective, C.SEED)}
        fit_s = {}
        fit_log_path = C.MODEL_DIR_B / "fit_seconds.json"
        fit_log = json.loads(fit_log_path.read_text()) if fit_log_path.exists() else {}
        for name, m in [("point", models["point"])] + [(n, C.booster("quantile", C.SEED, alpha=q)) for n, q in C.QUANTILES.items()]:
            path = C.MODEL_DIR_B / f"P3F_{k}_{name}.joblib"
            if path.exists() and f"{k}_{name}" in fit_log:  # fitted on the training block by script 53a
                m = joblib.load(path); fit_s[name] = fit_log[f"{k}_{name}"]
                assert m.get_params()["objective"] == ("regression_l1" if name == "point" else "quantile") and objective == "l1"
            else:
                s = time.perf_counter(); m.fit(train[features], train[task["delta"]]); fit_s[name] = time.perf_counter() - s
                joblib.dump(m, path, compress=3)
            models[name] = m
        s = time.perf_counter(); models["point"].predict(test[features].iloc[:5000]); lat_point = (time.perf_counter() - s) * 200  # ms per 1000 rows
        s = time.perf_counter()
        for n in C.QUANTILES:
            models[n].predict(test[features].iloc[:5000])
        lat_heads = (time.perf_counter() - s) * 200
        params = {n: int(sum(t["num_leaves"] for t in m.booster_.dump_model()["tree_info"])) for n, m in models.items()}
        compute_rows.append({"task": k, "fit_point_s": fit_s["point"], "fit_heads_s": sum(v for n, v in fit_s.items() if n != "point"),
                             "latency_point_ms_per_1000": lat_point, "latency_heads_ms_per_1000": lat_heads,
                             "leaves_point": params["point"], "leaves_heads": sum(v for n, v in params.items() if n != "point")})

        ptr = np.maximum(train[task["target"]].to_numpy(float) + models["point"].predict(train[features]), 0)
        train_metrics[k] = metrics(train[task["y"]], ptr, train[task["target"]])
        pc = C.predict_block(models, cal, features, task); pc["y"] = cal[task["y"]].to_numpy(float)
        pt = C.predict_block(models, test, features, task)
        y, cur = test[task["y"]].to_numpy(float), test[task["target"]].to_numpy(float)
        pred.loc[cal_mask, f"pred_P3F_{k}"] = pc["point"]; pred.loc[test_mask, f"pred_P3F_{k}"] = pt["point"]
        for n in C.QUANTILES:
            pred.loc[cal_mask, f"{n}_{k}"] = pc["heads"][n]; pred.loc[test_mask, f"{n}_{k}"] = pt["heads"][n]
        thr = float(pred[f"event_threshold_{k}"].iloc[0])
        high = y >= thr

        # interval layers on the test block: all candidates, v1 reference and LightGBM-direct + SC
        lgb_cal = pred.loc[cal_mask, f"pred_LightGBMdirect_{k}"].to_numpy(float)
        lgb_test = pred.loc[test_mask, f"pred_LightGBMdirect_{k}"].to_numpy(float)
        candidates = {}
        for m in C.INTERVAL_METHODS:
            for level in C.LEVELS:
                lo, hi, info = C.calibrate(m, level, pc, pt)
                lo_c, hi_c, _ = C.calibrate(m, level, pc, pc)
                candidates[(m, level)] = (lo, hi)
                info_rows.append({"key": k, "method": m, "level": level, **info})
                pred.loc[test_mask, f"lo{level}_{m}_{k}"] = lo; pred.loc[test_mask, f"hi{level}_{m}_{k}"] = hi
                pred.loc[cal_mask, f"lo{level}_{m}_{k}"] = lo_c; pred.loc[cal_mask, f"hi{level}_{m}_{k}"] = hi_c
        for level in C.LEVELS:
            candidates[("QR", level)] = C.raw_quantile_interval(level, pt)
            candidates[("v1 SC (Huber point)", level)] = (pred.loc[test_mask, f"lo{level}_v1_{k}"].to_numpy(float), pred.loc[test_mask, f"hi{level}_v1_{k}"].to_numpy(float))
            lc = {"y": pc["y"], "point": lgb_cal, "phase": pc["phase"]}
            lo, hi, _ = C.calibrate("SC", level, lc, {"point": lgb_test, "phase": pt["phase"]})
            candidates[("LightGBM-direct SC", level)] = (lo, hi)
        weeks = ((test.index - test.index.min()) / pd.Timedelta(days=7)).astype(int).to_numpy()
        pers_mae = float(np.mean(np.abs(y - cur)))
        for (m, level), (lo, hi) in candidates.items():
            if level != "90":
                continue
            lo50, hi50 = candidates[(m, "50")]
            cov, cov50 = (y >= lo) & (y <= hi), (y >= lo50) & (y <= hi50)
            wk = pd.Series(cov.astype(float)).groupby(weeks).mean()
            method_rows.append({"key": k, "target": task["target"], "horizon": task["horizon"], "method": m,
                                "cov90": cov.mean(), "cov50": cov50.mean(), "width90": np.mean(hi - lo), "width50": np.mean(hi50 - lo50),
                                "scaled_is90": float(np.mean(C.interval_score(y, lo, hi, 0.10)) / pers_mae),
                                "cov90_high": cov[high].mean(), "cov90_low": cov[~high].mean(),
                                "width90_high": np.mean((hi - lo)[high]), "width90_low": np.mean((hi - lo)[~high]),
                                "cov90_phase1": cov[pt["phase"] == 1].mean(), "cov90_phase2": cov[pt["phase"] == 2].mean(),
                                "cov50_phase1": cov50[pt["phase"] == 1].mean(), "cov50_phase2": cov50[pt["phase"] == 2].mean(),
                                "weeks_below_090": int((wk < 0.90).sum()), "min_weekly_cov90": float(wk.min()), "n_high": int(high.sum())})

        # selected layer -> canonical interval columns used by the figures
        for level in C.LEVELS:
            lo, hi = candidates[(method, level)]
            pred.loc[test_mask, f"lo{level}_{k}"] = lo; pred.loc[test_mask, f"hi{level}_{k}"] = hi
            pred.loc[cal_mask, f"lo{level}_{k}"] = pred.loc[cal_mask, f"lo{level}_{method}_{k}"]
            pred.loc[cal_mask, f"hi{level}_{k}"] = pred.loc[cal_mask, f"hi{level}_{method}_{k}"]
        q_in = C.conformal_quantile(np.abs(train[task["y"]].to_numpy(float) - ptr), 0.10)
        pred.loc[test_mask, f"lo90_in_sample_{k}"] = np.maximum(pt["point"] - q_in, 0); pred.loc[test_mask, f"hi90_in_sample_{k}"] = pt["point"] + q_in
        # exceedance score: empirical survival of signed calibration residuals (definition unchanged from v1)
        res = np.sort(pc["y"] - pc["point"])
        for mask, p in [(cal_mask, pc["point"]), (test_mask, pt["point"])]:
            pred.loc[mask, f"event_probability_{k}"] = 1 - np.searchsorted(res, thr - p, side="right") / len(res)
        # cross-stratification (Table S11) at 4 h
        if task["horizon"] == "4h":
            lo, hi = candidates[(method, "90")]
            cov = (y >= lo) & (y <= hi)
            ad = pred.loc[test_mask, "ad_class"].astype(str).eq("in-domain").to_numpy()
            for conc, cm in [("Below training Q90", ~high), ("At/above training Q90", high)]:
                for adn, am in [("In-domain", ad), ("Borderline/far-OOD", ~ad)]:
                    mm = cm & am
                    cross_rows.append({"target": task["target"], "horizon": "4h", "concentration": conc, "ad_class": adn,
                                       "n": int(mm.sum()), "coverage90": float(cov[mm].mean())})
        print(f"{k}: skill={C.mae_skill(y, pt['point'], cur):.4f} " + " ".join(f"{m}:{r['cov90']:.3f}/{r['cov90_high']:.3f}" for r in method_rows[-7:] for m in [r['method']]), f"{time.perf_counter() - t0:.0f}s", flush=True)

    pred.to_parquet(C.NEW_PRED_PATH, compression="zstd")
    test_pred = pred.loc[test_mask]
    pd.DataFrame(method_rows).to_csv(C.TABLE_DIR / "TableS17_interval_layers_test.csv", index=False)
    pd.DataFrame(info_rows).to_csv(C.TABLE_DIR / "TableS17_conformal_corrections.csv", index=False)
    pd.DataFrame(cross_rows).to_csv(C.TABLE_DIR / "TableS11_cross_strata.csv", index=False)

    # point metrics (Table 2 / S7)
    cv = pd.read_csv(C.TABLE_DIR / "TableS16_validation_selection.csv")
    cvp = cv[cv.stage == "point"]
    old_metrics = pd.read_csv(V1_DIR / "Table3_model_metrics.csv")
    rows = [r for r in old_metrics.to_dict("records") if r["model"] != "P3F"]
    for r in old_metrics.to_dict("records"):
        if r["model"] == "P3F":
            rows.append({**r, "model": "P3F v1 (Huber)"})
    for task in tasks:
        k = task["key"]
        y, cur = test[task["y"]].to_numpy(float), test[task["target"]].to_numpy(float)
        tm = metrics(y, test_pred[f"pred_P3F_{k}"].to_numpy(float), cur)
        c = cvp[cvp.key == k]
        rows.append({"model": "P3F", **task, **{f"train_{a}": b for a, b in train_metrics[k].items()}, **{f"test_{a}": b for a, b in tm.items()},
                     "cv_mean_r2": c[f"r2_{objective}"].mean(), "cv_min_r2": c[f"r2_{objective}"].min(),
                     "cv_mean_skill": c[f"skill_{objective}"].mean(), "cv_min_skill": c[f"skill_{objective}"].min(),
                     "train_test_gap": train_metrics[k]["r2"] - tm["r2"]})
    met = pd.DataFrame(rows)
    met.to_csv(C.TABLE_DIR / "Table3_model_metrics_with_v1.csv", index=False, encoding="utf-8-sig")
    met[met.model != "P3F v1 (Huber)"].to_csv(C.TABLE_DIR / "Table3_model_metrics.csv", index=False, encoding="utf-8-sig")

    # weekly block bootstrap of skill, same seeds/order as v1 script 03
    boot_rows = []
    idx = test_pred.index
    for task in tasks:
        k = task["key"]
        y = test_pred[f"obs_{k}"].to_numpy(float); pers = test_pred[f"pred_Persistence_{k}"].to_numpy(float)
        for model_name, tok in MODEL_TOKENS.items():
            b = weekly_block_bootstrap(idx, y, test_pred[f"pred_{tok}_{k}"].to_numpy(float), pers, seed=C.SEED + len(boot_rows))
            boot_rows.append({"model": model_name, **task, **b})
    boot = pd.DataFrame(boot_rows)
    m = boot["model"].eq("P3F")
    boot.loc[m, "q_bh"] = benjamini_hochberg(boot.loc[m, "p_one_sided"])
    boot.to_csv(C.TABLE_DIR / "Table3_block_bootstrap_skill.csv", index=False, encoding="utf-8-sig")

    # uncertainty subgroups (Table 4 / S10) for the selected layer
    urows = []
    for task in tasks:
        k = task["key"]; sub = test_pred
        y, p = sub[f"obs_{k}"].to_numpy(float), sub[f"pred_P3F_{k}"].to_numpy(float)
        thr = sub[f"event_threshold_{k}"].iloc[0]
        ph = C.phase_group(sub); tr = sub["transition_recent_1h"].gt(0).to_numpy(); ad = sub["ad_class"].astype(str).eq("in-domain").to_numpy()
        for gtype, gname, mask in [("all", "all", np.ones(len(sub), bool)), ("phase", "phase 1", ph == 1), ("phase", "phase 2", ph == 2),
                                   ("transition", "stable", ~tr), ("transition", "recent transition", tr),
                                   ("concentration", "below training Q90", y < thr), ("concentration", "at/above training Q90", y >= thr),
                                   ("AD", "in-domain", ad), ("AD", "borderline/far-OOD", ~ad)]:
            lo50, hi50, lo90, hi90 = (sub[f"{c}_{k}"].to_numpy(float)[mask] for c in ("lo50", "hi50", "lo90", "hi90"))
            yy = y[mask]
            urows.append({"target": task["target"], "horizon": task["horizon"], "group_type": gtype, "group": gname, "n": int(mask.sum()),
                          "mae": mean_absolute_error(yy, p[mask]), "coverage50": np.mean((yy >= lo50) & (yy <= hi50)),
                          "coverage90": np.mean((yy >= lo90) & (yy <= hi90)), "width50": np.mean(hi50 - lo50), "width90": np.mean(hi90 - lo90)})
    pd.DataFrame(urows).to_csv(C.TABLE_DIR / "Table5_uncertainty_subgroups.csv", index=False, encoding="utf-8-sig")

    # volatility-stratified four-hour skill (Fig. 3e)
    vrows = []
    for target in C.TARGETS:
        task = next(t for t in tasks if t["target"] == target and t["horizon"] == "4h")
        vol = f"{target}_roll36_std"
        edges = np.unique(np.quantile(train[vol], np.linspace(0, 1, 6)))
        bins = pd.cut(test[vol], bins=edges, include_lowest=True, duplicates="drop")
        for phase in [1, 2]:
            for b in bins.cat.categories:
                mask = bins.eq(b).to_numpy() & test["PROCESSPHASE_INLET"].round().eq(phase).to_numpy()
                if mask.sum() < 50:
                    continue
                y = test[task["y"]].to_numpy()[mask]; p = test_pred[f"pred_P3F_{task['key']}"].to_numpy()[mask]; pers = test[target].to_numpy()[mask]
                vrows.append({"target": target, "phase": phase, "volatility_bin": str(b), "volatility_median": float(test.loc[mask, vol].median()),
                              "n": int(mask.sum()), "skill_mae": metrics(y, p, pers)["skill_mae"]})
    pd.DataFrame(vrows).to_csv(C.TABLE_DIR / "Fig03_volatility_skill.csv", index=False, encoding="utf-8-sig")

    # compute profile: P3F counted as point model + four quantile heads
    comp = pd.DataFrame(compute_rows)
    comp.to_csv(C.TABLE_DIR / "TableS5_p3f_v2_compute_by_task.csv", index=False)
    old_comp = pd.read_csv(V1_DIR / "TableS_compute_profile.csv")
    old_comp = old_comp[old_comp.model != "P3F"]
    new_row = {"model": "P3F", "fit_seconds": comp.fit_point_s.sum() + comp.fit_heads_s.sum(),
               "latency_ms_per_1000": comp.latency_point_ms_per_1000.sum() + comp.latency_heads_ms_per_1000.sum(),
               "parameters_proxy": int(comp.leaves_point.sum() + comp.leaves_heads.sum()), "feature_count": len(features)}
    pd.concat([old_comp, pd.DataFrame([new_row])], ignore_index=True).to_csv(C.TABLE_DIR / "TableS_compute_profile.csv", index=False, encoding="utf-8-sig")

    # weekly four-hour coverage and weekly rolling update of the selected layer (Table S14)
    weeks = ((test.index - test.index.min()) / pd.Timedelta(days=7)).astype(int).to_numpy()
    wrows, rrows = [], []
    both = pd.concat([cal, test])
    for task in tasks:
        k = task["key"]; steps = task["steps"]
        y = test_pred[f"obs_{k}"].to_numpy(float)
        thr = test_pred[f"event_threshold_{k}"].iloc[0]; high = y >= thr
        fixed = (y >= test_pred[f"lo90_{k}"].to_numpy()) & (y <= test_pred[f"hi90_{k}"].to_numpy())
        v1 = (y >= test_pred[f"lo90_v1_{k}"].to_numpy()) & (y <= test_pred[f"hi90_v1_{k}"].to_numpy())
        # rolling: recompute the selected conformal correction weekly from the most recent n_cal scores with observed labels
        allp = pred.loc[cal_mask | test_mask]
        yb = allp[f"obs_{k}"].to_numpy(float); phb = C.phase_group(allp)
        if method.startswith("CQR"):
            score = np.maximum(allp[f"q05_{k}"].to_numpy(float) - yb, yb - allp[f"q95_{k}"].to_numpy(float))
            base_lo, base_hi = test_pred[f"q05_{k}"].to_numpy(float), test_pred[f"q95_{k}"].to_numpy(float)
        else:
            score = np.abs(yb - allp[f"pred_P3F_{k}"].to_numpy(float))
            base_lo = base_hi = test_pred[f"pred_P3F_{k}"].to_numpy(float)
        label_time = allp.index + pd.Timedelta(minutes=10 * (steps + 1))
        avail = np.searchsorted(label_time.values, (test.index + pd.Timedelta(minutes=10)).values, side="right")
        n_window = int(cal_mask.sum())
        q = np.empty(len(test))
        pht = C.phase_group(test_pred)
        for w in np.unique(weeks):
            first = np.where(weeks == w)[0][0]
            mwin = avail[first]
            sl = slice(max(0, mwin - n_window), mwin)
            for g in ([None] if not method.endswith("phase") else [1, 2]):
                sel_w = (weeks == w) if g is None else (weeks == w) & (pht == g)
                sc = score[sl] if g is None else score[sl][phb[sl] == g]
                q[sel_w] = C.conformal_quantile(sc, 0.10)
        lo, hi = np.maximum(base_lo - q, 0), np.maximum(base_hi + q, 0)
        roll = (y >= lo) & (y <= hi)
        rrows.append({"target": task["target"], "horizon": task["horizon"], "fixed_coverage": fixed.mean(), "rolling_coverage": roll.mean(),
                      "fixed_width": np.mean(test_pred[f"hi90_{k}"] - test_pred[f"lo90_{k}"]), "rolling_width": np.mean(hi - lo),
                      "fixed_coverage_high": fixed[high].mean(), "rolling_coverage_high": roll[high].mean(),
                      "fixed_coverage_low": fixed[~high].mean(), "rolling_coverage_low": roll[~high].mean()})
        if task["horizon"] == "4h":
            wk = pd.DataFrame({"v2": fixed.astype(float), "v1": v1.astype(float), "rolling": roll.astype(float), "week": weeks}, index=test.index)
            agg = wk.groupby("week").agg(week_start=("v2", lambda s: s.index.min()), v2=("v2", "mean"), v1=("v1", "mean"), rolling=("rolling", "mean"))
            agg["target"] = task["target"]
            wrows.append(agg.reset_index())
    pd.DataFrame(rrows).to_csv(C.TABLE_DIR / "TableS14_rolling_recalibration.csv", index=False, encoding="utf-8-sig")
    pd.concat(wrows).to_csv(C.TABLE_DIR / "Fig06f_weekly_coverage_4h.csv", index=False, encoding="utf-8-sig")

    summary = pd.DataFrame(method_rows).groupby("method")[["cov90", "cov50", "width90", "scaled_is90", "cov90_high", "weeks_below_090"]].mean()
    print(summary.to_string(), flush=True)
    p3 = met[met.model == "P3F"]; lg = met[met.model == "LightGBM-direct"]; v1m = met[met.model == "P3F v1 (Huber)"]
    print(json.dumps({"mean_skill_P3F_v2": p3.test_skill_mae.mean(), "mean_skill_v1": v1m.test_skill_mae.mean(), "mean_skill_lgbm": lg.test_skill_mae.mean(),
                      "mean_rmse_skill_v2": p3.test_skill_rmse.mean(), "mean_r2_v2": p3.test_r2.mean()}, indent=1), flush=True)


if __name__ == "__main__":
    main()
