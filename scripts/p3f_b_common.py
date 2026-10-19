"""Shared components of the revised P3F: median (L1) point model, quantile heads and
conformal interval layers (split-conformal and conformalised quantile regression,
pooled or Mondrian by inlet phase at issuance)."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("P3F_ROOT", Path(__file__).resolve().parents[1]))
DATA_PATH = ROOT / "data" / "processed" / "forecast_10min_features.parquet"
META_PATH = ROOT / "data" / "processed" / "forecast_metadata.json"
TABLE_DIR = ROOT / "tables"
PRED_DIR = ROOT / "data" / "predictions"
MODEL_DIR_B = ROOT / "models" / "p3f_v2"
OLD_PRED_PATH = PRED_DIR / "P3F_calibration_test_predictions.parquet"
NEW_PRED_PATH = PRED_DIR / "P3F_v2_calibration_test_predictions.parquet"
PROTOCOL_PATH = TABLE_DIR / "B_selection_protocol.json"
SELECTED_PATH = TABLE_DIR / "B_selected_configuration.json"

TARGETS = ["T1_PO4", "T1_NH4"]
HORIZON_ORDER = ["0.5h", "1h", "2h", "4h", "6h"]
SEED = 20260809
PURGE = pd.Timedelta(hours=78)
FOLD_CAL_DAYS = 30
QUANTILES = {"q05": 0.05, "q25": 0.25, "q75": 0.75, "q95": 0.95}
LEVELS = {"50": ("q25", "q75", 0.50), "90": ("q05", "q95", 0.10)}
INTERVAL_METHODS = ["SC", "SC-phase", "CQR", "CQR-phase"]

TREE_PARAMS = dict(
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
    n_jobs=-1,
    verbosity=-1,
)


def suffix(horizon: str) -> str:
    return horizon.replace(".", "p")


def load_meta() -> dict:
    return json.loads(META_PATH.read_text(encoding="utf-8"))


def task_list(meta: dict) -> list[dict]:
    return [
        {
            "target": t,
            "horizon": h,
            "steps": meta["horizons"][h],
            "y": f"y_{t}_{suffix(h)}",
            "delta": f"delta_{t}_{suffix(h)}",
            "key": f"{t}_{suffix(h)}",
        }
        for t in TARGETS
        for h in HORIZON_ORDER
    ]


def booster(objective: str, seed: int = SEED, alpha: float | None = None) -> lgb.LGBMRegressor:
    """objective: 'huber' (v1, alpha=0.9 mg/L), 'l1' (median) or 'quantile' (needs alpha)."""
    params = dict(TREE_PARAMS, random_state=seed)
    if objective == "huber":
        params.update(objective="huber", alpha=0.9)
    elif objective == "l1":
        params.update(objective="regression_l1")
    elif objective == "quantile":
        params.update(objective="quantile", alpha=float(alpha))
    else:
        raise ValueError(objective)
    return lgb.LGBMRegressor(**params)


def phase_group(frame: pd.DataFrame) -> np.ndarray:
    """Inlet phase observed at issuance, rounded to the nearest code (1 or 2)."""
    return np.clip(frame["PROCESSPHASE_INLET"].round().to_numpy(), 1, 2).astype(int)


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample split-conformal quantile: the ceil((n+1)(1-alpha))-th smallest score."""
    s = np.sort(np.asarray(scores, dtype=float))
    n = len(s)
    k = int(math.ceil((n + 1) * (1 - alpha)))
    return float(s[min(k, n) - 1])


def mae_skill(y, p, pers) -> float:
    return 1.0 - np.mean(np.abs(y - p)) / np.mean(np.abs(y - pers))


def interval_score(y, lo, hi, alpha: float) -> np.ndarray:
    y, lo, hi = (np.asarray(v, dtype=float) for v in (y, lo, hi))
    return (hi - lo) + (2 / alpha) * np.maximum(lo - y, 0) + (2 / alpha) * np.maximum(y - hi, 0)


def sort_heads(heads: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Rearrange the four quantile heads so that q05 <= q25 <= q75 <= q95 in every row."""
    names = ["q05", "q25", "q75", "q95"]
    stacked = np.sort(np.column_stack([heads[n] for n in names]), axis=1)
    return {n: stacked[:, i] for i, n in enumerate(names)}


def calibrate(method: str, level: str, cal: dict, new: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (lo, hi, info) on the `new` block.

    cal/new are dicts with arrays: y (calibration only), point, heads (sorted dict), phase.
    Bounds are absolute concentrations; lower limits are truncated at zero.
    """
    lo_name, hi_name, alpha = LEVELS[level]
    groups = [None] if not method.endswith("phase") else [1, 2]
    lo = np.empty(len(new["point"]))
    hi = np.empty(len(new["point"]))
    info = {}
    for g in groups:
        cmask = np.ones(len(cal["point"]), bool) if g is None else cal["phase"] == g
        nmask = np.ones(len(new["point"]), bool) if g is None else new["phase"] == g
        if method.startswith("SC"):
            q = conformal_quantile(np.abs(cal["y"][cmask] - cal["point"][cmask]), alpha)
            lo[nmask] = new["point"][nmask] - q
            hi[nmask] = new["point"][nmask] + q
        else:  # CQR (Romano et al., 2019)
            clo, chi = cal["heads"][lo_name][cmask], cal["heads"][hi_name][cmask]
            scores = np.maximum(clo - cal["y"][cmask], cal["y"][cmask] - chi)
            q = conformal_quantile(scores, alpha)
            lo[nmask] = new["heads"][lo_name][nmask] - q
            hi[nmask] = new["heads"][hi_name][nmask] + q
        info[f"Q_{'all' if g is None else 'phase' + str(g)}"] = q
        info[f"n_cal_{'all' if g is None else 'phase' + str(g)}"] = int(cmask.sum())
    lo = np.maximum(lo, 0.0)
    hi = np.maximum(hi, lo)
    return lo, hi, info


def raw_quantile_interval(level: str, new: dict) -> tuple[np.ndarray, np.ndarray]:
    lo_name, hi_name, _ = LEVELS[level]
    lo = np.maximum(new["heads"][lo_name], 0.0)
    return lo, np.maximum(new["heads"][hi_name], lo)


def fit_task_models(train: pd.DataFrame, features: list[str], task: dict, point_objective: str,
                    seed: int, with_heads: bool = True, label: str = "delta") -> dict:
    """Fit the point model and (optionally) the four quantile heads on the increment target."""
    col = task[label]
    models = {"point": booster(point_objective, seed)}
    models["point"].fit(train[features], train[col])
    if with_heads:
        for name, q in QUANTILES.items():
            m = booster("quantile", seed, alpha=q)
            m.fit(train[features], train[col])
            models[name] = m
    return models


def predict_block(models: dict, frame: pd.DataFrame, features: list[str], task: dict, label: str = "delta") -> dict:
    base = frame[task["target"]].to_numpy(float) if label == "delta" else 0.0
    out = {"point": np.maximum(base + models["point"].predict(frame[features]), 0.0)}
    heads = {n: base + models[n].predict(frame[features]) for n in QUANTILES if n in models}
    if heads:
        out["heads"] = sort_heads(heads)
    out["phase"] = phase_group(frame)
    return out
