"""Refit P3F v2 (point model and the quantile heads used by the selected interval layer)
with three additional seeds; LightGBM-direct seed results are taken from the v1 seed table
(its data, features and settings are unchanged)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p3f_b_common as C

SEEDS = [20260810, 20260811, 20260812]
OUT = C.TABLE_DIR / "TableS15_seed_sensitivity_p3f_v2.csv"


def main() -> None:
    sel = json.loads(C.SELECTED_PATH.read_text(encoding="utf-8"))
    objective, method = sel["point_objective"], sel["interval_method"]
    meta = C.load_meta(); features = meta["features"]; tasks = C.task_list(meta)
    data = pd.read_parquet(C.DATA_PATH)
    train, cal, test = (data.loc[data["split"].eq(s)] for s in ("train", "calibration", "test"))
    pred = pd.read_parquet(C.NEW_PRED_PATH); tp = pred.loc[pred["split"].eq("test")]
    rows = pd.read_csv(OUT).to_dict("records") if OUT.exists() else []
    done = {(r["seed"], r["key"]) for r in rows}
    if not any(r["seed"] == C.SEED for r in rows):
        for task in tasks:
            k = task["key"]; y = tp[f"obs_{k}"].to_numpy(float)
            cov = (y >= tp[f"lo90_{k}"]) & (y <= tp[f"hi90_{k}"]); high = y >= tp[f"event_threshold_{k}"].iloc[0]
            rows.append({"seed": C.SEED, "key": k, "target": task["target"], "horizon": task["horizon"],
                         "skill_mae": C.mae_skill(y, tp[f"pred_P3F_{k}"].to_numpy(float), test[task["target"]].to_numpy(float)),
                         "coverage90": cov.mean(), "coverage90_high": cov[high].mean(), "width90": float(np.mean(tp[f"hi90_{k}"] - tp[f"lo90_{k}"]))})
    for seed in SEEDS:
        for task in tasks:
            k = task["key"]
            if (seed, k) in done:
                continue
            t0 = time.perf_counter()
            models = {"point": C.booster(objective, seed)}
            models["point"].fit(train[features], train[task["delta"]])
            if method.startswith("CQR"):
                for n in ["q05", "q95"]:
                    models[n] = C.booster("quantile", seed, alpha=C.QUANTILES[n]); models[n].fit(train[features], train[task["delta"]])
            def block(frame):
                cur = frame[task["target"]].to_numpy(float)
                out = {"point": np.maximum(cur + models["point"].predict(frame[features]), 0), "phase": C.phase_group(frame)}
                if "q05" in models:
                    h = np.sort(np.column_stack([cur + models[n].predict(frame[features]) for n in ["q05", "q95"]]), axis=1)
                    out["heads"] = {"q05": h[:, 0], "q95": h[:, 1]}
                return out
            pc = block(cal); pc["y"] = cal[task["y"]].to_numpy(float); pt = block(test)
            lo, hi, _ = C.calibrate(method, "90", pc, pt)
            y = test[task["y"]].to_numpy(float); cov = (y >= lo) & (y <= hi); high = y >= tp[f"event_threshold_{k}"].iloc[0]
            rows.append({"seed": seed, "key": k, "target": task["target"], "horizon": task["horizon"],
                         "skill_mae": C.mae_skill(y, pt["point"], test[task["target"]].to_numpy(float)),
                         "coverage90": cov.mean(), "coverage90_high": cov[high].mean(), "width90": float(np.mean(hi - lo)),
                         "fit_seconds": time.perf_counter() - t0})
            pd.DataFrame(rows).to_csv(OUT, index=False)
            print(seed, k, round(rows[-1]["skill_mae"], 4), round(rows[-1]["coverage90_high"], 3), flush=True)


if __name__ == "__main__":
    main()
