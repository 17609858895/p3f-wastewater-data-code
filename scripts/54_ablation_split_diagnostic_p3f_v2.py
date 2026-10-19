"""Matched ablations and the four-hour split-protocol diagnostic for P3F v2.

Each ablation refits the selected point objective (and the quantile heads needed by the
selected interval layer) with the same seed and tree settings, recalibrates on the
calibration block and is evaluated on the future-test block.
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p3f_b_common as C

spec = importlib.util.spec_from_file_location("train_v1", Path(__file__).resolve().parent / "03_train_p3f_models.py")
v1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v1)
V1_DIR = C.TABLE_DIR / "v1_huber_split_conformal"


def main() -> None:
    sel = json.loads(C.SELECTED_PATH.read_text(encoding="utf-8"))
    objective, method = sel["point_objective"], sel["interval_method"]
    for name in ["Table4_ablation_metrics.csv", "Fig03_split_protocol_diagnostic.csv"]:
        if (C.TABLE_DIR / name).exists() and not (V1_DIR / name).exists():
            shutil.copy2(C.TABLE_DIR / name, V1_DIR / name)
    meta = C.load_meta(); features = meta["features"]; tasks = C.task_list(meta)
    sets = v1.select_feature_sets(features)
    data = pd.read_parquet(C.DATA_PATH)
    train, cal, test = (data.loc[data["split"].eq(s)] for s in ("train", "calibration", "test"))
    pred = pd.read_parquet(C.NEW_PRED_PATH)
    tp = pred.loc[pred["split"].eq("test")]
    heads_needed = {"q05": 0.05, "q95": 0.95} if method.startswith("CQR") else {}

    rows, preds = [], []
    for task in tasks:
        k = task["key"]
        y, cur = test[task["y"]].to_numpy(float), test[task["target"]].to_numpy(float)
        thr = tp[f"event_threshold_{k}"].iloc[0]; high = y >= thr
        full = tp[f"pred_P3F_{k}"].to_numpy(float)
        cov = (y >= tp[f"lo90_{k}"]) & (y <= tp[f"hi90_{k}"])
        m = v1.metrics(y, full, cur)
        rows.append({"variant": "Full P3F", **task, **m, "fit_seconds": np.nan, "feature_count": len(features),
                     "coverage90": cov.mean(), "width90": float(np.mean(tp[f"hi90_{k}"] - tp[f"lo90_{k}"])), "coverage90_high": cov[high].mean()})
        for variant, feats, label in [("No phase context", sets["no_phase"], "delta"), ("Short history only", sets["short_history"], "delta"),
                                      ("Target history only", sets["target_history_only"], "delta"), ("Direct target", sets["full"], "y")]:
            t0 = time.perf_counter()
            point = C.booster(objective, C.SEED); point.fit(train[feats], train[task[label]])
            fit_s = time.perf_counter() - t0
            models = {"point": point}
            for n, q in heads_needed.items():
                hm = C.booster("quantile", C.SEED, alpha=q); hm.fit(train[feats], train[task[label]]); models[n] = hm
            base_c = cal[task["target"]].to_numpy(float) if label == "delta" else 0.0
            base_t = cur if label == "delta" else 0.0
            pc = {"y": cal[task["y"]].to_numpy(float), "point": np.maximum(base_c + point.predict(cal[feats]), 0), "phase": C.phase_group(cal)}
            pt = {"point": np.maximum(base_t + point.predict(test[feats]), 0), "phase": C.phase_group(test)}
            if heads_needed:
                hc = {n: base_c + models[n].predict(cal[feats]) for n in heads_needed}
                ht = {n: base_t + models[n].predict(test[feats]) for n in heads_needed}
                pc["heads"] = dict(zip(["q05", "q95"], np.sort(np.column_stack([hc["q05"], hc["q95"]]), axis=1).T))
                pt["heads"] = dict(zip(["q05", "q95"], np.sort(np.column_stack([ht["q05"], ht["q95"]]), axis=1).T))
            lo, hi, _ = C.calibrate(method, "90", pc, pt)
            cv = (y >= lo) & (y <= hi)
            m = v1.metrics(y, pt["point"], cur)
            rows.append({"variant": variant, **task, **m, "fit_seconds": fit_s, "fit_seconds_all": time.perf_counter() - t0,
                         "feature_count": len(feats), "coverage90": cv.mean(), "width90": float(np.mean(hi - lo)), "coverage90_high": cv[high].mean()})
            preds.append(pd.DataFrame({"date": test.index, "variant": variant, "target": task["target"], "horizon": task["horizon"],
                                       "observed": y, "prediction": pt["point"], "phase": test["PROCESSPHASE_INLET"].to_numpy(),
                                       "transition": test["transition_inlet_recent_1h"].to_numpy()}))
            print(f"{variant:20s} {k}: skill={m['skill_mae']:.4f} cov90={cv.mean():.3f} high={cv[high].mean():.3f} fit={fit_s:.1f}s", flush=True)
            pd.DataFrame(rows).to_csv(C.TABLE_DIR / "Table4_ablation_metrics.csv", index=False, encoding="utf-8-sig")
    pd.concat(preds, ignore_index=True).to_parquet(C.PRED_DIR / "P3F_v2_ablation_test_predictions.parquet", compression="zstd")

    # four-hour split-protocol diagnostic (Fig. 3d) with the v2 point model
    srows = []
    cv = pd.read_csv(C.TABLE_DIR / "TableS16_validation_selection.csv")
    for task in [t for t in tasks if t["horizon"] == "4h"]:
        target = task["target"]
        for r in cv[(cv.stage == "point") & (cv.key == task["key"])].itertuples():
            srows.append({"protocol": "Purged rolling", "repeat": int(r.fold), "target": target, "r2": getattr(r, f"r2_{objective}"), "skill_mae": getattr(r, f"skill_{objective}")})
        for rep, rs in enumerate([17, 29, 41, 53], start=1):
            itr, iva = train_test_split(np.arange(len(train)), test_size=0.2, random_state=rs, shuffle=True)
            tr, va = train.iloc[itr], train.iloc[iva]
            mdl = C.booster(objective, C.SEED + rs); mdl.fit(tr[features], tr[task["delta"]])
            p = np.maximum(va[target].to_numpy() + mdl.predict(va[features]), 0)
            mm = v1.metrics(va[task["y"]].to_numpy(), p, va[target].to_numpy())
            srows.append({"protocol": "Random rows", "repeat": rep, "target": target, "r2": mm["r2"], "skill_mae": mm["skill_mae"]})
        for rep, vs in enumerate(["2022-07-01", "2022-08-01", "2022-09-01", "2022-10-01"], start=1):
            vs = pd.Timestamp(vs + " 00:00:00+00:00")
            ve = min(vs + pd.offsets.MonthEnd(1) + pd.Timedelta(hours=23, minutes=50), train.index.max())
            tr = train.loc[train.index < vs]; va = train.loc[(train.index >= vs) & (train.index <= ve)]
            mdl = C.booster(objective, C.SEED + rep + 100); mdl.fit(tr[features], tr[task["delta"]])
            p = np.maximum(va[target].to_numpy() + mdl.predict(va[features]), 0)
            mm = v1.metrics(va[task["y"]].to_numpy(), p, va[target].to_numpy())
            srows.append({"protocol": "Chronological, no purge", "repeat": rep, "target": target, "r2": mm["r2"], "skill_mae": mm["skill_mae"]})
        print("split diagnostic", target, flush=True)
    pd.DataFrame(srows).to_csv(C.TABLE_DIR / "Fig03_split_protocol_diagnostic.csv", index=False, encoding="utf-8-sig")
    print("done", flush=True)


if __name__ == "__main__":
    main()
