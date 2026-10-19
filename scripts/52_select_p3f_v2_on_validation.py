"""Select the revised P3F point objective and interval layer on the four rolling-validation
folds (July-October 2022) using rules written to tables/B_selection_protocol.json before any
fold model is fitted. Only training-pool data are read; calibration and test rows are dropped
on load.

Outputs: tables/TableS16_validation_selection.csv (fold x task x candidate metrics),
tables/TableS16_validation_selection_summary.csv and tables/B_selected_configuration.json.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p3f_b_common as C

PROTOCOL = {
    "purpose": "Pre-specified selection of the revised P3F point objective and interval layer.",
    "data_used": "Training/rolling-validation pool only (2021-08-03 to 2022-10-28). Calibration and test blocks are not read by this script.",
    "point_candidates": {
        "huber": "LightGBM Huber, transition threshold 0.9 mg/L (P3F v1)",
        "l1": "LightGBM L1 (conditional median) on the same increment target",
    },
    "tree_settings": C.TREE_PARAMS,
    "quantile_heads": "LightGBM quantile objective at 0.05, 0.25, 0.75 and 0.95 on the increment target, same tree settings; heads sorted row-wise to remove crossing.",
    "interval_candidates": {
        "SC": "split-conformal absolute residual around the point forecast, one quantile per task",
        "SC-phase": "split-conformal absolute residual, separate quantile for inlet phase 1 and 2 at issuance (Mondrian)",
        "CQR": "conformalised quantile regression (Romano et al., 2019), one correction per task",
        "CQR-phase": "conformalised quantile regression, separate correction for inlet phase 1 and 2 at issuance (Mondrian)",
    },
    "reference_only": {"QR": "raw quantile heads without conformal correction (not eligible)"},
    "conformal_quantile": "ceil((n+1)(1-alpha))-th smallest calibration score; lower limits truncated at zero",
    "fold_design": {
        "point_selection": "each fold's model fitted on the full fold-training window and evaluated on the validation month (as in Table S6)",
        "interval_selection": "fold-training window split into an inner training part and the final 30 days as fold calibration, separated by a 78 h purge; validation month as above",
        "seeds": "base seed 20260809 + fold number",
    },
    "rule_point": "Select the objective with the higher MAE skill averaged over 4 folds x 10 tasks.",
    "rule_interval": "Among SC, SC-phase, CQR and CQR-phase (built on the selected objective), keep candidates whose mean 90 % validation coverage over folds and tasks is at least 0.88; select the one with the lowest mean 90 % interval score (Gneiting and Raftery, 2007) divided by the fold-task persistence MAE. The selected method is used for both 50 % and 90 % intervals.",
    "test_rule": "The future-test block is evaluated only after B_selected_configuration.json is written. Non-selected candidates are also reported on the test block without altering the selection.",
    "disclosure": "The future-test block of P3F v1 was examined before this protocol was written (diagnosis of high-concentration and summer coverage shortfalls).",
}


def write_protocol() -> None:
    if C.PROTOCOL_PATH.exists():
        old = json.loads(C.PROTOCOL_PATH.read_text(encoding="utf-8"))
        assert {k: v for k, v in old.items() if k != "written_utc"} == json.loads(json.dumps(PROTOCOL)), "protocol changed after it was written"
        return
    C.PROTOCOL_PATH.write_text(json.dumps({"written_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), **PROTOCOL}, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    write_protocol()
    meta = C.load_meta()
    features = meta["features"]
    tasks = C.task_list(meta)
    data = pd.read_parquet(C.DATA_PATH)
    data = data.loc[data["split"].eq("train")]  # calibration and test rows are never loaded here
    out_rows = C.TABLE_DIR / "TableS16_validation_selection.csv"
    rows = pd.read_csv(out_rows).to_dict("records") if out_rows.exists() else []
    done = {(r["fold"], r["key"], r["stage"]) for r in rows}

    # Stage 1: point objective on full fold-training windows
    for fold in meta["rolling_folds"]:
        f = fold["fold"]
        tr = data.loc[(data.index >= pd.Timestamp(fold["train_start"])) & (data.index <= pd.Timestamp(fold["train_end"]))]
        va = data.loc[(data.index >= pd.Timestamp(fold["valid_start"])) & (data.index <= pd.Timestamp(fold["valid_end"]))]
        for task in tasks:
            if (f, task["key"], "point") in done:
                continue
            y, cur = va[task["y"]].to_numpy(float), va[task["target"]].to_numpy(float)
            rec = {"fold": f, "key": task["key"], "target": task["target"], "horizon": task["horizon"], "stage": "point"}
            for obj in ["huber", "l1"]:
                t0 = time.perf_counter()
                m = C.booster(obj, C.SEED + f)
                m.fit(tr[features], tr[task["delta"]])
                p = np.maximum(cur + m.predict(va[features]), 0)
                rec[f"skill_{obj}"] = C.mae_skill(y, p, cur)
                rec[f"mae_{obj}"] = float(np.mean(np.abs(y - p)))
                rec[f"r2_{obj}"] = float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))
                rec[f"fit_s_{obj}"] = time.perf_counter() - t0
            rows.append(rec)
            pd.DataFrame(rows).to_csv(out_rows, index=False)
            print(f"point fold {f} {task['key']} huber={rec['skill_huber']:.4f} l1={rec['skill_l1']:.4f}", flush=True)

    df = pd.DataFrame(rows)
    pt = df[df.stage == "point"]
    mean_skill = {o: float(pt[f"skill_{o}"].mean()) for o in ["huber", "l1"]}
    objective = max(mean_skill, key=mean_skill.get)
    print("point selection", mean_skill, "->", objective, flush=True)

    # Stage 2: interval layer on inner-train / fold-calibration / validation month
    for fold in meta["rolling_folds"]:
        f = fold["fold"]
        tr_end = pd.Timestamp(fold["train_end"])
        cal_start = tr_end - pd.Timedelta(days=C.FOLD_CAL_DAYS)
        inner = data.loc[(data.index >= pd.Timestamp(fold["train_start"])) & (data.index <= cal_start - C.PURGE)]
        cal = data.loc[(data.index >= cal_start) & (data.index <= tr_end)]
        va = data.loc[(data.index >= pd.Timestamp(fold["valid_start"])) & (data.index <= pd.Timestamp(fold["valid_end"]))]
        for task in tasks:
            if (f, task["key"], "interval") in done:
                continue
            t0 = time.perf_counter()
            models = C.fit_task_models(inner, features, task, objective, C.SEED + f)
            pc = C.predict_block(models, cal, features, task)
            pc["y"] = cal[task["y"]].to_numpy(float)
            pv = C.predict_block(models, va, features, task)
            y, cur = va[task["y"]].to_numpy(float), va[task["target"]].to_numpy(float)
            pers_mae = float(np.mean(np.abs(y - cur)))
            rec = {"fold": f, "key": task["key"], "target": task["target"], "horizon": task["horizon"], "stage": "interval",
                   "n_inner": len(inner), "n_cal": len(cal), "n_valid": len(va), "skill_point": C.mae_skill(y, pv["point"], cur)}
            for method in C.INTERVAL_METHODS + ["QR"]:
                for level, (_, _, alpha) in C.LEVELS.items():
                    if method == "QR":
                        lo, hi = C.raw_quantile_interval(level, pv)
                    else:
                        lo, hi, _ = C.calibrate(method, level, pc, pv)
                    cov = (y >= lo) & (y <= hi)
                    rec[f"{method}_cov{level}"] = float(cov.mean())
                    rec[f"{method}_width{level}"] = float(np.mean(hi - lo))
                    rec[f"{method}_is{level}"] = float(np.mean(C.interval_score(y, lo, hi, alpha)) / pers_mae)
                    for g in (1, 2):
                        rec[f"{method}_cov{level}_phase{g}"] = float(cov[pv["phase"] == g].mean())
            rec["fit_s"] = time.perf_counter() - t0
            rows.append(rec)
            pd.DataFrame(rows).to_csv(out_rows, index=False)
            print(f"interval fold {f} {task['key']} " + " ".join(f"{m}={rec[m + '_cov90']:.3f}/{rec[m + '_is90']:.3f}" for m in C.INTERVAL_METHODS + ['QR']), flush=True)

    df = pd.DataFrame(rows)
    iv = df[df.stage == "interval"]
    summary = []
    for method in C.INTERVAL_METHODS + ["QR"]:
        summary.append({"method": method, "eligible": method != "QR",
                        "mean_cov90": iv[f"{method}_cov90"].mean(), "min_task_mean_cov90": iv.groupby("key")[f"{method}_cov90"].mean().min(),
                        "mean_cov50": iv[f"{method}_cov50"].mean(),
                        "mean_scaled_is90": iv[f"{method}_is90"].mean(), "mean_scaled_is50": iv[f"{method}_is50"].mean(),
                        "mean_width90": iv[f"{method}_width90"].mean(),
                        "mean_abs_phase_gap90": np.mean(np.abs(iv[f"{method}_cov90_phase1"] - iv[f"{method}_cov90_phase2"]))})
    summary = pd.DataFrame(summary)
    point_summary = pd.DataFrame([{"method": f"point:{o}", "mean_skill": v} for o, v in mean_skill.items()])
    pd.concat([point_summary, summary], ignore_index=True).to_csv(C.TABLE_DIR / "TableS16_validation_selection_summary.csv", index=False)
    eligible = summary[summary.eligible & (summary.mean_cov90 >= 0.88)]
    chosen = eligible.sort_values("mean_scaled_is90").iloc[0]["method"]
    C.SELECTED_PATH.write_text(json.dumps({
        "written_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "point_objective": objective, "point_mean_validation_skill": mean_skill,
        "interval_method": chosen,
        "interval_summary": summary.to_dict("records"),
    }, indent=2, default=float), encoding="utf-8")
    print("interval selection ->", chosen, flush=True)
    print(summary.to_string(), flush=True)



if __name__ == "__main__":
    main()
