"""Paired weekly block-bootstrap contrasts for P3F v2 (Table S12) and paired coverage
contrasts between the v2 and v1 interval layers (Table S18).

All contrasts reuse one set of week draws (seed 20260809, 1000 resamples of the 26 test
weeks), so the mean over ten tasks is computed from the same draws.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p3f_b_common as C

N_BOOT = 1000
V1_DIR = C.TABLE_DIR / "v1_huber_split_conformal"


def week_groups(index):
    return ((index - index.min()) / pd.Timedelta(days=7)).astype(int).to_numpy()


def main() -> None:
    for name in ["TableS12_paired_skill_contrasts.csv"]:
        if (C.TABLE_DIR / name).exists() and not (V1_DIR / name).exists():
            shutil.copy2(C.TABLE_DIR / name, V1_DIR / name)
    pred = pd.read_parquet(C.NEW_PRED_PATH)
    test = pred.loc[pred["split"].eq("test")]
    abl = pd.read_parquet(C.PRED_DIR / "P3F_v2_ablation_test_predictions.parquet")
    g = week_groups(test.index); weeks = np.unique(g)
    draw = np.random.default_rng(C.SEED).integers(0, len(weeks), size=(N_BOOT, len(weeks)))
    W = np.zeros((len(weeks), len(test))); W[g, np.arange(len(test))] = 1  # week membership

    rows, store = [], {}
    comps = ["LightGBM-direct", "P3F v1 (Huber)", "XGBoost-delta", "No phase context", "Short history only", "Target history only", "Direct target"]
    for target in C.TARGETS:
        for h in C.HORIZON_ORDER:
            k = f"{target}_{C.suffix(h)}"
            y = test[f"obs_{k}"].to_numpy(float); full = test[f"pred_P3F_{k}"].to_numpy(float); pers = test[f"pred_Persistence_{k}"].to_numpy(float)
            ep = W @ np.abs(y - pers); ef = W @ np.abs(y - full)
            for name in comps:
                if name == "LightGBM-direct":
                    other = test[f"pred_LightGBMdirect_{k}"].to_numpy(float)
                elif name == "XGBoost-delta":
                    other = test[f"pred_XGBoostdelta_{k}"].to_numpy(float)
                elif name == "P3F v1 (Huber)":
                    other = test[f"pred_P3Fv1_{k}"].to_numpy(float)
                else:
                    sub = abl.loc[(abl.variant == name) & (abl.target == target) & (abl.horizon == h)]
                    other = sub.set_index(pd.DatetimeIndex(sub["date"])).reindex(test.index)["prediction"].to_numpy(float)
                eo = W @ np.abs(y - other)
                point = (1 - eo.sum() / ep.sum()) - (1 - ef.sum() / ep.sum())
                boot = (1 - eo[draw].sum(1) / ep[draw].sum(1)) - (1 - ef[draw].sum(1) / ep[draw].sum(1))
                store.setdefault(name, []).append(boot)
                rows.append({"comparison": f"{name} minus P3F", "target": target, "horizon": h, "skill_difference": point,
                             "ci_low": np.quantile(boot, 0.025), "ci_high": np.quantile(boot, 0.975), "fraction_resamples_above_zero": float(np.mean(boot > 0))})
    for name in comps:
        mean_boot = np.vstack(store[name]).mean(0)
        point = np.mean([r["skill_difference"] for r in rows if r["comparison"] == f"{name} minus P3F"])
        rows.append({"comparison": f"{name} minus P3F", "target": "mean of 10 tasks", "horizon": "all", "skill_difference": point,
                     "ci_low": np.quantile(mean_boot, 0.025), "ci_high": np.quantile(mean_boot, 0.975), "fraction_resamples_above_zero": float(np.mean(mean_boot > 0))})
    pd.DataFrame(rows).to_csv(C.TABLE_DIR / "TableS12_paired_skill_contrasts.csv", index=False, encoding="utf-8-sig")

    # coverage contrasts, v2 selected layer minus v1 fixed-width layer: all and high-state observations
    crow, cstore = [], {}
    for target in C.TARGETS:
        for h in C.HORIZON_ORDER:
            k = f"{target}_{C.suffix(h)}"
            y = test[f"obs_{k}"].to_numpy(float)
            high = (y >= test[f"event_threshold_{k}"].iloc[0]).astype(float)
            c2 = ((y >= test[f"lo90_{k}"]) & (y <= test[f"hi90_{k}"])).to_numpy(float)
            c1 = ((y >= test[f"lo90_v1_{k}"]) & (y <= test[f"hi90_v1_{k}"])).to_numpy(float)
            w2 = (test[f"hi90_{k}"] - test[f"lo90_{k}"]).to_numpy(float); w1 = (test[f"hi90_v1_{k}"] - test[f"lo90_v1_{k}"]).to_numpy(float)
            for sub, wt in [("all", np.ones(len(y))), ("at/above training Q90", high)]:
                n = W @ wt; a2 = W @ (c2 * wt); a1 = W @ (c1 * wt)
                point = a2.sum() / n.sum() - a1.sum() / n.sum()
                boot = a2[draw].sum(1) / n[draw].sum(1) - a1[draw].sum(1) / n[draw].sum(1)
                cstore.setdefault(sub, []).append(boot)
                crow.append({"stratum": sub, "target": target, "horizon": h, "coverage_v1": a1.sum() / n.sum(), "coverage_v2": a2.sum() / n.sum(),
                             "difference": point, "ci_low": np.quantile(boot, 0.025), "ci_high": np.quantile(boot, 0.975),
                             "width_ratio_v2_over_v1": float(np.sum(w2 * wt) / np.sum(w1 * wt))})
    for sub in ["all", "at/above training Q90"]:
        mb = np.vstack(cstore[sub]).mean(0)
        rs = [r for r in crow if r["stratum"] == sub]
        crow.append({"stratum": sub, "target": "mean of 10 tasks", "horizon": "all", "coverage_v1": np.mean([r["coverage_v1"] for r in rs]),
                     "coverage_v2": np.mean([r["coverage_v2"] for r in rs]), "difference": np.mean([r["difference"] for r in rs]),
                     "ci_low": np.quantile(mb, 0.025), "ci_high": np.quantile(mb, 0.975), "width_ratio_v2_over_v1": np.mean([r["width_ratio_v2_over_v1"] for r in rs])})
    pd.DataFrame(crow).to_csv(C.TABLE_DIR / "TableS18_paired_coverage_contrasts.csv", index=False, encoding="utf-8-sig")
    print(pd.DataFrame(rows).query("target == 'mean of 10 tasks'").to_string())
    print(pd.DataFrame(crow).query("target == 'mean of 10 tasks'").to_string())


if __name__ == "__main__":
    main()
