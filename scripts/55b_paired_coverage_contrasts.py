"""Paired weekly block-bootstrap coverage contrasts (Table S18), same week draws as script 55.

Comparisons: P3F (CQR by phase) minus P3F-Huber (objective and interval layer both differ), and
P3F minus fixed-width split-conformal intervals around the same median forecast (interval layer
only). Strata: all observations; observed future concentration at or above the training-label
90th percentile (retrospective); current concentration at issuance at or above the same threshold
(available at issuance)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p3f_b_common as C

N_BOOT = 1000


def main() -> None:
    pred = pd.read_parquet(C.NEW_PRED_PATH); test = pred.loc[pred["split"].eq("test")]
    g = ((test.index - test.index.min()) / pd.Timedelta(days=7)).astype(int).to_numpy(); weeks = np.unique(g)
    draw = np.random.default_rng(C.SEED).integers(0, len(weeks), size=(N_BOOT, len(weeks)))
    W = np.zeros((len(weeks), len(test))); W[g, np.arange(len(test))] = 1
    refs = {"P3F-Huber": ("lo90_v1_{k}", "hi90_v1_{k}"), "Median, fixed width": ("lo90_SC_{k}", "hi90_SC_{k}")}
    rows, store = [], {}
    for ref, (lo_t, hi_t) in refs.items():
        for target in C.TARGETS:
            for h in C.HORIZON_ORDER:
                k = f"{target}_{C.suffix(h)}"
                y = test[f"obs_{k}"].to_numpy(float); thr = test[f"event_threshold_{k}"].iloc[0]
                c2 = ((y >= test[f"lo90_{k}"]) & (y <= test[f"hi90_{k}"])).to_numpy(float)
                c1 = ((y >= test[lo_t.format(k=k)]) & (y <= test[hi_t.format(k=k)])).to_numpy(float)
                w2 = (test[f"hi90_{k}"] - test[f"lo90_{k}"]).to_numpy(float); w1 = (test[hi_t.format(k=k)] - test[lo_t.format(k=k)]).to_numpy(float)
                strata = {"all": np.ones(len(y)), "at/above training Q90": (y >= thr).astype(float),
                          "current at/above training Q90": (test[f"current_{target}"].to_numpy(float) >= thr).astype(float)}
                for st, wt in strata.items():
                    n = W @ wt; a2 = W @ (c2 * wt); a1 = W @ (c1 * wt)
                    boot = a2[draw].sum(1) / np.maximum(n[draw].sum(1), 1) - a1[draw].sum(1) / np.maximum(n[draw].sum(1), 1)
                    store.setdefault((ref, st), []).append(boot)
                    rows.append({"reference": ref, "stratum": st, "target": target, "horizon": h, "n": int(wt.sum()),
                                 "coverage_v1": a1.sum() / n.sum(), "coverage_v2": a2.sum() / n.sum(), "difference": a2.sum() / n.sum() - a1.sum() / n.sum(),
                                 "ci_low": np.quantile(boot, 0.025), "ci_high": np.quantile(boot, 0.975),
                                 "width_ratio_v2_over_v1": float(np.sum(w2 * wt) / np.sum(w1 * wt))})
        for st in ["all", "at/above training Q90", "current at/above training Q90"]:
            mb = np.vstack(store[(ref, st)]).mean(0)
            rs = [r for r in rows if r["reference"] == ref and r["stratum"] == st and r["target"] != "mean of 10 tasks"]
            rows.append({"reference": ref, "stratum": st, "target": "mean of 10 tasks", "horizon": "all", "n": int(np.mean([r["n"] for r in rs])),
                         "coverage_v1": np.mean([r["coverage_v1"] for r in rs]), "coverage_v2": np.mean([r["coverage_v2"] for r in rs]),
                         "difference": np.mean([r["difference"] for r in rs]), "ci_low": np.quantile(mb, 0.025), "ci_high": np.quantile(mb, 0.975),
                         "width_ratio_v2_over_v1": np.mean([r["width_ratio_v2_over_v1"] for r in rs])})
    out = pd.DataFrame(rows)
    out.to_csv(C.TABLE_DIR / "TableS18_paired_coverage_contrasts.csv", index=False, encoding="utf-8-sig")
    print(out[out.target == "mean of 10 tasks"].round(3).to_string())


if __name__ == "__main__":
    main()
