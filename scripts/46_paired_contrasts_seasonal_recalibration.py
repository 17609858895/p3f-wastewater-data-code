"""Paired weekly block-bootstrap contrasts, horizon-aligned seasonal persistence,
and rolling residual recalibration, computed from the stored calibration/test
predictions and the 10-min feature table (no model refitting)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("P3F_ROOT", Path(__file__).resolve().parents[1]))
PRED = ROOT / "data" / "predictions" / "P3F_calibration_test_predictions.parquet"
ABL = ROOT / "data" / "predictions" / "P3F_ablation_test_predictions.parquet"
FEAT = ROOT / "data" / "processed" / "forecast_10min_features.parquet"
OUT = ROOT / "tables"

SEED = 20260809
N_BOOT = 1000
TARGETS = ["T1_PO4", "T1_NH4"]
HORIZONS = {"0.5h": ("0p5h", 3), "1h": ("1h", 6), "2h": ("2h", 12), "4h": ("4h", 24), "6h": ("6h", 36)}


def week_groups(index: pd.DatetimeIndex) -> np.ndarray:
    return ((index - index.min()) / pd.Timedelta(days=7)).astype(int).to_numpy()


def paired_bootstrap(index, y, pred_a, pred_b, pers, draw):
    """Skill(a) - Skill(b) with whole weeks resampled jointly for both models."""
    g = week_groups(index)
    weeks = np.unique(g)
    ea = np.array([np.abs(y[g == w] - pred_a[g == w]).sum() for w in weeks])
    eb = np.array([np.abs(y[g == w] - pred_b[g == w]).sum() for w in weeks])
    ep = np.array([np.abs(y[g == w] - pers[g == w]).sum() for w in weeks])
    point = (1 - ea.sum() / ep.sum()) - (1 - eb.sum() / ep.sum())
    boot = (1 - ea[draw].sum(1) / ep[draw].sum(1)) - (1 - eb[draw].sum(1) / ep[draw].sum(1))
    return point, boot


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pred = pd.read_parquet(PRED)
    test = pred.loc[pred["split"].eq("test")].copy()
    abl = pd.read_parquet(ABL)
    n_weeks = len(np.unique(week_groups(test.index)))
    rng = np.random.default_rng(SEED)
    draw = rng.integers(0, n_weeks, size=(N_BOOT, n_weeks))  # shared across tasks and contrasts

    # 1. Paired contrasts against full P3F (same resampled weeks for every task)
    contrasts = {"LightGBM-direct": None, "No phase context": None, "Short history only": None,
                 "Target history only": None, "Direct target": None}
    rows, task_boot = [], {c: [] for c in contrasts}
    for target in TARGETS:
        for horizon, (hk, _) in HORIZONS.items():
            k = f"{target}_{hk}"
            y = test[f"obs_{k}"].to_numpy(float)
            full = test[f"pred_P3F_{k}"].to_numpy(float)
            pers = test[f"pred_Persistence_{k}"].to_numpy(float)
            for name in contrasts:
                if name == "LightGBM-direct":
                    other = test[f"pred_LightGBMdirect_{k}"].to_numpy(float)
                else:
                    sub = abl.loc[(abl["variant"] == name) & (abl["target"] == target) & (abl["horizon"] == horizon)]
                    sub = sub.set_index(pd.DatetimeIndex(sub["date"])).reindex(test.index)
                    other = sub["prediction"].to_numpy(float)
                point, boot = paired_bootstrap(test.index, y, other, full, pers, draw)
                task_boot[name].append(boot)
                rows.append({"comparison": f"{name} minus P3F", "target": target, "horizon": horizon,
                             "skill_difference": point, "ci_low": np.quantile(boot, 0.025),
                             "ci_high": np.quantile(boot, 0.975), "fraction_resamples_above_zero": float(np.mean(boot > 0))})
    for name, boots in task_boot.items():
        stacked = np.vstack(boots)  # tasks x resamples
        mean_boot = stacked.mean(0)
        point = np.mean([r["skill_difference"] for r in rows if r["comparison"] == f"{name} minus P3F"])
        rows.append({"comparison": f"{name} minus P3F", "target": "mean of 10 tasks", "horizon": "all",
                     "skill_difference": point, "ci_low": np.quantile(mean_boot, 0.025),
                     "ci_high": np.quantile(mean_boot, 0.975), "fraction_resamples_above_zero": float(np.mean(mean_boot > 0))})
    pd.DataFrame(rows).to_csv(OUT / "TableS12_paired_skill_contrasts.csv", index=False, encoding="utf-8-sig")

    # 2. Horizon-aligned seasonal persistence y(t + h - 24 h)
    feat = pd.read_parquet(FEAT, columns=TARGETS + [f"y_{t}_{hk}" for t in TARGETS for hk, _ in HORIZONS.values()] + [f"{t}_lag144" for t in TARGETS] + ["split"])
    ft = feat.loc[feat["split"].eq("test")]
    srows = []
    for target in TARGETS:
        for horizon, (hk, steps) in HORIZONS.items():
            y = ft[f"y_{target}_{hk}"].to_numpy(float)
            pers = ft[target].to_numpy(float)
            aligned = feat[target].shift(144 - steps).loc[ft.index].to_numpy(float)
            issuance = ft[f"{target}_lag144"].to_numpy(float)
            mp = np.mean(np.abs(y - pers))
            srows.append({"target": target, "horizon": horizon,
                          "skill_issuance_minus_24h": 1 - np.mean(np.abs(y - issuance)) / mp,
                          "skill_target_minus_24h": 1 - np.mean(np.abs(y - aligned)) / mp})
    pd.DataFrame(srows).to_csv(OUT / "TableS13_seasonal_persistence_alignment.csv", index=False, encoding="utf-8-sig")

    # 3. Rolling recalibration: q90 from the most recent residuals whose labels are observed at issuance
    rrows, weekly_rows = [], []
    cal = pred.loc[pred["split"].eq("calibration")]
    for target in TARGETS:
        for horizon, (hk, steps) in HORIZONS.items():
            k = f"{target}_{hk}"
            both = pd.concat([cal, test])
            resid = (both[f"obs_{k}"] - both[f"pred_P3F_{k}"]).abs()
            label_time = both.index + pd.Timedelta(minutes=10 * (steps + 1))  # end of target bin
            n_window = int(cal.shape[0])
            y = test[f"obs_{k}"].to_numpy(float)
            p = test[f"pred_P3F_{k}"].to_numpy(float)
            avail_count = np.searchsorted(label_time.values, (test.index + pd.Timedelta(minutes=10)).values, side="right")
            r = resid.to_numpy(float)
            q = np.empty(len(test))
            # update weekly (start of each test week) to mimic a periodic recalibration schedule
            g = week_groups(test.index)
            for w in np.unique(g):
                first = np.where(g == w)[0][0]
                m = avail_count[first]
                window = r[max(0, m - n_window):m]
                q[g == w] = np.quantile(window, 0.90, method="higher")
            lo, hi = np.maximum(p - q, 0), p + q
            covered = (y >= lo) & (y <= hi)
            fixed_cov = (y >= test[f"lo90_{k}"].to_numpy()) & (y <= test[f"hi90_{k}"].to_numpy())
            thr = test[f"event_threshold_{k}"].iloc[0]
            high = y >= thr
            rrows.append({"target": target, "horizon": horizon,
                          "fixed_coverage": fixed_cov.mean(), "rolling_coverage": covered.mean(),
                          "fixed_width": np.mean(test[f"hi90_{k}"] - test[f"lo90_{k}"]), "rolling_width": np.mean(hi - lo),
                          "fixed_coverage_high": fixed_cov[high].mean(), "rolling_coverage_high": covered[high].mean(),
                          "fixed_coverage_low": fixed_cov[~high].mean(), "rolling_coverage_low": covered[~high].mean()})
            if horizon == "4h":
                wk = pd.DataFrame({"fixed": fixed_cov.astype(float), "rolling": covered.astype(float), "week": g}, index=test.index)
                agg = wk.groupby("week").agg(week_start=("fixed", lambda s: s.index.min()), fixed=("fixed", "mean"), rolling=("rolling", "mean"))
                agg["target"] = target
                weekly_rows.append(agg.reset_index())
    pd.DataFrame(rrows).to_csv(OUT / "TableS14_rolling_recalibration.csv", index=False, encoding="utf-8-sig")
    pd.concat(weekly_rows).to_csv(OUT / "Fig06f_weekly_coverage_4h.csv", index=False, encoding="utf-8-sig")
    print(json.dumps({"status": "ok", "weeks": int(n_weeks)}))


if __name__ == "__main__":
    main()
