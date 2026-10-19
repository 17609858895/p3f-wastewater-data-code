"""Diagnostics for the v1 -> v2 point-objective change: share of v1 training residuals inside
the Huber quadratic region (|r| <= 0.9 mg/L), and signed errors of v1/v2 forecasts in the
high-concentration stratum of the future-test block."""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p3f_b_common as C


def main() -> None:
    meta = C.load_meta(); features = meta["features"]; tasks = C.task_list(meta)
    data = pd.read_parquet(C.DATA_PATH)
    train = data.loc[data["split"].eq("train")]
    pred = pd.read_parquet(C.NEW_PRED_PATH); test = pred.loc[pred["split"].eq("test")]
    rows = []
    for task in tasks:
        k = task["key"]
        m1 = joblib.load(C.ROOT / "models" / "p3f" / f"P3F_{k}.joblib")
        r1 = train[task["delta"]].to_numpy(float) - m1.predict(train[features])
        y = test[f"obs_{k}"].to_numpy(float); high = y >= test[f"event_threshold_{k}"].iloc[0]
        e1 = y - test[f"pred_P3Fv1_{k}"].to_numpy(float); e2 = y - test[f"pred_P3F_{k}"].to_numpy(float)
        rows.append({"target": task["target"], "horizon": task["horizon"],
                     "share_train_increments_abs_le_0p9": float(np.mean(np.abs(train[task["delta"]]) <= 0.9)),
                     "share_v1_train_residuals_abs_le_0p9": float(np.mean(np.abs(r1) <= 0.9)),
                     "v1_test_mean_signed_error_high": float(e1[high].mean()), "v2_test_mean_signed_error_high": float(e2[high].mean()),
                     "v1_test_median_signed_error_all": float(np.median(e1)), "v2_test_median_signed_error_all": float(np.median(e2)),
                     "v1_test_mean_signed_error_all": float(e1.mean()), "v2_test_mean_signed_error_all": float(e2.mean())})
    out = pd.DataFrame(rows)
    out.to_csv(C.TABLE_DIR / "TableS19_loss_alignment_diagnostics.csv", index=False)
    print(out.round(3).to_string())


if __name__ == "__main__":
    main()
