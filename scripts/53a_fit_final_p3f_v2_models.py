"""Fit the P3F v2 median (L1) models and quantile heads on the full training block and save them
with their fitting times. Reads training rows only; the L1 objective was fixed by stage 1 of
script 52, and the heads are needed by every candidate CQR layer."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p3f_b_common as C


def main() -> None:
    meta = C.load_meta(); features = meta["features"]; tasks = C.task_list(meta)
    data = pd.read_parquet(C.DATA_PATH)
    train = data.loc[data["split"].eq("train")]
    del data
    C.MODEL_DIR_B.mkdir(parents=True, exist_ok=True)
    log_path = C.MODEL_DIR_B / "fit_seconds.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else {}
    for task in tasks:
        k = task["key"]
        specs = [("point", C.booster("l1", C.SEED))] + [(n, C.booster("quantile", C.SEED, alpha=q)) for n, q in C.QUANTILES.items()]
        for name, m in specs:
            path = C.MODEL_DIR_B / f"P3F_{k}_{name}.joblib"
            if path.exists() and f"{k}_{name}" in log:
                continue
            t0 = time.perf_counter(); m.fit(train[features], train[task["delta"]]); log[f"{k}_{name}"] = time.perf_counter() - t0
            joblib.dump(m, path, compress=3)
            log_path.write_text(json.dumps(log, indent=1))
            print(k, name, round(log[f"{k}_{name}"], 1), flush=True)


if __name__ == "__main__":
    main()
