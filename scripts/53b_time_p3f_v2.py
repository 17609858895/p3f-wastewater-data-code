"""Uncontended fitting and inference timing of P3F v2 (ten median models + 40 quantile heads),
refitting on the training block without saving; updates the compute tables written by 53."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p3f_b_common as C


def main() -> None:
    meta = C.load_meta(); features = meta["features"]; tasks = C.task_list(meta)
    data = pd.read_parquet(C.DATA_PATH)
    train, test = data.loc[data["split"].eq("train")], data.loc[data["split"].eq("test")]
    X5 = test[features].iloc[:5000]
    rows = []
    for task in tasks:
        rec = {"task": task["key"]}
        models = {}
        for name in ["point", *C.QUANTILES]:
            m = C.booster("l1", C.SEED) if name == "point" else C.booster("quantile", C.SEED, alpha=C.QUANTILES[name])
            t0 = time.perf_counter(); m.fit(train[features], train[task["delta"]]); rec[f"fit_{name}_s"] = time.perf_counter() - t0
            models[name] = m
        t0 = time.perf_counter(); models["point"].predict(X5); rec["latency_point_ms_per_1000"] = (time.perf_counter() - t0) * 200  # 5000 rows -> ms per 1000
        t0 = time.perf_counter()
        for n in C.QUANTILES:
            models[n].predict(X5)
        rec["latency_heads_ms_per_1000"] = (time.perf_counter() - t0) * 200
        rec["leaves_point"] = int(sum(t["num_leaves"] for t in models["point"].booster_.dump_model()["tree_info"]))
        rec["leaves_heads"] = int(sum(sum(t["num_leaves"] for t in models[n].booster_.dump_model()["tree_info"]) for n in C.QUANTILES))
        rec["fit_point_s"] = rec["fit_point_s"]
        rec["fit_heads_s"] = sum(rec[f"fit_{n}_s"] for n in C.QUANTILES)
        rows.append(rec); print(rec, flush=True)
    comp = pd.DataFrame(rows)
    comp.to_csv(C.TABLE_DIR / "TableS5_p3f_v2_compute_by_task.csv", index=False)
    prof = pd.read_csv(C.TABLE_DIR / "TableS_compute_profile.csv")
    prof.loc[prof.model.eq("P3F"), ["fit_seconds", "latency_ms_per_1000", "parameters_proxy"]] = [
        comp.fit_point_s.sum() + comp.fit_heads_s.sum(), comp.latency_point_ms_per_1000.sum() + comp.latency_heads_ms_per_1000.sum(),
        int(comp.leaves_point.sum() + comp.leaves_heads.sum())]
    prof.to_csv(C.TABLE_DIR / "TableS_compute_profile.csv", index=False, encoding="utf-8-sig")
    print(prof.to_string())


if __name__ == "__main__":
    main()
