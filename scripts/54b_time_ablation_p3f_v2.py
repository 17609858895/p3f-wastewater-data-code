"""Uncontended fitting time of the ablation median models (same conditions as script 53b);
updates fit_seconds in tables/Table4_ablation_metrics.csv. Predictions are not changed."""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p3f_b_common as C

spec = importlib.util.spec_from_file_location("train_v1", Path(__file__).resolve().parent / "03_train_p3f_models.py")
v1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v1)


def main() -> None:
    meta = C.load_meta(); features = meta["features"]; tasks = C.task_list(meta)
    sets = v1.select_feature_sets(features)
    data = pd.read_parquet(C.DATA_PATH); train = data.loc[data["split"].eq("train")]
    path = C.TABLE_DIR / "Table4_ablation_metrics.csv"
    ab = pd.read_csv(path)
    for task in tasks:
        for variant, feats, label in [("No phase context", sets["no_phase"], "delta"), ("Short history only", sets["short_history"], "delta"),
                                      ("Target history only", sets["target_history_only"], "delta"), ("Direct target", sets["full"], "y")]:
            m = C.booster("l1", C.SEED)
            t0 = time.perf_counter(); m.fit(train[feats], train[task[label]]); fs = time.perf_counter() - t0
            ab.loc[(ab.variant == variant) & (ab.key == task["key"]), "fit_seconds"] = fs
            print(variant, task["key"], round(fs, 2), flush=True)
    ab.to_csv(path, index=False, encoding="utf-8-sig")
    print(ab.groupby("variant").fit_seconds.sum())


if __name__ == "__main__":
    main()
