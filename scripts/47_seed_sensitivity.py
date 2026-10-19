"""Refit P3F and LightGBM-direct with additional random seeds and report
future-test MAE skill, holding data, features and hyperparameters fixed."""
from __future__ import annotations

import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("P3F_ROOT", Path(__file__).resolve().parents[1]))
SCRIPT_DIR = Path(os.environ.get("P3F_SCRIPTS", ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("train", SCRIPT_DIR / "03_train_p3f_models.py")
train_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_mod)

SEEDS = [20260809, 20260810, 20260811, 20260812]
OUT = ROOT / "tables" / "TableS15_seed_sensitivity.csv"


def main() -> None:
    meta = json.loads((ROOT / "data" / "processed" / "forecast_metadata.json").read_text(encoding="utf-8"))
    data = pd.read_parquet(ROOT / "data" / "processed" / "forecast_10min_features.parquet")
    features = meta["features"]
    tasks = train_mod.task_list(meta)
    train = data.query("split == 'train'")
    test = data.query("split == 'test'")
    rows = []
    done = set()
    if OUT.exists():
        prev = pd.read_csv(OUT)
        rows = prev.to_dict("records")
        done = {(r["model"], r["seed"], r["target"], r["horizon"]) for r in rows}
    for seed in SEEDS:
        for task in tasks:
            current = test[task["target"]].to_numpy()
            y = test[task["y"]].to_numpy()
            for model_name in ["P3F", "LightGBM-direct"]:
                if (model_name, seed, task["target"], task["horizon"]) in done:
                    continue
                if model_name == "P3F":
                    model, label, is_delta = train_mod.p3f_model(seed), task["delta"], True
                else:
                    model, label, is_delta = train_mod.direct_lgbm(seed), task["y"], False
                t0 = time.perf_counter()
                model.fit(train[features], train[label])
                pred = train_mod.predict_absolute(model, test[features], current, is_delta)
                m = train_mod.metrics(y, pred, current)
                rows.append({"model": model_name, "seed": seed, "target": task["target"], "horizon": task["horizon"],
                             "skill_mae": m["skill_mae"], "r2": m["r2"], "fit_seconds": time.perf_counter() - t0})
                pd.DataFrame(rows).to_csv(OUT, index=False)
                print(model_name, seed, task["key"], round(m["skill_mae"], 4), flush=True)


if __name__ == "__main__":
    main()
