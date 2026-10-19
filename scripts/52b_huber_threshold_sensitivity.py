"""Post hoc sensitivity of the Huber point objective to its transition threshold on the four
rolling-validation folds (training pool only), for comparison with the L1 objective selected by
the pre-specified rule. Thresholds 0.1 and 0.3 mg/L are added to the 0.9 mg/L of P3F-Huber."""
from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p3f_b_common as C

OUT = C.TABLE_DIR / "TableS20_huber_threshold_validation.csv"


def main() -> None:
    meta = C.load_meta(); features = meta["features"]; tasks = C.task_list(meta)
    data = pd.read_parquet(C.DATA_PATH); data = data.loc[data["split"].eq("train")]
    rows = pd.read_csv(OUT).to_dict("records") if OUT.exists() else []
    done = {(r["fold"], r["key"], r["delta"]) for r in rows}
    for fold in meta["rolling_folds"]:
        f = fold["fold"]
        tr = data.loc[(data.index >= pd.Timestamp(fold["train_start"])) & (data.index <= pd.Timestamp(fold["train_end"]))]
        va = data.loc[(data.index >= pd.Timestamp(fold["valid_start"])) & (data.index <= pd.Timestamp(fold["valid_end"]))]
        for task in tasks:
            y, cur = va[task["y"]].to_numpy(float), va[task["target"]].to_numpy(float)
            for delta in [0.1, 0.3]:
                if (f, task["key"], delta) in done:
                    continue
                m = lgb.LGBMRegressor(**dict(C.TREE_PARAMS, random_state=C.SEED + f, objective="huber", alpha=delta))
                m.fit(tr[features], tr[task["delta"]])
                p = np.maximum(cur + m.predict(va[features]), 0)
                rows.append({"fold": f, "key": task["key"], "target": task["target"], "horizon": task["horizon"], "delta": delta,
                             "skill_mae": C.mae_skill(y, p, cur)})
                pd.DataFrame(rows).to_csv(OUT, index=False)
                print(f, task["key"], delta, round(rows[-1]["skill_mae"], 4), flush=True)
    df = pd.DataFrame(rows)
    s16 = pd.read_csv(C.TABLE_DIR / "TableS16_validation_selection.csv"); s16 = s16[s16.stage == "point"]
    summ = [{"objective": f"Huber, {d} mg/L", "mean_skill": df[df.delta == d].skill_mae.mean()} for d in [0.1, 0.3]]
    summ += [{"objective": "Huber, 0.9 mg/L", "mean_skill": s16.skill_huber.mean()}, {"objective": "L1", "mean_skill": s16.skill_l1.mean()}]
    pd.DataFrame(summ).to_csv(C.TABLE_DIR / "TableS20_huber_threshold_summary.csv", index=False)
    print(pd.DataFrame(summ))


if __name__ == "__main__":
    main()
