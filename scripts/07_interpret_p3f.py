from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "forecast_10min_features.parquet"
META_PATH = ROOT / "data" / "processed" / "forecast_metadata.json"
MODEL_DIR = ROOT / "models" / "p3f"
TABLE_DIR = ROOT / "tables"

TARGETS = ["T1_PO4", "T1_NH4"]
HORIZONS = ["0.5h", "1h", "2h", "4h", "6h"]


def suffix(horizon: str) -> str:
    return horizon.replace(".", "p")


def ale_curve(model, frame: pd.DataFrame, feature: str, bins: int = 12) -> pd.DataFrame:
    values = frame[feature].to_numpy()
    edges = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)))
    if len(edges) < 4:
        return pd.DataFrame(columns=["feature_value", "ale", "n"])
    effects = []
    counts = []
    centers = []
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (values >= left) & (values <= right if right == edges[-1] else values < right)
        subset = frame.loc[mask].copy()
        counts.append(len(subset))
        centers.append(float(np.median(values[mask])) if mask.any() else float((left + right) / 2))
        if len(subset) == 0:
            effects.append(0.0)
            continue
        low = subset.copy()
        high = subset.copy()
        low[feature] = left
        high[feature] = right
        effects.append(float(np.mean(model.predict(high) - model.predict(low))))
    accumulated = np.cumsum(effects)
    weights = np.asarray(counts, dtype=float)
    centered = accumulated - np.average(accumulated, weights=np.maximum(weights, 1))
    return pd.DataFrame({"feature_value": centers, "ale": centered, "n": counts})


def main() -> None:
    metadata = json.loads(META_PATH.read_text(encoding="utf-8"))
    features = metadata["features"]
    feature_map = pd.read_csv(TABLE_DIR / "Table1_feature_dictionary.csv")
    family = feature_map.set_index("feature")["family"].to_dict()
    lag_band = feature_map.set_index("feature")["lag_band"].to_dict()
    data = pd.read_parquet(DATA_PATH)
    train = data.query("split == 'train'")
    test = data.query("split == 'test'")
    stride = max(1, len(test) // 3500)
    sample = test.iloc[::stride].iloc[:3500].copy()
    train_medians = train[features].median()

    shap_rows: list[dict] = []
    occlusion_rows: list[dict] = []
    for target in TARGETS:
        for horizon in HORIZONS:
            key = f"{target}_{suffix(horizon)}"
            model = joblib.load(MODEL_DIR / f"P3F_{key}.joblib")
            contributions = model.predict(sample[features], pred_contrib=True)[:, :-1]
            absolute = np.abs(contributions)
            phase_values = sample["PROCESSPHASE_INLET"].round().to_numpy()
            transition = sample["transition_inlet_recent_1h"].gt(0).to_numpy()
            for i, feature in enumerate(features):
                for subgroup, mask in [
                    ("all", np.ones(len(sample), dtype=bool)),
                    ("phase 1", phase_values == 1),
                    ("phase 2", phase_values == 2),
                    ("stable", ~transition),
                    ("recent transition", transition),
                ]:
                    if mask.sum() == 0:
                        continue
                    shap_rows.append(
                        {
                            "target": target,
                            "horizon": horizon,
                            "feature": feature,
                            "family": family[feature],
                            "lag_band": lag_band[feature],
                            "subgroup": subgroup,
                            "mean_abs_contribution": float(absolute[mask, i].mean()),
                            "signed_mean_contribution": float(contributions[mask, i].mean()),
                            "n": int(mask.sum()),
                        }
                    )

            base_prediction = model.predict(sample[features])
            for band in ["state/current", "recent <=1 h", "short 1-6 h", "circadian 12-24 h", "long 48 h"]:
                columns = [c for c in features if lag_band[c] == band]
                occluded = sample[features].copy()
                occluded.loc[:, columns] = train_medians[columns].to_numpy()
                change = np.abs(base_prediction - model.predict(occluded))
                occlusion_rows.append({"target": target, "horizon": horizon, "lag_band": band, "mean_abs_prediction_change": float(change.mean()), "median_abs_prediction_change": float(np.median(change)), "n": len(change), "occluded_features": len(columns)})

    shap_df = pd.DataFrame(shap_rows)
    shap_df.to_parquet(TABLE_DIR / "Fig07_feature_contributions.parquet", compression="zstd")
    pd.DataFrame(occlusion_rows).to_csv(TABLE_DIR / "Fig07_temporal_occlusion.csv", index=False, encoding="utf-8-sig")

    ale_rows: list[pd.DataFrame] = []
    ale_features = ["IN_METAL_Q", "T1_O2", "IN_Q", "TEMPERATURE"]
    ale_sample = test.iloc[:: max(1, len(test) // 12000)].iloc[:12000]
    for target in TARGETS:
        key = f"{target}_4h"
        model = joblib.load(MODEL_DIR / f"P3F_{key}.joblib")
        for phase in [1, 2]:
            phase_frame = ale_sample.loc[ale_sample["PROCESSPHASE_INLET"].round().eq(phase), features]
            for feature in ale_features:
                curve = ale_curve(model, phase_frame, feature)
                curve["target"] = target
                curve["horizon"] = "4h"
                curve["phase"] = phase
                curve["feature"] = feature
                ale_rows.append(curve)
    pd.concat(ale_rows, ignore_index=True).to_csv(TABLE_DIR / "Fig07_ale_curves.csv", index=False, encoding="utf-8-sig")

    network_columns = ["T1_PO4", "T1_NH4", "T1_O2", "IN_METAL_Q", "METAL_Q", "IN_Q", "TEMPERATURE", "MAX_CF", "PROCESSPHASE_INLET"]
    hourly = test[network_columns].resample("1h").median()
    hourly["season"] = pd.Categorical.from_codes(((hourly.index.month % 12) // 3), ["DJF", "MAM", "JJA", "SON"])
    edge_rows: list[dict] = []
    variables = network_columns[:-1]
    for phase in [1, 2]:
        for season in ["DJF", "MAM", "JJA", "SON"]:
            subset = hourly.loc[hourly["PROCESSPHASE_INLET"].round().eq(phase) & hourly["season"].eq(season), variables]
            if len(subset) < 50:
                continue
            corr = subset.corr(method="spearman")
            for source, target in combinations(variables, 2):
                edge_rows.append({"phase": phase, "season": season, "source": source, "target": target, "correlation": float(corr.loc[source, target]), "n": len(subset)})
    edges = pd.DataFrame(edge_rows)
    edges.to_csv(TABLE_DIR / "Fig07_dynamic_network_edges.csv", index=False, encoding="utf-8-sig")
    network_summary = (
        edges.assign(sign=lambda d: np.sign(d["correlation"]))
        .groupby(["phase", "source", "target"], as_index=False)
        .agg(median_correlation=("correlation", "median"), min_abs_correlation=("correlation", lambda x: float(np.min(np.abs(x)))), sign_consistency=("sign", lambda x: float(max(np.mean(x > 0), np.mean(x < 0)))), seasons=("season", "nunique"))
    )
    network_summary.to_csv(TABLE_DIR / "Fig07_dynamic_network_summary.csv", index=False, encoding="utf-8-sig")
    print(json.dumps({"status": "ok", "shap_rows": len(shap_df), "occlusion_rows": len(occlusion_rows), "ale_rows": sum(len(x) for x in ale_rows), "network_edges": len(edges)}))


if __name__ == "__main__":
    main()
