from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "IOPTQCfFiFoNPo_2min_Agtrup_Aug_2023.csv"
OUT_DIR = ROOT / "data" / "processed"
TABLE_DIR = ROOT / "tables"

TARGETS = ["T1_PO4", "T1_NH4"]
HORIZONS = {"0.5h": 3, "1h": 6, "2h": 12, "4h": 24, "6h": 36}
LAGS = [1, 3, 6, 12, 24, 36, 72, 144, 288]
ROLL_WINDOWS = [6, 36, 144, 288]
CONTINUOUS = [
    "IN_METAL_Q",
    "T1_O2",
    "METAL_Q",
    "TEMPERATURE",
    "IN_Q",
    "MAX_CF",
    "T1_NH4",
    "T1_PO4",
]
PHASES = ["PROCESSPHASE_INLET", "PROCESSPHASE_OUTLET"]


def phase_elapsed_hours(series: pd.Series) -> pd.Series:
    change_group = series.ne(series.shift()).cumsum()
    return series.groupby(change_group).cumcount().astype("float32") / 6.0


def feature_family(name: str) -> str:
    if name.startswith("T1_PO4"):
        return "PO4 history"
    if name.startswith("T1_NH4"):
        return "NH4 history"
    if "O2" in name:
        return "Dissolved O2"
    if "METAL_Q" in name:
        return "Metal dosing"
    if name.startswith("IN_Q"):
        return "Influent flow"
    if name.startswith("TEMPERATURE"):
        return "Temperature"
    if name.startswith("MAX_CF"):
        return "Control factor"
    if "PHASE" in name or name.startswith("phase_") or name.startswith("transition_"):
        return "Process phase"
    return "Calendar"


def lag_band(name: str) -> str:
    if name.startswith(("hour_", "doy_", "dow_")) or "phase" in name.lower() or "transition" in name.lower():
        return "state/current"
    if "_lag" in name:
        lag = int(name.rsplit("_lag", 1)[1])
        if lag <= 6:
            return "recent <=1 h"
        if lag <= 36:
            return "short 1-6 h"
        if lag <= 144:
            return "circadian 12-24 h"
        return "long 48 h"
    if "_roll" in name:
        window = int(name.split("_roll", 1)[1].split("_", 1)[0])
        if window <= 6:
            return "recent <=1 h"
        if window <= 36:
            return "short 1-6 h"
        if window <= 144:
            return "circadian 12-24 h"
        return "long 48 h"
    return "state/current"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(RAW, parse_dates=["date"])
    raw = raw.sort_values("date").set_index("date")
    if raw.index.has_duplicates:
        raise ValueError("Raw timestamps are duplicated; stop before modeling.")

    agg = {column: "mean" for column in CONTINUOUS}
    agg.update({column: "last" for column in PHASES})
    data = raw.resample("10min").agg(agg)
    if data.isna().any().any():
        missing = data.isna().sum()
        raise ValueError(f"Unexpected missing values after 10-min aggregation: {missing[missing > 0].to_dict()}")

    X = pd.DataFrame(index=data.index)
    for column in CONTINUOUS + PHASES:
        X[column] = data[column].astype("float32")

    for column in CONTINUOUS + PHASES:
        for lag in LAGS:
            X[f"{column}_lag{lag}"] = data[column].shift(lag).astype("float32")

    rolling_columns = ["IN_METAL_Q", "T1_O2", "METAL_Q", "TEMPERATURE", "IN_Q", "T1_NH4", "T1_PO4"]
    for column in rolling_columns:
        for window in ROLL_WINDOWS:
            roll = data[column].rolling(window=window, min_periods=window)
            X[f"{column}_roll{window}_mean"] = roll.mean().astype("float32")
            X[f"{column}_roll{window}_std"] = roll.std(ddof=0).astype("float32")
            X[f"{column}_roll{window}_range"] = (roll.max() - roll.min()).astype("float32")

    inlet = data["PROCESSPHASE_INLET"]
    outlet = data["PROCESSPHASE_OUTLET"]
    inlet_change = inlet.ne(inlet.shift()).astype("float32")
    outlet_change = outlet.ne(outlet.shift()).astype("float32")
    X["phase_inlet_elapsed_h"] = phase_elapsed_hours(inlet)
    X["phase_outlet_elapsed_h"] = phase_elapsed_hours(outlet)
    X["transition_inlet_recent_1h"] = inlet_change.rolling(6, min_periods=1).max().astype("float32")
    X["transition_outlet_recent_1h"] = outlet_change.rolling(6, min_periods=1).max().astype("float32")
    X["phase_pair_code"] = (10 * inlet + outlet).astype("float32")

    minute_of_day = X.index.hour * 60 + X.index.minute
    day_of_year = X.index.dayofyear
    day_of_week = X.index.dayofweek
    X["hour_sin"] = np.sin(2 * np.pi * minute_of_day / 1440).astype("float32")
    X["hour_cos"] = np.cos(2 * np.pi * minute_of_day / 1440).astype("float32")
    X["doy_sin"] = np.sin(2 * np.pi * day_of_year / 365.25).astype("float32")
    X["doy_cos"] = np.cos(2 * np.pi * day_of_year / 365.25).astype("float32")
    X["dow_sin"] = np.sin(2 * np.pi * day_of_week / 7).astype("float32")
    X["dow_cos"] = np.cos(2 * np.pi * day_of_week / 7).astype("float32")

    target_columns: list[str] = []
    delta_columns: list[str] = []
    for target in TARGETS:
        for horizon_name, steps in HORIZONS.items():
            suffix = horizon_name.replace(".", "p")
            y_name = f"y_{target}_{suffix}"
            d_name = f"delta_{target}_{suffix}"
            X[y_name] = data[target].shift(-steps).astype("float32")
            X[d_name] = (X[y_name] - data[target]).astype("float32")
            target_columns.append(y_name)
            delta_columns.append(d_name)

    feature_columns = [c for c in X.columns if c not in target_columns + delta_columns]
    X = X.dropna(subset=feature_columns + target_columns + delta_columns).copy()

    purge = pd.Timedelta(hours=78)
    split = pd.Series("unused", index=X.index, dtype="object")
    train_start = X.index.min()
    train_end = pd.Timestamp("2022-10-28 18:00:00+00:00")
    calibration_start = pd.Timestamp("2022-11-01 00:00:00+00:00")
    calibration_end = pd.Timestamp("2023-01-28 18:00:00+00:00")
    test_start = pd.Timestamp("2023-02-01 00:00:00+00:00")
    test_end = X.index.max()
    split.loc[(X.index >= train_start) & (X.index <= train_end)] = "train"
    split.loc[(X.index >= calibration_start) & (X.index <= calibration_end)] = "calibration"
    split.loc[(X.index >= test_start) & (X.index <= test_end)] = "test"
    X.insert(0, "split", split)

    folds = []
    fold_starts = [
        pd.Timestamp("2022-07-01 00:00:00+00:00"),
        pd.Timestamp("2022-08-01 00:00:00+00:00"),
        pd.Timestamp("2022-09-01 00:00:00+00:00"),
        pd.Timestamp("2022-10-01 00:00:00+00:00"),
    ]
    for i, valid_start in enumerate(fold_starts, start=1):
        valid_end = min(valid_start + pd.offsets.MonthEnd(1) + pd.Timedelta(hours=23, minutes=50), train_end)
        fold_train_end = valid_start - purge
        folds.append(
            {
                "fold": i,
                "train_start": str(train_start),
                "train_end": str(fold_train_end),
                "valid_start": str(valid_start),
                "valid_end": str(valid_end),
                "purge_hours": 78,
            }
        )

    out_path = OUT_DIR / "forecast_10min_features.parquet"
    X.to_parquet(out_path, compression="zstd")

    feature_map = pd.DataFrame(
        {
            "feature": feature_columns,
            "family": [feature_family(c) for c in feature_columns],
            "lag_band": [lag_band(c) for c in feature_columns],
            "available_at_issuance": True,
        }
    )
    feature_map.to_csv(TABLE_DIR / "Table1_feature_dictionary.csv", index=False, encoding="utf-8-sig")

    split_summary = (
        X.loc[X["split"].ne("unused")]
        .groupby("split", observed=True)
        .agg(start=("split", lambda s: str(s.index.min())), end=("split", lambda s: str(s.index.max())), n=("split", "size"))
        .reset_index()
    )
    split_summary.to_csv(TABLE_DIR / "Table2_split_summary.csv", index=False, encoding="utf-8-sig")

    metadata = {
        "source": str(RAW),
        "processed": str(out_path),
        "resolution": "10min",
        "rows": int(len(X)),
        "features": feature_columns,
        "feature_count": len(feature_columns),
        "targets": TARGETS,
        "horizons": HORIZONS,
        "target_columns": target_columns,
        "delta_columns": delta_columns,
        "purge_hours": 78,
        "splits": split_summary.to_dict(orient="records"),
        "rolling_folds": folds,
        "claim_boundary": "Single-plant future-time forecasting; no external-site or causal claim.",
    }
    (OUT_DIR / "forecast_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ok", "rows": len(X), "features": len(feature_columns), "output": str(out_path)}))


if __name__ == "__main__":
    main()
