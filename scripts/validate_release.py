"""Validate released bytes and recompute scientific summaries without full retraining."""
from __future__ import annotations
import argparse
import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from assemble_data import digest, main as assemble

ROOT = Path(__file__).resolve().parents[1]


def close(a, b, label, tol=1e-6):
    if not np.allclose(a, b, rtol=tol, atol=tol, equal_nan=True):
        raise AssertionError(f"{label}: {a} != {b}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preprocess", action="store_true", help="Also reconstruct every feature and label from raw CSV")
    args = ap.parse_args()
    assemble()
    report = {"full_retraining_performed": False}
    manifest = json.loads((ROOT / "release_manifest.json").read_text(encoding="utf-8"))
    for item in manifest["files"]:
        p = ROOT / item["path"]
        assert p.stat().st_size == item["bytes"] and digest(p) == item["sha256"], item["path"]
    report["verified_manifest_files"] = len(manifest["files"])
    sources = sorted((ROOT / "scripts").rglob("*.py"))
    for p in sources:
        ast.parse(p.read_text(encoding="utf-8-sig"), filename=str(p))
    report["parsed_python_files"] = len(sources)
    rawpath = ROOT / "data/raw/IOPTQCfFiFoNPo_2min_Agtrup_Aug_2023.csv"
    assert digest(rawpath) == "6734fa9e7a351b5823c501d2d297b27923783f93ed7aa1c9c68b8e0ec0143a07"
    raw = pd.read_csv(rawpath, parse_dates=["date"])
    assert raw.shape == (525600, 11) and raw.isna().sum().sum() == 0
    assert raw.date.is_unique and (raw.date.diff().dropna() == pd.Timedelta(minutes=2)).all()
    meta = json.loads((ROOT / "data/processed/forecast_metadata.json").read_text(encoding="utf-8"))
    data = pd.read_parquet(ROOT / "data/processed/forecast_10min_features.parquet")
    assert len(data) == meta["rows"] == 104796 and len(meta["features"]) == meta["feature_count"] == 195
    assert data.index.is_unique and data.index.is_monotonic_increasing and not data.isna().any().any()
    assert data["split"].value_counts().to_dict() == {"train": 65053, "test": 26028, "calibration": 12781, "unused": 934}
    for before, after in [("train", "calibration"), ("calibration", "test")]:
        assert data.index[data.split.eq(after)].min() - data.index[data.split.eq(before)].max() == pd.Timedelta(hours=78)
    report["raw_rows"] = len(raw)
    report["feature_rows"] = len(data)
    report["feature_count"] = meta["feature_count"]
    pred = pd.read_parquet(ROOT / "data/predictions/P3F_v2_calibration_test_predictions.parquet")
    test = pred.loc[pred.split.eq("test")]
    assert pred.index.equals(data.index[data.split.isin(["calibration", "test"])])
    tokens = {"Persistence": "Persistence", "Seasonal-24h": "Seasonal24h", "Ridge-delta": "RidgeDelta",
              "LightGBM-direct": "LightGBMdirect", "XGBoost-delta": "XGBoostdelta", "P3F": "P3F", "P3F v1 (Huber)": "P3Fv1"}
    metrics = pd.read_csv(ROOT / "tables/Table3_model_metrics_with_v1.csv")
    for r in metrics.itertuples():
        y = test[f"obs_{r.key}"].to_numpy(float)
        p = test[f"pred_{tokens[r.model]}_{r.key}"].to_numpy(float)
        pers = test[f"pred_Persistence_{r.key}"].to_numpy(float)
        mae, rmse = np.abs(y - p).mean(), np.sqrt(np.square(y - p).mean())
        values = [mae, rmse, 1 - np.square(y-p).sum()/np.square(y-y.mean()).sum(),
                  1-mae/np.abs(y-pers).mean(), 1-rmse/np.sqrt(np.square(y-pers).mean())]
        close(values, [r.test_mae,r.test_rmse,r.test_r2,r.test_skill_mae,r.test_skill_rmse], f"point metrics {r.model} {r.key}")
    report["recomputed_point_metric_rows"] = len(metrics)
    subgroups = pd.read_csv(ROOT / "tables/Table5_uncertainty_subgroups.csv")
    for r in subgroups.itertuples():
        k = r.target + "_" + r.horizon.replace(".", "p")
        y, p = test[f"obs_{k}"].to_numpy(float), test[f"pred_P3F_{k}"].to_numpy(float)
        threshold = test[f"event_threshold_{k}"].iloc[0]
        if r.group_type == "all": mask = np.ones(len(test), bool)
        elif r.group_type == "phase": mask = test.PROCESSPHASE_INLET.round().eq(int(r.group[-1])).to_numpy()
        elif r.group_type == "transition": mask = (test.transition_recent_1h.eq(0) if r.group == "stable" else test.transition_recent_1h.gt(0)).to_numpy()
        elif r.group_type == "concentration": mask = y < threshold if r.group.startswith("below") else y >= threshold
        else: mask = (test.ad_class.astype(str).eq("in-domain") if r.group == "in-domain" else test.ad_class.astype(str).ne("in-domain")).to_numpy()
        assert mask.sum() == r.n
        close(np.abs(y[mask]-p[mask]).mean(), r.mae, f"subgroup MAE {k}")
        for level in [50,90]:
            lo,hi = test[f"lo{level}_{k}"].to_numpy(float)[mask],test[f"hi{level}_{k}"].to_numpy(float)[mask]
            assert (lo >= 0).all() and (hi >= lo).all()
            close(((y[mask]>=lo)&(y[mask]<=hi)).mean(), getattr(r,f"coverage{level}"), f"coverage {k}", 1e-12)
            close((hi-lo).mean(), getattr(r,f"width{level}"), f"width {k}", 1e-12)
    report["recomputed_uncertainty_rows"] = len(subgroups)
    cv = pd.read_csv(ROOT / "tables/TableS16_validation_selection.csv")
    assert len(cv) == 80 and (cv.stage.value_counts() == 40).all()
    means = {o:cv.loc[cv.stage.eq("point"),f"skill_{o}"].mean() for o in ["huber","l1"]}
    selected = json.loads((ROOT / "tables/B_selected_configuration.json").read_text(encoding="utf-8"))
    assert selected["point_objective"] == max(means,key=means.get) == "l1"
    interval = cv.loc[cv.stage.eq("interval")]
    candidates = [m for m in ["SC","SC-phase","CQR","CQR-phase"] if interval[f"{m}_cov90"].mean()>=0.88]
    assert selected["interval_method"] == min(candidates,key=lambda m:interval[f"{m}_is90"].mean()) == "CQR-phase"
    report["validation_selection"] = "l1 + CQR-phase"
    # Existing analysis scripts run in isolated temporary roots; compare all bootstrap outputs.
    with tempfile.TemporaryDirectory(prefix="p3f-validate-") as directory:
        work = Path(directory)
        (work / "data/predictions").mkdir(parents=True)
        (work / "tables").mkdir()
        for name in ["P3F_v2_calibration_test_predictions.parquet","P3F_v2_ablation_test_predictions.parquet"]:
            shutil.copy2(ROOT / "data/predictions" / name, work / "data/predictions" / name)
        env = dict(os.environ, P3F_ROOT=str(work))
        for name in ["55_paired_contrasts_p3f_v2.py","55b_paired_coverage_contrasts.py"]:
            subprocess.run([sys.executable,str(ROOT / "scripts" / name)],env=env,check=True,capture_output=True,text=True)
        for name in ["TableS12_paired_skill_contrasts.csv","TableS18_paired_coverage_contrasts.csv"]:
            actual,expected = pd.read_csv(work / "tables" / name),pd.read_csv(ROOT / "tables" / name)
            assert_frame_equal(actual,expected,check_exact=False,rtol=1e-10,atol=1e-10)
        report["weekly_bootstrap_tables_reproduced"] = 2
        if args.preprocess:
            (work / "scripts").mkdir()
            (work / "data/raw").mkdir()
            shutil.copy2(ROOT / "scripts/02_build_forecast_dataset.py",work / "scripts")
            shutil.copy2(rawpath,work / "data/raw")
            subprocess.run([sys.executable,str(work / "scripts/02_build_forecast_dataset.py")],check=True,capture_output=True,text=True)
            reconstructed = pd.read_parquet(work / "data/processed/forecast_10min_features.parquet")
            assert_frame_equal(data,reconstructed,check_exact=True)
            report["full_feature_reconstruction"] = "exact dataframe equality"
    report["status"] = "PASS"
    (ROOT / "validation_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
