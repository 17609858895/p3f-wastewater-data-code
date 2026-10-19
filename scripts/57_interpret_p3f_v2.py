"""Re-run the model-dependence analyses of script 07 (temporal occlusion, phase-stratified
TreeSHAP, ALE and the phase-conditioned association networks) on the P3F v2 point models.
v1 outputs are copied to tables/v1_huber_split_conformal/ first."""
from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p3f_b_common as C

spec = importlib.util.spec_from_file_location("interp_v1", Path(__file__).resolve().parent / "07_interpret_p3f.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

V1_DIR = C.TABLE_DIR / "v1_huber_split_conformal"
for name in ["Fig07_feature_contributions.parquet", "Fig07_temporal_occlusion.csv", "Fig07_ale_curves.csv",
             "Fig07_dynamic_network_edges.csv", "Fig07_dynamic_network_summary.csv"]:
    if (C.TABLE_DIR / name).exists() and not (V1_DIR / name).exists():
        shutil.copy2(C.TABLE_DIR / name, V1_DIR / name)

_load = joblib.load
mod.MODEL_DIR = C.MODEL_DIR_B
mod.joblib.load = lambda p: _load(str(p).replace(".joblib", "_point.joblib"))
mod.main()
