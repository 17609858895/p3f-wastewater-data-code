"""Run expensive model reproduction in a NEW directory, preserving released results."""
from __future__ import annotations
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True, help="New, nonexistent run directory")
    ap.add_argument("--extended", action="store_true", help="Also refit seed/sensitivity/timing analyses")
    ap.add_argument("--xgb-device", choices=["cuda", "cpu"], default="cuda", help="Original XGBoost used CUDA; CPU may differ numerically")
    args = ap.parse_args()
    dest = args.output.resolve()
    if dest.exists():
        raise FileExistsError("Use a new output directory to protect released outputs")
    dest.mkdir(parents=True)
    shutil.copytree(ROOT / "scripts", dest / "scripts", ignore=shutil.ignore_patterns("__pycache__"))
    (dest / "data/raw").mkdir(parents=True)
    (dest / "tables").mkdir()
    shutil.copy2(ROOT / "data/raw/IOPTQCfFiFoNPo_2min_Agtrup_Aug_2023.csv", dest / "data/raw")
    env = dict(os.environ, P3F_ROOT=str(dest), P3F_SCRIPTS=str(dest / "scripts"), P3F_XGB_DEVICE=args.xgb_device)
    def run(name):
        print("Running", name, flush=True)
        subprocess.run([sys.executable, str(dest / "scripts" / name)], cwd=dest, env=env, check=True)
    run("02_build_forecast_dataset.py")
    run("03_train_p3f_models.py")
    run("46_paired_contrasts_seasonal_recalibration.py")
    run("07_interpret_p3f.py")
    # Preserve the actual initial Huber outputs before the v2 scripts replace them.
    archive = dest / "tables/v1_huber_split_conformal"
    archive.mkdir()
    for p in (dest / "tables").iterdir():
        if p.is_file():
            shutil.copy2(p, archive / p.name)
    for name in ["50_phase_cycle_descriptors.py", "52_select_p3f_v2_on_validation.py", "53_fit_test_p3f_v2.py",
                 "54_ablation_split_diagnostic_p3f_v2.py", "55_paired_contrasts_p3f_v2.py",
                 "55b_paired_coverage_contrasts.py", "57_interpret_p3f_v2.py", "58_loss_alignment_diagnostics.py"]:
        run(name)
    if args.extended:
        for name in ["52b_huber_threshold_sensitivity.py", "53b_time_p3f_v2.py", "54b_time_ablation_p3f_v2.py", "56_seed_sensitivity_p3f_v2.py"]:
            run(name)


if __name__ == "__main__":
    main()
