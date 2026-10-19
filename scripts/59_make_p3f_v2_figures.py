"""Render Figures 3-7 for P3F v2 with the refined styling of script 36.

Inputs are the v2 tables in tables/ and data/predictions/P3F_v2_*.parquet. Fig. 6b, 6e and 6f
are redrawn to compare the v2 interval layer with the v1 fixed-width layer; all other panel
builders are those of scripts 08/36. Output: figures/20261019_P3Fv2/.

Usage: python 59_make_p3f_v2_figures.py [--only Fig03|...|Fig07] [--dpi 600]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import matplotlib.dates as mdates
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("refine36", HERE / "36_refine_fig03_07.py")
R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
S = R.source  # panel builders of script 08
ROOT = HERE.parents[0]
S.PRED_PATH = ROOT / "data" / "predictions" / "P3F_v2_calibration_test_predictions.parquet"
S.ABLATION_PRED_PATH = ROOT / "data" / "predictions" / "P3F_v2_ablation_test_predictions.parquet"
R.OUTPUT = ROOT / "figures" / "20261019_P3Fv2"
V1_DIR = ROOT / "tables" / "v1_huber_split_conformal"


def fig06_v2(predictions, uncertainty, uncertainty_v1):
    panels = S.fig06(predictions, uncertainty)
    a, _, c, d, _, _ = panels
    test = predictions.query("split == 'test'").copy()
    short = {"T1_PO4": r"PO$_4$", "T1_NH4": r"NH$_4$"}
    hx = np.arange(len(S.HORIZONS))

    def cover(k, lo, hi, mask=None):
        y = test[f"obs_{k}"]; c_ = (y >= test[lo]) & (y <= test[hi])
        return float(c_[mask].mean()) if mask is not None else float(c_.mean())

    def b(ax, standalone):
        for target in S.TARGETS:
            for version, ls, marker in [("v1", ":", "x"), ("v2", "-", "o")]:
                vals = []
                for h in S.HORIZONS:
                    k = S.key(target, h)
                    high = test[f"obs_{k}"] >= test[f"event_threshold_{k}"]
                    lo, hi = (f"lo90_v1_{k}", f"hi90_v1_{k}") if version == "v1" else (f"lo90_{k}", f"hi90_{k}")
                    vals.append(cover(k, lo, hi, high))
                label = f"{short[target]} {'fixed width' if version == 'v1' else 'phase CQR'}"
                ax.plot(hx, vals, linestyle=ls, marker=marker, color=S.TARGET_COLOR[target], linewidth=1.9,
                        markerfacecolor="white" if version == "v2" else S.TARGET_COLOR[target], markersize=6.5, label=label)
        ax.axhline(0.9, color=S.GRAY, linestyle="--", linewidth=1.2)
        ax.set_xticks(hx, ["0.5", "1", "2", "4", "6"])
        ax.set_ylim(0.0, 1.0)
        S.style_axis(ax, xlabel="Horizon (h)", ylabel="90% coverage,\nobserved ≥ training Q90")
        S.numeric_ticks(ax, x=False, y=True)
        ax.legend(loc="lower left", ncol=2, fontsize=7.5)

    def wilson(p, n, z=1.96):
        den = 1 + z * z / n; center = (p + z * z / (2 * n)) / den
        half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
        return center - half, center + half

    def e(ax, standalone):
        groups = ["all", "phase 1", "phase 2", "recent transition", "at/above training Q90", "borderline/far-OOD"]
        y_pos = np.arange(len(groups))
        for target, offset in [("T1_PO4", -0.13), ("T1_NH4", 0.13)]:
            vals, low, high, old = [], [], [], []
            for g in groups:
                row = uncertainty.query("target == @target and horizon == '4h' and group == @g").iloc[0]
                lo, hi = wilson(row.coverage90, row.n)
                vals.append(row.coverage90); low.append(row.coverage90 - lo); high.append(hi - row.coverage90)
                old.append(uncertainty_v1.query("target == @target and horizon == '4h' and group == @g").iloc[0].coverage90)
            ax.scatter(old, y_pos + offset, marker="x", color=S.TARGET_COLOR[target], s=34, linewidth=1.3, zorder=2,
                       label="_nolegend_")
            ax.errorbar(vals, y_pos + offset, xerr=[low, high], fmt="o" if target == "T1_PO4" else "s", color=S.TARGET_COLOR[target],
                        markerfacecolor="white", capsize=3, linewidth=1.5, label=S.TARGET_LABEL[target], zorder=3)
        ax.axvline(0.9, color=S.GRAY, linestyle="--", linewidth=1.2)
        ax.set_yticks(y_pos, groups); ax.invert_yaxis()
        S.style_axis(ax, xlabel="Four-hour 90% coverage (Wilson CI)", ylabel="")
        S.numeric_ticks(ax, x=True, y=False)
        ax.legend(loc="upper center", ncol=2, fontsize=8.1)

    def f(ax, standalone):
        weekly = test.resample("7D").agg(ad=("ad_score", "median"))
        ad_min, ad_max = weekly.ad.min(), weekly.ad.max()
        weekly["drift_scaled"] = (weekly.ad - ad_min) / max(ad_max - ad_min, 1e-9)
        for target in S.TARGETS:
            k = S.key(target, "4h")
            for version, lo, hi, ls, lw in [("v1", f"lo90_v1_{k}", f"hi90_v1_{k}", ":", 1.4), ("v2", f"lo90_{k}", f"hi90_{k}", "-", 1.9)]:
                cv = ((test[f"obs_{k}"] >= test[lo]) & (test[f"obs_{k}"] <= test[hi])).astype(float)
                ax.plot(weekly.index, cv.resample("7D").mean(), color=S.TARGET_COLOR[target], linestyle=ls, linewidth=lw,
                        label="_nolegend_" if version == "v1" else f"{S.TARGET_LABEL[target]} coverage")
        ax.plot(weekly.index, weekly["drift_scaled"], color=S.PURPLE, linestyle="--", linewidth=1.5, label="scaled drift score")
        ax.axhline(0.9, color=S.GRAY, linestyle=":", linewidth=1.2)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2)); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
        S.style_axis(ax, xlabel="Future-test week", ylabel="Weekly coverage or scaled drift")
        ax.set_ylim(0, 1.05)
        ax.legend(loc="lower left", fontsize=7.4)

    return [a, b, c, d, e, f]


def runtime_panel_v2(ablation, compute):
    """Script 36 runtime panel with label positions that separate Full P3F from Direct target."""
    import matplotlib
    from matplotlib.ticker import LogLocator, NullFormatter
    summary = ablation.groupby("variant", as_index=False).agg(skill=("skill_mae", "mean"), coverage=("coverage90", "mean"),
                                                             fit=("fit_seconds", "sum"), features=("feature_count", "max"))
    measured_full = float(compute.loc[compute.model.eq("P3F"), "fit_seconds"].iloc[0])
    summary.loc[summary.variant.eq("Full P3F"), "fit"] = measured_full
    frontier = S.pareto_front(summary, "fit", "skill")
    colors = {"Full P3F": R.BLUE, "No phase context": R.GREEN, "Short history only": R.GOLD, "Target history only": R.PURPLE, "Direct target": R.TEAL}
    names = {"Full P3F": "Full P³F", "No phase context": "No phase", "Short history only": "Short history",
             "Target history only": "Target history", "Direct target": "Direct target"}
    pos = {"Full P3F": (0.93, 0.56, "right"), "Short history only": (0.33, 0.77, "center"), "Direct target": (0.70, 0.93, "center"),
           "Target history only": (0.43, 0.20, "center"), "No phase context": (0.83, 0.055, "center")}

    def draw(ax, standalone):
        ax.plot(frontier.fit, frontier.skill, color=R.MAUVE, linestyle="--", linewidth=1.5, zorder=1)
        for row in summary.itertuples():
            size = 45 + row.features * 0.45
            ax.scatter(row.fit, row.skill, s=size, color=colors[row.variant], edgecolor="white", linewidth=1, zorder=3)
            if row.variant in set(frontier.variant):
                ax.scatter(row.fit, row.skill, s=size + 90, facecolors="none", edgecolors=R.INK, linewidth=1.25, zorder=2)
            x, y, ha = pos[row.variant]
            ax.annotate(names[row.variant] + chr(10) + f"cov={row.coverage:.2f}", (row.fit, row.skill), xytext=(x, y), textcoords="axes fraction",
                        ha=ha, va="bottom", fontsize=11.1, linespacing=1.0,
                        arrowprops={"arrowstyle": "-", "color": R.STRUCTURAL_GRAY, "linewidth": 0.75, "shrinkA": 4, "shrinkB": 5})
        ax.set_xscale("log")
        ax.set_xlim(summary.fit.min() * 0.80, summary.fit.max() * 1.26)
        ax.set_ylim(summary.skill.min() - 0.018, summary.skill.max() + 0.030)
        S.style_axis(ax, xlabel="Total fit time across tasks (s)", ylabel="Mean future-test MAE skill")
        S.numeric_ticks(ax, x=False, y=True)
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=[4, 6, 8], numticks=4))
        ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
    return draw, summary.to_dict(orient="records"), measured_full


def widen_latency(builders):
    """P3F (median model + four quantile heads) needs ~200 ms per 1000 windows; extend the 3f axis."""
    f = builders[5]

    def g(ax, standalone):
        f(ax, standalone)
        ax.set_xlim(0.006, 1200)
    builders[5] = g
    return builders


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["all", *R.STEMS], default="all")
    ap.add_argument("--dpi", type=int, default=600)
    args = ap.parse_args()
    R.configure_publication_style()
    R.plt.rcParams.update({"font.weight": "normal", "axes.labelweight": "normal"})
    meta = json.loads(S.META_PATH.read_text(encoding="utf-8"))
    read = lambda name: pd.read_csv(S.TABLE_DIR / name)
    metrics = read("Table3_model_metrics.csv")
    compute = read("TableS_compute_profile.csv")
    ablation = read("Table4_ablation_metrics.csv")
    # Fig. 4e compares point-model fitting cost across matched variants
    by_task = read("TableS5_p3f_v2_compute_by_task.csv")
    compute_point = compute.copy()
    compute_point.loc[compute_point.model.eq("P3F"), "fit_seconds"] = by_task.fit_point_s.sum()
    predictions = pd.read_parquet(S.PRED_PATH)
    abl_pred = pd.read_parquet(S.ABLATION_PRED_PATH); abl_pred["date"] = pd.to_datetime(abl_pred["date"])
    runtime, runtime_summary, full_runtime = runtime_panel_v2(ablation, compute_point)
    unc = read("Table5_uncertainty_subgroups.csv"); unc_v1 = pd.read_csv(V1_DIR / "Table5_uncertainty_subgroups.csv")
    choices = {
        "Fig03": lambda: widen_latency(S.fig03(metrics, read("Fig03_split_protocol_diagnostic.csv"), read("Fig03_volatility_skill.csv"), compute, meta)),
        "Fig04": lambda: S.fig04(ablation, abl_pred, predictions)[:-1] + [runtime],
        "Fig05": lambda: S.fig05(metrics, read("Table3_block_bootstrap_skill.csv"), predictions),
        "Fig06": lambda: fig06_v2(predictions, unc, unc_v1),
        "Fig07": lambda: S.fig07(read("Fig07_temporal_occlusion.csv"), pd.read_parquet(S.TABLE_DIR / "Fig07_feature_contributions.parquet"),
                                 read("Fig07_ale_curves.csv"), read("Fig07_dynamic_network_summary.csv")),
    }
    for fid, factory in choices.items():
        if args.only in ["all", fid]:
            R.render(fid, factory(), args.dpi)
    (R.OUTPUT / "Fig03_07_v2_inputs.json").write_text(json.dumps({
        "predictions": str(S.PRED_PATH.relative_to(ROOT)), "ablation_predictions": str(S.ABLATION_PRED_PATH.relative_to(ROOT)),
        "tables": "tables/ (v2); v1 comparison from tables/v1_huber_split_conformal/Table5_uncertainty_subgroups.csv",
        "runtime_panel": runtime_summary, "p3f_point_fit_seconds": full_runtime}, ensure_ascii=False, indent=2, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
