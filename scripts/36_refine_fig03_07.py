"""Refine Figures 3–7 from locked existing tables and panel builders.

No model training, table writes or new analysis is performed. All scientific
curves and observations come from 08_make_remaining_figures.py. The runtime
panel uses the existing measured Full P3F aggregate from TableS_compute_profile
instead of the original median placeholder. Originals are never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap, to_hex, to_rgba
from matplotlib.text import Text, Annotation
from matplotlib.patches import Patch
from matplotlib.ticker import LogLocator, NullFormatter
from matplotlib.transforms import Bbox

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts" / "08_make_remaining_figures.py"
OUTPUT = ROOT / "figures" / "20261015_精修"
spec = importlib.util.spec_from_file_location("locked_figure_source", SOURCE)
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)
from ml_publication_style import (configure_publication_style, style_axis,
    numeric_ticks, export_subfigures, save_figure_with_subfigures,
    align_subplot_columns, align_subplot_rows, _axes_visual_group,
    _visual_footprint)
from scientific_palettes import STYLE_COLORS

# The project retained a historical ninth neutral entry. Use the current global
# canonical registry for all new data marks and audits, without editing either.
CANONICAL_PALETTE_PATH = ROOT / "scripts/plot_helpers/canonical_scientific_palettes.py"
canonical_spec = importlib.util.spec_from_file_location("canonical_scientific_palettes", CANONICAL_PALETTE_PATH)
canonical_palette = importlib.util.module_from_spec(canonical_spec)
canonical_spec.loader.exec_module(canonical_palette)
get_palette = canonical_palette.get_palette

AUDIT_PATH = ROOT / "scripts/plot_helpers/audit_palette_usage.py"
audit_spec = importlib.util.spec_from_file_location("figure_palette_audit", AUDIT_PATH)
palette_audit = importlib.util.module_from_spec(audit_spec)
audit_spec.loader.exec_module(palette_audit)
palette_audit.get_palette = get_palette

COLORS = get_palette("nature_new", n=8)
CORAL, BLUE, GREEN, TEAL, PURPLE, BLUSH, GOLD, MAUVE = COLORS
INK = STYLE_COLORS["ink"]
STRUCTURAL_GRAY = STYLE_COLORS["muted"]
source.MODEL_COLOR["Persistence"] = BLUSH
FAMILY_COLORS = {"PO4 history": BLUE, "NH4 history": CORAL,
    "Dissolved O2": TEAL, "Metal dosing": GOLD, "Influent flow": GREEN,
    "Process phase": PURPLE, "Calendar": MAUVE,
    "Temperature": BLUSH, "Control factor": MAUVE}
SERIES_COLORS = palette_audit.validate_series_colors("nature_new", {
    **source.TARGET_COLOR, **source.MODEL_COLOR,
    **{f"feature:{k}": v for k, v in FAMILY_COLORS.items()},
    "Target history only": PURPLE,
    "Direct target": TEAL,
    "No phase context": GREEN,
    "Short history only": GOLD,
    "Pareto front": MAUVE,
    "availability:not used": GOLD,
})

STEMS = {"Fig03": "Fig03_leakage_baseline_benchmark",
    "Fig04": "Fig04_P3F_ablation", "Fig05": "Fig05_future_test_performance",
    "Fig06": "Fig06_uncertainty_domain", "Fig07": "Fig07_phase_conditioned_dependence"}
PLACEMENTS = {"Fig03": [(0,0,1),(0,1,2),(0,2,3),(1,0,1),(1,1,2),(1,2,3)],
    "Fig04": [(0,0,2),(0,2,3),(1,0,1),(1,1,2),(1,2,3)],
    "Fig05": [(0,0,1),(0,1,2),(0,2,3),(1,0,1),(1,1,2),(1,2,3)],
    "Fig06": [(0,0,1),(0,1,2),(0,2,3),(1,0,1),(1,1,2),(1,2,3)],
    "Fig07": [(0,0,1),(0,1,2),(0,2,3),(1,0,1),(1,1,3)]}
LABEL_WRAP = {"Prior 6-h target volatility (mg/L)": "Prior 6-h volatility\n(mg/L)",
    "Chronological evaluation blocks": "Chronological\nevaluation blocks",
    "Four-hour evaluation protocol": "Four-hour evaluation\nprotocol",
    "Inference latency (ms per 1000 windows)": "Inference latency\n(ms per 1000 windows)",
    "Skill gain from phase context": "Skill gain from\nphase context",
    "Benefit of persistence correction": "Benefit of persistence\ncorrection",
    "Mean future-test MAE skill": "Mean future-test\nMAE skill",
    "Total fit time across tasks (s)": "Total fit time\nacross tasks (s)",
    "Normalized concentration + offset": "Normalized concentration\n+ offset",
    "P³F MAE / persistence MAE": "MAE / overall\npersistence MAE",
    "Observed-concentration quintile": "Observed concentration\nquintile",
    "MAE skill with block-bootstrap CI": "MAE skill\n(95% block-bootstrap CI)",
    "Empirical future-test coverage": "Future-test coverage",
    "Empirical 90% coverage": "Future-test 90% coverage",
    "Mean 90% width / target IQR": "Mean 90% interval width\n/ target IQR",
    "Applicability-domain distance": "Applicability-domain\ndistance",
    "Untouched future-test time": "Untouched future-test\ntime",
    "Retained fraction from most in-domain": "Fraction retained from\nmost in-domain observations",
    "Four-hour 90% coverage (Wilson CI)": "Four-hour 90% coverage (Wilson CI)",
    "Weekly coverage or scaled drift": "Weekly coverage\nor scaled distance",
    "Normalized mean |TreeSHAP|": "TreeSHAP share of shown families",
    "Within-feature empirical quantile": "Ordered ALE bins (scaled 0–1)",
    "Centered ALE / feature maximum": "Scaled ALE"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def input_inventory() -> dict:
    paths = [SOURCE, ROOT/"figures"/"ml_publication_style.py",
        ROOT/"figures"/"scientific_palettes.py", source.META_PATH,
        source.PRED_PATH, source.ABLATION_PRED_PATH]
    paths += [source.TABLE_DIR / name for name in ["Table3_model_metrics.csv",
        "Table3_block_bootstrap_skill.csv", "Fig03_split_protocol_diagnostic.csv",
        "Fig03_volatility_skill.csv", "TableS_compute_profile.csv",
        "Table4_ablation_metrics.csv", "Table5_uncertainty_subgroups.csv",
        "Fig07_temporal_occlusion.csv", "Fig07_feature_contributions.parquet",
        "Fig07_ale_curves.csv", "Fig07_dynamic_network_summary.csv"]]
    return {str(p.relative_to(ROOT)): digest(p) for p in paths}


def above_legend(ax, ncol=2, fontsize=11.0):
    handles, labels = ax.get_legend_handles_labels()
    labels=[label.replace("T1-", "").replace("Short history only", "Short history").replace("Target history only", "Target history").replace(", inlet metal dose", " metal dose").replace(", T1 O", " O") for label in labels]
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    if handles:
        return ax.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.025),
            ncol=ncol, fontsize=fontsize, frameon=False, borderaxespad=0,
            columnspacing=0.85, handletextpad=0.40, handlelength=2.0,
            labelspacing=0.35)


def improve_fonts(ax, figure_id, panel, standalone):
    for text in ax.texts:
        text.set_fontfamily("Arial")
        text.set_fontweight("normal")
        text.set_fontsize(11.2 if not standalone else 12.2)
    if ax.axison:
        ax.xaxis.label.set_text(LABEL_WRAP.get(ax.get_xlabel(), ax.get_xlabel()))
        ax.yaxis.label.set_text(LABEL_WRAP.get(ax.get_ylabel(), ax.get_ylabel()))
        style_axis(ax, tick_size=15.0 if standalone else 16.0, label_size=17.0 if standalone else 17.5)
        ax.xaxis.labelpad = 9
        ax.yaxis.labelpad = 9
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontweight("normal")
        if ax.images:
            for spine in ax.spines.values():
                spine.set_visible(False)
        for artist in [*ax.images, *ax.collections]:
            colorbar = getattr(artist, "colorbar", None)
            if colorbar is not None:
                colorbar.ax.set_box_aspect(None)
                colorbar.ax.set_aspect("auto")
                colorbar.ax.yaxis.label.set_fontsize(12.0)
                colorbar.ax.yaxis.label.set_fontweight("normal")
                colorbar.ax.yaxis.labelpad = 7
                colorbar.ax.tick_params(labelsize=11, length=4, width=1.1)
                for tick in colorbar.ax.get_yticklabels():
                    tick.set_fontweight("normal")
                colorbar.outline.set_linewidth(1.0)


def refinement(ax, figure_id, panel, standalone):
    improve_fonts(ax, figure_id, panel, standalone)
    if (figure_id, panel) == ("Fig03", "a"):
        ax.images[0].set_cmap(ListedColormap([CORAL, GOLD, BLUE]))
        ax.tick_params(axis="y", labelsize=13.5)
        ax.set_xticks(ax.get_xticks(),["issue t","+0.5 h","+1 h","+2 h","+4 h","+6 h"],rotation=90,ha="center")
        ax.tick_params(axis="x", labelsize=13.2, pad=6)
        for text in list(ax.texts):
            text.remove()
        ax.legend(handles=[Patch(facecolor=BLUE,label="available"),
            Patch(facecolor=GOLD,label="not used"),Patch(facecolor=CORAL,label="forbidden")],
            loc="lower center",bbox_to_anchor=(0.5,1.025),ncol=1,fontsize=11.8,frameon=False)
    if (figure_id, panel) in [("Fig03", "c"), ("Fig04", "a")]:
        # Two-line labels remove the crowded diagonal task labels.
        labels = [label.get_text().split("\n")[-1] for label in ax.get_xticklabels()]
        ax.set_xticks(ax.get_xticks(), labels, rotation=0, ha="center")
        ax.tick_params(axis="x", labelsize=12.3, pad=7)
        ax.tick_params(axis="y", labelsize=12.6)
        ax.text(0.25,1.018,r"PO$_4$",transform=ax.transAxes,ha="center",va="bottom",fontsize=14,color=BLUE)
        ax.text(0.75,1.018,r"NH$_4$",transform=ax.transAxes,ha="center",va="bottom",fontsize=14,color=CORAL)
        if figure_id == "Fig03":
            ax.set_yticks(ax.get_yticks(),["Persist.","Seasonal","Ridge","LightGBM","XGBoost","P³F"])
        ax.set_xticks(np.arange(-0.5,10,1),minor=True)
        ax.set_yticks(np.arange(-0.5,len(ax.get_yticks()),1),minor=True)
        ax.grid(which="minor",color="white",linewidth=0.8)
        ax.tick_params(which="minor",bottom=False,left=False)
        for text in ax.texts:
            if text.get_transform() != ax.transAxes:
                text.set_fontsize((9.6 if text.get_text().startswith("-") else 10.5) if not standalone else 12)
    if (figure_id, panel) == ("Fig03", "d"):
        ax.set_xticks(ax.get_xticks(),["Random\nrows","Chronol.\nno purge","Purged\nrolling"])
        ax.tick_params(axis="x",labelsize=12.8)
    if (figure_id, panel) == ("Fig03", "b"):
        for patch in ax.patches:
            patch.set_alpha(1.0)
        for text in ax.texts:
            if "78 h" in text.get_text():
                text.set_text("78 h purge at both boundaries")
                text.set_position((text.get_position()[0], 3.57))
                text.set_fontsize(10.3 if not standalone else 11.6)
        ax.set_ylim(-0.34, 3.96)
    if (figure_id, panel) == ("Fig03", "f"):
        for line in ax.lines:
            if line.get_label() == "Pareto-efficient front":
                line.set_color(MAUVE)
        for artist in ax.collections:
            artist.set_alpha(1.0)
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=5))
        ax.xaxis.set_minor_formatter(NullFormatter())
        for text in ax.texts:
            if isinstance(text, Annotation):
                offsets = {"Persistence": (7, 8), "Seasonal\n24h": (7, 6),
                    "Ridge\ndelta": (-7, 9), "LightGBM\ndirect": (-42, 22),
                    "P³F": (8, -9), "XGBoost\ndelta": (9, -18)}
                if text.get_text() in offsets:
                    text.set_position(offsets[text.get_text()])
                text.set_fontsize(10.3 if not standalone else 11.5)
    if (figure_id, panel) == ("Fig04", "d"):
        for line in ax.lines:
            if line.get_label() == "Target history only":
                line.set_color(PURPLE)
    if (figure_id, panel) == ("Fig04", "b"):
        for line in ax.lines:
            if line.get_label().endswith(", stable"):
                line.set_label(line.get_label().replace(", stable", ", no recent\ntransition").replace("T1-", ""))
            elif line.get_label().endswith(", transition"):
                line.set_label(line.get_label().replace(", transition", ", recent\ntransition").replace("T1-", ""))
    if (figure_id, panel) == ("Fig05", "a"):
        ax.set_yticks(ax.get_yticks(),[r"PO$_4$",r"NH$_4$"])
        for text in ax.texts:
            is_r2 = text.get_text().startswith("R²=")
            text.set_fontsize(9.3 if is_r2 else 12.3)
            if not is_r2:
                x, y = text.get_position()
                text.set_position((x, y - 0.10))
    if (figure_id, panel) == ("Fig05", "b"):
        for artist in ax.collections:
            if hasattr(artist, "get_offsets") and len(artist.get_offsets()) > 100:
                artist.set_alpha(0.12)
        for line in ax.lines:
            if line.get_label().startswith("T1-"):
                line.set_linewidth(2.0)
    if (figure_id, panel) == ("Fig05", "c"):
        data_lines = [line for line in ax.lines if len(line.get_xdata()) > 100]
        for target, observed, forecast in zip(source.TARGETS, data_lines[::2], data_lines[1::2]):
            observed.set_linewidth(1.0)
            observed.set_alpha(0.70)
            observed.set_label(f"{source.TARGET_LABEL[target]} observed")
            forecast.set_linewidth(2.2)
            forecast.set_label(f"{source.TARGET_LABEL[target]} forecast")
    if (figure_id, panel) == ("Fig06", "a"):
        # Preserve all existing low-coverage points that were below the old 0.38 limit.
        ax.set_ylim(0.30, 0.98)
    if (figure_id, panel) == ("Fig06", "b"):
        for line in ax.lines:
            line.set_label(line.get_label().replace("in-sample", "training\nresidual").replace("conformal", "calibrated"))
    if (figure_id, panel) == ("Fig06", "f"):
        for line in ax.lines:
            if line.get_label() == "scaled drift score":
                line.set_label("scaled median\ndistance")
    if (figure_id, panel) == ("Fig06", "e"):
        labels = [label.get_text() for label in ax.get_yticklabels()]
        replacements = {"recent transition": "Recent transition",
            "at/above training Q90": "Observed ≥ training Q90",
            "borderline/far-OOD": "Borderline / far OOD"}
        ax.set_yticks(ax.get_yticks(), [replacements.get(x, x) for x in labels])
        ax.tick_params(axis="y", labelsize=11.5)
        ax.set_xlabel("Four-hour 90% coverage\n(Wilson CI)", fontsize=17.5)
    if (figure_id, panel) == ("Fig07", "a"):
        ax.tick_params(axis="y", labelsize=11.5)
        ax.tick_params(axis="x", labelsize=12.1)
        for text in ax.texts:
            text.set_fontsize(10.8 if not standalone else 12)
    if (figure_id, panel) == ("Fig07", "b"):
        ax.tick_params(axis="y", labelsize=12)
        ax.set_xlabel("TreeSHAP share of\nshown families", fontsize=17.5)
    if (figure_id, panel) == ("Fig07", "c"):
        for family, artist in zip(sorted(FAMILY_COLORS), ax.collections):
            artist.set_facecolor(FAMILY_COLORS[family])
            artist.set_edgecolor("white")
            if family == "Calendar":
                artist.set_paths([matplotlib.markers.MarkerStyle("^").get_path().transformed(
                    matplotlib.markers.MarkerStyle("^").get_transform())])
                artist.set_sizes([190])
                artist.set_facecolor("none")
                artist.set_edgecolor(MAUVE)
                artist.set_linewidth(1.3)
                artist.set_zorder(4)
            elif family == "Control factor":
                artist.set_sizes([150])
                artist.set_facecolor("none")
                artist.set_edgecolor(MAUVE)
                artist.set_linewidth(1.3)
                artist.set_zorder(4)
        # Explicit offsets and leader lines separate the low-share annotations.
        positions = {"PO$_4$ history": (0.98,0.09,"right"),
            "NH$_4$ history": (0.09,0.90,"left"),
            "Process phase": (0.62,0.56,"left"),
            "Influent flow": (0.58,0.34,"left"),
            "Dissolved O$_2$": (0.35,0.27,"left"),
            "Calendar": (0.40,0.72,"left"),
            "Metal dosing": (0.30,0.18,"left"),
            "Control factor": (0.01,0.02,"left"),
            "Temperature": (0.03,0.41,"left")}
        for text in list(ax.texts):
            if not isinstance(text, Annotation):
                continue
            label=text.get_text(); point=text.xy
            x,y,ha=positions[label]
            text.remove()
            connector={"arrowstyle":"-","color":STRUCTURAL_GRAY,
                    "linewidth":0.7,"shrinkA":3,"shrinkB":5}
            if label == "Calendar":
                connector["connectionstyle"]="angle,angleA=90,angleB=0,rad=0"
                connector["relpos"]=(0,0)
            ax.annotate(label,point,xytext=(x,y),textcoords="axes fraction",
                ha=ha,va="bottom",fontsize=11.5 if not standalone else 12.0,
                annotation_clip=False,
                arrowprops=connector)
        ax.set_xlim(-0.014, 0.43)
        ax.set_ylim(-0.04, 0.42)
    if (figure_id, panel) == ("Fig07", "d"):
        ax.set_xlabel("Ordered ALE bins\n(scaled 0–1)", fontsize=17.5)
    if (figure_id, panel) == ("Fig07", "e"):
        for text in ax.texts:
            text.set_fontsize(15.0 if text.get_text().startswith("phase") else 16.0)
            if text.get_text() == "control":
                text.set_fontsize(14.5)
        for artist in ax.collections:
            if len(artist.get_offsets()) == 1:
                artist.set_sizes([2250])
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.035), ncol=2,
            fontsize=13.0, frameon=False, handlelength=2.4,
            columnspacing=1.2, handletextpad=0.5)

    if (figure_id,panel) in [("Fig04","c"),("Fig04","d"),("Fig05","d")]:
        ax.set_xticks([0.5,1,2,4,6],["0.5","1","2","4","6"])
        ax.tick_params(axis="x",labelsize=14.5 if not standalone else 15)
        if not standalone:
            ax.get_xticklabels()[0].set_ha("right")

    if ax.axison and ax.get_legend() is not None:
        if (figure_id,panel) == ("Fig03","a"):
            pass
        elif (figure_id,panel) == ("Fig03","f"):
            ax.get_legend().remove()
            ax.legend(loc="lower right",fontsize=12.2,frameon=False)
        else:
            count = len(ax.get_legend_handles_labels()[1])
            above_legend(ax, ncol=2 if count > 1 else 1,
                fontsize=12.0 if count >= 4 else 13.0)


def runtime_panel(ablation, compute):
    summary = ablation.groupby("variant", as_index=False).agg(
        skill=("skill_mae", "mean"), coverage=("coverage90", "mean"),
        fit=("fit_seconds", "sum"), features=("feature_count", "max"))
    measured_full = float(compute.loc[compute.model.eq("P3F"), "fit_seconds"].iloc[0])
    summary.loc[summary.variant.eq("Full P3F"), "fit"] = measured_full
    frontier = source.pareto_front(summary, "fit", "skill")
    colors = {"Full P3F": BLUE, "No phase context": GREEN,
        "Short history only": GOLD, "Target history only": PURPLE, "Direct target": TEAL}
    names = {"Full P3F": "Full P³F", "No phase context": "No phase",
        "Short history only": "Short history", "Target history only": "Target history",
        "Direct target": "Direct target"}
    label_positions = {"Full P3F": (0.62, 0.95, "center"),
        "Short history only": (0.33, 0.77, "center"),
        "Direct target": (0.96, 0.80, "right"),
        "Target history only": (0.43, 0.20, "center"),
        "No phase context": (0.83, 0.055, "center")}

    def draw(ax, standalone):
        ax.plot(frontier.fit, frontier.skill, color=MAUVE, linestyle="--", linewidth=1.5, zorder=1)
        for row in summary.itertuples():
            size = 45 + row.features * 0.45
            ax.scatter(row.fit, row.skill, s=size, color=colors[row.variant], edgecolor="white", linewidth=1, zorder=3)
            if row.variant in set(frontier.variant):
                ax.scatter(row.fit, row.skill, s=size+90, facecolors="none", edgecolors=INK, linewidth=1.25, zorder=2)
            x, y, ha = label_positions[row.variant]
            ax.annotate(names[row.variant] + f"\ncov={row.coverage:.2f}",
                (row.fit, row.skill), xytext=(x, y), textcoords="axes fraction",
                ha=ha, va="bottom", fontsize=11.1, linespacing=1.0,
                arrowprops={"arrowstyle": "-", "color": STRUCTURAL_GRAY,
                    "linewidth": 0.75, "shrinkA": 4, "shrinkB": 5})
        ax.set_xscale("log")
        ax.set_xlim(summary.fit.min()*0.80, summary.fit.max()*1.26)
        ax.set_ylim(summary.skill.min()-0.018, summary.skill.max()+0.030)
        style_axis(ax, xlabel="Total fit time across tasks (s)", ylabel="Mean future-test MAE skill")
        numeric_ticks(ax, x=False, y=True)
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=[4,6,8], numticks=4))
        ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
    return draw, summary.to_dict(orient="records"), measured_full


def fit_footprint(ax, target, tolerance=0.00006):
    fig = ax.figure
    for _ in range(16):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        footprint = _visual_footprint(ax, renderer)
        x0, y0, x1, y1 = target
        errors = [abs(footprint.x0-x0), abs(footprint.x1-x1), abs(footprint.y0-y0), abs(footprint.y1-y1)]
        if max(errors) < tolerance:
            break
        group=_axes_visual_group(ax)
        frame=Bbox.union([item.get_position().frozen() for item in group])
        # Text has fixed physical dimensions. Subtract its measured margins
        # instead of repeatedly shrinking axes as if the text scaled with them.
        gx0=x0+(frame.x0-footprint.x0)
        gx1=x1-(footprint.x1-frame.x1)
        gy0=y0+(frame.y0-footprint.y0)
        gy1=y1-(footprint.y1-frame.y1)
        if gx1-gx0 < 0.065 or gy1-gy0 < 0.10:
            raise ValueError(f"Labels cannot fit target footprint: {ax.get_xlabel()}, {ax.get_ylabel()}")
        sx=(gx1-gx0)/frame.width
        sy=(gy1-gy0)/frame.height
        for item in group:
            p = item.get_position().frozen()
            item.set_position([gx0+(p.x0-frame.x0)*sx,
                gy0+(p.y0-frame.y0)*sy, p.width*sx, p.height*sy])
    fig.canvas.draw()
    footprint = _visual_footprint(ax, fig.canvas.get_renderer())
    return {"actual": list(footprint.extents), "target": list(target),
        "max_error": float(max(abs(a-b) for a,b in zip(footprint.extents,target)))}


def numeric_signature(ax):
    """Evidence record: coordinates/arrays are hashed without style attributes."""
    arrays = []
    for line in ax.lines:
        arrays.append([np.asarray(line.get_xdata(),dtype=object).astype(str).tolist(),
            np.asarray(line.get_ydata(),dtype=object).astype(str).tolist()])
    for image in ax.images:
        arrays.append(np.asarray(image.get_array()).astype(str).tolist())
    for collection in ax.collections:
        arrays.append(np.asarray(collection.get_offsets()).astype(str).tolist())
    return hashlib.sha256(json.dumps(arrays, sort_keys=True).encode()).hexdigest()


def palette_check(ax):
    roles = {}
    all_artists = [*ax.lines, *ax.collections, *ax.patches]
    legend = ax.get_legend()
    if legend is not None:
        all_artists += list(getattr(legend, "legend_handles", getattr(legend, "legendHandles", [])))
    allowed = {to_hex(x).lower() for x in COLORS}
    neutral = {to_hex(INK).lower(), to_hex(STRUCTURAL_GRAY).lower(), "#ffffff", "#000000"}
    excluded = []
    for i, artist in enumerate(all_artists):
        try:
            raw = palette_audit._series_colours(artist)
        except TypeError:
            continue
        if not raw:
            continue
        colors = {to_hex(x).lower() for x in raw if to_rgba(x)[3]>0}
        if colors and colors.issubset(allowed):
            roles[f"artist_{i}:{artist.get_label()}"] = artist
        elif colors.issubset(neutral):
            excluded.append({"artist": type(artist).__name__, "label": artist.get_label(),
                "colors": sorted(colors), "role": "structural reference/outline"})
        else:
            raise ValueError(f"Unregistered actual artist colors: {colors}, {artist.get_label()}")
    report = palette_audit.audit_series_artists("nature_new", roles) if roles else {"passed": True, "checks": []}
    report["structural_exclusions"] = excluded
    report["image_colormaps"] = [im.get_cmap().name for im in ax.images]
    return report


def notebook(folder, stem, only):
    obj = {"cells": [{"cell_type":"markdown", "metadata":{}, "source":[
        f"# {stem}\n", "Regenerate the refined combined figure and independent panels from locked project results.\n",
        "This entry performs no model fitting or table modification.\n"]},
        {"cell_type":"code", "execution_count":None, "metadata":{}, "outputs":[],
            "source":["from pathlib import Path\n", "import runpy\n", "import sys\n",
                "project = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'scripts/36_refine_fig03_07.py').exists())\n",
                f"sys.argv = ['36_refine_fig03_07.py', '--only', '{only}']\n",
                "runpy.run_path(str(project / 'scripts/36_refine_fig03_07.py'), run_name='__main__')\n"]}],
        "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
            "language_info":{"name":"python","version":"3.12"}}, "nbformat":4,"nbformat_minor":5}
    (folder/f"{stem}_jupyter_code.ipynb").write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding="utf-8")


def render(figure_id, builders, dpi):
    stem = STEMS[figure_id]
    folder = OUTPUT/stem
    folder.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(13.2, 9.2))
    axes=[]
    rows=[[],[]]
    reports=[]
    lefts=[0.012,0.351,0.690]
    rights=[0.310,0.649,0.988]
    if figure_id == "Fig03":
        lefts=[0.012,0.300,0.598]
        rights=[0.262,0.560,0.988]
    row_bounds=[(0.554,0.975),(0.032,0.467)]
    for letter, builder, (row, start, stop) in zip("abcdef", builders, PLACEMENTS[figure_id]):
        ax=fig.add_axes([lefts[start]+0.060, row_bounds[row][0]+0.080,
            rights[stop-1]-lefts[start]-0.10, row_bounds[row][1]-row_bounds[row][0]-0.17])
        builder(ax, False)
        signature_before=numeric_signature(ax)
        refinement(ax, figure_id, letter, False)
        signature_after=numeric_signature(ax)
        if signature_before != signature_after:
            raise ValueError(f"Style edit changed numerical marks for {figure_id}{letter}")
        axes.append(ax); rows[row].append(ax)
        reports.append({"panel":letter, "numerical_signature":signature_after,
            "palette":palette_check(ax)})
    # Required helpers are reapplied on each run; final fitting uses complete
    # measured footprints so spanning heatmaps and network panels also align.
    align_subplot_rows(rows,iterations=1)
    if all(len(row)==3 for row in rows):
        align_subplot_columns(rows,iterations=1)
    for ax, report, (row,start,stop) in zip(axes,reports,PLACEMENTS[figure_id]):
        report["alignment"] = fit_footprint(ax,
            (lefts[start],row_bounds[row][0],rights[stop-1],row_bounds[row][1]))
    save_figure_with_subfigures(fig, folder, stem, dpi=dpi, close=False)
    fig.savefig(folder/f"{stem}.pdf", bbox_inches="tight", facecolor="white", metadata={"Author":"CHONG LIU","Creator":"CHONG LIU"})
    plt.close(fig)
    subdir=folder/"subfigures"
    for letter,builder in zip("abcdef",builders):
        wide=(figure_id,panel:=letter) in [("Fig04","a"),("Fig07","e")]
        size=(10.2,5.9) if wide else (7.2,5.7)
        single=plt.figure(figsize=size)
        ax=single.add_axes([0.18,0.20,0.72,0.64])
        builder(ax, True)
        signature_before=numeric_signature(ax)
        refinement(ax,figure_id,letter,True)
        if signature_before != numeric_signature(ax):
            raise ValueError(f"Standalone style changed marks: {figure_id}{letter}")
        fit_footprint(ax,(0.025,0.025,0.975,0.975))
        palette_check(ax)
        single.savefig(subdir/f"{stem}_{letter}.png",dpi=dpi,bbox_inches="tight",facecolor="white")
        single.savefig(subdir/f"{stem}_{letter}.pdf",bbox_inches="tight",facecolor="white",metadata={"Author":"CHONG LIU","Creator":"CHONG LIU"})
        plt.close(single)
    lines=["# Subfigures", "", "Independently rendered standalone panels from the same locked panel builders and styling as the combined figure; these files are not crops.", ""]
    lines += [f"- {letter}: `{stem}_{letter}.png`, `{stem}_{letter}.pdf`" for letter in "abcdef"[:len(builders)]]
    (subdir/"README.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    notebook(folder,stem,figure_id)
    (folder/"style_qa.json").write_text(json.dumps({"figure":figure_id,"font_weight":"normal",
        "palette":"nature_new","dpi":dpi,"panels":reports},ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"figure":figure_id,"status":"exported","panels":len(builders)},ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--only",choices=["all",*STEMS],default="all")
    parser.add_argument("--dpi",type=int,default=600)
    args=parser.parse_args()
    configure_publication_style()
    plt.rcParams.update({"font.weight":"normal", "axes.labelweight":"normal"})
    initial=input_inventory()
    metadata=json.loads(source.META_PATH.read_text(encoding="utf-8"))
    read=lambda name: pd.read_csv(source.TABLE_DIR/name)
    metrics=read("Table3_model_metrics.csv")
    compute=read("TableS_compute_profile.csv")
    ablation=read("Table4_ablation_metrics.csv")
    predictions=pd.read_parquet(source.PRED_PATH)
    ablation_pred=pd.read_parquet(source.ABLATION_PRED_PATH)
    ablation_pred["date"]=pd.to_datetime(ablation_pred["date"])
    runtime,runtime_summary,full_runtime=runtime_panel(ablation,compute)
    choices={"Fig03":lambda:source.fig03(metrics,read("Fig03_split_protocol_diagnostic.csv"),read("Fig03_volatility_skill.csv"),compute,metadata),
        "Fig04":lambda:source.fig04(ablation,ablation_pred,predictions)[:-1]+[runtime],
        "Fig05":lambda:source.fig05(metrics,read("Table3_block_bootstrap_skill.csv"),predictions),
        "Fig06":lambda:source.fig06(predictions,read("Table5_uncertainty_subgroups.csv")),
        "Fig07":lambda:source.fig07(read("Fig07_temporal_occlusion.csv"),pd.read_parquet(source.TABLE_DIR/"Fig07_feature_contributions.parquet"),read("Fig07_ale_curves.csv"),read("Fig07_dynamic_network_summary.csv"))}
    for figure_id,factory in choices.items():
        if args.only in ["all",figure_id]:
            render(figure_id,factory(),args.dpi)
    final=input_inventory()
    if initial != final:
        raise ValueError("A locked source/table changed during rendering")
    OUTPUT.mkdir(parents=True,exist_ok=True)
    (OUTPUT/"Fig03_07_locked_input_qa.json").write_text(json.dumps({
        "inputs_unchanged":True,"sha256":final,"series_map":SERIES_COLORS,
        "runtime_source":"tables/TableS_compute_profile.csv: model=P3F, fit_seconds",
        "measured_full_runtime_seconds":full_runtime,"runtime_panel_coordinates":runtime_summary,
        "scientific_display_corrections":["Fig06a y-range expanded to show existing low coverage points",
            "Fig05f denominator explicitly overall persistence MAE",
            "Fig07d x is ordered retained ALE bins scaled 0–1; y is within-curve scaled ALE",
            "Fig07b shares normalized over six displayed feature families"],
        "palette_encoding":"Canonical eight-color nature_new. Calendar and Control factor use the same mauve hue with triangle/circle distinction and direct labels because Fig07c has nine families. The derived Pareto front also uses canonical mauve. Numeric heatmaps interpolate palette endpoints through a near-white midpoint. Documented transparency applies to intervals, dense point clouds, auxiliary observation traces, phase overlays and network edges; categorical fills remain exact opaque colors.",
        "originals_preserved":True},ensure_ascii=False,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
