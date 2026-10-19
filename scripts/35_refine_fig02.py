"""Refine Fig. 2 from locked inputs without changing analyses or original files.

The six panels retain their original order and numerical values.  The split
timeline uses the actual locked block boundaries rather than month approximations.
Run this file or the notebook in the new figure folder to recreate all exports.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedFormatter, FixedLocator

ROOT = Path(__file__).resolve().parents[1]
HELPERS = ROOT / "scripts/plot_helpers"
sys.path.insert(0, str(HELPERS))
from ml_publication_style import (configure_publication_style, style_axis,
                                  numeric_ticks, export_subfigures,
                                  save_figure_with_subfigures)
from scientific_palettes import STYLE_COLORS, get_palette
from audit_palette_usage import audit_series_artists, validate_series_colors

OUT = ROOT / "figures" / "20261015_精修" / "Fig02_data_dynamics"
STEM = "Fig02_data_dynamics"
PALETTE = get_palette("nature_new")
INK = STYLE_COLORS["ink"]
SERIES = validate_series_colors("nature_new", {
    "PO4": PALETTE[1], "NH4": PALETTE[0], "O2": PALETTE[3],
    "metal": PALETTE[6], "flow": PALETTE[4],
    "phase1": PALETTE[2], "phase2": PALETTE[3],
    "train": PALETTE[2], "calibration": PALETTE[6],
    "test": PALETTE[0], "purge": PALETTE[7],
})
TARGETS = [("T1_PO4", "PO4", "T1-PO$_4$"),
           ("T1_NH4", "NH4", "T1-NH$_4$")]
ARTISTS = {role: [] for role in SERIES}
PROTECTED = [
    ROOT / "scripts" / "generate_initial_figures.py",
    ROOT / "figures" / "Fig02_data_dynamics" / f"{STEM}.png",
    ROOT / "figures" / "Fig02_data_dynamics" / f"{STEM}.pdf",
    ROOT / "tables" / "Fig02_daily_target_summary.csv",
    ROOT / "tables" / "Fig02_autocorrelation_10min.csv",
    ROOT / "tables" / "Fig02_phase_transition_alignment.csv",
    ROOT / "tables" / "Fig02_analysis_summary.json",
    ROOT / "data" / "processed" / "forecast_metadata.json",
    ROOT / "data" / "raw" / "IOPTQCfFiFoNPo_2min_Agtrup_Aug_2023.csv",
]


def fingerprint(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def register(role, artist):
    ARTISTS[role].append(artist)
    return artist


def combined(ax):
    return getattr(ax.figure, "_fig02_combined", False)


def reading_size(ax, standalone, combined_size):
    return combined_size if combined(ax) else standalone


def restyle(ax, xlabel=None, ylabel=None, tick=12.5, label=15.5):
    if combined(ax):
        tick, label = 16.0, 17.5
    style_axis(ax, xlabel=xlabel, ylabel=ylabel, tick_size=tick, label_size=label)
    # Preserve the project manifest's explicit regular-weight typography.
    for text in ax.get_xticklabels() + ax.get_yticklabels():
        text.set_fontweight("normal")
    ax.xaxis.label.set_fontweight("normal")
    ax.yaxis.label.set_fontweight("normal")
    ax.tick_params(axis="both", pad=5)


def target_legend(ax, **kwargs):
    handles = [register(role, Line2D([], [], color=SERIES[role], lw=2.3,
                                   label=label)) for _, role, label in TARGETS]
    return ax.legend(handles=handles, fontsize=reading_size(ax, 12, 13),
                     handlelength=2.4, **kwargs)


def draw_daily(ax, daily):
    for target, role, _ in TARGETS:
        color = SERIES[role]
        register(role, ax.fill_between(daily.index, daily[f"{target}_q10"],
                                      daily[f"{target}_q90"], color=color,
                                      alpha=0.17, lw=0))
        register(role, ax.plot(daily.index, daily[f"{target}_median"],
                              color=color, lw=1.7)[0])
    ax.set_ylim(0, 13.5)
    ax.set_xlim(pd.Timestamp("2021-08-01", tz="UTC"),
                pd.Timestamp("2023-08-01", tz="UTC"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    numeric_ticks(ax, x=False, n=5)
    restyle(ax, "Date", "Concentration (mg/L)")
    target_legend(ax, loc="lower left", bbox_to_anchor=(0, 1.015), ncol=2,
                  borderaxespad=0, columnspacing=1.2)


def draw_week(ax, week):
    variables = ["T1_PO4", "T1_NH4", "T1_O2", "METAL_Q", "IN_Q"]
    roles = ["PO4", "NH4", "O2", "metal", "flow"]
    labels = ["T1-PO$_4$", "T1-NH$_4$", "O$_2$", "Metal dose", "Inflow"]
    offsets = np.arange(5)[::-1]
    for off, var, role in zip(offsets, variables, roles):
        values = week[var].astype(float)
        iqr = values.quantile(.75) - values.quantile(.25)
        z = ((values - values.median()) / (iqr if iqr > 0 else 1)).clip(-2.5, 2.5)
        register(role, ax.plot(week.index, off + .24 * z,
                              color=SERIES[role], lw=1.5)[0])
    phase = week["PROCESSPHASE_INLET"].round().clip(1, 2)
    for number, role in [(1, "phase1"), (2, "phase2")]:
        register(role, ax.fill_between(week.index, 4.76, 5.00,
                                      where=phase.eq(number), step="mid",
                                      color=SERIES[role], lw=0))
    ax.set_yticks(offsets, labels)
    ax.set_ylim(-.65, 5.15)
    ax.set_xlim(week.index[0], week.index[-1])
    ax.xaxis.set_major_locator(FixedLocator([
        mdates.date2num(pd.Timestamp(f"2022-08-{day:02d}", tz="UTC"))
        for day in [1, 3, 5, 7]]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    restyle(ax, "Date (August 2022)", label=15)
    handles = [register(role, Patch(facecolor=SERIES[role], edgecolor=INK,
                                   linewidth=.55, label=f"Inlet phase {number}"))
               for number, role in [(1, "phase1"), (2, "phase2")]]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.015),
              ncol=2, fontsize=reading_size(ax, 11.5, 13), borderaxespad=0, handlelength=1.2,
              columnspacing=1.0)


def draw_ecdf(ax, raw):
    for target, role, _ in TARGETS:
        values = np.sort(raw[target].to_numpy(dtype=float))
        y = (np.arange(len(values)) + .5) / len(values)
        register(role, ax.plot(values, y, color=SERIES[role], lw=2.3)[0])
        q50, q95 = np.quantile(values, [.50, .95])
        register(role, ax.scatter(q50, .50, s=44, marker="o", color=SERIES[role],
                                  edgecolors=INK, linewidths=.55, zorder=4))
        register(role, ax.scatter(q95, .95, s=62, marker="^", color=SERIES[role],
                                  edgecolors=INK, linewidths=.55, zorder=4))
    ax.set_xscale("symlog", linthresh=.1, linscale=.7)
    ticks = [0, .1, .5, 2, 5, 20]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FixedFormatter(["0", "0.1", "0.5", "2", "5", "20"]))
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 1.025)
    ax.set_yticks([0, .25, .5, .75, 1])
    restyle(ax, "Concentration (mg/L)", "Cumulative probability", label=15)
    target_legend(ax, loc="lower right", bbox_to_anchor=(.995, .025),
                  borderaxespad=0)
    # Shape definitions are reading information; no result values are added.
    definitions = ("$\\circ$ Median\n$\\triangle$ 95th percentile" if combined(ax)
                   else r"$\circ$ Median    $\triangle$ 95th percentile")
    ax.text(.98, .23 if combined(ax) else .21, definitions, transform=ax.transAxes,
            fontsize=reading_size(ax, 10.5, 12), ha="right", va="bottom", color=INK)


def draw_acf(ax, acf):
    ax.axhline(0, color=INK, lw=.85, alpha=.55, zorder=0)
    for target, role, _ in TARGETS:
        register(role, ax.plot(acf["lag_h"], acf[target],
                              color=SERIES[role], lw=2.3)[0])
    ax.set_xlim(0, 24)
    ax.set_ylim(-.05, 1.03)
    ax.set_xticks([0, 6, 12, 18, 24])
    ax.set_yticks([0, .25, .5, .75, 1])
    restyle(ax, "Lag (h)", "Autocorrelation")
    target_legend(ax, loc="upper right", bbox_to_anchor=(.99, .99),
                  borderaxespad=0)


def draw_phase(ax, phase, counts):
    ax.axvline(0, color=INK, lw=1, ls=(0, (3, 3)), zorder=1)
    ax.axhline(0, color=INK, lw=.85, alpha=.55, zorder=1)
    x = phase["elapsed_h"].to_numpy()
    handles = []
    for target, role, label in TARGETS:
        color = SERIES[role]
        for key, direction, linestyle, opacity in [
            ("1to2", "1→2", "-", .15),
            ("2to1", "2→1", (0, (4, 2.5)), .10),
        ]:
            lower = phase[f"{target}_{key}_q25"].to_numpy()
            upper = phase[f"{target}_{key}_q75"].to_numpy()
            mid = phase[f"{target}_{key}_median"].to_numpy()
            register(role, ax.fill_between(x, lower, upper, color=color,
                                          alpha=opacity, lw=0, zorder=0))
            # Exact target-colour boundaries make overlapping IQR bands legible.
            for boundary in (lower, upper):
                register(role, ax.plot(x, boundary, color=color, lw=.6,
                                      alpha=.55, ls=linestyle, zorder=1)[0])
            register(role, ax.plot(x, mid, color=color, lw=2.3,
                                  ls=linestyle, zorder=3)[0])
            legend_label = label.replace("T1-", "") if combined(ax) else label
            handles.append(register(role, Line2D([], [], color=color, lw=2.3,
                                                 ls=linestyle,
                                                 label=f"{legend_label}  {direction}")))
    ax.set_xlim(-6, 6)
    # Original limits clipped both ends of the true 25th–75th percentile bands.
    ax.set_ylim(-.95, 1.15)
    ax.set_xticks([-6, -3, 0, 3, 6])
    ax.set_yticks([-.8, -.4, 0, .4, .8])
    restyle(ax, "Time from transition (h)", "Concentration change (IQR)", label=15)
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.015, 1, .19),
              mode="expand", ncol=2, fontsize=reading_size(ax, 11.5, 13), borderaxespad=0,
              handlelength=2.3, columnspacing=.8, labelspacing=.35)
    for x_pos, direction, key, ha in [(0.01, "1→2", "1to2", "left"),
                                      (.99, "2→1", "2to1", "right")]:
        ax.text(x_pos, .025, f"{direction}: n = {counts[key]:,}",
                transform=ax.transAxes, ha=ha, va="bottom", fontsize=reading_size(ax, 10.5, 12),
                color=INK)


def draw_split(ax, metadata):
    blocks = {row["split"]: row for row in metadata["splits"]}
    for role, y, label in [("train", 2.7, "Training / rolling validation"),
                           ("calibration", 1.65, "Calibration"),
                           ("test", .6, "Future test")]:
        row = blocks[role]
        start, end = pd.Timestamp(row["start"]), pd.Timestamp(row["end"])
        xs, xe = mdates.date2num(start), mdates.date2num(end)
        register(role, ax.barh(y, xe - xs, left=xs, height=.38,
                              color=SERIES[role], edgecolor=INK, linewidth=.7))
        label_x = (xs + xe) / 2
        ax.text(label_x, y + .30, f"{label}\nn = {row['n']:,}",
                ha="center", va="bottom", fontsize=reading_size(ax, 11.5, 13), color=INK)
    for earlier, later in [("train", "calibration"), ("calibration", "test")]:
        start = mdates.date2num(pd.Timestamp(blocks[earlier]["end"]))
        end = mdates.date2num(pd.Timestamp(blocks[later]["start"]))
        register("purge", ax.axvspan(start, end, color=SERIES["purge"],
                                      alpha=.55, lw=0, zorder=-1))
    proxy = register("purge", Patch(facecolor=SERIES["purge"], edgecolor=INK,
                                    linewidth=.5,
                                    label=f"{metadata['purge_hours']} h purge"))
    ax.legend(handles=[proxy], loc="lower left", bbox_to_anchor=(0, 1.015),
              fontsize=reading_size(ax, 11.5, 13), borderaxespad=0, handlelength=1.1)
    ax.set_xlim(mdates.date2num(pd.Timestamp("2021-08-01", tz="UTC")),
                mdates.date2num(pd.Timestamp("2023-08-01", tz="UTC")))
    ax.set_ylim(.1, 3.6)
    dates = ["2021-08-01", "2022-02-01", "2022-08-01", "2023-02-01", "2023-08-01"]
    ax.xaxis.set_major_locator(FixedLocator([mdates.date2num(pd.Timestamp(d)) for d in dates]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.set_yticks([])
    restyle(ax, "Date", label=15)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", left=False)


def notebook():
    code = ["from pathlib import Path\n", "import runpy\n",
            "root = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'scripts' / '35_refine_fig02.py').exists())\n",
            "runpy.run_path(str(root / 'scripts' / '35_refine_fig02.py'), run_name='__main__')\n"]
    nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": [
                "# Fig. 2: process dynamics\n",
                "Run the source-of-truth script to recreate the combined PNG/PDF and six matching subfigures. Locked source values and original files are preserved.\n"]},
            {"cell_type": "code", "metadata": {}, "execution_count": None,
             "outputs": [], "source": code}]}
    (OUT / f"{STEM}_jupyter_code.ipynb").write_text(json.dumps(nb, indent=2), encoding="utf-8")


def render_independent(drawer, args, suffix):
    """Draw a fresh standalone canvas with the same locked panel builder."""
    fig, ax = plt.subplots(figsize=(6.7, 5.0))
    drawer(ax, *args)
    fig.subplots_adjust(left=.17, right=.97, bottom=.18, top=.82)
    subdir = OUT / "subfigures"
    subdir.mkdir(parents=True, exist_ok=True)
    # The mandatory combined exporter also produces intermediate crops.  These
    # filenames are deliberately replaced by newly drawn standalone canvases.
    fig.savefig(subdir / f"{STEM}_{suffix}.png", dpi=600,
                bbox_inches="tight", facecolor="white")
    fig.savefig(subdir / f"{STEM}_{suffix}.pdf",
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


def set_pdf_metadata():
    import fitz
    paths = [OUT / f"{STEM}.pdf", *sorted((OUT / "subfigures").glob("*.pdf"))]
    for path in paths:
        with fitz.open(path) as document:
            values = document.metadata.copy()
            values.update({"author": "CHONG LIU", "creator": "CHONG LIU"})
            document.set_metadata(values)
            document.saveIncr()
        with fitz.open(path) as document:
            assert document.metadata["author"] == "CHONG LIU"
            assert document.metadata["creator"] == "CHONG LIU"


def main():
    before = {str(p.relative_to(ROOT)): fingerprint(p) for p in PROTECTED}
    configure_publication_style()
    plt.rcParams.update({"font.weight": "normal", "axes.labelweight": "normal",
                         "axes.titleweight": "normal", "legend.fontsize": 12})
    daily = pd.read_csv(ROOT / "tables" / "Fig02_daily_target_summary.csv",
                        parse_dates=["date"]).set_index("date")
    acf = pd.read_csv(ROOT / "tables" / "Fig02_autocorrelation_10min.csv")
    phase = pd.read_csv(ROOT / "tables" / "Fig02_phase_transition_alignment.csv")
    summary = json.loads((ROOT / "tables" / "Fig02_analysis_summary.json").read_text())
    metadata = json.loads((ROOT / "data" / "processed" / "forecast_metadata.json").read_text())
    raw = pd.read_csv(PROTECTED[-1], parse_dates=["date"]).sort_values("date").set_index("date")
    ten = raw.resample("10min").median()  # same display-only aggregation as original
    week = ten.loc["2022-08-01":"2022-08-07 23:59:59"]
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 9.2))
    fig._fig02_combined = True
    draw_daily(axes[0, 0], daily)
    draw_week(axes[0, 1], week)
    draw_ecdf(axes[0, 2], raw)
    draw_acf(axes[1, 0], acf)
    draw_phase(axes[1, 1], phase, summary["phase_transition_counts"])
    draw_split(axes[1, 2], metadata)
    fig.subplots_adjust(left=.075, right=.985, bottom=.11, top=.905,
                        wspace=.48, hspace=.53)
    OUT.mkdir(parents=True, exist_ok=True)
    save_figure_with_subfigures(fig, OUT, STEM, dpi=600, close=True)
    specs = [
        (draw_daily, (daily,), "a"), (draw_week, (week,), "b"),
        (draw_ecdf, (raw,), "c"), (draw_acf, (acf,), "d"),
        (draw_phase, (phase, summary["phase_transition_counts"]), "e"),
        (draw_split, (metadata,), "f"),
    ]
    for drawer, args, suffix in specs:
        render_independent(drawer, args, suffix)
    set_pdf_metadata()
    audit = audit_series_artists("nature_new", ARTISTS, expected_colors=SERIES)
    (OUT / "subfigures" / "README.md").write_text(
        "# Subfigures\n\nIndependently rendered on fresh 6.7 × 5.0 inch canvases "
        "with the same final panel builders and locked inputs as the combined figure. "
        "The final panel files are independently drawn, not crops.\n\n"
        + "\n".join(f"- {suffix}: {STEM}_{suffix}.png and {STEM}_{suffix}.pdf"
                     for _, _, suffix in specs) + "\n", encoding="utf-8")
    notebook()
    after = {str(p.relative_to(ROOT)): fingerprint(p) for p in PROTECTED}
    if before != after:
        raise RuntimeError("A protected original or locked input changed")
    audit.update({"series_colors": SERIES, "input_sha256": before,
                  "protected_files_unchanged": True,
                  "visible_panel_letters": False,
                  "font_weight": "normal",
                  "combined_canvas_inches": [13.2, 9.2],
                  "combined_font_points": {"axis_labels": 17.5,
                                           "ticks": 16, "legends": 13},
                  "standalone_rendering": "fresh 6.7 × 5.0 inch canvases",
                  "pdf_author": "CHONG LIU", "pdf_creator": "CHONG LIU",
                  "pdf_producer": "preserved from the actual exporter",
                  "panels": ["daily median and 10th–90th percentile", "fixed week",
                             "ECDF", "autocorrelation", "phase-aligned median and IQR",
                             "actual locked chronological blocks"],
                  "transparency": {"daily_range": .17,
                                   "phase_1to2_IQR": .15, "phase_2to1_IQR": .10,
                                   "phase_boundary": .55, "purge": .55},
                  "phase_display_limits": [-.95, 1.15],
                  "phase_data_extrema": [float(phase.filter(regex="_q25$").min().min()),
                                         float(phase.filter(regex="_q75$").max().max())],
                  "split_source": "data/processed/forecast_metadata.json"})
    (OUT / "palette_and_input_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps({"folder": str(OUT), "panels": len(fig.axes),
                      "palette_audit_passed": audit["passed"],
                      "protected_files_unchanged": True}))


if __name__ == "__main__":
    main()
