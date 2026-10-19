from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from scipy.ndimage import gaussian_filter
from sklearn.metrics import average_precision_score, precision_recall_curve


ROOT = Path(__file__).resolve().parents[1]
FIGURE_ROOT = ROOT / "figures"
TABLE_DIR = ROOT / "tables"
PRED_PATH = ROOT / "data" / "predictions" / "P3F_calibration_test_predictions.parquet"
ABLATION_PRED_PATH = ROOT / "data" / "predictions" / "P3F_ablation_test_predictions.parquet"
META_PATH = ROOT / "data" / "processed" / "forecast_metadata.json"
sys.path.insert(0, str(ROOT / "scripts/plot_helpers"))

from ml_publication_style import (  # noqa: E402
    align_subplot_columns,
    align_subplot_rows,
    configure_publication_style,
    export_subfigures,
    numeric_ticks,
    save_figure_with_subfigures,
    scale_figure_text,
    set_figure_text_weight,
    style_axis,
)
from scientific_palettes import STYLE_COLORS, get_palette  # noqa: E402


configure_publication_style()
SOFT = get_palette("nature_new", n=9)
# Exact canonical nature_new values, without hue-derived darkening.
CORAL, BLUE, GREEN, TEAL, PURPLE, BLUSH, GOLD, MAUVE, PALETTE_GRAY = SOFT
NAVY = STYLE_COLORS["ink"]
GRAY = STYLE_COLORS["muted"]
LIGHT_GRAY = SOFT[8]
PALETTE = SOFT[:8]
LOW_SAT_DIVERGING = LinearSegmentedColormap.from_list(
    "nature_new_diverging", [BLUE, "#F8F7F4", CORAL]
)
LOW_SAT_SEQUENTIAL = LinearSegmentedColormap.from_list(
    "nature_new_sequential", ["#F8F7F4", SOFT[2], SOFT[3], BLUE]
)
TARGET_COLOR = {"T1_PO4": BLUE, "T1_NH4": CORAL}
TARGET_LABEL = {"T1_PO4": r"T1-PO$_4$", "T1_NH4": r"T1-NH$_4$"}
TARGETS = ["T1_PO4", "T1_NH4"]
MODEL_COLOR = {
    "Persistence": GRAY,
    "Seasonal-24h": GOLD,
    "Ridge-delta": GREEN,
    "LightGBM-direct": TEAL,
    "XGBoost-delta": CORAL,
    "P3F": BLUE,
}
HORIZONS = ["0.5h", "1h", "2h", "4h", "6h"]
HOURS = {"0.5h": 0.5, "1h": 1, "2h": 2, "4h": 4, "6h": 6}


def key(target: str, horizon: str) -> str:
    return f"{target}_{horizon.replace('.', 'p')}"


def clean_label(text: str) -> str:
    return text.replace("T1_PO4", r"T1-PO$_4$").replace("T1_NH4", r"T1-NH$_4$")


def display_model(text: str) -> str:
    return text.replace("P3F", "P³F")


def display_family(text: str) -> str:
    return {
        "PO4 history": r"PO$_4$ history",
        "NH4 history": r"NH$_4$ history",
        "Dissolved O2": r"Dissolved O$_2$",
    }.get(text, text)


def add_colorbar(fig, ax, image, label: str, ticks=None):
    cb = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03, ticks=ticks)
    cb.set_label(label, fontsize=11.5, fontweight="bold")
    cb.ax.tick_params(labelsize=9.5, width=1.2, length=4)
    for tick in cb.ax.get_yticklabels():
        tick.set_fontweight("bold")
    return cb


def heatmap(ax, values, row_labels, col_labels, cmap=LOW_SAT_DIVERGING, center=0.0, fmt=".2f", cbar_label=""):
    arr = np.asarray(values, dtype=float)
    vmax = np.nanmax(np.abs(arr - center))
    norm = TwoSlopeNorm(vmin=center - vmax, vcenter=center, vmax=center + vmax) if vmax > 0 else None
    image = ax.imshow(arr, cmap=cmap, norm=norm, aspect="auto")
    dense_columns = len(col_labels) > 6
    ax.set_xticks(np.arange(len(col_labels)), col_labels)
    plt.setp(
        ax.get_xticklabels(),
        rotation=50 if dense_columns else 0,
        ha="center",
        rotation_mode="default",
        multialignment="center",
    )
    ax.set_yticks(np.arange(len(row_labels)), row_labels)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            value = arr[i, j]
            if np.isfinite(value):
                ax.text(j, i, format(value, fmt), ha="center", va="center", fontsize=8.8, fontweight="bold", color="white" if abs(value - center) > 0.55 * max(vmax, 1e-9) else STYLE_COLORS["ink"])
    ax.tick_params(
        bottom=True,
        left=True,
        length=5,
        width=1.3,
        labelsize=9.4 if dense_columns else 10.5,
    )
    if dense_columns:
        ax.tick_params(axis="x", pad=9)
    ax.tick_params(axis="y", pad=8)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_visible(False)
    add_colorbar(ax.figure, ax, image, cbar_label)


def draw_box(ax, xy, width, height, text, edge, fill="white", fontsize=10.5):
    patch = FancyBboxPatch(xy, width, height, boxstyle="round,pad=0.018,rounding_size=0.025", facecolor=fill, edgecolor=edge, linewidth=1.7)
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=fontsize, fontweight="bold", color=STYLE_COLORS["ink"])
    return patch


def arrow(ax, start, end, color=GRAY, style="-"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14, linewidth=1.6, linestyle=style, color=color, connectionstyle="arc3,rad=0.0"))


def pareto_front(frame: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    """Return the non-dominated lower-cost / higher-skill frontier."""
    ordered = frame.sort_values([x_col, y_col], ascending=[True, False]).copy()
    keep = []
    best_skill = -np.inf
    for idx, row in ordered.iterrows():
        if row[y_col] > best_skill + 1e-12:
            keep.append(idx)
            best_skill = row[y_col]
    return ordered.loc[keep]


def write_notebook(folder: Path, stem: str, only: str) -> None:
    obj = {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": [f"# {stem}\n", "Rebuilds the combined figure and independently rendered standalone panels from locked result tables.\n"]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [f"%run ../../scripts/08_make_remaining_figures.py --only {only}\n"]},
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.12"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (folder / f"{stem}_jupyter_code.ipynb").write_text(json.dumps(obj, indent=2), encoding="utf-8")


def export_figure(stem: str, only: str, builders: list, combined_size=(16.0, 9.2), wspace=0.31) -> None:
    folder = FIGURE_ROOT / stem
    folder.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=combined_size)
    for ax, builder in zip(axes.flat, builders):
        builder(ax, False)
    scale_figure_text(fig, factor=1.15)
    set_figure_text_weight(fig, "normal")
    fig.subplots_adjust(left=0.065, right=0.955, top=0.97, bottom=0.085, wspace=wspace, hspace=0.34)
    align_subplot_columns(axes)
    align_subplot_rows(axes)
    align_subplot_columns(axes)
    align_subplot_rows(axes)
    save_figure_with_subfigures(fig, folder, stem, dpi=600, close=False)
    plt.close(fig)

    subdir = folder / "subfigures"
    subdir.mkdir(parents=True, exist_ok=True)
    for letter, builder in zip("abcdef", builders):
        single, ax = plt.subplots(figsize=(6.7, 5.0))
        builder(ax, True)
        scale_figure_text(single, factor=1.15)
        set_figure_text_weight(single, "normal")
        single.subplots_adjust(left=0.15, right=0.94, top=0.94, bottom=0.16)
        single.savefig(subdir / f"{stem}_{letter}.png", dpi=600, bbox_inches="tight", facecolor="white")
        single.savefig(subdir / f"{stem}_{letter}.pdf", bbox_inches="tight", facecolor="white")
        plt.close(single)
    lines = ["# Subfigures", "", "Each panel is independently rendered on a standalone canvas from the same locked source tables and panel builder as the combined figure.", ""]
    for letter in "abcdef":
        lines.append(f"- {letter}: `{stem}_{letter}.png`, `{stem}_{letter}.pdf`")
    (subdir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_notebook(folder, stem, only)


def export_figure_mosaic(
    stem: str,
    only: str,
    builders: list,
    placements: list[tuple[int, int, int]],
    *,
    combined_size=(16.0, 9.2),
    wspace=0.42,
    standalone_sizes: list[tuple[float, float]] | None = None,
) -> None:
    """Export a two-row, three-column mosaic with optional column spans."""
    if len(builders) != len(placements):
        raise ValueError("builders and placements must have equal length")
    folder = FIGURE_ROOT / stem
    folder.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=combined_size)
    grid = fig.add_gridspec(2, 3)
    row_axes: list[list] = [[], []]
    for builder, (row, col_start, col_stop) in zip(builders, placements):
        ax = fig.add_subplot(grid[row, col_start:col_stop])
        builder(ax, False)
        row_axes[row].append(ax)
    scale_figure_text(fig, factor=1.15)
    set_figure_text_weight(fig, "normal")
    fig.subplots_adjust(left=0.065, right=0.955, top=0.97, bottom=0.085, wspace=wspace, hspace=0.34)
    align_subplot_rows(row_axes)
    align_subplot_rows(row_axes)
    save_figure_with_subfigures(fig, folder, stem, dpi=600, close=False)
    plt.close(fig)

    subdir = folder / "subfigures"
    subdir.mkdir(parents=True, exist_ok=True)
    sizes = standalone_sizes or [(6.7, 5.0)] * len(builders)
    letters = list("abcdefghijklmnopqrstuvwxyz"[:len(builders)])
    for letter, builder, size in zip(letters, builders, sizes):
        single, ax = plt.subplots(figsize=size)
        builder(ax, True)
        scale_figure_text(single, factor=1.15)
        set_figure_text_weight(single, "normal")
        single.subplots_adjust(left=0.15, right=0.94, top=0.94, bottom=0.16)
        single.savefig(subdir / f"{stem}_{letter}.png", dpi=600, bbox_inches="tight", facecolor="white")
        single.savefig(subdir / f"{stem}_{letter}.pdf", bbox_inches="tight", facecolor="white")
        plt.close(single)

    current = set(letters)
    for letter in "abcdefghijklmnopqrstuvwxyz":
        if letter in current:
            continue
        removed_any = False
        for suffix in ("png", "pdf"):
            stale = subdir / f"{stem}_{letter}.{suffix}"
            if stale.exists():
                stale.unlink()
                removed_any = True
        if not removed_any and ord(letter) > ord(letters[-1]) + 1:
            break

    lines = ["# Subfigures", "", "Each panel is independently rendered on a standalone canvas from the same locked source tables and panel builder as the combined figure.", ""]
    for letter in letters:
        lines.append(f"- {letter}: `{stem}_{letter}.png`, `{stem}_{letter}.pdf`")
    (subdir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_notebook(folder, stem, only)


def fig03(metrics_df, split_diag, volatility, compute, metadata):
    def a(ax, standalone):
        rows = ["target history", "process sensors", "phase/state", "calendar", "future controls"]
        cols = ["issue t", "+0.5 h", "+1 h", "+2 h", "+4 h", "+6 h"]
        matrix = np.array([[1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1], [0, -1, -1, -1, -1, -1]])
        cmap = ListedColormap([SOFT[0], "#F8F7F4", SOFT[1]])
        ax.imshow(matrix, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(cols)), cols, rotation=35, ha="right")
        ax.set_yticks(range(len(rows)), rows)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                label = "available" if matrix[i, j] == 1 else ("forbidden" if matrix[i, j] == -1 else "not used")
                ax.text(j, i, label, ha="center", va="center", fontsize=8.1, fontweight="bold", color=STYLE_COLORS["ink"])
        ax.tick_params(bottom=True, left=True, length=5, width=1.3, labelsize=10.3)
        for t in ax.get_xticklabels() + ax.get_yticklabels(): t.set_fontweight("bold")
        for s in ax.spines.values(): s.set_visible(False)

    def b(ax, standalone):
        ranges = [("training", "2021-08-03", "2022-10-28", GREEN, 3), ("calibration", "2022-11-01", "2023-01-28", GOLD, 2), ("future test", "2023-02-01", "2023-07-31", CORAL, 1)]
        for label, start, end, color, y in ranges:
            start, end = pd.Timestamp(start), pd.Timestamp(end)
            ax.barh(y, (end - start).days, left=mdates.date2num(start), height=0.54, color=color, alpha=0.72, edgecolor=color, linewidth=1.5)
            ax.text(start + (end - start) / 2, y, label, ha="center", va="center", fontsize=10.2, fontweight="bold")
        for date in [pd.Timestamp("2022-10-28"), pd.Timestamp("2022-11-01"), pd.Timestamp("2023-01-28"), pd.Timestamp("2023-02-01")]:
            ax.axvline(date, color=PURPLE, linestyle="--", linewidth=1.2, alpha=0.8)
        ax.text(pd.Timestamp("2022-12-15"), 3.45, "78 h purge at both boundaries", ha="center", color=PURPLE, fontsize=9.2, fontweight="bold")
        fold_ranges = []
        for i, fold in enumerate(metadata["rolling_folds"], start=1):
            start = pd.Timestamp(fold["valid_start"]).tz_localize(None)
            end = pd.Timestamp(fold["valid_end"]).tz_localize(None)
            fold_ranges.append((start, end))
            ax.plot([start, end], [0.32, 0.32], color=BLUE, linewidth=5, solid_capstyle="butt")
            if standalone:
                ax.text(start + (end - start) / 2, 0.05, f"V{i}", ha="center", va="top", fontsize=8.2, color=BLUE, fontweight="bold")
        if fold_ranges and not standalone:
            fold_start = min(item[0] for item in fold_ranges)
            fold_end = max(item[1] for item in fold_ranges)
            ax.text(
                fold_start + (fold_end - fold_start) / 2,
                0.05,
                "V1–V4",
                ha="center",
                va="top",
                fontsize=8.2,
                color=BLUE,
                fontweight="bold",
            )
        ax.set_yticks([])
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
        ax.set_ylim(-0.3, 3.8)
        style_axis(ax, xlabel="Chronological evaluation blocks", ylabel="")

    def c(ax, standalone):
        order = ["Persistence", "Seasonal-24h", "Ridge-delta", "LightGBM-direct", "XGBoost-delta", "P3F"]
        columns = [f"{t}\n{h}" for t in [r"PO$_4$", r"NH$_4$"] for h in HORIZONS]
        arr = []
        for model in order:
            row = []
            for target in TARGETS:
                for horizon in HORIZONS:
                    value = metrics_df.query("model == @model and target == @target and horizon == @horizon")["test_skill_mae"].iloc[0]
                    row.append(value)
            arr.append(row)
        heatmap(ax, arr, [display_model(model) for model in order], columns, cbar_label="MAE skill vs persistence")

    def d(ax, standalone):
        protocols = ["Random rows", "Chronological, no purge", "Purged rolling"]
        x = np.arange(len(protocols))
        for target, marker, offset in [("T1_PO4", "o", -0.08), ("T1_NH4", "s", 0.08)]:
            sub = split_diag.query("target == @target")
            for i, protocol in enumerate(protocols):
                values = sub.query("protocol == @protocol")["skill_mae"].to_numpy()
                jitter = np.linspace(-0.04, 0.04, len(values))
                ax.scatter(np.full(len(values), x[i] + offset) + jitter, values, s=42, marker=marker, facecolors="white", edgecolors=TARGET_COLOR[target], linewidth=1.5, zorder=3)
                ax.plot([x[i] + offset - 0.07, x[i] + offset + 0.07], [np.median(values)] * 2, color=TARGET_COLOR[target], linewidth=2.4)
        ax.axhline(0, color=GRAY, linestyle="--", linewidth=1.3)
        ax.set_xticks(x, ["random\nrows", "chronological\nno purge", "purged\nrolling"])
        style_axis(ax, xlabel="Four-hour evaluation protocol", ylabel="MAE skill")
        numeric_ticks(ax, x=False, y=True, n=5)
        ax.scatter([], [], marker="o", facecolors="white", edgecolors=BLUE, label=r"T1-PO$_4$")
        ax.scatter([], [], marker="s", facecolors="white", edgecolors=CORAL, label=r"T1-NH$_4$")
        ax.legend(loc="best")

    def e(ax, standalone):
        for target in TARGETS:
            for phase, style in [(1, "-"), (2, "--")]:
                sub = volatility.query("target == @target and phase == @phase").sort_values("volatility_median")
                ax.plot(sub["volatility_median"], sub["skill_mae"], linestyle=style, marker="o" if phase == 1 else "s", color=TARGET_COLOR[target], linewidth=2.1, markersize=5.3, label=f"{TARGET_LABEL[target]}, phase {phase}")
        ax.axhline(0, color=GRAY, linestyle=":", linewidth=1.3)
        style_axis(ax, xlabel="Prior 6-h target volatility (mg/L)", ylabel="Four-hour MAE skill")
        numeric_ticks(ax, n=5)
        ax.legend(loc="best", fontsize=9.2)

    def f(ax, standalone):
        mean_skill = metrics_df.groupby("model")["test_skill_mae"].mean()
        merged = compute.set_index("model").join(mean_skill.rename("skill"), how="inner")
        merged["latency_plot"] = merged["latency_ms_per_1000"].clip(lower=0.01)
        frontier = pareto_front(merged, "latency_plot", "skill")
        ax.plot(frontier["latency_plot"], frontier["skill"], color=NAVY, linestyle="--", linewidth=1.5, zorder=1, label="Pareto-efficient front")
        offsets = {"Persistence": (5, 5), "Seasonal-24h": (5, 5), "Ridge-delta": (5, 7), "LightGBM-direct": (-52, 18), "P3F": (8, 4), "XGBoost-delta": (8, -19)}
        for model, row in merged.iterrows():
            size = 45 + 28 * np.log10(max(row["parameters_proxy"], 1) + 1)
            x_value = row["latency_plot"]
            ax.scatter(x_value, row["skill"], s=size, color=MODEL_COLOR.get(model, GRAY), edgecolor="white", linewidth=1.0, alpha=0.9, zorder=3)
            if model in frontier.index:
                ax.scatter(x_value, row["skill"], s=size + 90, facecolors="none", edgecolors=NAVY, linewidth=1.25, zorder=2)
            ax.annotate(display_model(model).replace("-", "\n"), (x_value, row["skill"]), xytext=offsets.get(model, (4, 4)), textcoords="offset points", fontsize=8.4, fontweight="bold")
        ax.set_xscale("log")
        ax.set_xlim(0.006, 160)
        ax.set_ylim(-0.225, 0.375)
        ax.axhline(0, color=GRAY, linestyle="--", linewidth=1.2)
        style_axis(ax, xlabel="Inference latency (ms per 1000 windows)", ylabel="Mean future-test MAE skill")
        numeric_ticks(ax, y=True, x=False)
        ax.legend(loc="lower right", fontsize=8.2)

    return [a, b, c, d, e, f]


def fig04(ablation, ablation_pred, predictions):
    variants = ["Full P3F", "No phase context", "Short history only", "Target history only", "Direct target"]

    def b(ax, standalone):
        columns = [f"{t}\n{h}" for t in [r"PO$_4$", r"NH$_4$"] for h in HORIZONS]
        full = ablation.query("variant == 'Full P3F'").set_index(["target", "horizon"])["skill_mae"]
        arr = []
        row_labels = []
        for variant in variants[1:]:
            sub = ablation.query("variant == @variant").set_index(["target", "horizon"])["skill_mae"]
            arr.append([sub.loc[(target, horizon)] - full.loc[(target, horizon)] for target in TARGETS for horizon in HORIZONS])
            row_labels.append(variant)
        heatmap(ax, arr, row_labels, columns, cbar_label="Ablation minus full skill")

    def subgroup_skill(frame, target, horizon, variant, transition):
        sub = frame.query("target == @target and horizon == @horizon and variant == @variant")
        mask = sub["transition"].gt(0) if transition else sub["transition"].eq(0)
        y = sub.loc[mask, "observed"].to_numpy(); p = sub.loc[mask, "prediction"].to_numpy()
        full_key = key(target, horizon)
        dates = pd.DatetimeIndex(sub.loc[mask, "date"])
        pers = predictions.loc[dates, f"pred_Persistence_{full_key}"].to_numpy()
        return 1 - np.mean(np.abs(y - p)) / np.mean(np.abs(y - pers))

    def c(ax, standalone):
        x = np.arange(len(HORIZONS))
        for target in TARGETS:
            for transition, linestyle in [(False, "-"), (True, "--")]:
                gain = []
                for horizon in HORIZONS:
                    full_sub = predictions.query("split == 'test'")
                    mask = full_sub["transition_recent_1h"].gt(0) if transition else full_sub["transition_recent_1h"].eq(0)
                    k = key(target, horizon)
                    y = full_sub.loc[mask, f"obs_{k}"].to_numpy(); pers = full_sub.loc[mask, f"pred_Persistence_{k}"].to_numpy(); pfull = full_sub.loc[mask, f"pred_P3F_{k}"].to_numpy()
                    full_skill = 1 - np.mean(np.abs(y - pfull)) / np.mean(np.abs(y - pers))
                    no_phase = subgroup_skill(ablation_pred, target, horizon, "No phase context", transition)
                    gain.append(full_skill - no_phase)
                ax.plot(x, gain, color=TARGET_COLOR[target], linestyle=linestyle, marker="o" if not transition else "s", linewidth=2.0, label=f"{TARGET_LABEL[target]}, {'transition' if transition else 'stable'}")
        ax.axhline(0, color=GRAY, linestyle=":", linewidth=1.2)
        ax.set_xticks(x, HORIZONS)
        style_axis(ax, xlabel="Forecast horizon", ylabel="Skill gain from phase context")
        numeric_ticks(ax, x=False, y=True)
        ax.legend(loc="best", fontsize=8.8)

    def d(ax, standalone):
        full = ablation.query("variant == 'Full P3F'")
        direct = ablation.query("variant == 'Direct target'")
        for target in TARGETS:
            f = full.query("target == @target").set_index("horizon").loc[HORIZONS]
            dsub = direct.query("target == @target").set_index("horizon").loc[HORIZONS]
            ax.plot([HOURS[h] for h in HORIZONS], f["skill_mae"] - dsub["skill_mae"], color=TARGET_COLOR[target], marker="o", linewidth=2.2, label=TARGET_LABEL[target])
        ax.axhline(0, color=GRAY, linestyle="--", linewidth=1.2)
        style_axis(ax, xlabel="Forecast horizon (h)", ylabel="Benefit of persistence correction")
        numeric_ticks(ax, n=5)
        ax.legend(loc="best")

    def e(ax, standalone):
        styles = [("Full P3F", "-", "o"), ("Short history only", "--", "s"), ("Target history only", ":", "^")]
        for variant, linestyle, marker in styles:
            sub = ablation.query("variant == @variant").groupby("horizon")["skill_mae"].mean().reindex(HORIZONS)
            ax.plot([HOURS[h] for h in HORIZONS], sub, linestyle=linestyle, marker=marker, color={"Full P3F": BLUE, "Short history only": GOLD, "Target history only": GRAY}[variant], linewidth=2.2, label=display_model(variant))
        ax.axhline(0, color=GRAY, linestyle="--", linewidth=1.1)
        style_axis(ax, xlabel="Forecast horizon (h)", ylabel="Mean MAE skill")
        numeric_ticks(ax, n=5)
        ax.legend(loc="best", fontsize=9.2)

    def f(ax, standalone):
        summary = ablation.groupby("variant", as_index=False).agg(skill=("skill_mae", "mean"), coverage=("coverage90", "mean"), fit=("fit_seconds", "sum"), features=("feature_count", "max"))
        full_mask = summary["variant"].eq("Full P3F")
        if full_mask.any() and (summary.loc[full_mask, "fit"].isna().all() or summary.loc[full_mask, "fit"].max() <= 0):
            summary.loc[full_mask, "fit"] = summary.loc[~full_mask & summary["fit"].gt(0), "fit"].median()
        frontier = pareto_front(summary, "fit", "skill")
        ax.plot(frontier["fit"], frontier["skill"], color=NAVY, linestyle="--", linewidth=1.5, zorder=1, label="Pareto-efficient front")
        offsets = {
            "Full P3F": (0, 28),
            "No phase context": (-10, 10),
            "Short history only": (-12, 10),
            "Target history only": (12, -12),
            "Direct target": (0, 16),
        }
        align = {
            "Full P3F": "center",
            "No phase context": "right",
            "Short history only": "right",
            "Target history only": "left",
            "Direct target": "center",
        }
        vertical_align = {
            "Full P3F": "bottom",
            "No phase context": "bottom",
            "Short history only": "bottom",
            "Target history only": "top",
            "Direct target": "bottom",
        }
        short_names = {
            "Full P3F": "Full P³F",
            "No phase context": "No phase",
            "Short history only": "Short history",
            "Target history only": "Target history",
            "Direct target": "Direct target",
        }
        variant_colors = {"Full P3F": BLUE, "No phase context": GREEN, "Short history only": GOLD, "Target history only": PURPLE, "Direct target": CORAL}
        for row in summary.itertuples():
            size = 45 + row.features * 0.45
            ax.scatter(row.fit, row.skill, s=size, color=variant_colors[row.variant], edgecolor="white", linewidth=1.0, zorder=3)
            if row.variant in set(frontier["variant"]):
                ax.scatter(row.fit, row.skill, s=size + 90, facecolors="none", edgecolors=NAVY, linewidth=1.25, zorder=2)
            label = short_names[row.variant] + f"\ncov={row.coverage:.2f}"
            ax.annotate(
                label,
                (row.fit, row.skill),
                xytext=offsets[row.variant],
                textcoords="offset points",
                fontsize=8.1,
                fontweight="bold",
                ha=align[row.variant],
                va=vertical_align[row.variant],
                linespacing=0.94,
                annotation_clip=False,
            )
        ax.set_xscale("log")
        ax.set_xlim(summary["fit"].min() * 0.84, summary["fit"].max() * 1.18)
        ax.set_ylim(summary["skill"].min() - 0.010, summary["skill"].max() + 0.014)
        style_axis(ax, xlabel="Total fit time across tasks (s)", ylabel="Mean future-test MAE skill")
        numeric_ticks(ax, x=False, y=True)
        ax.text(0.02, 0.025, "dashed = Pareto front\npoint size = feature count", transform=ax.transAxes, ha="left", va="bottom", fontsize=8.1, color=GRAY, fontweight="bold")

    return [b, c, d, e, f]


def fig05(metrics_df, boot, predictions):
    test = predictions.query("split == 'test'").copy()

    def a(ax, standalone):
        p3f = metrics_df.query("model == 'P3F'")
        arr = []
        labels = []
        for target in TARGETS:
            row = []
            for horizon in HORIZONS:
                r = p3f.query("target == @target and horizon == @horizon").iloc[0]
                row.append(r.test_skill_mae)
            arr.append(row); labels.append(TARGET_LABEL[target])
        heatmap(ax, arr, labels, HORIZONS, cbar_label="Future-test MAE skill")
        for i, target in enumerate(TARGETS):
            for j, horizon in enumerate(HORIZONS):
                r = p3f.query("target == @target and horizon == @horizon").iloc[0]
                ax.text(j, i + 0.25, f"R²={r.test_r2:.2f}", ha="center", va="center", fontsize=7.6, fontweight="bold", color="white" if r.test_skill_mae > 0.32 else STYLE_COLORS["ink"])

    def b(ax, standalone):
        rng = np.random.default_rng(7)
        series = []
        for target, horizon, marker in [("T1_PO4", "1h", "o"), ("T1_NH4", "4h", "s")]:
            k = key(target, horizon)
            idx = rng.choice(len(test), size=min(3500, len(test)), replace=False)
            y = test[f"obs_{k}"].to_numpy()[idx]; p = test[f"pred_P3F_{k}"].to_numpy()[idx]
            series.append((target, horizon, marker, y, p))
        maxv = max(test[f"obs_{key('T1_PO4','1h')}"] .quantile(0.997), test[f"obs_{key('T1_NH4','4h')}"] .quantile(0.997))
        for target, horizon, marker, y, p in series:
            color = TARGET_COLOR[target]
            visible = (y >= 0) & (p >= 0) & (y <= maxv) & (p <= maxv)
            yv, pv = y[visible], p[visible]
            ax.scatter(yv, pv, s=7, marker=marker, color=color, alpha=0.075, linewidth=0, rasterized=True, zorder=1)
            density, x_edges, y_edges = np.histogram2d(yv, pv, bins=44, range=[[0, maxv], [0, maxv]])
            density = gaussian_filter(density, sigma=1.15)
            positive = density[density > 0]
            levels = np.unique(np.quantile(positive, [0.58, 0.78, 0.92])) if positive.size else np.array([])
            if levels.size:
                x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
                y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
                ax.contour(x_centers, y_centers, density.T, levels=levels, colors=[color], linewidths=np.linspace(0.7, 1.35, levels.size), alpha=0.86, zorder=2)
            bin_edges = np.unique(np.quantile(yv, np.linspace(0.02, 0.98, 11)))
            med_x, q25, med_y, q75 = [], [], [], []
            for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
                mask = (yv >= lo) & (yv <= hi)
                if mask.sum() < 10:
                    continue
                med_x.append(np.median(yv[mask]))
                q25.append(np.quantile(pv[mask], 0.25))
                med_y.append(np.median(pv[mask]))
                q75.append(np.quantile(pv[mask], 0.75))
            ax.fill_between(med_x, q25, q75, color=color, alpha=0.10, linewidth=0, zorder=2)
            ax.plot(med_x, med_y, color=color, marker=marker, markersize=4.2, linewidth=1.7, zorder=3, label=f"{TARGET_LABEL[target]}, {horizon}")
        ax.plot([0, maxv], [0, maxv], color=GRAY, linestyle="--", linewidth=1.3)
        ax.set_xlim(0, maxv); ax.set_ylim(0, maxv)
        style_axis(ax, xlabel="Observed concentration (mg/L)", ylabel="P³F prediction (mg/L)")
        numeric_ticks(ax, n=5)
        ax.legend(loc="upper left", fontsize=9.2)

    def c(ax, standalone):
        window = test.loc["2023-02-06":"2023-02-08 23:50"].copy()
        offsets = {"T1_PO4": 1.2, "T1_NH4": 0.0}
        for target in TARGETS:
            k = key(target, "4h")
            center = window[f"obs_{k}"].median(); scale = window[f"obs_{k}"].quantile(0.75) - window[f"obs_{k}"].quantile(0.25)
            scale = max(scale, 1e-6)
            obs = (window[f"obs_{k}"] - center) / scale + offsets[target]
            mid = (window[f"pred_P3F_{k}"] - center) / scale + offsets[target]
            lo = (window[f"lo90_{k}"] - center) / scale + offsets[target]
            hi = (window[f"hi90_{k}"] - center) / scale + offsets[target]
            ax.fill_between(window.index, lo, hi, color=TARGET_COLOR[target], alpha=0.16, linewidth=0)
            ax.plot(window.index, obs, color=TARGET_COLOR[target], linewidth=1.0, alpha=0.55)
            ax.plot(window.index, mid, color=TARGET_COLOR[target], linewidth=1.8, label=TARGET_LABEL[target])
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d\n%b"))
        style_axis(ax, xlabel="Future-test date", ylabel="Normalized concentration + offset")
        ax.legend(loc="best")

    def d(ax, standalone):
        p3f = boot.query("model == 'P3F'")
        for target in TARGETS:
            sub = p3f.query("target == @target").set_index("horizon").loc[HORIZONS]
            x = np.array([HOURS[h] for h in HORIZONS])
            ax.plot(x, sub["median"], marker="o", color=TARGET_COLOR[target], linewidth=2.2, label=TARGET_LABEL[target])
            ax.fill_between(x, sub["ci_low"], sub["ci_high"], color=TARGET_COLOR[target], alpha=0.18)
        ax.axhline(0, color=GRAY, linestyle="--", linewidth=1.2)
        style_axis(ax, xlabel="Forecast horizon (h)", ylabel="MAE skill with block-bootstrap CI")
        numeric_ticks(ax, n=5)
        ax.legend(loc="best")

    def e(ax, standalone):
        for target in TARGETS:
            for horizon, linestyle in [("1h", "-"), ("4h", "--")]:
                k = key(target, horizon)
                precision, recall, _ = precision_recall_curve(test[f"event_label_{k}"], test[f"event_probability_{k}"])
                ap = average_precision_score(test[f"event_label_{k}"], test[f"event_probability_{k}"])
                ax.plot(recall, precision, color=TARGET_COLOR[target], linestyle=linestyle, linewidth=2.0, label=f"{TARGET_LABEL[target]}, {horizon} (AP={ap:.2f})")
        style_axis(ax, xlabel="Recall", ylabel="Precision")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        numeric_ticks(ax, n=5)
        ax.legend(loc="best", fontsize=8.5)

    def f(ax, standalone):
        for target in TARGETS:
            k = key(target, "4h")
            quantile = pd.qcut(test[f"obs_{k}"], q=5, labels=False, duplicates="drop")
            values = []
            for q in sorted(quantile.dropna().unique()):
                mask = quantile.eq(q)
                values.append(np.mean(np.abs(test.loc[mask, f"obs_{k}"] - test.loc[mask, f"pred_P3F_{k}"])))
            scale = np.mean(np.abs(test[f"obs_{k}"] - test[f"pred_Persistence_{k}"]))
            ax.plot(np.arange(1, len(values) + 1), np.asarray(values) / scale, marker="o", color=TARGET_COLOR[target], linewidth=2.2, label=TARGET_LABEL[target])
        ax.axhline(1, color=GRAY, linestyle="--", linewidth=1.2)
        ax.set_xticks(range(1, 6), ["Q1", "Q2", "Q3", "Q4", "Q5"])
        style_axis(ax, xlabel="Observed-concentration quintile", ylabel="P³F MAE / persistence MAE")
        numeric_ticks(ax, x=False, y=True)
        ax.legend(loc="best")

    return [a, b, c, d, e, f]


def fig06(predictions, uncertainty):
    test = predictions.query("split == 'test'").copy()
    short_target = {"T1_PO4": r"PO$_4$", "T1_NH4": r"NH$_4$"}

    def legend_box(ax, **kwargs):
        legend = ax.legend(
            frameon=True,
            facecolor="white",
            framealpha=0.94,
            edgecolor="none",
            borderpad=0.30,
            labelspacing=0.28,
            handletextpad=0.45,
            columnspacing=0.75,
            **kwargs,
        )
        legend.set_zorder(10)
        return legend

    def coverage(target, horizon, level="90", sample=None):
        sub = test if sample is None else sample
        k = key(target, horizon)
        return np.mean((sub[f"obs_{k}"] >= sub[f"lo{level}_{k}"]) & (sub[f"obs_{k}"] <= sub[f"hi{level}_{k}"]))

    def a(ax, standalone):
        horizon_offsets = np.linspace(-0.024, 0.024, len(HORIZONS))
        for target in TARGETS:
            for level, nominal, marker in [("50", 0.5, "o"), ("90", 0.9, "s")]:
                empirical = [coverage(target, h, level) for h in HORIZONS]
                x_values = nominal + horizon_offsets
                ax.scatter(x_values, empirical, s=42, marker=marker, facecolors="white", edgecolors=TARGET_COLOR[target], linewidth=1.5, label=f"{short_target[target]}, {level}%", zorder=3)
                for x, y in zip(x_values, empirical):
                    ax.plot([nominal, x], [nominal, y], color=TARGET_COLOR[target], alpha=0.32, linewidth=0.9, zorder=1)
        ax.plot([0.4, 0.96], [0.4, 0.96], color=GRAY, linestyle="--", linewidth=1.2)
        ax.set_xlim(0.4, 0.96); ax.set_ylim(0.38, 0.96)
        style_axis(ax, xlabel="Nominal coverage", ylabel="Empirical future-test coverage")
        numeric_ticks(ax, n=5)
        legend_box(ax, loc="upper left", ncol=2, fontsize=7.5)

    def b(ax, standalone):
        for target in TARGETS:
            for calibrated, marker in [(False, "x"), (True, "o")]:
                cover = []; width = []
                for horizon in HORIZONS:
                    k = key(target, horizon)
                    if calibrated:
                        lo, hi = test[f"lo90_{k}"], test[f"hi90_{k}"]
                    else:
                        lo, hi = test[f"lo90_in_sample_{k}"], test[f"hi90_in_sample_{k}"]
                    cover.append(np.mean((test[f"obs_{k}"] >= lo) & (test[f"obs_{k}"] <= hi)))
                    iqr = test[f"obs_{k}"].quantile(0.75) - test[f"obs_{k}"].quantile(0.25)
                    width.append(np.mean(hi - lo) / max(iqr, 1e-6))
                ax.plot(width, cover, marker=marker, linestyle="-" if calibrated else ":", color=TARGET_COLOR[target], linewidth=1.9, label=f"{short_target[target]} {'conformal' if calibrated else 'in-sample'}")
        ax.axhline(0.9, color=GRAY, linestyle="--", linewidth=1.2)
        style_axis(ax, xlabel="Mean 90% width / target IQR", ylabel="Empirical 90% coverage")
        numeric_ticks(ax, n=5)
        legend_box(ax, loc="lower right", fontsize=7.6)

    def c(ax, standalone):
        bins = pd.qcut(test["ad_score"], 8, duplicates="drop")
        for target in TARGETS:
            k = key(target, "4h")
            base_mae = np.mean(np.abs(test[f"obs_{k}"] - test[f"pred_P3F_{k}"]))
            x=[]; error=[]; width=[]
            for category in bins.cat.categories:
                mask = bins.eq(category)
                x.append(test.loc[mask, "ad_score"].median())
                error.append(np.mean(np.abs(test.loc[mask, f"obs_{k}"] - test.loc[mask, f"pred_P3F_{k}"])) / base_mae)
                width.append(np.mean(test.loc[mask, f"hi90_{k}"] - test.loc[mask, f"lo90_{k}"]) / np.mean(test[f"hi90_{k}"] - test[f"lo90_{k}"]))
            ax.plot(x, error, color=TARGET_COLOR[target], marker="o", linewidth=2.1, label=f"{short_target[target]} error")
            ax.plot(x, width, color=TARGET_COLOR[target], marker="s", linestyle="--", linewidth=1.7, alpha=0.75, label=f"{short_target[target]} width")
        ax.axhline(1, color=GRAY, linestyle=":", linewidth=1.2)
        style_axis(ax, xlabel="Applicability-domain distance", ylabel="Ratio to full-test mean")
        numeric_ticks(ax, n=5)
        legend_box(ax, loc="lower left", ncol=2, fontsize=7.4)

    def d(ax, standalone):
        ordered = test.sort_values("ad_score")
        fractions = np.linspace(0.2, 1.0, 9)
        for target in TARGETS:
            k = key(target, "4h")
            full_mae = np.mean(np.abs(ordered[f"obs_{k}"] - ordered[f"pred_P3F_{k}"]))
            ratios=[]; covers=[]
            for fraction in fractions:
                sub = ordered.iloc[: max(10, int(len(ordered) * fraction))]
                ratios.append(np.mean(np.abs(sub[f"obs_{k}"] - sub[f"pred_P3F_{k}"])) / full_mae)
                covers.append(np.mean((sub[f"obs_{k}"] >= sub[f"lo90_{k}"]) & (sub[f"obs_{k}"] <= sub[f"hi90_{k}"])))
            ax.plot(fractions, ratios, color=TARGET_COLOR[target], marker="o", linewidth=2.0, label=f"{short_target[target]} error")
            ax.plot(fractions, covers, color=TARGET_COLOR[target], marker="s", linestyle="--", linewidth=1.7, label=f"{short_target[target]} coverage")
        ax.axhline(0.9, color=GRAY, linestyle=":", linewidth=1.1)
        style_axis(ax, xlabel="Retained fraction from most in-domain", ylabel="Error ratio or coverage")
        ax.set_xlim(0.18, 1.02)
        numeric_ticks(ax, n=5)
        legend_box(
            ax,
            loc="upper right",
            bbox_to_anchor=(0.98, 0.98),
            ncol=2,
            fontsize=7.1,
        )

    def wilson(p, n, z=1.96):
        den=1+z*z/n; center=(p+z*z/(2*n))/den; half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
        return center-half, center+half

    def e(ax, standalone):
        groups = ["all", "phase 1", "phase 2", "recent transition", "at/above training Q90", "borderline/far-OOD"]
        y_pos = np.arange(len(groups))
        for target, offset in [("T1_PO4", -0.10), ("T1_NH4", 0.10)]:
            values=[]; low=[]; high=[]
            for group in groups:
                row = uncertainty.query("target == @target and horizon == '4h' and group == @group").iloc[0]
                lo, hi = wilson(row.coverage90, row.n)
                values.append(row.coverage90); low.append(row.coverage90-lo); high.append(hi-row.coverage90)
            ax.errorbar(values, y_pos+offset, xerr=[low, high], fmt="o" if target=="T1_PO4" else "s", color=TARGET_COLOR[target], markerfacecolor="white", capsize=3, linewidth=1.5, label=TARGET_LABEL[target])
        ax.axvline(0.9, color=GRAY, linestyle="--", linewidth=1.2)
        ax.set_yticks(y_pos, groups); ax.invert_yaxis()
        style_axis(ax, xlabel="Four-hour 90% coverage (Wilson CI)", ylabel="")
        numeric_ticks(ax, x=True, y=False)
        legend_box(ax, loc="upper center", ncol=2, fontsize=8.1)

    def f(ax, standalone):
        weekly = test.resample("7D").agg(ad=("ad_score", "median"))
        ad_min, ad_max = weekly.ad.min(), weekly.ad.max()
        weekly["drift_scaled"] = (weekly.ad - ad_min) / max(ad_max - ad_min, 1e-9)
        for target in TARGETS:
            k = key(target, "4h")
            covered = ((test[f"obs_{k}"] >= test[f"lo90_{k}"]) & (test[f"obs_{k}"] <= test[f"hi90_{k}"])).astype(float)
            weekly[f"coverage_{target}"] = covered.resample("7D").mean()
            ax.plot(weekly.index, weekly[f"coverage_{target}"], color=TARGET_COLOR[target], linewidth=1.9, label=f"{TARGET_LABEL[target]} coverage")
        ax.plot(weekly.index, weekly["drift_scaled"], color=PURPLE, linestyle="--", linewidth=1.7, label="scaled drift score")
        ax.axhline(0.9, color=GRAY, linestyle=":", linewidth=1.2)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2)); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
        style_axis(ax, xlabel="Untouched future-test time", ylabel="Weekly coverage or scaled drift")
        ax.set_ylim(0, 1.05)
        legend_box(ax, loc="lower right", fontsize=7.7)

    return [a, b, c, d, e, f]


def fig07(occlusion, contributions, ale, network):
    contrib = contributions.groupby(["target", "horizon", "family", "subgroup"], as_index=False)["mean_abs_contribution"].sum()

    def a(ax, standalone):
        rows=[]; labels=[]
        for target in TARGETS:
            for horizon in HORIZONS:
                sub = occlusion.query("target == @target and horizon == @horizon").set_index("lag_band")
                bands=["state/current", "recent <=1 h", "short 1-6 h", "circadian 12-24 h", "long 48 h"]
                values=sub.loc[bands,"mean_abs_prediction_change"].to_numpy()
                values=values/max(values.sum(),1e-9)
                rows.append(values); labels.append(f"{TARGET_LABEL[target]} {horizon}")
        image=ax.imshow(rows, cmap=LOW_SAT_SEQUENTIAL, aspect="auto", vmin=0)
        ax.set_xticks(range(5), ["state", "≤1 h", "1–6 h", "12–24 h", "48 h"], rotation=28, ha="right")
        ax.set_yticks(range(len(labels)), labels)
        for i,row in enumerate(rows):
            for j,value in enumerate(row): ax.text(j,i,f"{value:.2f}",ha="center",va="center",fontsize=8.0,fontweight="bold",color="white" if value>0.35 else STYLE_COLORS["ink"])
        for t in ax.get_xticklabels()+ax.get_yticklabels(): t.set_fontweight("bold")
        ax.tick_params(bottom=True,left=True,length=5,width=1.2,labelsize=9.5)
        for s in ax.spines.values(): s.set_visible(False)
        add_colorbar(ax.figure,ax,image,"Normalized occlusion effect")

    def b(ax, standalone):
        families=["PO4 history","NH4 history","Dissolved O2","Metal dosing","Influent flow","Process phase"]
        y=np.arange(len(families))
        for target,offset,marker in [("T1_PO4",-0.12,"o"),("T1_NH4",0.12,"s")]:
            phase_values = {}
            for phase,alpha in [(1,1.0),(2,0.45)]:
                subgroup = f"phase {phase}"
                sub=contrib.query("target == @target and horizon == '4h' and subgroup == @subgroup").set_index("family")
                values=np.array([sub.loc[f,"mean_abs_contribution"] if f in sub.index else 0 for f in families])
                values=values/max(values.sum(),1e-9)
                phase_values[phase] = values
            for row_y, v1, v2 in zip(y + offset, phase_values[1], phase_values[2]):
                ax.plot([v1, v2], [row_y, row_y], color=TARGET_COLOR[target], alpha=0.36, linewidth=1.4, zorder=1)
            ax.scatter(phase_values[1],y+offset,s=70,marker=marker,facecolors=TARGET_COLOR[target],edgecolors=TARGET_COLOR[target],linewidth=1.3,zorder=3,label=f"{TARGET_LABEL[target]}, phase 1")
            ax.scatter(phase_values[2],y+offset,s=70,marker=marker,facecolors="white",edgecolors=TARGET_COLOR[target],linewidth=1.4,zorder=4,label=f"{TARGET_LABEL[target]}, phase 2")
        ax.set_yticks(y,[display_family(family) for family in families]); ax.invert_yaxis()
        style_axis(ax,xlabel="Normalized mean |TreeSHAP|",ylabel="")
        numeric_ticks(ax,x=True,y=False)
        ax.legend(loc="lower right",fontsize=8.0)

    def c(ax, standalone):
        families=sorted(contrib["family"].unique())
        po=contrib.query("target == 'T1_PO4' and subgroup == 'all'").groupby("family")["mean_abs_contribution"].mean()
        nh=contrib.query("target == 'T1_NH4' and subgroup == 'all'").groupby("family")["mean_abs_contribution"].mean()
        po=po/po.sum(); nh=nh/nh.sum()
        label_offsets = {
            "PO4 history": (5, -2),
            "NH4 history": (5, 8),
            "Process phase": (6, 6),
            "Influent flow": (6, 5),
            "Dissolved O2": (6, -15),
            "Calendar": (6, 10),
            "Metal dosing": (6, -14),
            "Control factor": (-5, 7),
            "Temperature": (6, 10),
        }
        label_align = {"Control factor": "right"}
        for i,family in enumerate(families):
            ax.scatter(po.get(family,0),nh.get(family,0),s=92,color=PALETTE[i%len(PALETTE)],edgecolor="white",linewidth=0.9)
            ax.annotate(
                display_family(family),
                (po.get(family, 0), nh.get(family, 0)),
                xytext=label_offsets.get(family, (4, 3)),
                textcoords="offset points",
                fontsize=7.5,
                fontweight="bold",
                ha=label_align.get(family, "left"),
            )
        limit=max(max(po.max(),nh.max())*1.08,0.05)
        ax.plot([0,limit],[0,limit],color=GRAY,linestyle="--",linewidth=1.2)
        ax.set_xlim(0,limit); ax.set_ylim(0,limit)
        style_axis(ax,xlabel=r"PO$_4$ contribution share",ylabel=r"NH$_4$ contribution share")
        numeric_ticks(ax,n=5)

    def d(ax, standalone):
        feature_labels={"IN_METAL_Q":"inlet metal dose","T1_O2":r"T1 O$_2$","IN_Q":"influent flow","TEMPERATURE":"temperature"}
        styles={"IN_METAL_Q":"-","T1_O2":"--","IN_Q":"-.","TEMPERATURE":":"}
        for target in TARGETS:
            for feature in ["IN_METAL_Q", "T1_O2"]:
                sub=ale.query("target == @target and feature == @feature")
                for phase,alpha in [(1,1.0),(2,0.48)]:
                    curve=sub.query("phase == @phase").sort_values("feature_value")
                    scale=max(curve["ale"].abs().max(),1e-9)
                    label=f"{TARGET_LABEL[target]}, {feature_labels[feature]}" if phase == 1 else None
                    ax.plot(np.linspace(0,1,len(curve)),curve["ale"]/scale,color=TARGET_COLOR[target],linestyle=styles[feature],alpha=alpha,linewidth=2.0,label=label)
        ax.axhline(0,color=GRAY,linestyle=":",linewidth=1.1)
        style_axis(ax,xlabel="Within-feature empirical quantile",ylabel="Centered ALE / feature maximum")
        numeric_ticks(ax,n=5)
        ax.legend(loc="lower left", fontsize=8.2)

    def draw_network(ax, phase, center_x):
        names=["T1_PO4","T1_NH4","T1_O2","IN_METAL_Q","IN_Q","TEMPERATURE","MAX_CF"]
        labels={"T1_PO4":r"PO$_4$","T1_NH4":r"NH$_4$","T1_O2":r"O$_2$","IN_METAL_Q":"dose","IN_Q":"flow","TEMPERATURE":"temp","MAX_CF":"control"}
        node_colors={
            "T1_PO4": BLUE,
            "T1_NH4": CORAL,
            "T1_O2": TEAL,
            "IN_METAL_Q": GOLD,
            "IN_Q": GREEN,
            "TEMPERATURE": BLUSH,
            "MAX_CF": MAUVE,
        }
        angles=np.linspace(0,2*np.pi,len(names),endpoint=False)+np.pi/2
        pos={name:(center_x+0.30*np.cos(angle),0.50+0.32*np.sin(angle)) for name,angle in zip(names,angles)}
        subset=network.loc[(network["phase"] == phase) & (network["seasons"] >= 2) & (network["sign_consistency"] >= 0.75) & (network["median_correlation"].abs() >= 0.25)]
        for row in subset.itertuples():
            if row.source not in pos or row.target not in pos: continue
            x0,y0=pos[row.source]; x1,y1=pos[row.target]
            ax.plot([x0,x1],[y0,y1],color=BLUE if row.median_correlation>0 else CORAL,linewidth=0.6+3.2*abs(row.median_correlation),alpha=0.50,zorder=1)
        for name,(x,y) in pos.items():
            color=node_colors[name]
            ax.scatter([x],[y],s=1000,color=color,edgecolor=NAVY,linewidth=1.35,zorder=3)
            ax.text(x,y,labels[name],ha="center",va="center",fontsize=9.2,fontweight="bold",zorder=4)
        ax.text(center_x,0.94,f"phase {phase}",ha="center",va="center",fontsize=10.5,fontweight="bold")

    def e(ax, standalone):
        ax.set_xlim(0.12,1.88); ax.set_ylim(0,1); ax.axis("off")
        draw_network(ax,1,0.60); draw_network(ax,2,1.40)
        ax.plot([],[],color=BLUE,linewidth=2,label="stable positive association")
        ax.plot([],[],color=CORAL,linewidth=2,label="stable negative association")
        ax.legend(loc="lower center",ncol=2,fontsize=8.4)

    return [a,b,c,d,e]


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--only",choices=["all","Fig03","Fig04","Fig05","Fig06","Fig07"],default="all")
    args=parser.parse_args()
    metadata=json.loads(META_PATH.read_text(encoding="utf-8"))
    metrics_df=pd.read_csv(TABLE_DIR/"Table3_model_metrics.csv")
    boot=pd.read_csv(TABLE_DIR/"Table3_block_bootstrap_skill.csv")
    split_diag=pd.read_csv(TABLE_DIR/"Fig03_split_protocol_diagnostic.csv")
    volatility=pd.read_csv(TABLE_DIR/"Fig03_volatility_skill.csv")
    compute=pd.read_csv(TABLE_DIR/"TableS_compute_profile.csv")
    ablation=pd.read_csv(TABLE_DIR/"Table4_ablation_metrics.csv")
    uncertainty=pd.read_csv(TABLE_DIR/"Table5_uncertainty_subgroups.csv")
    predictions=pd.read_parquet(PRED_PATH)
    ablation_pred=pd.read_parquet(ABLATION_PRED_PATH)
    ablation_pred["date"]=pd.to_datetime(ablation_pred["date"])
    if args.only in ["all","Fig03"]: export_figure("Fig03_leakage_baseline_benchmark","Fig03",fig03(metrics_df,split_diag,volatility,compute,metadata), wspace=0.36)
    if args.only in ["all","Fig04"]:
        export_figure_mosaic(
            "Fig04_P3F_ablation",
            "Fig04",
            fig04(ablation,ablation_pred,predictions),
            [(0,0,2),(0,2,3),(1,0,1),(1,1,2),(1,2,3)],
            wspace=0.48,
            standalone_sizes=[(10.2,5.0),(6.7,5.0),(6.7,5.0),(6.7,5.0),(6.7,5.0)],
        )
    if args.only in ["all","Fig05"]: export_figure("Fig05_future_test_performance","Fig05",fig05(metrics_df,boot,predictions), wspace=0.48)
    if args.only in ["all","Fig06"]: export_figure("Fig06_uncertainty_domain","Fig06",fig06(predictions,uncertainty))
    if args.only in ["all","Fig07"]:
        occlusion=pd.read_csv(TABLE_DIR/"Fig07_temporal_occlusion.csv")
        contributions=pd.read_parquet(TABLE_DIR/"Fig07_feature_contributions.parquet")
        ale=pd.read_csv(TABLE_DIR/"Fig07_ale_curves.csv")
        network=pd.read_csv(TABLE_DIR/"Fig07_dynamic_network_summary.csv")
        export_figure_mosaic(
            "Fig07_phase_conditioned_dependence",
            "Fig07",
            fig07(occlusion,contributions,ale,network),
            [(0,0,1),(0,1,2),(0,2,3),(1,0,1),(1,1,3)],
            combined_size=(16.8,9.2),
            wspace=0.44,
            standalone_sizes=[(6.7,5.0),(6.7,5.0),(6.7,5.0),(6.7,5.0),(9.2,5.0)],
        )
    print(json.dumps({"status":"ok","only":args.only}))


if __name__=="__main__":
    main()
