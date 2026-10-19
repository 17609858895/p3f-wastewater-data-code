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
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle, FancyBboxPatch
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FixedFormatter


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
TABLES = ROOT / "tables"
DATA = ROOT / "data" / "raw" / "IOPTQCfFiFoNPo_2min_Agtrup_Aug_2023.csv"

# The installed Nature-style helpers live in the figures root.
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
TABLES.mkdir(parents=True, exist_ok=True)

INK = STYLE_COLORS["ink"]
MUTED = STYLE_COLORS["muted"]
LIGHT = STYLE_COLORS["light"]
SOFT = get_palette("nature_new", n=9)
# Use the exact canonical nature_new registry values for every semantic color.
CORAL, BLUE, GREEN, TEAL, PURPLE, _, GOLD, _, _ = SOFT


def _box(ax, xy, width, height, text, fc, ec=INK, fontsize=10.5,
         radius=0.03, lw=1.4, text_color=INK, zorder=3):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zorder,
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=text_color, zorder=zorder + 1)
    return patch


def _arrow(ax, start, end, color=INK, lw=1.6, rad=0.0, style="-|>", zorder=4):
    patch = FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=13,
        linewidth=lw, color=color,
        connectionstyle=f"arc3,rad={rad}", zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def _clean_schematic_axis(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def draw_process_panel(ax):
    _clean_schematic_axis(ax)
    # Influent and chemical-dosing path.
    ax.add_patch(Rectangle((0.02, 0.48), 0.13, 0.12, facecolor="#D9DEE4", edgecolor=INK, lw=1.4))
    ax.text(0.085, 0.64, "Influent", ha="center", va="bottom", fontsize=10, fontweight="bold")
    _arrow(ax, (0.15, 0.54), (0.26, 0.54), color=BLUE)
    _box(ax, (0.02, 0.12), 0.18, 0.13, "Metal-\nsalt\ndosing", SOFT[6], fontsize=9.5)
    _arrow(ax, (0.20, 0.185), (0.36, 0.43), color=CORAL, rad=-0.12)

    # Two biological tanks; only Tank 1 targets are highlighted.
    for i, (x, label, fill) in enumerate([(0.26, "Tank 1", SOFT[1]), (0.60, "Tank 2", SOFT[3])]):
        ax.add_patch(FancyBboxPatch((x, 0.36), 0.27, 0.34,
                                   boxstyle="round,pad=0.01,rounding_size=0.035",
                                   facecolor=fill, edgecolor=INK, lw=1.5))
        ax.plot([x + 0.03, x + 0.24], [0.47, 0.47], color="#FFFFFF", lw=2.0, alpha=0.85)
        bubbles_x = np.linspace(x + 0.05, x + 0.22, 5)
        for j, bx in enumerate(bubbles_x):
            ax.add_patch(Circle((bx, 0.43 + 0.035 * (j % 2)), 0.012,
                                facecolor="white", edgecolor=BLUE, lw=0.8))
        ax.text(x + 0.135, 0.75, label, ha="center", va="center",
                fontsize=10.5, fontweight="bold", color=INK)
        # Sensor glyph.
        ax.plot([x + 0.135, x + 0.135], [0.70, 0.84], color=INK, lw=1.5)
        ax.add_patch(Circle((x + 0.135, 0.86), 0.023, facecolor="white", edgecolor=BLUE, lw=1.4))

    _arrow(ax, (0.53, 0.54), (0.60, 0.54), color=BLUE)
    _arrow(ax, (0.87, 0.54), (0.98, 0.54), color=BLUE)
    ax.text(0.94, 0.59, "Process\nstream", ha="center", va="bottom",
            fontsize=9.5, fontweight="bold", color=MUTED)

    # SCADA variables and targets.
    _box(ax, (0.20, 0.05), 0.65, 0.17,
         "SCADA: flow  |  O$_2$  |  temperature\nphase  |  targets: T1-PO$_4$, T1-NH$_4$",
         "#FFFFFF", ec=BLUE, fontsize=8.9, radius=0.025)
    _arrow(ax, (0.395, 0.36), (0.44, 0.22), color=BLUE)
    _arrow(ax, (0.735, 0.36), (0.64, 0.22), color=TEAL)


def draw_forecast_panel(ax):
    _clean_schematic_axis(ax)
    t_hist = np.linspace(-48, 0, 260)
    y1 = 0.53 + 0.075 * np.sin(t_hist / 4.7) + 0.025 * np.sin(t_hist / 1.6)
    y2 = 0.31 + 0.065 * np.cos(t_hist / 5.4 + 0.8) + 0.02 * np.sin(t_hist / 1.2)
    ax.axvspan(0.71, 0.98, color="#FCEDEB", zorder=0)
    ax.axvline(0.71, color=INK, lw=1.6, ls="--", dashes=(4, 3))
    x_hist = 0.05 + (t_hist + 48) / 48 * 0.66
    ax.plot(x_hist, y1, color=BLUE, lw=2.0)
    ax.plot(x_hist, y2, color=CORAL, lw=2.0)
    ax.text(0.71, 0.90, "issuance time  t", ha="center", va="center",
            fontsize=10, fontweight="bold", color=INK)
    ax.text(0.38, 0.79, "history available at t", ha="center", fontsize=10.5,
            fontweight="bold", color=BLUE)
    ax.text(0.845, 0.79, "future controls\nnot observed", ha="center", fontsize=9.5,
            fontweight="bold", color=CORAL)
    ax.text(0.035, 0.53, "PO$_4$", ha="right", va="center", fontsize=10, fontweight="bold", color=BLUE)
    ax.text(0.035, 0.31, "NH$_4$", ha="right", va="center", fontsize=10, fontweight="bold", color=CORAL)
    for h, xpos in zip([0.5, 1, 2, 4, 6], [0.735, 0.782, 0.835, 0.900, 0.965]):
        ax.plot([xpos, xpos], [0.16, 0.23], color=PURPLE, lw=1.6)
        ax.scatter([xpos], [0.23], s=36, color=PURPLE, edgecolor="white", lw=0.7, zorder=3)
        ax.text(xpos, 0.105, f"{h:g}", ha="center", va="center", fontsize=7.6, fontweight="bold")
    _arrow(ax, (0.72, 0.20), (0.97, 0.20), color=PURPLE, lw=1.4)
    ax.text(0.845, 0.04, "horizon (h)", ha="center", va="center",
            fontsize=9.3, fontweight="bold", color=PURPLE)


def draw_architecture_panel(ax):
    _clean_schematic_axis(ax)
    # Multiscale numeric tokens.
    branch_y = [0.73, 0.49, 0.25]
    branch_names = ["short\n2 h", "medium\n12 h", "long\n48 h"]
    fills = [SOFT[1], SOFT[3], SOFT[4]]
    for y, name, fill in zip(branch_y, branch_names, fills):
        _box(ax, (0.02, y - 0.08), 0.14, 0.16, name, fill, fontsize=9.3)
        for j in range(3):
            ax.add_patch(Rectangle((0.19 + j * 0.035, y - 0.045), 0.025, 0.09,
                                   facecolor=fill, edgecolor=INK, lw=0.9))
        _arrow(ax, (0.16, y), (0.185, y), color=INK, lw=1.2)
        _arrow(ax, (0.29, y), (0.39, 0.56), color=BLUE, lw=1.2)

    _box(ax, (0.31, 0.04), 0.22, 0.15, "phase + process\ncontext features", SOFT[6], ec=CORAL, fontsize=9.1)
    _arrow(ax, (0.42, 0.19), (0.45, 0.39), color=CORAL, lw=1.6)

    _box(ax, (0.39, 0.34), 0.25, 0.44, "P$^3$F\n\nrobust gradient experts\n────────\nhorizon-specific Δy",
         "#FFFFFF", ec=BLUE, fontsize=10.2, radius=0.035, lw=1.8)
    _arrow(ax, (0.16, 0.15), (0.78, 0.58), color=TEAL, lw=1.6, rad=0.22)
    ax.text(0.65, 0.045, "current-target persistence skip", ha="center", va="center",
            fontsize=8.8, fontweight="bold", color=TEAL)

    _box(ax, (0.70, 0.56), 0.26, 0.20, "restore absolute level\ny(t) + Δŷ(t+h)", SOFT[0], ec=PURPLE, fontsize=9.6)
    _arrow(ax, (0.64, 0.56), (0.70, 0.66), color=PURPLE)
    _box(ax, (0.76, 0.27), 0.20, 0.15, "PO$_4$ private\nhead", SOFT[1], ec=BLUE, fontsize=9.3)
    _box(ax, (0.76, 0.06), 0.20, 0.15, "NH$_4$ private\nhead", SOFT[0], ec=CORAL, fontsize=9.3)
    _arrow(ax, (0.83, 0.56), (0.85, 0.42), color=BLUE)
    _arrow(ax, (0.87, 0.56), (0.86, 0.21), color=CORAL)


def draw_output_panel(ax):
    _clean_schematic_axis(ax)
    _box(ax, (0.02, 0.37), 0.19, 0.25, "multi-horizon\npoint forecast", SOFT[1], ec=BLUE, fontsize=10)
    _box(ax, (0.02, 0.07), 0.19, 0.21, "event\nprobability", SOFT[6], ec=CORAL, fontsize=10)
    _box(ax, (0.28, 0.37), 0.19, 0.25, "calibration\nresiduals", SOFT[3], ec=TEAL, fontsize=10)
    _box(ax, (0.28, 0.07), 0.19, 0.21, "independent\nsplit conformal", SOFT[2], ec=GREEN, fontsize=10)
    _arrow(ax, (0.21, 0.49), (0.28, 0.49), color=INK)
    _arrow(ax, (0.21, 0.18), (0.28, 0.18), color=INK)

    # Forecast mini-panels.
    for y0, color, label in [(0.59, BLUE, "T1-PO$_4$"), (0.22, CORAL, "T1-NH$_4$")]:
        x = np.linspace(0.57, 0.96, 100)
        trend = y0 + 0.05 * np.sin(np.linspace(0, 4.6, 100)) + np.linspace(-0.02, 0.08, 100)
        band = 0.035 + np.linspace(0.01, 0.045, 100)
        ax.fill_between(x, trend - band, trend + band, color=color, alpha=0.20, lw=0)
        ax.plot(x, trend, color=color, lw=2.1)
        ax.plot([0.57, 0.96], [y0 + 0.08, y0 + 0.08], color=MUTED, lw=1.2, ls="--")
        ax.text(0.53, y0, label, ha="right", va="center", fontsize=10, fontweight="bold", color=color)
    ax.add_patch(Polygon([[0.91, 0.76], [0.95, 0.68], [0.87, 0.68]], closed=True,
                         facecolor="#FFFFFF", edgecolor=CORAL, lw=1.5))
    ax.text(0.91, 0.707, "!", ha="center", va="center", fontsize=11, fontweight="bold", color=CORAL)
    _box(ax, (0.68, 0.02), 0.24, 0.14, "warning or abstain", "#FFFFFF", ec=PURPLE, fontsize=9.5)


def draw_validation_panel(ax):
    _clean_schematic_axis(ax)
    segments = [
        (0.03, 0.31, SOFT[2], "training"),
        (0.35, 0.20, SOFT[1], "rolling validation"),
        (0.56, 0.16, SOFT[6], "calibration"),
        (0.73, 0.24, SOFT[0], "future test"),
    ]
    for x, w, fc, label in segments:
        ax.add_patch(Rectangle((x, 0.67), w, 0.15, facecolor=fc, edgecolor=INK, lw=1.2))
        ax.text(x + w / 2, 0.745, label, ha="center", va="center", fontsize=9.2, fontweight="bold")
    for x in [0.34, 0.55, 0.72]:
        ax.plot([x, x], [0.63, 0.86], color="white", lw=4.5, zorder=3)
        ax.plot([x, x], [0.63, 0.86], color=MUTED, lw=1.1, ls="--", zorder=4)
    ax.text(0.515, 0.89, "purged chronological protocol", ha="center", va="center",
            fontsize=10.3, fontweight="bold", color=INK)

    gates = [
        (0.05, "skill > 0\nvs persistence", BLUE),
        (0.38, "coverage-width\ncalibration", GREEN),
        (0.70, "applicability\ndomain", PURPLE),
    ]
    for x, label, color in gates:
        _box(ax, (x, 0.25), 0.24, 0.22, label, "#FFFFFF", ec=color, fontsize=9.7)
        _arrow(ax, (x + 0.12, 0.67), (x + 0.12, 0.47), color=color, lw=1.4)
    ax.text(0.50, 0.09, "credible early warning only inside the validated domain",
            ha="center", va="center", fontsize=10.5, fontweight="bold", color=INK)


def _save_standalone_panel(draw_fn, folder: Path, stem: str, suffix: str, figsize=(7.2, 5.0)):
    fig, ax = plt.subplots(figsize=figsize)
    draw_fn(ax)
    scale_figure_text(fig, factor=1.15)
    set_figure_text_weight(fig, "normal")
    fig.subplots_adjust(left=0.04, right=0.98, bottom=0.04, top=0.98)
    out = folder / "subfigures"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{stem}_{suffix}.png", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(out / f"{stem}_{suffix}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def generate_fig01():
    folder = FIGURES / "Fig01_P3FM_framework"
    folder.mkdir(parents=True, exist_ok=True)
    stem = "Fig01_P3FM_framework"
    fig = plt.figure(figsize=(15.5, 8.8))
    gs = fig.add_gridspec(2, 12, height_ratios=[1.03, 0.97], hspace=0.09, wspace=0.08)
    axes = [
        fig.add_subplot(gs[0, 0:3]),
        fig.add_subplot(gs[0, 3:6]),
        fig.add_subplot(gs[0, 6:12]),
        fig.add_subplot(gs[1, 0:6]),
        fig.add_subplot(gs[1, 6:12]),
    ]
    drawers = [draw_process_panel, draw_forecast_panel, draw_architecture_panel,
               draw_output_panel, draw_validation_panel]
    for ax, fn in zip(axes, drawers):
        fn(ax)
    scale_figure_text(fig, factor=1.15)
    set_figure_text_weight(fig, "normal")
    fig.subplots_adjust(left=0.018, right=0.986, bottom=0.025, top=0.985)
    # Required shared helper: creates combined PNG/PDF and an initial panel set.
    save_figure_with_subfigures(fig, folder, stem, dpi=600, close=True)
    # User preference: overwrite crops with independently rendered standalone panels.
    sizes = [(5.8, 4.6), (5.8, 4.6), (9.3, 4.6), (8.4, 4.3), (8.4, 4.3)]
    for suffix, fn, size in zip("abcde", drawers, sizes):
        _save_standalone_panel(fn, folder, stem, suffix, figsize=size)
    (folder / "subfigures" / "README.md").write_text(
        "# Subfigures\n\nIndependently rendered from the same panel builders used by the final combined figure.\n",
        encoding="utf-8",
    )


def acf_fft(values: np.ndarray, nlags: int) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    x = x - x.mean()
    n = len(x)
    fft_n = 1 << int(math.ceil(math.log2(max(2, 2 * n - 1))))
    f = np.fft.rfft(x, n=fft_n)
    cov = np.fft.irfft(f * np.conjugate(f), n=fft_n)[: nlags + 1]
    denom = np.arange(n, n - nlags - 1, -1, dtype=float)
    cov = cov / denom
    return cov / cov[0]


def load_data():
    df = pd.read_csv(DATA, parse_dates=["date"])
    df = df.sort_values("date").set_index("date")
    df.index = pd.DatetimeIndex(df.index)
    ten = df.resample("10min").median()
    return df, ten


def prepare_figure2_data(df: pd.DataFrame, ten: pd.DataFrame):
    targets = ["T1_PO4", "T1_NH4"]
    daily_median = df[targets].resample("1D").median()
    daily_q10 = df[targets].resample("1D").quantile(0.10)
    daily_q90 = df[targets].resample("1D").quantile(0.90)

    week = ten.loc["2022-08-01":"2022-08-07 23:59:59"].copy()

    nlags = 24 * 6
    acf = pd.DataFrame({
        "lag_h": np.arange(nlags + 1) / 6.0,
        "T1_PO4": acf_fft(ten["T1_PO4"].to_numpy(), nlags),
        "T1_NH4": acf_fft(ten["T1_NH4"].to_numpy(), nlags),
    })

    phase = ten["PROCESSPHASE_INLET"].round().clip(1, 2).astype(int).to_numpy()
    transitions = np.where(np.diff(phase) != 0)[0] + 1
    # Retain events at least 1 h apart and with a full +/- 6 h window.
    kept = []
    last = -10_000
    window = 36
    for idx in transitions:
        if idx < window or idx + window >= len(ten):
            continue
        if idx - last < 6:
            continue
        kept.append(idx)
        last = idx

    rel = np.arange(-window, window + 1) / 6.0
    phase_rows = {"elapsed_h": rel}
    direction_counts = {}
    for old_phase, new_phase, key in [(1, 2, "1to2"), (2, 1, "2to1")]:
        ids = [idx for idx in kept if phase[idx - 1] == old_phase and phase[idx] == new_phase]
        direction_counts[key] = len(ids)
        for target in targets:
            arr = ten[target].to_numpy(dtype=float)
            scale = np.nanpercentile(arr, 75) - np.nanpercentile(arr, 25)
            aligned = []
            for idx in ids:
                seg = arr[idx - window: idx + window + 1]
                baseline = np.nanmedian(arr[idx - 6: idx])
                aligned.append((seg - baseline) / scale)
            mat = np.vstack(aligned)
            phase_rows[f"{target}_{key}_q25"] = np.nanquantile(mat, 0.25, axis=0)
            phase_rows[f"{target}_{key}_median"] = np.nanmedian(mat, axis=0)
            phase_rows[f"{target}_{key}_q75"] = np.nanquantile(mat, 0.75, axis=0)
    phase_aligned = pd.DataFrame(phase_rows)
    return daily_median, daily_q10, daily_q90, week, acf, phase_aligned, direction_counts


def draw_two_year_panel(ax, daily_median, daily_q10, daily_q90):
    style_axis(ax, xlabel="Date", ylabel="Daily concentration (mg/L)", tick_size=10.5, label_size=12.8)
    for target, color, label in [
        ("T1_PO4", BLUE, "T1-PO$_4$"),
        ("T1_NH4", CORAL, "T1-NH$_4$"),
    ]:
        ax.fill_between(daily_median.index, daily_q10[target], daily_q90[target],
                        color=color, alpha=0.13, linewidth=0)
        ax.plot(daily_median.index, daily_median[target], color=color, lw=1.35, label=label)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.legend(loc="upper left", bbox_to_anchor=(0.01, 0.99), ncol=2,
              handlelength=2.3, columnspacing=1.2, borderaxespad=0.0)


def draw_week_panel(ax, week):
    variables = ["T1_PO4", "T1_NH4", "T1_O2", "METAL_Q", "IN_Q"]
    labels = ["T1-PO$_4$", "T1-NH$_4$", "O$_2$", "metal dose", "inflow"]
    colors = [BLUE, CORAL, TEAL, GOLD, PURPLE]
    offsets = np.arange(len(variables))[::-1]
    phase = week["PROCESSPHASE_INLET"].round().clip(1, 2)
    for off, var, color in zip(offsets, variables, colors):
        vals = week[var].astype(float)
        med = vals.median()
        iqr = vals.quantile(0.75) - vals.quantile(0.25)
        z = ((vals - med) / (iqr if iqr > 0 else 1.0)).clip(-2.5, 2.5)
        ax.plot(week.index, off + 0.24 * z, color=color, lw=1.15)
    ax.set_yticks(offsets)
    ax.set_yticklabels(labels)
    phase_y0, phase_y1 = len(variables) - 0.22, len(variables) + 0.02
    ax.fill_between(week.index, phase_y0, phase_y1, where=(phase == 1),
                    color=SOFT[2], alpha=0.85, step="mid", lw=0)
    ax.fill_between(week.index, phase_y0, phase_y1, where=(phase == 2),
                    color=SOFT[3], alpha=0.85, step="mid", lw=0)
    ax.set_ylim(-0.7, len(variables) + 0.08)
    ax.set_xlabel("Fixed seven-day window", fontweight="bold", fontsize=12.8)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d\n%b"))
    style_axis(ax, tick_size=10.2, label_size=12.8)
    ax.text(0.985, 0.975, "inlet phase 1 / 2", transform=ax.transAxes,
            ha="right", va="top", fontsize=8.8, fontweight="bold", color=MUTED)


def draw_ecdf_panel(ax, df):
    style_axis(ax, xlabel="Sensor concentration (mg/L)", ylabel="Empirical cumulative probability",
               tick_size=10.5, label_size=12.8)
    for target, color, label in [
        ("T1_PO4", BLUE, "T1-PO$_4$"),
        ("T1_NH4", CORAL, "T1-NH$_4$"),
    ]:
        values = np.sort(df[target].to_numpy(dtype=float))
        y = (np.arange(len(values)) + 0.5) / len(values)
        ax.plot(values, y, color=color, lw=1.8, label=label)
        q50, q95 = np.quantile(values, [0.50, 0.95])
        ax.scatter([q50, q95], [0.5, 0.95], s=[38, 52], color=color,
                   edgecolor="white", lw=0.8, zorder=4)
    ax.set_xscale("symlog", linthresh=0.1, linscale=0.7)
    ticks = [0, 0.1, 0.5, 1, 2, 5, 10, 20]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FixedFormatter([str(v) for v in ticks]))
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 1.01)
    ax.legend(loc="lower right", bbox_to_anchor=(0.99, 0.02), borderaxespad=0.0)


def draw_acf_panel(ax, acf):
    style_axis(ax, xlabel="Lag (h)", ylabel="Autocorrelation", tick_size=10.5, label_size=12.8)
    ax.axhline(0, color=MUTED, lw=1.0)
    ax.plot(acf["lag_h"], acf["T1_PO4"], color=BLUE, lw=1.8, label="T1-PO$_4$")
    ax.plot(acf["lag_h"], acf["T1_NH4"], color=CORAL, lw=1.8, label="T1-NH$_4$")
    ax.set_xlim(0, 24)
    ax.set_ylim(-0.05, 1.03)
    ax.set_xticks([0, 4, 8, 12, 16, 20, 24])
    ax.legend(loc="upper right", bbox_to_anchor=(0.99, 0.99), borderaxespad=0.0)


def draw_phase_panel(ax, phase_aligned, direction_counts):
    style_axis(ax, xlabel="Time from inlet-phase transition (h)",
               ylabel="Change from pre-event baseline (IQR)", tick_size=10.5, label_size=12.2)
    ax.axvline(0, color=INK, lw=1.3, ls="--", dashes=(4, 3))
    ax.axhline(0, color=MUTED, lw=1.0)
    x = phase_aligned["elapsed_h"].to_numpy()
    for target, color in [("T1_PO4", BLUE), ("T1_NH4", CORAL)]:
        for key, linestyle in [("1to2", "-"), ("2to1", "--")]:
            lower = phase_aligned[f"{target}_{key}_q25"].to_numpy()
            mid = phase_aligned[f"{target}_{key}_median"].to_numpy()
            upper = phase_aligned[f"{target}_{key}_q75"].to_numpy()
            ax.fill_between(x, lower, upper, color=color, alpha=0.055, linewidth=0)
            ax.plot(x, mid, color=color, lw=1.8, ls=linestyle)
    ax.set_xlim(-6, 6)
    ax.set_xticks([-6, -3, 0, 3, 6])
    ax.set_ylim(-0.50, 0.68)
    target_legend = ax.legend(
        handles=[Line2D([0], [0], color=BLUE, lw=2, label="T1-PO$_4$"),
                 Line2D([0], [0], color=CORAL, lw=2, label="T1-NH$_4$")],
        loc="upper left", bbox_to_anchor=(0.01, 0.99), borderaxespad=0.0,
    )
    ax.add_artist(target_legend)
    ax.legend(
        handles=[Line2D([0], [0], color=INK, lw=1.8, ls="-", label="phase 1→2"),
                 Line2D([0], [0], color=INK, lw=1.8, ls="--", label="phase 2→1")],
        loc="upper right", bbox_to_anchor=(0.99, 0.99), borderaxespad=0.0,
    )
    count_text = f"n = {direction_counts['1to2']} / {direction_counts['2to1']}"
    ax.text(0.98, 0.05, count_text, transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9.2, fontweight="bold", color=MUTED)


def draw_split_panel(ax):
    ax.set_xlim(pd.Timestamp("2021-08-01").value, pd.Timestamp("2023-08-01").value)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    spans = [
        ("training / rolling validation", "2021-08-01", "2022-10-31", SOFT[2], GREEN, 2.65),
        ("calibration", "2022-11-01", "2023-01-31", SOFT[6], GOLD, 1.65),
        ("untouched future test", "2023-02-01", "2023-07-31", SOFT[0], CORAL, 0.65),
    ]
    for label, start, end, fc, ec, y in spans:
        xs, xe = pd.Timestamp(start).value, pd.Timestamp(end).value
        ax.add_patch(Rectangle((xs, y), xe - xs, 0.65, facecolor=fc, edgecolor=ec, lw=1.5))
        ax.text((xs + xe) / 2, y + 0.325, label, ha="center", va="center",
                fontsize=10.2, fontweight="bold", color=INK)
    for dt in ["2021-08-01", "2022-01-01", "2022-07-01", "2023-01-01", "2023-07-31"]:
        x = pd.Timestamp(dt).value
        ax.plot([x, x], [0.15, 3.55], color="#D9DEE4", lw=0.9, zorder=0)
        ax.text(x, 0.08, pd.Timestamp(dt).strftime("%b\n%Y"), ha="center", va="top",
                fontsize=9.2, fontweight="bold", color=MUTED)
    ax.text(0.5, 0.94, "purge gap between adjacent blocks", transform=ax.transAxes,
            ha="center", va="center", fontsize=10.0, fontweight="bold", color=PURPLE)


def _render_data_panel(drawer, args, folder: Path, stem: str, suffix: str, figsize=(7.0, 5.2)):
    fig, ax = plt.subplots(figsize=figsize)
    drawer(ax, *args)
    scale_figure_text(fig, factor=1.15)
    set_figure_text_weight(fig, "normal")
    fig.subplots_adjust(left=0.17, right=0.97, bottom=0.17, top=0.97)
    out = folder / "subfigures"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{stem}_{suffix}.png", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(out / f"{stem}_{suffix}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def generate_fig02():
    df, ten = load_data()
    daily_median, daily_q10, daily_q90, week, acf, phase_aligned, direction_counts = prepare_figure2_data(df, ten)
    folder = FIGURES / "Fig02_data_dynamics"
    folder.mkdir(parents=True, exist_ok=True)
    stem = "Fig02_data_dynamics"

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.0))
    draw_two_year_panel(axes[0, 0], daily_median, daily_q10, daily_q90)
    draw_week_panel(axes[0, 1], week)
    draw_ecdf_panel(axes[0, 2], df)
    draw_acf_panel(axes[1, 0], acf)
    draw_phase_panel(axes[1, 1], phase_aligned, direction_counts)
    draw_split_panel(axes[1, 2])
    scale_figure_text(fig, factor=1.15)
    set_figure_text_weight(fig, "normal")
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.08, top=0.975, wspace=0.32, hspace=0.32)
    align_subplot_columns(axes)
    align_subplot_rows(axes)
    align_subplot_columns(axes)
    align_subplot_rows(axes)
    save_figure_with_subfigures(fig, folder, stem, dpi=600, close=True)

    panel_specs = [
        (draw_two_year_panel, (daily_median, daily_q10, daily_q90), "a", (6.4, 4.6)),
        (draw_week_panel, (week,), "b", (6.4, 4.6)),
        (draw_ecdf_panel, (df,), "c", (6.4, 4.6)),
        (draw_acf_panel, (acf,), "d", (6.4, 4.6)),
        (draw_phase_panel, (phase_aligned, direction_counts), "e", (6.4, 4.6)),
        (draw_split_panel, tuple(), "f", (6.4, 4.6)),
    ]
    for drawer, args, suffix, size in panel_specs:
        _render_data_panel(drawer, args, folder, stem, suffix, figsize=size)
    (folder / "subfigures" / "README.md").write_text(
        "# Subfigures\n\nIndependently rendered from the same panel builders used by the final combined figure.\n",
        encoding="utf-8",
    )

    daily_export = daily_median.add_suffix("_median")
    daily_export = daily_export.join(daily_q10.add_suffix("_q10")).join(daily_q90.add_suffix("_q90"))
    daily_export.to_csv(TABLES / "Fig02_daily_target_summary.csv", index_label="date")
    acf.to_csv(TABLES / "Fig02_autocorrelation_10min.csv", index=False)
    phase_aligned.to_csv(TABLES / "Fig02_phase_transition_alignment.csv", index=False)
    summary = {
        "raw_rows": int(len(df)),
        "ten_min_rows": int(len(ten)),
        "fixed_week_start": "2022-08-01",
        "fixed_week_end": "2022-08-07",
        "retained_phase_transitions": int(sum(direction_counts.values())),
        "phase_transition_counts": direction_counts,
        "phase_alignment_window_h": 6,
        "targets": ["T1_PO4", "T1_NH4"],
        "units_status": "authoritative data descriptor reports nutrient concentrations in mg/L",
    }
    (TABLES / "Fig02_analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def write_notebook(path: Path, only: str):
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [f"# Regenerate {only}\n", "This notebook calls the publication figure source-of-truth script.\n"],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [f"%run ../../scripts/generate_initial_figures.py --only {only}\n"],
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["Fig01", "Fig02", "all"], default="all")
    args = parser.parse_args()
    if args.only in {"Fig01", "all"}:
        generate_fig01()
        write_notebook(FIGURES / "Fig01_P3FM_framework" / "Fig01_P3FM_framework_jupyter_code.ipynb", "Fig01")
    if args.only in {"Fig02", "all"}:
        generate_fig02()
        write_notebook(FIGURES / "Fig02_data_dynamics" / "Fig02_data_dynamics_jupyter_code.ipynb", "Fig02")


if __name__ == "__main__":
    main()
