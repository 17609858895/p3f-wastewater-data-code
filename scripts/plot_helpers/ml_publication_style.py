"""
Shared publication figure helpers for Nature-style ML figures.

Use these helpers in every generated figure script unless the journal requires
a different style. They enforce visible tick marks, readable regular-weight
text, high-resolution export, and per-panel subfigure export.
"""

from __future__ import annotations

import string
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import Bbox

from scientific_palettes import PALETTES, STYLE_COLORS, get_palette


def configure_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.dpi": 160,
            "savefig.dpi": 600,
            "axes.labelsize": 15.5,
            "font.weight": "normal",
            "axes.labelweight": "normal",
            "axes.titleweight": "normal",
            "xtick.labelsize": 12.5,
            "ytick.labelsize": 12.5,
            "legend.fontsize": 11.5,
            "axes.linewidth": 1.7,
            "xtick.major.size": 7.2,
            "ytick.major.size": 7.2,
            "xtick.major.width": 1.55,
            "ytick.major.width": 1.55,
            "xtick.minor.size": 4.2,
            "ytick.minor.size": 4.2,
            "xtick.minor.width": 1.15,
            "ytick.minor.width": 1.15,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def scale_figure_text(fig: plt.Figure, factor: float = 1.15) -> None:
    """Enlarge every text artist after panel construction.

    This catches explicit per-panel font sizes as well as rcParam-controlled
    axis, tick, legend, annotation and colorbar text.  Calling it immediately
    before layout/export keeps the change consistent across combined figures
    and independently rendered subfigures.
    """
    if factor <= 0:
        raise ValueError("factor must be positive")
    for artist in fig.findobj(match=Text):
        size = artist.get_fontsize()
        if size and size > 0:
            artist.set_fontsize(size * factor)


def set_figure_text_weight(fig: plt.Figure, weight: str = "normal") -> None:
    """Apply one text weight to axes, legends, annotations and colorbars."""
    for artist in fig.findobj(match=Text):
        artist.set_fontweight(weight)


def _axes_visual_group(ax) -> list:
    """Return a main axes together with colorbar axes created from its data."""
    group = [ax]
    for artist in [*ax.images, *ax.collections]:
        colorbar = getattr(artist, "colorbar", None)
        colorbar_ax = getattr(colorbar, "ax", None)
        if colorbar_ax is not None and colorbar_ax not in group:
            group.append(colorbar_ax)
    return group


def _visual_footprint(ax, renderer) -> Bbox:
    """Measure an axes including labels, legend, annotations and colorbar."""
    figure = ax.figure
    boxes = []
    for grouped_ax in _axes_visual_group(ax):
        bbox = grouped_ax.get_tightbbox(renderer)
        if bbox is not None:
            boxes.append(bbox.transformed(figure.transFigure.inverted()))
    if not boxes:
        return ax.get_position().frozen()
    return Bbox.union(boxes)


def align_subplot_columns(axes, iterations: int = 8,
                          tolerance: float = 2e-5) -> dict[int, tuple[float, float]]:
    """Align each column by its complete visual footprint.

    The measured boundary includes tick labels, axis labels, legends,
    annotations and any attached colorbar.  This avoids the misleading case
    where plotting rectangles align numerically but the two rows still look
    unequal.  Each row is expanded to the column-wise visual union so that
    text is not squeezed toward the plotting area.

    Returns the final ``{column: (left, right)}`` visual bounds for QA.
    """
    grid = [list(row) for row in axes]
    if not grid or not grid[0]:
        raise ValueError("axes must be a non-empty two-dimensional grid")
    ncols = len(grid[0])
    if any(len(row) != ncols for row in grid):
        raise ValueError("axes rows must have equal length")

    figure = grid[0][0].figure
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    bounds: dict[int, tuple[float, float]] = {}
    for col in range(ncols):
        column_axes = [row[col] for row in grid if row[col].get_visible()]
        if not column_axes:
            continue
        footprints = [_visual_footprint(ax, renderer) for ax in column_axes]
        left = min(bbox.x0 for bbox in footprints)
        right = max(bbox.x1 for bbox in footprints)
        bounds[col] = (left, right)

    for _ in range(iterations):
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        max_error = 0.0
        for col, (target_left, target_right) in bounds.items():
            target_width = target_right - target_left
            for row in grid:
                ax = row[col]
                if not ax.get_visible():
                    continue
                footprint = _visual_footprint(ax, renderer)
                current_width = footprint.x1 - footprint.x0
                if current_width <= 0:
                    continue
                scale = target_width / current_width
                for grouped_ax in _axes_visual_group(ax):
                    pos = grouped_ax.get_position().frozen()
                    new_left = target_left + (pos.x0 - footprint.x0) * scale
                    new_right = target_left + (pos.x1 - footprint.x0) * scale
                    grouped_ax.set_position(
                        [new_left, pos.y0, new_right - new_left, pos.height]
                    )
                max_error = max(
                    max_error,
                    abs(footprint.x0 - target_left),
                    abs(footprint.x1 - target_right),
                )
        if max_error <= tolerance:
            break
    return bounds


def align_subplot_rows(axes, iterations: int = 8,
                       tolerance: float = 2e-5) -> dict[int, tuple[float, float]]:
    """Align each row by its complete visual top and bottom boundaries.

    Rows may contain different numbers of axes, which supports mosaics with a
    two-column heatmap or network panel.  Tick labels, axis labels, legends,
    annotations and attached colorbars are included in the measured footprint.

    Returns the final ``{row: (bottom, top)}`` visual bounds for QA.
    """
    grid = [list(row) for row in axes]
    if not grid or not any(grid):
        raise ValueError("axes must contain at least one subplot")

    figure = next(ax.figure for row in grid for ax in row)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    bounds: dict[int, tuple[float, float]] = {}
    for row_index, row in enumerate(grid):
        row_axes = [ax for ax in row if ax.get_visible()]
        if not row_axes:
            continue
        footprints = [_visual_footprint(ax, renderer) for ax in row_axes]
        bottom = min(bbox.y0 for bbox in footprints)
        top = max(bbox.y1 for bbox in footprints)
        bounds[row_index] = (bottom, top)

    for _ in range(iterations):
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        max_error = 0.0
        for row_index, (target_bottom, target_top) in bounds.items():
            target_height = target_top - target_bottom
            for ax in grid[row_index]:
                if not ax.get_visible():
                    continue
                footprint = _visual_footprint(ax, renderer)
                current_height = footprint.y1 - footprint.y0
                if current_height <= 0:
                    continue
                scale = target_height / current_height
                for grouped_ax in _axes_visual_group(ax):
                    pos = grouped_ax.get_position().frozen()
                    new_bottom = target_bottom + (pos.y0 - footprint.y0) * scale
                    new_top = target_bottom + (pos.y1 - footprint.y0) * scale
                    grouped_ax.set_position(
                        [pos.x0, new_bottom, pos.width, new_top - new_bottom]
                    )
                max_error = max(
                    max_error,
                    abs(footprint.y0 - target_bottom),
                    abs(footprint.y1 - target_top),
                )
        if max_error <= tolerance:
            break
    return bounds


def style_axis(ax, xlabel: str | None = None, ylabel: str | None = None,
               tick_size: float = 12.5, label_size: float = 15.5) -> None:
    ax.grid(False)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color(STYLE_COLORS["ink"])
        ax.spines[spine].set_linewidth(1.7)
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    ax.tick_params(axis="x", which="major", bottom=True, top=False, colors=STYLE_COLORS["ink"],
                   width=1.55, length=7.2, direction="out", labelsize=tick_size)
    ax.tick_params(axis="y", which="major", left=True, right=False, colors=STYLE_COLORS["ink"],
                   width=1.55, length=7.2, direction="out", labelsize=tick_size)
    ax.tick_params(axis="both", which="minor", colors=STYLE_COLORS["ink"], width=1.15,
                   length=4.2, direction="out")
    for tick in ax.xaxis.get_major_ticks():
        tick.tick1line.set_visible(True)
        tick.tick1line.set_markersize(7.2)
        tick.tick1line.set_markeredgewidth(1.55)
    for tick in ax.yaxis.get_major_ticks():
        tick.tick1line.set_visible(True)
        tick.tick1line.set_markersize(7.2)
        tick.tick1line.set_markeredgewidth(1.55)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("normal")
        label.set_fontsize(tick_size)
    ax.xaxis.label.set_fontweight("normal")
    ax.yaxis.label.set_fontweight("normal")
    ax.xaxis.label.set_size(label_size)
    ax.yaxis.label.set_size(label_size)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontweight="normal", fontsize=label_size, labelpad=7)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontweight="normal", fontsize=label_size, labelpad=7)


def numeric_ticks(ax, x: bool = True, y: bool = True, n: int = 5) -> None:
    """Keep numeric axes readable; use 3-6 major ticks unless the domain differs."""
    if x:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=n, prune=None))
    if y:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=n, prune=None))


def _panel_letters() -> list[str]:
    letters = list(string.ascii_lowercase)
    return letters + [a + b for a in letters for b in letters]


def export_subfigures(fig: plt.Figure, folder: str | Path, stem: str,
                      dpi: int = 600, pad: float = 1.06) -> Path:
    """Save each visible subplot into `folder/subfigures`.

    The crop comes from the final combined figure, so labels, ticks, legends,
    and colorbars match the exported main figure. Colorbar-only axes are skipped.
    """
    folder = Path(folder)
    subdir = folder / "subfigures"
    subdir.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    labels = _panel_letters()
    exported = []
    for ax in fig.axes:
        if not ax.get_visible() or str(ax.get_label()).startswith("<colorbar"):
            continue
        bbox = ax.get_tightbbox(renderer)
        if bbox is None:
            continue
        bbox = bbox.expanded(pad, pad).transformed(fig.dpi_scale_trans.inverted())
        suffix = labels[len(exported)]
        png = subdir / f"{stem}_{suffix}.png"
        pdf = subdir / f"{stem}_{suffix}.pdf"
        fig.savefig(png, dpi=dpi, bbox_inches=bbox, facecolor="white")
        fig.savefig(pdf, bbox_inches=bbox, facecolor="white")
        exported.append((suffix, png.name, pdf.name))
    readme = subdir / "README.md"
    lines = ["# Subfigures", "", "Extracted from the final combined figure export.", ""]
    for suffix, png_name, pdf_name in exported:
        lines.append(f"- {suffix}: `{png_name}`, `{pdf_name}`")
    readme.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return subdir


def save_figure_with_subfigures(fig: plt.Figure, folder: str | Path, stem: str,
                                dpi: int = 600, close: bool = True) -> None:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"{stem}.png", dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(folder / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    export_subfigures(fig, folder, stem, dpi=dpi)
    if close:
        plt.close(fig)
