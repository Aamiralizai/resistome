"""
plotstyle.py
============
House figure standard for the W1 resistome project.

Every figure produced by this pipeline is:
  * multi-panel, laid out in 2-3 rows
  * set in Times New Roman
  * given bold, explicitly labelled x and y axes
  * saved at publication resolution in both raster and vector form

Import `multipanel` and `save` in every plotting script; do not call
matplotlib.pyplot.subplots directly, or the style will drift between figures.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Font resolution
# ---------------------------------------------------------------------------

_PREFERRED = ["Times New Roman", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"]
_SANS = ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"]

# Nature-family journals specify sans-serif (Helvetica or Arial) for figures.
# Set W1_FIGURE_FONT=sans to comply; the default keeps the serif house style.
import os as _os
USE_SANS = _os.environ.get("W1_FIGURE_FONT", "serif").lower().startswith("sans")


def _resolve_serif() -> list[str]:
    """Return an ordered serif stack, warning if true Times New Roman is absent.

    On WSL/Ubuntu install the real font with:
        sudo apt-get install -y ttf-mscorefonts-installer fonts-liberation
        rm -rf ~/.cache/matplotlib
    Nimbus Roman and Liberation Serif are metric-compatible substitutes; most
    journals accept them, but check before submission.
    """
    available = {f.name for f in fm.fontManager.ttflist}
    stack = [f for f in _PREFERRED if f in available]
    if not stack:
        warnings.warn("No serif font from the preferred list found; falling back to default.")
        return ["serif"]
    if stack[0] != "Times New Roman":
        warnings.warn(
            f"Times New Roman not installed; using '{stack[0]}' as a substitute. "
            "Install ttf-mscorefonts-installer for the real thing."
        )
    return stack


def apply_style(base_size: int = 11) -> None:
    """Set global rcParams. Call once at the top of any plotting script."""
    if USE_SANS:
        available = {f.name for f in fm.fontManager.ttflist}
        stack = [f for f in _SANS if f in available] or ["sans-serif"]
        fam, key = "sans-serif", "font.sans-serif"
    else:
        stack, fam, key = _resolve_serif(), "serif", "font.serif"
    matplotlib.rcParams.update(
        {
            # --- typography -------------------------------------------------
            "font.family": fam,
            key: stack,
            "mathtext.fontset": "stix",
            "font.size": base_size,
            "axes.titlesize": base_size + 1,
            "axes.titleweight": "bold",
            "axes.labelsize": base_size + 1,
            "axes.labelweight": "bold",      # <- bold axis labels, house rule
            "xtick.labelsize": base_size - 1,
            "ytick.labelsize": base_size - 1,
            "legend.fontsize": base_size - 1,
            "legend.title_fontsize": base_size - 1,
            # --- axes and ticks ---------------------------------------------
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#B4BCC8",
            "axes.labelcolor": "#232936",
            "text.color": "#232936",
            "axes.labelpad": 6.0,
            "axes.titlepad": 8.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "both",
            "grid.color": "#E6EAF0",
            "grid.linewidth": 0.7,
            "grid.alpha": 1.0,
            "axes.axisbelow": True,          # gridlines behind the data
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.color": "#5A6474",
            "ytick.color": "#5A6474",
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.major.pad": 4.0,
            "ytick.major.pad": 4.0,
            "xtick.minor.size": 2,
            "ytick.minor.size": 2,
            # --- output -----------------------------------------------------
            "figure.dpi": 110,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "savefig.transparent": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "pdf.fonttype": 42,              # editable text in Illustrator
            "ps.fonttype": 42,
            # 'none' keeps SVG text as TEXT rather than converting it to
            # outlines, so labels stay editable and re-typeable in Inkscape
            # or Illustrator. Converted paths cannot be corrected later.
            "svg.fonttype": "none",
            "legend.frameon": False,
            "legend.handlelength": 1.4,
            "legend.handletextpad": 0.5,
            "legend.columnspacing": 1.0,
            "lines.linewidth": 1.6,
            "lines.markersize": 4,
            "lines.solid_capstyle": "round",
            "patch.linewidth": 0.0,     # flat fills, no heavy outlines
            "hatch.linewidth": 0.6,
        }
    )


# ---------------------------------------------------------------------------
# Colour palette (colour-blind safe, print-safe)
# ---------------------------------------------------------------------------

PALETTE = {
    "primary":   "#3D6FB4",   # clear blue
    "secondary": "#E4694E",   # coral
    "tertiary":  "#3FA786",   # jade
    "accent":    "#E8B33C",   # gold
    "purple":    "#8B6BB7",
    "neutral":   "#8A94A6",   # cool grey
    "light":     "#D9E5F2",
    "ink":       "#232936",   # soft charcoal, never pure black
}
# Muted-but-saturated, colour-blind safe, and monotonic in luminance across
# the first four so a greyscale print still separates them.
SERIES = ["#3D6FB4", "#E4694E", "#3FA786", "#E8B33C", "#8B6BB7", "#8A94A6"]


def diverging():
    return plt.get_cmap("RdBu_r")


def sequential():
    return plt.get_cmap("viridis")


# ---------------------------------------------------------------------------
# Multi-panel construction
# ---------------------------------------------------------------------------

def multipanel(
    nrows: int = 3,
    ncols: int = 2,
    panel_width: float = 3.4,
    panel_height: float = 2.7,
    labels: bool = True,
    label_size: int = 13,
    sharex: bool = False,
    sharey: bool = False,
    height_ratios: Sequence[float] | None = None,
    width_ratios: Sequence[float] | None = None,
    hspace: float = 0.48,
    wspace: float = 0.42,
):
    """Create a multi-panel figure in house style.

    House rule: figures are laid out in 2 or 3 rows. A ValueError is raised
    for anything else so that single-panel figures do not creep in.

    Returns
    -------
    fig  : matplotlib Figure
    axes : flat numpy array of Axes, row-major order
    """
    if nrows not in (2, 3):
        raise ValueError(
            f"House style requires 2 or 3 rows, got nrows={nrows}. "
            "Combine related panels rather than making a single-row figure."
        )
    apply_style()
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(panel_width * ncols, panel_height * nrows),
        sharex=sharex,
        sharey=sharey,
        layout="constrained",
        gridspec_kw={"height_ratios": height_ratios, "width_ratios": width_ratios},
    )
    fig.get_layout_engine().set(w_pad=wspace / 6, h_pad=hspace / 6,
                                wspace=0.02, hspace=0.02)
    axes = np.atleast_1d(axes).ravel()
    if labels:
        add_panel_labels(fig, axes, size=label_size)
    return fig, axes


def add_panel_labels(fig, axes: Iterable, size: int = 12,
                     dx: float = -0.50, dy: float = 0.20) -> None:
    """Bold (A), (B), (C)... offset from each panel's top-left corner.

    The offset is in INCHES, not axes fractions. Axes-fraction offsets move
    with the panel, so a long bold y-label or wide tick labels push the letter
    into the neighbouring panel - which is exactly what happened before.
    """
    from matplotlib.transforms import ScaledTranslation
    for i, ax in enumerate(axes):
        ax.text(
            0.0, 1.0, f"({chr(65 + i)})",
            transform=ax.transAxes + ScaledTranslation(dx, dy, fig.dpi_scale_trans),
            fontsize=size, fontweight="bold", va="baseline", ha="left",
            color="#232936",
        )


def no_grid(*axes) -> None:
    """Turn the grid off - for heatmaps, hexbins and images, where gridlines
    sit on top of the data and add nothing."""
    for ax in axes:
        ax.grid(False)


def grid_axis(ax, axis: str = "y") -> None:
    """Restrict the grid to one axis. Bar charts read better with gridlines
    only along the value axis."""
    ax.grid(False)
    ax.grid(True, axis=axis)


def label_axes(ax, xlabel: str, ylabel: str, title: str | None = None) -> None:
    """Set both axis labels in bold. Never leave an axis unlabelled."""
    ax.set_xlabel(xlabel, fontweight="bold")
    ax.set_ylabel(ylabel, fontweight="bold")
    if title:
        ax.set_title(title, fontweight="bold")


def bar_values(ax, bars, fmt="{:.2f}", horizontal=False, size=7,
               color="#232936", pad=0.01) -> None:
    """Print each bar's value at its end.

    Direct labelling removes the need for the reader to trace a bar back to a
    gridline, and lets the axis be de-emphasised instead of competing with the
    data.
    """
    span = (ax.get_xlim() if horizontal else ax.get_ylim())
    off = pad * (span[1] - span[0])
    for b in bars:
        if horizontal:
            v = b.get_width()
            ax.text(v + (off if v >= 0 else -off), b.get_y() + b.get_height() / 2,
                    fmt.format(v), va="center", fontsize=size, color=color,
                    ha="left" if v >= 0 else "right")
        else:
            v = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, v + (off if v >= 0 else -off),
                    fmt.format(v), ha="center", fontsize=size, color=color,
                    va="bottom" if v >= 0 else "top")


def despine(ax, left=False, bottom=False) -> None:
    """Drop the remaining spines. With a grid present the axis lines are
    redundant; removing them is the single biggest modernising change."""
    ax.spines["left"].set_visible(not left)
    ax.spines["bottom"].set_visible(not bottom)


def shorten(labels, n: int = 20) -> list[str]:
    """Truncate long category labels so they do not run into other panels."""
    out = []
    for x in labels:
        x = str(x)
        out.append(x if len(x) <= n else x[: n - 1] + "\u2026")
    return out


def hide_unused(axes, n_used: int) -> None:
    """Blank any trailing panels in the grid that were not filled."""
    for ax in axes[n_used:]:
        ax.axis("off")


def save(fig, outdir, stem: str, formats: Sequence[str] = ("png", "pdf"), dpi: int = 600) -> list[Path]:
    """Write the figure to disk in every requested format."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in formats:
        p = outdir / f"{stem}.{ext}"
        fig.savefig(p, dpi=dpi, format=ext)
        written.append(p)
    plt.close(fig)
    return written
