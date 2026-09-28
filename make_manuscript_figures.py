"""
make_manuscript_figures.py
==========================
The five consolidated main figures, built from the pipeline's own tables.

Design decisions, and why
-------------------------
*Thematic rather than per-stage.* Ten figures that each report one analysis
read as a sequence of outputs. Five figures that each carry one argument read
as a paper. Panels are grouped by the claim they support, not by the script
that produced them.

*One result, one place.* Cross-habitat variance appears in Figure 1 alongside
the other variance results and nowhere else; Figure 5 carries habitat
PERFORMANCE, which is a different claim. Showing the same numbers twice in a
main figure invites a reviewer to ask which one is the result.

*Adult stratum throughout, unless stated.* The variance partition, model
comparison, conservation analysis and carriage analysis all use the adult
cohort. Mixing strata between panels would make them incommensurable - the
combined-stratum carriage figures, for instance, differ substantially from the
adult ones and must not be substituted silently.

*No truncated labels.* Study names are shortened to author and year by rule,
not clipped with an ellipsis, and dense panels label only outlying points.

*Colour-blind safe, and legible in greyscale.* The palette has monotonic
luminance across the first four series, and categorical distinctions are
carried by position or marker as well as hue wherever a panel allows it.

Usage
-----
    python make_manuscript_figures.py \\
        --tables /mnt/x/w1_resistome/tables \\
        --habitats /mnt/x/w1_resistome/habitats \\
        --outdir /mnt/x/w1_resistome/manuscript_figures

All paths are arguments; nothing is hard-coded, so the script runs unchanged
for anyone who has the supplementary tables.
"""

from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.transforms import ScaledTranslation

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------ style --
INK = "#1F2430"
GREY = "#8A94A6"
GRID = "#E6EAF0"
# Colour-blind safe; luminance increases monotonically across the first four,
# so the ordering survives greyscale printing.
C = {"blue": "#3B6EA8", "red": "#D1495B", "green": "#3B8B6B",
     "gold": "#E0A526", "purple": "#7D5BA6", "grey": GREY,
     "light": "#D8E3F0", "dark": "#1F2430"}
SERIES = [C["blue"], C["red"], C["green"], C["gold"], C["purple"], C["grey"]]


def style(font: str = "serif") -> None:
    fam = ["Times New Roman", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"] \
        if font == "serif" else ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"]
    matplotlib.rcParams.update({
        "font.family": "serif" if font == "serif" else "sans-serif",
        ("font.serif" if font == "serif" else "font.sans-serif"): fam,
        "font.size": 9, "axes.labelsize": 9.5, "axes.titlesize": 9.5,
        "axes.labelweight": "bold", "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 7.5, "figure.dpi": 110, "savefig.dpi": 300,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.06,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": "#B4BCC8", "axes.linewidth": 0.8,
        "axes.labelcolor": INK, "text.color": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
        "axes.axisbelow": True, "xtick.color": "#5A6474", "ytick.color": "#5A6474",
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "patch.linewidth": 0.0, "lines.linewidth": 1.4,
        "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "none",     # keeps SVG text editable, not outlined
    })


def panel(fig, axes, skip=()):
    for i, ax in enumerate(axes):
        if i in skip:
            continue
        ax.text(0.0, 1.0, "ABCDEFGHIJ"[i],
                transform=ax.transAxes + ScaledTranslation(-0.42, 0.20,
                                                           fig.dpi_scale_trans),
                fontsize=13, fontweight="bold", va="baseline", color=INK)


def grid_axis(ax, axis="y"):
    ax.grid(False); ax.grid(True, axis=axis)


def despine(ax, left=False, bottom=False):
    ax.spines["left"].set_visible(not left)
    ax.spines["bottom"].set_visible(not bottom)
    if left:
        ax.tick_params(axis="y", length=0)
    if bottom:
        ax.tick_params(axis="x", length=0)


def label(ax, x, y, title=None):
    ax.set_xlabel(x, fontweight="bold"); ax.set_ylabel(y, fontweight="bold")
    if title:
        ax.set_title(title, fontweight="bold", fontsize=8.5, pad=6)


def bar_text(ax, bars, fmt="{:.3f}", horiz=False, size=7):
    span = (ax.get_xlim() if horiz else ax.get_ylim())
    off = 0.012 * (span[1] - span[0])
    for b in bars:
        v = b.get_width() if horiz else b.get_height()
        if horiz:
            ax.text(v + (off if v >= 0 else -off),
                    b.get_y() + b.get_height() / 2, fmt.format(v),
                    va="center", ha="left" if v >= 0 else "right",
                    fontsize=size, color=INK)
        else:
            ax.text(b.get_x() + b.get_width() / 2, v + (off if v >= 0 else -off),
                    fmt.format(v), ha="center", fontsize=size, color=INK,
                    va="bottom" if v >= 0 else "top")


def short_study(name: str) -> str:
    """'Hayden_2020_infant_cystic_fibrosis' -> 'Hayden 2020'.

    Truncating with an ellipsis leaves figures looking machine-generated and
    the reader unable to identify the cohort. Reducing to author and year by
    rule is both shorter and unambiguous.
    """
    s = str(name)
    m = re.match(r"([A-Za-z\-]+)[_\s]*((?:19|20)\d{2})", s)
    if m:
        return f"{m.group(1)} {m.group(2)}"
    m = re.match(r"(PRJ[A-Z]+\d+)", s)
    if m:
        return m.group(1)
    return s.replace("_", " ")[:18]


# ------------------------------------------------------- drawing idioms ----
# The visual grammar below - annotated heatmaps, dumbbells, bubble scatters
# with an inset statistics box, and slope plots - reads better than paired
# bars for these comparisons. A dumbbell shows the COLLAPSE from marginal to
# unique as a line, which is the actual claim; two adjacent bars leave the
# reader to compute the difference.

def heatmap(ax, M, row_lab, col_lab, cbar_label, cmap="Reds", fmt="{:.2f}",
            fig=None, dashes="—"):
    """Annotated matrix. Cell text switches to white on dark backgrounds so
    every value stays legible regardless of the colour scale."""
    A = np.asarray(M, dtype=float)
    im = ax.imshow(A, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(col_lab))); ax.set_xticklabels(col_lab)
    ax.set_yticks(range(len(row_lab))); ax.set_yticklabels(row_lab)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    # Text colour is chosen from the RENDERED cell luminance rather than from
    # the value, because a fixed threshold puts dark text on dark cells
    # whenever the colour scale is narrow.
    norm = im.norm
    cm = im.cmap
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            v = A[i, j]
            if not np.isfinite(v):
                ax.text(j, i, dashes, ha="center", va="center", fontsize=8.5,
                        color=INK)
                continue
            r, g, b, _ = cm(norm(v))
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            ax.text(j, i, fmt.format(v), ha="center", va="center",
                    fontsize=8.5, fontweight="bold",
                    color="white" if lum < 0.55 else INK)
    if fig is not None:
        cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
        cb.set_label(cbar_label, fontweight="bold", fontsize=8)
        cb.outline.set_visible(False)
    return im


def dumbbell(ax, labels, left, right, left_name, right_name,
             lc=None, rc=None, fmt="{:.3f}", ref=None):
    """Two values per row joined by a line. The gap is the message."""
    lc = lc or C["blue"]; rc = rc or C["gold"]
    y = np.arange(len(labels))[::-1]
    for yi, a, b in zip(y, left, right):
        ax.plot([a, b], [yi, yi], color="#C6CCD6", lw=1.6, zorder=1,
                solid_capstyle="round")
    ax.scatter(left, y, s=54, color=lc, zorder=3, label=left_name,
               edgecolors="white", linewidths=0.9)
    ax.scatter(right, y, s=54, color=rc, zorder=3, label=right_name,
               marker="s", edgecolors="white", linewidths=0.9)
    for yi, a, b in zip(y, left, right):
        ax.annotate(fmt.format(a), (a, yi), textcoords="offset points",
                    xytext=(-4, 7), ha="right", fontsize=6.8, color=INK)
        ax.annotate(fmt.format(b), (b, yi), textcoords="offset points",
                    xytext=(4, -11), ha="left", fontsize=6.8, color=INK)
    if ref is not None:
        ax.axvline(ref, ls="--", lw=1.2, color=GREY, zorder=0)
    ax.set_yticks(y); ax.set_yticklabels(labels)
    grid_axis(ax, "x"); despine(ax, left=True)


def statbox(ax, text, loc="lower right"):
    """Inset statistics, so the number sits with the data rather than in the
    caption where the reader has to look for it."""
    xy = {"lower right": (0.97, 0.03, "right", "bottom"),
          "upper left": (0.03, 0.97, "left", "top"),
          "lower left": (0.03, 0.03, "left", "bottom")}[loc]
    ax.text(xy[0], xy[1], text, transform=ax.transAxes, ha=xy[2], va=xy[3],
            fontsize=7.5, color=INK,
            bbox=dict(boxstyle="round,pad=0.34", facecolor="white",
                      edgecolor="#C6CCD6", linewidth=0.7))


def slope(ax, xlabels, series, colors=None, fmt="{:.3f}"):
    """Paired before/after lines with end labels."""
    colors = colors or SERIES
    x = np.arange(len(xlabels))
    for i, (name, vals) in enumerate(series.items()):
        ax.plot(x, vals, "-o", lw=1.8, ms=7, color=colors[i % len(colors)],
                mec="white", mew=1.2, label=name)
        for xi, v in zip(x, vals):
            ax.annotate(fmt.format(v), (xi, v), textcoords="offset points",
                        xytext=(0, -14 if i % 2 else 9), ha="center",
                        fontsize=7, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(xlabels)
    ax.set_xlim(-0.35, len(xlabels) - 0.65)
    grid_axis(ax, "y")


def _safe_legends(fig):
    """Drop legend calls on axes that ended up with nothing plotted.

    A legend over an empty collection raises during rendering rather than at
    call time, so the failure appears as an opaque matplotlib traceback at
    savefig. Checking here keeps one sparse panel from killing a whole figure.
    """
    for ax in fig.get_axes():
        leg = ax.get_legend()
        if leg is not None and not ax.get_legend_handles_labels()[0]:
            leg.remove()


def save(fig, outdir: Path, name: str):
    _safe_legends(fig)
    outdir.mkdir(parents=True, exist_ok=True)
    out = []
    for ext in ("png", "pdf", "svg"):
        f = outdir / f"{name}.{ext}"
        fig.savefig(f, format=ext)
        out.append(f)
    plt.close(fig)
    print(f"  {name:<44} {out[0]}")
    return out


def rd(d: Path, name: str):
    p = d / name
    return pd.read_csv(p) if p.exists() else None


# ------------------------------------------------------- Figure 1: variance --
def figure1(T: Path, H: Path, out: Path, work: Path | None = None):
    """Variance structure: what composition explains, and how robustly."""
    fig, ax = plt.subplots(3, 2, figsize=(11.6, 13.4))
    fig.subplots_adjust(hspace=0.40, wspace=0.34)
    ax = ax.ravel()

    vp = rd(T, "variance_partition_adult.csv")
    ucol = ("unique_total_fraction_adj" if "unique_total_fraction_adj" in vp.columns
            else "unique_R2_adj")
    mcol = "marginal_R2_adj" if "marginal_R2_adj" in vp.columns else "R2_adj"

    # (A) unique variance across strata, as an annotated matrix
    st = rd(T, "stratum_comparison_variance.csv")
    if st is not None:
        st = st.set_index("block")
        order = ["taxonomy", "study", "disease", "life_stage", "geography",
                 "host", "exposure", "technical"]
        rows = [b for b in order if b in st.index]
        cols = [c for c in ("adult", "all", "infant") if c in st.columns]
        heatmap(ax[0], st.loc[rows, cols].values,
                [r.replace("_", " ").capitalize() for r in rows],
                [c.capitalize() for c in cols],
                "Partial variance explained", cmap="Blues", fmt="{:.3f}", fig=fig)
        ax[0].set_title("Variance explained by block and life stage",
                        fontweight="bold", fontsize=9.5, pad=8)

    # (B) marginal collapsing to unique - the compositional-overlap argument
    v = vp.sort_values(ucol, ascending=False)
    dumbbell(ax[1], [b.replace("_", " ").capitalize() for b in v["block"]],
             v[ucol].values, v[mcol].values,
             "Unique share of total", "Marginal", lc=C["blue"], rc=C["gold"])
    ax[1].legend(loc="lower right")
    label(ax[1], "Adjusted $R^2$", "",
          "Taxonomy retains the largest unique contribution")

    # (C) sensitivity, each variant against the primary analysis
    sens = rd(T, "sensitivity_taxonomy_adult.csv")
    if sens is not None:
        prim = float(sens["taxonomy_unique"].iloc[0])
        lab_map = {"primary (5% ARG, 50 PCs, all samples)": "Primary",
                   "25 taxonomy components": "25 tax. PCs",
                   "100 taxonomy components": "100 tax. PCs"}
        labs = [lab_map.get(v, v.replace("ARG prevalence >= ", "ARG prev ")
                            .replace("one sample per subject", "1 sample/subject")
                            .split(" (")[0]) for v in sens["variant"]]
        y = np.arange(len(sens))[::-1]
        for yi, val in zip(y, sens["taxonomy_unique"]):
            ax[2].plot([prim, val], [yi, yi], color="#C6CCD6", lw=1.6, zorder=1)
        ax[2].scatter(sens["taxonomy_unique"], y, s=58, color=C["blue"],
                      zorder=3, edgecolors="white", linewidths=0.9)
        for yi, val in zip(y, sens["taxonomy_unique"]):
            ax[2].annotate(f"{val:.3f}", (val, yi), textcoords="offset points",
                           xytext=(9, -3), fontsize=7, color=INK)
        ax[2].axvline(prim, ls="--", lw=1.3, color=GREY)
        ax[2].set_yticks(y); ax[2].set_yticklabels(labs)
        ax[2].set_xlim(sens["taxonomy_unique"].min() - 0.008,
                       sens["taxonomy_unique"].max() + 0.014)
        grid_axis(ax[2], "x"); despine(ax[2], left=True)
        label(ax[2], "Unique share of total variance (taxonomy)", "",
              f"Robustness: {sens['taxonomy_unique'].min():.3f}–"
              f"{sens['taxonomy_unique'].max():.3f} across variants")

    # (D) relative against load-adjusted
    ra = rd(T, "variance_relative_vs_absolute_adult.csv")
    if ra is not None and "unique_R2_adj_relative" in ra.columns:
        d = ra.sort_values("unique_R2_adj_relative", ascending=False)
        dumbbell(ax[3], [b.replace("_", " ").capitalize() for b in d["block"]],
                 d["unique_R2_adj_relative"].values,
                 d["unique_R2_adj_absolute"].values,
                 "Relative", "Load-adjusted", lc=C["blue"], rc=C["gold"])
        ax[3].legend(loc="lower right")
        # the columns in this table are PARTIAL R2, not the unique share of
        # total; mislabelling them would contradict Figure 1B
        label(ax[3], "Partial adjusted $R^2$", "",
              "Estimates are insensitive to microbial load")

    # (E) cross-habitat structure - reported here and nowhere else
    hv = rd(H, "cross_habitat_variance.csv")
    if hv is not None:
        PRETTY = {"ocean": "Marine", "human gut": "Human gut",
                  "animal": "Animal", "environmental": "Environmental"}
        blocks = ["taxonomy", "study", "geography", "technical"]
        habs = ["human gut", "animal", "environmental", "ocean"]
        habs = [h for h in habs if h in set(hv["habitat"])]
        x = np.arange(len(blocks)); w = 0.8 / len(habs)
        for i, h in enumerate(habs):
            g = hv[hv["habitat"] == h].set_index("block")
            vals = [g.loc[b, "unique"] if b in g.index else np.nan for b in blocks]
            ax[4].bar(x + i * w - 0.4 + w / 2, vals, width=w,
                      label=PRETTY.get(h, h), color=SERIES[i])
        ax[4].set_xticks(x)
        ax[4].set_xticklabels([b.capitalize() for b in blocks])
        ax[4].legend(ncol=2)
        grid_axis(ax[4], "y"); despine(ax[4], bottom=True)
        label(ax[4], "", "Unique adjusted $R^2$",
              "Cross-habitat variance structure")

        # (F) is the habitat difference a power effect?
        # The human-gut estimate is from the adult analysis, so its sample size
        # must be the adult count. cross_habitat_variance.csv carries the full
        # cohort size, which would misplace the point on the size axis.
        t = hv[hv["block"] == "taxonomy"].copy()
        adult_n = None
        stp = (work / "sample_table.parquet") if work else \
            Path(str(T).replace("/tables", "/work")) / "sample_table.parquet"
        if stp.exists():
            stt = pd.read_parquet(stp)
            if "age_category" in stt.columns:
                sub = stt[stt["age_category"].astype(str) == "adult"]
                # intersect with the aligned matrix: the estimate comes from
                # samples carrying both layers, not from every adult sample
                tp = (work / "taxa_clr.parquet") if work else None
                ap = (work / "arg_abundance.parquet") if work else None
                if tp is not None and tp.exists() and ap.exists():
                    al = (set(pd.read_parquet(tp).index.astype(str))
                          & set(pd.read_parquet(ap).index.astype(str)))
                    sub = sub[sub["sample"].astype(str).isin(al)]
                adult_n = int(len(sub))
        if adult_n:
            t.loc[t["habitat"] == "human gut", "n_samples"] = adult_n
            print(f"    Figure 2F: human-gut sample size set to the adult "
                  f"analysis population ({adult_n:,}), not the full cohort.")
        ax[5].grid(False)
        ax[5].scatter(t["n_samples"], t["unique"],
                      s=np.clip(pd.to_numeric(t["n_studies"], errors="coerce").fillna(20) * 3.2, 60, 420),
                      color=C["blue"], alpha=0.85, edgecolors="white",
                      linewidths=1.4, zorder=3)
        for _, r in t.iterrows():
            ax[5].annotate(PRETTY.get(r["habitat"], r["habitat"]),
                           (r["n_samples"], r["unique"]),
                           textcoords="offset points", xytext=(11, 7), fontsize=8)
        z = np.polyfit(np.log10(t["n_samples"]), t["unique"], 1)
        xs = np.linspace(np.log10(t["n_samples"].min() * 0.7),
                         np.log10(t["n_samples"].max() * 1.5), 30)
        ax[5].plot(10 ** xs, np.polyval(z, xs), ls="--", lw=1.5, color=C["blue"])
        ax[5].set_xscale("log")
        ax[5].set_xlim(t["n_samples"].min() * 0.5, t["n_samples"].max() * 2.6)
        from scipy.stats import spearmanr
        rr = spearmanr(t["n_samples"], t["unique"])
        ax[5].set_ylim(t["unique"].min() * 0.90, t["unique"].max() * 1.06)
        statbox(ax[5], f"Spearman rho = {rr.statistic:.2f}\nn = {len(t)} habitats",
                "upper left")
        label(ax[5], "Number of samples (log scale)",
              "Taxonomy unique adjusted $R^2$",
              "Signal does not track habitat sample size")

    panel(fig, ax)
    return save(fig, out, "Figure2_variance_structure")


# ---------------------------------------- Figure 2: prediction & conservation --
def figure2(T: Path, out: Path):
    """Prediction degrades under cohort shift, and genomic conservation
    explains which genes survive that degradation."""
    fig, ax = plt.subplots(3, 2, figsize=(11.6, 13.4))
    fig.subplots_adjust(hspace=0.40, wspace=0.34)
    ax = ax.ravel()

    NICE = {"xgb": "XGBoost", "rf": "RF", "ridge": "Ridge",
            "taxa_nn": "Taxa NN", "mlp": "MLP", "mean": "Mean",
            "ft": "FT-Transformer"}

    def model_matrix(scheme, value):
        rows, strata = [], []
        for st in ("adult", "all", "infant"):
            m = rd(T, f"model_summary_{st}.csv")
            if m is None:
                continue
            g = m[(m["scheme"] == scheme) & (m["model"] != "mean")]
            if not len(g):
                continue
            rows.append(g.set_index("model")[value]); strata.append(st)
        if not rows:
            return None, None, None
        M = pd.DataFrame(rows, index=[s.capitalize() for s in strata])
        return M, M.index.tolist()

    # A single column order everywhere: heatmaps whose columns are reordered
    # per panel cannot be compared across panels.
    _base = rd(T, "model_summary_adult.csv")
    MODEL_ORDER = (_base[(_base["scheme"] == "leave_one_study_out")
                         & (_base["model"] != "mean")]
                   .sort_values("median_spearman", ascending=False)["model"].tolist()
                   if _base is not None else [])

    # (A,B) predictability by model and stratum, under both schemes
    for k, (scheme, ttl) in enumerate([("random", "Random cross-validation"),
                                       ("leave_one_study_out",
                                        "Leave-one-study-out")]):
        M, rows = model_matrix(scheme, "median_spearman")
        if M is None:
            continue
        cols = [c for c in MODEL_ORDER if c in M.columns]
        heatmap(ax[k], M[cols].values, rows, [NICE.get(c, c) for c in cols],
                r"Median Spearman $\rho$", cmap="Reds", fmt="{:.2f}", fig=fig)
        ax[k].set_title(f"{ttl} predictability", fontweight="bold",
                        fontsize=9.5, pad=8)

    # (C) held-out studies: the estimate free of selection
    rows, strata = [], []
    for st in ("adult", "all", "infant"):
        h = rd(T, f"holdout_summary_{st}.csv")
        if h is None:
            continue
        g = h[h["model"] != "mean"].set_index("model")["median_spearman"]
        rows.append(g); strata.append(st)
    if rows:
        M = pd.DataFrame(rows, index=[s.capitalize() for s in strata])
        cols = [c for c in MODEL_ORDER if c in M.columns]
        heatmap(ax[2], M[cols].values, M.index.tolist(),
                [NICE.get(c, c) for c in cols], r"Median Spearman $\rho$",
                cmap="Reds", fmt="{:.2f}", fig=fig)
        ax[2].set_title("Held-out-study performance", fontweight="bold",
                        fontsize=9.5, pad=8)

    # (D) the central result
    # Prefer the adult-population evidence set; fall back to the earlier
    # all-ages one only if the adult run has not been performed. Reading
    # whichever file happens to exist would silently mix populations.
    mob = rd(T, "mobility_vs_predictability_adult_adult.csv")
    if mob is None:
        mob = rd(T, "mobility_vs_predictability_independent_adult.csv")
    if mob is not None:
        from scipy.stats import spearmanr
        core = mob["conservation"] >= 0.5
        ax[3].grid(True)
        # A NaN marker size produces a legend handle with an empty path, and
        # matplotlib then fails at savefig with "need at least one array to
        # concatenate" - a traceback that points at the renderer rather than
        # at the data. Genes present in the model output but absent from the
        # stratum-restricted abundance matrix have no prevalence, so this is a
        # live possibility whenever the gene universe widens.
        _prev = pd.to_numeric(mob.get("prevalence", pd.Series(np.nan,
                                      index=mob.index)), errors="coerce")
        _n_nan = int(_prev.isna().sum())
        if _n_nan:
            print(f"    NOTE: {_n_nan} of {len(mob)} gene families have no "
                  f"prevalence value; plotted at the median marker size.")
        _prev = _prev.fillna(_prev.median() if _prev.notna().any() else 0.2)
        sz = np.clip(_prev * 900, 28, 320).astype(float)
        # drop any row whose coordinates are not finite, for the same reason
        _ok = np.isfinite(mob["conservation"]) & np.isfinite(mob["rho"])
        if not _ok.all():
            print(f"    NOTE: {int((~_ok).sum())} gene families dropped from "
                  f"Figure 3D for non-finite coordinates.")
            mob, sz = mob[_ok].reset_index(drop=True), sz[_ok].reset_index(drop=True)
            core = mob["conservation"] >= 0.5
        # An empty series cannot be drawn or legended: matplotlib raises
        # "need at least one array to concatenate" when it computes the
        # legend's extents. Plot each group only if it has members, and label
        # only what was plotted.
        if (~core).any():
            ax[3].scatter(mob.loc[~core, "conservation"], mob.loc[~core, "rho"],
                          s=sz[~core], color=C["blue"], alpha=0.72,
                          edgecolors="white", linewidths=0.8,
                          label=f"Accessory-enriched (n={int((~core).sum())})",
                          zorder=3)
        if core.any():
            ax[3].scatter(mob.loc[core, "conservation"], mob.loc[core, "rho"],
                          s=sz[core], color=C["gold"], alpha=0.9, marker="s",
                          edgecolors="white", linewidths=0.8,
                          label=f"Mostly core (n={int(core.sum())})", zorder=3)
        else:
            print("    NOTE: no gene family reaches conservation >= 0.5; the "
                  "core-like group is empty and is omitted from Figure 3D.")
        # Labelling every core gene produces overlapping text in the dense
        # upper-right cluster; label the five most conserved and alternate the
        # offset so adjacent labels do not collide.
        lab = mob[core].nlargest(5, "conservation").sort_values("rho")
        for k2, (_, r) in enumerate(lab.iterrows()):
            ax[3].annotate(r["gene"], (r["conservation"], r["rho"]),
                           textcoords="offset points",
                           xytext=(9, 7) if k2 % 2 == 0 else (9, -12),
                           fontsize=7, style="italic", color=INK)
        z = np.polyfit(mob["conservation"], mob["rho"], 1)
        xs = np.linspace(mob["conservation"].min(), mob["conservation"].max(), 40)
        ax[3].plot(xs, np.polyval(z, xs), ls="--", lw=1.5, color=C["blue"])
        rr = spearmanr(mob["conservation"], mob["rho"])
        statbox(ax[3], f"Spearman rho = {rr.statistic:.2f}\nP = {rr.pvalue:.1e}\n"
                       f"n = {len(mob)} ARG families")
        if ax[3].get_legend_handles_labels()[0]:
            ax[3].legend(loc="upper left")
        label(ax[3], "Within-species conservation",
              r"Cross-cohort Spearman $\rho$",
              "Genomic conservation predicts ARG predictability")

    # (E) generalisation by model and stratum
    piv = {}
    for st in ("adult", "all", "infant"):
        m = rd(T, f"model_summary_{st}.csv")
        if m is None:
            continue
        g = m[(m["scheme"] == "leave_one_study_out") & (m["model"] != "mean")]
        piv[st.capitalize()] = g.set_index("model")["median_spearman"]
    if piv:
        P = pd.DataFrame(piv)
        order = P.mean(axis=1).sort_values(ascending=False).index
        P = P.loc[order]
        for i, c in enumerate(P.columns):
            ax[4].plot(np.arange(len(P)), P[c], "-o", lw=1.8, ms=6.5,
                       color=SERIES[i], mec="white", mew=1.1, label=c)
            for xi, v in enumerate(P[c]):
                ax[4].annotate(f"{v:.2f}", (xi, v), textcoords="offset points",
                               xytext=(0, 8), ha="center", fontsize=6.8)
        ax[4].set_xticks(np.arange(len(P)))
        ax[4].set_xticklabels([NICE.get(m, m) for m in P.index])
        ax[4].legend(title="Life stage", title_fontsize=7.5)
        grid_axis(ax[4], "y")
        label(ax[4], "", r"Median Spearman $\rho$",
              "Cross-cohort generalisation by model")

    # (F) the joint model, with bootstrap intervals
    mv = rd(T, "mobility_multivariable_adult.csv")
    if mv is not None:
        d = mv[mv["term"] != "const"].copy().iloc[::-1]
        y = np.arange(len(d))
        ax[5].barh(y, d["beta"], height=0.55,
                   color=[C["blue"] if b > 0 else C["red"] for b in d["beta"]])
        if "ci_low" in d.columns:
            ax[5].errorbar(d["beta"], y,
                           xerr=[(d["beta"] - d["ci_low"]).abs(),
                                 (d["ci_high"] - d["beta"]).abs()],
                           fmt="none", ecolor=INK, capsize=3.5, lw=1.1)
        ax[5].axvline(0, lw=1.1, color=INK)
        ax[5].set_yticks(y)
        ax[5].set_yticklabels([t.replace("_", " ").replace("arg sd", "Abundance variance")
                               .capitalize() for t in d["term"]])
        grid_axis(ax[5], "x"); despine(ax[5], left=True)
        label(ax[5], "Standardised coefficient (95% CI)", "",
              "Conservation and host breadth act in opposite directions")

    panel(fig, ax)
    return save(fig, out, "Figure3_prediction_conservation")


# ------------------------------------------- Figure 3: within-subject & carriage --
def figure3(T: Path, out: Path):
    """Within-subject coupling and high-risk gene carriage."""
    fig, ax = plt.subplots(3, 2, figsize=(11.6, 13.4))
    fig.subplots_adjust(hspace=0.42, wspace=0.34)
    ax = ax.ravel()

    lv = rd(T, "longitudinal_variance_all.csv")
    if lv is not None:
        y = np.arange(len(lv))[::-1]
        ax[0].barh(y, lv["R2_adj"], height=0.5,
                   color=[C["blue"], C["grey"]][:len(lv)])
        for yi, v, n in zip(y, lv["R2_adj"], lv["n"]):
            ax[0].annotate(f"{v:.3f}   (n = {int(n):,})", (v, yi),
                           textcoords="offset points", xytext=(8, -3),
                           fontsize=8, color=INK)
        ax[0].set_yticks(y)
        ax[0].set_yticklabels([l.replace("_", "-").capitalize() for l in lv["level"]])
        ax[0].set_xlim(0, lv["R2_adj"].max() * 1.45)
        grid_axis(ax[0], "x"); despine(ax[0], left=True)
        label(ax[0], "Variance explained by taxonomy", "",
              "Coupling persists after removing all stable confounding")

    lp = rd(T, "longitudinal_paired_change_all.csv")
    if lp is not None and "spearman" in lp.columns:
        lp = lp[lp["study"].notna()]
        if "time_gap_band" in lp.columns:
            lp = lp[lp["time_gap_band"].isna()]
        lp = lp.dropna(subset=["spearman"]).sort_values("spearman")
        yy = np.arange(len(lp))
        ax[1].scatter(lp["spearman"], yy, s=np.clip(pd.to_numeric(lp["n_pairs"], errors="coerce").fillna(20) / 5, 16, 130),
                      color=C["blue"], alpha=0.78, edgecolors="white",
                      linewidths=0.8, zorder=3)
        ax[1].axvline(0, lw=1.1, color=INK)
        ax[1].axvline(lp["spearman"].median(), ls="--", lw=1.3, color=C["red"])
        for idx in list(lp.index[:2]) + list(lp.index[-3:]):
            i2 = list(lp.index).index(idx)
            ax[1].annotate(short_study(lp.loc[idx, "study"]),
                           (lp.loc[idx, "spearman"], i2),
                           textcoords="offset points", xytext=(8, -3),
                           fontsize=7, color=INK)
        ax[1].set_yticks([])
        n_pos = int((lp["spearman"] > 0).sum())
        statbox(ax[1], f"Positive in {n_pos} of {len(lp)} studies\n"
                       f"median rho = {lp['spearman'].median():.2f}", "upper left")
        grid_axis(ax[1], "x"); despine(ax[1], left=True)
        label(ax[1], r"Compositional vs resistome change, Spearman $\rho$",
              "Study (point size = pairs)", "Within-subject coupling by study")

    dr = rd(T, "dose_response_all.csv")
    if dr is not None and "spearman_adjusted" in dr.columns:
        d = dr.dropna(subset=["spearman_raw", "spearman_adjusted"]).copy()
        d = d.nlargest(8, "n_subjects").sort_values("spearman_raw")
        dumbbell(ax[2], [short_study(x) for x in d["study"]],
                 d["spearman_adjusted"].values, d["spearman_raw"].values,
                 "Adjusted", "Unadjusted", lc=C["blue"], rc=C["gold"], fmt="{:.2f}")
        ax[2].axvline(0, lw=1.0, color=INK)
        ax[2].set_xlim(min(0, d[["spearman_raw", "spearman_adjusted"]].min().min()) - 0.10,
                       d[["spearman_raw", "spearman_adjusted"]].max().max() + 0.14)
        ax[2].legend(loc="upper left", ncol=2)
        label(ax[2], r"Displacement coupling, Spearman $\rho$", "",
              "Adjusted for elapsed time and sequencing depth")

    ca = rd(T, "carriage_prediction_auc_adult.csv")
    if ca is not None:
        wp = ca["well_powered"].astype(bool)
        ax[3].grid(True)
        for mask, fc, ec, lab_ in ((wp, C["blue"], "white", "Well powered"),
                                   (~wp, "none", GREY, "Not well powered")):
            if not mask.any():
                continue
            ax[3].scatter(ca.loc[mask, "prevalence"] * 100,
                          ca.loc[mask, "median_auc"],
                          s=np.clip(pd.to_numeric(ca.loc[mask, "n_carriers"], errors="coerce").fillna(50) / 10, 30, 260),
                          facecolors=fc, edgecolors=ec, linewidths=1.3,
                          alpha=0.85, label=lab_, zorder=3)
        ax[3].axhline(0.5, ls="--", lw=1.0, color=INK)
        ax[3].axhline(0.7, ls=":", lw=1.2, color=C["red"])
        for _, r in ca.iterrows():
            if r["median_auc"] > 0.7 or r["prevalence"] > 0.3:
                ax[3].annotate(r["gene"], (r["prevalence"] * 100, r["median_auc"]),
                               textcoords="offset points", xytext=(9, 5),
                               fontsize=7.5, style="italic")
        ax[3].set_xscale("log")
        ax[3].set_ylim(0.45, max(0.99, ca["median_auc"].max() * 1.05))
        ax[3].legend(loc="lower left", framealpha=0.92, frameon=True,
                     edgecolor="#C6CCD6", facecolor="white")
        label(ax[3], "Carriage prevalence (%, log scale)",
              "Median leave-one-study-out AUC",
              "Cross-cohort predictability of high-risk ARG carriage")

    en = rd(T, "carriage_enterotype_associations_adult.csv")
    if en is not None and "odds_ratio" in en.columns:
        e = en.dropna(subset=["odds_ratio"]).copy()
        padj = "p_adj_BH" if "p_adj_BH" in e.columns else "p_adj"
        if padj in e.columns:
            e = e[e[padj] < 0.05]
        e = e.reindex(e["odds_ratio"].apply(np.log).abs().sort_values().index).tail(10)
        yy = np.arange(len(e))
        ax[4].scatter(e["odds_ratio"], yy,
                      s=np.clip(pd.to_numeric(e["lrt_chi2"], errors="coerce").fillna(10) * 1.6, 40, 240),
                      color=[C["red"] if v > 1 else C["blue"] for v in e["odds_ratio"]],
                      edgecolors="white", linewidths=1.1, zorder=3)
        ax[4].axvline(1, lw=1.1, color=INK)
        ax[4].set_xscale("log")
        ax[4].set_yticks(yy)

        def _lvl(v):
            return (str(v).replace("ent_enterotype_", "").replace("enterotype_", "")
                    .replace("dysbiosis_score", "dysbiosis score"))
        ax[4].set_yticklabels([f"{g} × {_lvl(t)}"
                               for g, t in zip(e["gene"], e["strongest_level"])],
                              fontsize=7.5)
        grid_axis(ax[4], "x"); despine(ax[4], left=True)
        label(ax[4], "Odds ratio (log scale, point size = $\\chi^2$)", "",
              "Carriage and community state (FDR < 0.05)")

    ag = rd(T, "absolute_vs_relative_groups_adult.csv")
    if ag is not None and "frac_reordered" in ag.columns:
        d = ag.sort_values("rank_rho")
        y = np.arange(len(d))
        ax[5].barh(y, d["rank_rho"], height=0.5, color=C["blue"])
        for yi, r, f in zip(y, d["rank_rho"], d["frac_reordered"]):
            ax[5].annotate(f"rho = {r:.3f}   ({100*f:.1f}% reordered)", (r, yi),
                           textcoords="offset points", xytext=(-8, -3),
                           ha="right", fontsize=7.5, color="white")
        ax[5].set_yticks(y)
        ax[5].set_yticklabels([g.replace("_", " ").capitalize() for g in d["grouping"]])
        ax[5].set_xlim(0, 1.05)
        grid_axis(ax[5], "x"); despine(ax[5], left=True)
        label(ax[5], "Rank agreement between relative and load-adjusted scales", "",
              "Population comparisons are robust to microbial load")

    panel(fig, ax)
    return save(fig, out, "Figure4_within_subject_carriage")


# ---------------------------------------------- Figure 4: hypotheses tested --
def figure4(T: Path, H: Path, out: Path):
    """Hypotheses tested in advance, and not supported."""
    fig, ax = plt.subplots(2, 2, figsize=(11.6, 9.0))
    fig.subplots_adjust(hspace=0.44, wspace=0.34)
    ax = ax.ravel()

    td = rd(T, "temporal_direction_all.csv")
    if td is not None:
        d = td.dropna(subset=["median_rho"])
        res = d[d["model"].str.startswith("resistome")]
        com = d[d["model"].str.startswith("composition")]
        series = {}
        if len(res) == 2:
            series["Future resistome"] = [res.iloc[0]["median_rho"],
                                          res.iloc[1]["median_rho"]]
        if len(com) == 2:
            series["Future composition"] = [com.iloc[0]["median_rho"],
                                            com.iloc[1]["median_rho"]]
        if series:
            slope(ax[0], ["Own past only", "Other layer added"], series,
                  colors=[C["blue"], C["gold"]], fmt="{:.2f}")
            ax[0].legend(loc="center right")
            gains = d["gain"].dropna()
            for k, g in enumerate(gains):
                ax[0].annotate(f"$\\Delta$ = {g:+.2f}", (0.5, 0.30 + 0.34 * k),
                               xycoords="axes fraction", ha="center", fontsize=8.5)
            label(ax[0], "", r"Median Spearman $\rho$",
                  "Adding the other layer reduces performance")

    eb = rd(T, "exposure_boundary_summary_all.csv")
    if eb is not None and "median_crossing" in eb.columns:
        d = eb.dropna(subset=["median_internal"]).head(2)
        lab_ = [c.split(":")[0].replace("d_taxa", "Composition")
                .replace("d_arg", "Resistome") for c in d["comparison"]]
        dumbbell(ax[1], lab_, d["median_internal"].values,
                 d["median_crossing"].values,
                 "Within exposure state", "Across boundary",
                 lc=C["grey"], rc=C["red"], fmt="{:.1f}")
        ax[1].legend(loc="lower right")
        ps = ",  ".join(f"P = {p:.3f}" for p in d["p_value"])
        label(ax[1], "Median displacement", "",
              f"No significant difference ({ps})")

    md = rd(T, "mediation_all.csv")
    if md is not None:
        r = md.iloc[0]
        terms = ["total_effect", "direct_effect", "indirect_effect"]
        vals = [r[t] for t in terms]
        y = np.arange(3)[::-1]
        ax[2].barh(y, vals, height=0.5,
                   color=[C["grey"], C["grey"], C["red"]])
        if {"indirect_ci_low", "indirect_ci_high"} <= set(md.columns):
            ax[2].errorbar(r["indirect_effect"], y[-1],
                           xerr=[[r["indirect_effect"] - r["indirect_ci_low"]],
                                 [r["indirect_ci_high"] - r["indirect_effect"]]],
                           fmt="none", ecolor=INK, capsize=4, lw=1.2)
        for yi, v in zip(y, vals):
            ax[2].annotate(f"{v:.3f}", (v, yi), textcoords="offset points",
                           xytext=(8, -3), fontsize=8)
        ax[2].axvline(0, lw=1.1, color=INK)
        ax[2].set_yticks(y)
        ax[2].set_yticklabels(["Total", "Direct", "Indirect (mediated)"])
        grid_axis(ax[2], "x"); despine(ax[2], left=True)
        pm = r.get("proportion_mediated", np.nan)
        label(ax[2], "Effect on resistome change", "",
              f"Mediated path spans zero ({100*pm:.1f}% mediated)")

    fh = rd(H, "fold_heterogeneity_by_habitat.csv")
    if fh is not None:
        PRETTY = {"ocean": "Marine"}
        d = fh.sort_values("rho_distance")
        y = np.arange(len(d))
        ax[3].barh(y, d["rho_distance"], height=0.5,
                   color=[C["red"] if v > 0 else C["blue"] for v in d["rho_distance"]])
        for yi, v, n in zip(y, d["rho_distance"], d["n_folds"]):
            ax[3].annotate(f"{v:+.2f}", (v, yi), textcoords="offset points",
                           xytext=(8 if v > 0 else -8, -3),
                           ha="left" if v > 0 else "right", fontsize=8)
        ax[3].axvline(0, lw=1.1, color=INK)
        ax[3].set_yticks(y)
        ax[3].set_yticklabels([f"{PRETTY.get(h, h.capitalize())} (n = {int(n)})"
                               for h, n in zip(d["habitat"], d["n_folds"])])
        lim = float(np.abs(d["rho_distance"]).max()) * 1.6
        ax[3].set_xlim(-lim, lim)
        grid_axis(ax[3], "x"); despine(ax[3], left=True)
        label(ax[3], r"Cohort distance vs predictability, Spearman $\rho$", "",
              "Distance does not predict poorer transferability")

    panel(fig, ax)
    return save(fig, out, "Figure5_hypotheses_tested")


def figure5(T: Path, H: Path, out: Path):
    """Habitat performance and the metric caveat. Habitat VARIANCE is in
    Figure 1 and is deliberately not repeated."""
    fig, ax = plt.subplots(2, 2, figsize=(11.6, 9.0))
    fig.subplots_adjust(hspace=0.44, wspace=0.34)
    ax = ax.ravel()
    PRETTY = {"ocean": "Marine", "human gut": "Human gut", "animal": "Animal",
              "environmental": "Environmental"}
    from scipy.stats import spearmanr

    hm = rd(H, "cross_habitat_models.csv")
    if hm is not None:
        col = "median_spearman" if "median_spearman" in hm.columns else "spearman"
        if "model" in hm.columns and (hm["model"] == "xgb").any():
            hm = hm[hm["model"] == "xgb"]
        d = hm.groupby("habitat")[col].median().rename("rho").reset_index()
        d["lab"] = d["habitat"].map(lambda h: PRETTY.get(h, h.capitalize()))
        d = d.sort_values("rho")
        y = np.arange(len(d))
        ax[0].barh(y, d["rho"], height=0.5,
                   color=[C["blue"] if h == "human gut" else C["grey"]
                          for h in d["habitat"]])
        for yi, v in zip(y, d["rho"]):
            ax[0].annotate(f"{v:.3f}", (v, yi), textcoords="offset points",
                           xytext=(8, -3), fontsize=8.5)
        ax[0].set_yticks(y); ax[0].set_yticklabels(d["lab"])
        ax[0].set_xlim(0, d["rho"].max() * 1.35)
        grid_axis(ax[0], "x"); despine(ax[0], left=True)
        label(ax[0], r"Median cross-cohort Spearman $\rho$ (gradient boosting)", "",
              "Cross-cohort performance by habitat")

    art = []
    for hd in sorted(x for x in H.iterdir() if x.is_dir() and x.name != "logs"):
        perf, argf, stf = (hd / "model_performance.csv",
                           hd / "arg_abundance.parquet", hd / "sample_table.parquet")
        if not (perf.exists() and argf.exists() and stf.exists()):
            continue
        pf = pd.read_csv(perf)
        pf = pf[(pf.get("model", "xgb") == "xgb") & pf["median_spearman"].notna()]
        arg = pd.read_parquet(argf)
        stt = pd.read_parquet(stf).set_index("sample")["study"].astype(str)
        for _, r in pf.iterrows():
            ids = [i for i in stt[stt == str(r["study"])].index if i in arg.index]
            if len(ids) < 20:
                continue
            art.append({"habitat": PRETTY.get(hd.name, hd.name.capitalize()),
                        "rho": r["median_spearman"], "n": len(ids),
                        "arg_sd": float(np.median(arg.loc[ids].std(axis=0)))})
    art = pd.DataFrame(art)
    if len(art):
        habs = sorted(art["habitat"].unique())
        cm = {h: SERIES[i % len(SERIES)] for i, h in enumerate(habs)}
        ax[1].grid(True)
        for h in habs:
            g = art[art["habitat"] == h]
            if not len(g):
                continue
            ax[1].scatter(g["arg_sd"], g["rho"], s=np.clip(pd.to_numeric(g["n"], errors="coerce").fillna(30) / 3, 26, 200),
                          alpha=0.78, color=cm[h], edgecolors="white",
                          linewidths=0.7, label=h, zorder=3)
        z = np.polyfit(art["arg_sd"], art["rho"], 1)
        xs = np.linspace(art["arg_sd"].min(), art["arg_sd"].max(), 40)
        ax[1].plot(xs, np.polyval(z, xs), ls="--", lw=1.5, color=INK)
        rr = spearmanr(art["arg_sd"], art["rho"])
        statbox(ax[1], f"Spearman rho = {rr.statistic:.2f}\n"
                       f"P = {rr.pvalue:.1e}\nn = {len(art)} folds", "upper left")
        ax[1].legend(loc="lower right", ncol=2)
        label(ax[1], "Held-out cohort resistome variance",
              r"Cross-cohort Spearman $\rho$",
              f"Metric caveat: performance scales with outcome variance "
              f"({len(art)} non-gut folds)")

    fh = rd(H, "fold_heterogeneity.csv")
    if fh is not None:
        d = fh.copy()
        d["lab"] = d["habitat"].map(lambda h: PRETTY.get(h, h.capitalize()))
        hb = sorted(d["lab"].unique())
        cm = {h: SERIES[i % len(SERIES)] for i, h in enumerate(hb)}
        ax[2].grid(True)
        for h in hb:
            g = d[d["lab"] == h]
            if not len(g):
                continue
            ax[2].scatter(g["n_test"], g["median_spearman"], s=30, alpha=0.75,
                          color=cm[h], edgecolors="white", linewidths=0.6,
                          label=h, zorder=3)
        ax[2].set_xscale("log")
        rr = spearmanr(d["n_test"], d["median_spearman"])
        ax[2].legend(loc="lower right", ncol=2)
        statbox(ax[2], f"Spearman rho = {rr.statistic:.2f}", "upper left")
        label(ax[2], "Held-out cohort size (log scale)",
              r"Cross-cohort Spearman $\rho$",
              "Performance is not explained by cohort size")

    hc = rd(T, "conservation_by_host_count_adult_adult.csv")
    if hc is None:
        hc = rd(T, "conservation_by_host_count_independent_adult.csv")
    if hc is not None:
        ax[3].plot(hc["min_host_species"], hc["spearman"], "-o", lw=2.0, ms=9,
                   color=C["blue"], mec="white", mew=1.4, zorder=3)
        for _, r in hc.iterrows():
            ax[3].annotate(f"{r['spearman']:.2f}\nn = {int(r['n_genes'])}",
                           (r["min_host_species"], r["spearman"]),
                           textcoords="offset points", xytext=(0, 13),
                           ha="center", fontsize=7.5)
        ax[3].set_ylim(0, 1.0)
        ax[3].set_xticks(hc["min_host_species"])
        grid_axis(ax[3], "y")
        label(ax[3], "Minimum host species per gene family",
              r"Conservation vs predictability, Spearman $\rho$",
              "Association persists after excluding single-host genes")

    panel(fig, ax)
    return save(fig, out, "Figure6_habitats_and_metric")



# --------------------------------------------- Figure 1: study workflow ----
def figure_workflow(T: Path, H: Path, out: Path, work: Path | None = None):
    """The study design as a portrait flow diagram, with every count read from
    the analysis outputs.

    Layout notes. The figure is portrait rather than landscape: a workflow is
    read top to bottom, and a tall single column keeps the reading order
    unambiguous while fitting a journal page without rotation. Arrows are drawn
    between box EDGES computed from geometry rather than to fixed coordinates,
    so nothing crosses a box or stops short of one. Attrition is shown beside
    the cascade so the losses are legible. The other-habitat branch leaves the
    flow sideways, because those analyses run in parallel to the human ones
    rather than downstream of them.
    """
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(7.6, 11.6))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    # ---- counts, read from the pipeline ---------------------------------
    c = {}
    cb = rd(T, "cohort_by_study.csv")
    if cb is not None:
        c["final"] = int(cb["n_samples"].sum()); c["studies"] = int(len(cb))
        if "countries" in cb.columns:
            c["countries"] = len({x.strip() for v in cb["countries"].dropna()
                                  for x in str(v).split(",")})
    if work is not None:
        rep = work / "join_report.txt"
        if rep.exists():
            txt = rep.read_text(encoding="utf-8", errors="ignore")
            for key, pat in (("zenodo_runs", r"Zenodo runs\s*:\s*([\d,]+)"),
                             ("bridged_runs", r"bridges -> runs\s*:\s*([\d,]+)"),
                             ("bridged_samples", r"bridges -> Metalog samples:\s*([\d,]+)"),
                             ("metalog", r"Metalog human samples with metadata\s*:\s*([\d,]+)"),
                             ("joined", r"curated metadata\s*:\s*([\d,]+)"),
                             ("faecal", r"ENVO:00002003\]\s*:\s*([\d,]+)"),
                             ("status", r"known disease status\s*:\s*([\d,]+)"),
                             ("final", r"FINAL USABLE SAMPLES:\s*([\d,]+)")):
                m = re.search(pat, txt)
                if m:
                    c[key] = int(m.group(1).replace(",", ""))
        for k2, f in (("arg", "arg_abundance.parquet"), ("taxa", "taxa_clr.parquet")):
            fp = work / f
            if fp.exists():
                d = pd.read_parquet(fp); c[f"{k2}_n"], c[f"{k2}_p"] = d.shape
        lp = work / "longitudinal_pairs_all.parquet"
        if lp.exists():
            c["subjects"] = int(pd.read_parquet(lp)["subject_id"].nunique())

    hv = rd(H, "cross_habitat_variance.csv")
    hab = {}
    if hv is not None:
        for _, r in hv[hv["block"] == "taxonomy"].iterrows():
            hab[{"ocean": "Marine"}.get(r["habitat"], r["habitat"].capitalize())] = \
                int(r["n_samples"])
    hab.pop("Human gut", None)

    def n(k, f="{:,}"):
        v = c.get(k)
        return f.format(v) if isinstance(v, (int, np.integer)) else "—"

    missing = [k for k in ("zenodo_runs", "bridged_runs", "arg_n", "subjects")
               if k not in c]
    if missing:
        print(f"    NOTE: {len(missing)} counts unavailable, shown as em-dash "
              f"({', '.join(missing)}). Pass --work to fill them.")

    BOXES = {}

    def box(name, x, y, w, h, lines, face, edge=None, fs=8.2, tc=None,
            weight="normal", lw=1.3, radius=1.0):
        ax.add_patch(FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle=f"round,pad=0.0,rounding_size={radius}",
            facecolor=face, edgecolor=edge or face, linewidth=lw, zorder=2))
        if lines:
            ax.text(x, y, lines, ha="center", va="center", fontsize=fs,
                    zorder=3, color=tc or INK, weight=weight, linespacing=1.5)
        BOXES[name] = (x, y, w, h)
        return name

    def edge(name, side):
        x, y, w, h = BOXES[name]
        return {"t": (x, y + h / 2), "b": (x, y - h / 2),
                "l": (x - w / 2, y), "r": (x + w / 2, y)}[side]

    def arrow(a, b, colour=None, lw=1.4, rad=0.0):
        ax.add_patch(FancyArrowPatch(
            a, b, arrowstyle="-|>", mutation_scale=12, linewidth=lw,
            color=colour or "#9BA5B4", zorder=1,
            connectionstyle=f"arc3,rad={rad}", shrinkA=1.5, shrinkB=1.5))

    NAVY, SKY = "#2E5C8A", "#DCE7F3"
    PANEL, PANEL_E = "#F2F4F7", "#C3CBD6"
    CX = 46          # centre of the main column, offset to leave a side margin

    # ---- sources ---------------------------------------------------------
    ax.text(CX, 98.4, "Public metagenome resources", ha="center", fontsize=10,
            fontweight="bold", color=INK)
    box("res", CX - 23, 93.4, 42, 5.6,
        f"Resistome resource · {n('zenodo_runs')} runs\nResFinder / KMA",
        SKY, NAVY, fs=7.8)
    box("met", CX + 23, 93.4, 42, 5.6,
        f"Curated metadata · {n('metalog')} samples\nMetaPhlAn 4", SKY, NAVY,
        fs=7.8)
    box("bridge", CX, 85.2, 76, 5.4,
        f"Bridged by run and by sample accession\n"
        f"{n('bridged_runs')} runs from {n('bridged_samples')} samples",
        PANEL, PANEL_E, fs=7.8)
    arrow(edge("res", "b"), (CX - 12, 88.2), rad=-0.12)
    arrow(edge("met", "b"), (CX + 12, 88.2), rad=0.12)

    # ---- filtering cascade ----------------------------------------------
    steps = [("Joined with curated metadata", "joined"),
             ("Faecal material [ENVO:00002003]", "faecal"),
             ("Recorded health status", "status"),
             (r"$\geq$10$^6$ post-QC reads", "final")]
    y, dy, prev = 76.6, 5.6, "bridged_samples"
    for k2, (lab, key) in enumerate(steps):
        nm = f"s{k2}"
        box(nm, CX, y, 64, 4.0, "", PANEL, PANEL_E)
        ax.text(CX - 29, y, lab, ha="left", va="center", fontsize=7.8,
                color=INK, zorder=3)
        ax.text(CX + 29, y, n(key), ha="right", va="center", fontsize=7.8,
                color=INK, fontweight="bold", zorder=3)
        if c.get(prev) and c.get(key) and c[prev] - c[key] > 0:
            ax.text(CX - 34, y, f"−{c[prev] - c[key]:,}", ha="right",
                    va="center", fontsize=6.8, color=C["red"], zorder=3)
        arrow(edge("bridge" if k2 == 0 else f"s{k2-1}", "b"), edge(nm, "t"),
              lw=1.2)
        prev = key
        y -= dy

    box("cohort", CX, 51.4, 76, 5.8,
        f"Analysed human cohort\n{n('final')} samples · {n('studies')} studies "
        f"· {n('countries')} countries",
        NAVY, tc="white", fs=8.6, weight="bold")
    arrow(edge("s3", "b"), edge("cohort", "t"), lw=1.5)

    # ---- parallel habitats ----------------------------------------------
    if hab:
        txt = "\n".join(f"{k}  {v:,}" for k, v in hab.items())
        box("hab", 92, 64.5, 15, 15, f"Other\nhabitats\n\n{txt}", "white",
            PANEL_E, fs=6.9, lw=1.1)
        ax.add_patch(FancyArrowPatch(
            edge("bridge", "r"), edge("hab", "t"), arrowstyle="-|>",
            mutation_scale=10, linewidth=1.2, color="#AEB7C4",
            linestyle=(0, (5, 3)), zorder=1,
            connectionstyle="arc3,rad=-0.35", shrinkA=2, shrinkB=2))
        # placed to the left of the connector, not beneath it
        ax.text(80.5, 78.6, "same procedure,\nno host filters", ha="center",
                fontsize=6.4, color=GREY, style="italic", zorder=3,
                linespacing=1.35)

    # ---- the two data layers --------------------------------------------
    box("arg", CX - 20, 42.4, 34, 5.8,
        f"Resistome matrix\n{n('arg_n')} × {n('arg_p')} gene families",
        SKY, C["red"], fs=7.6)
    box("taxa", CX + 20, 42.4, 34, 5.8,
        f"Taxonomic matrix\n{n('taxa_n')} × {n('taxa_p')} species",
        SKY, C["green"], fs=7.6)
    arrow((CX - 14, 48.5), edge("arg", "t"), C["red"], rad=0.16)
    arrow((CX + 14, 48.5), edge("taxa", "t"), C["green"], rad=-0.16)

    box("align", CX, 34.6, 40, 4.2,
        f"Aligned dataset · {n('taxa_n')} samples", PANEL, PANEL_E, fs=7.8)
    arrow(edge("arg", "b"), (CX - 8, 36.8), C["red"], lw=1.2, rad=-0.14)
    arrow(edge("taxa", "b"), (CX + 8, 36.8), C["green"], lw=1.2, rad=0.14)

    # ---- analyses, stacked in the portrait column -----------------------
    panels = [("Variance partitioning",
               "composition versus study, geography and host", C["gold"]),
              ("Cross-cohort prediction",
               "leave-one-study-out and fully held-out studies", C["gold"]),
              ("Within-subject analysis",
               f"{n('subjects')} repeatedly sampled subjects", C["gold"]),
              ("Genomic conservation",
               "pangenome frequency versus predictability", C["red"])]
    y = 26.2
    for k2, (title, sub, col) in enumerate(panels):
        nm = f"p{k2}"
        box(nm, CX, y, 64, 5.6, "", "white", col, lw=1.5)
        ax.text(CX - 29, y + 1.1, title, ha="left", va="center", fontsize=8.0,
                fontweight="bold", color=INK, zorder=3)
        ax.text(CX - 29, y - 1.3, sub, ha="left", va="center", fontsize=7.2,
                color="#4A5462", zorder=3)
        arrow(edge("align" if k2 == 0 else f"p{k2-1}", "b"), edge(nm, "t"),
              lw=1.1)
        y -= 7.2

    written = save(fig, out, "Figure1_study_workflow")
    return written


# ------------------------------------------------- supplementary figures ---
def supplementary(T: Path, H: Path, out: Path, work: Path | None = None,
                  exploratory: bool = False):
    """Cohort description, resistome composition, per-study held-out
    performance, and the exploratory attribution analysis."""

    # ---- S1: cohort landscape -------------------------------------------
    cb = rd(T, "cohort_by_study.csv")
    if cb is not None:
        fig, ax = plt.subplots(2, 2, figsize=(11.6, 8.6))
        fig.subplots_adjust(hspace=0.46, wspace=0.30)
        ax = ax.ravel()

        top = cb.nlargest(15, "n_samples").sort_values("n_samples")
        ax[0].barh(np.arange(len(top)), top["n_samples"], height=0.66, color=C["blue"])
        ax[0].set_yticks(np.arange(len(top)))
        ax[0].set_yticklabels([short_study(x) for x in top["study"]], fontsize=7.5)
        grid_axis(ax[0], "x"); despine(ax[0], left=True)
        label(ax[0], "Samples", "",
              f"Fifteen largest of {len(cb)} studies")

        if "countries" in cb.columns:
            cc = {}
            for _, r in cb.iterrows():
                for x in str(r["countries"]).split(","):
                    x = x.strip()
                    if x and x != "nan":
                        cc[x] = cc.get(x, 0) + int(r["n_samples"])
            cs = pd.Series(cc).nlargest(15).sort_values()
            ax[1].barh(np.arange(len(cs)), cs.values, height=0.66, color=C["green"])
            ax[1].set_yticks(np.arange(len(cs)))
            ax[1].set_yticklabels(cs.index, fontsize=7.5)
            grid_axis(ax[1], "x"); despine(ax[1], left=True)
            label(ax[1], "Samples", "", f"Geographic distribution ({len(cc)} countries)")

        if "median_reads" in cb.columns:
            v = cb["median_reads"].dropna()
            ax[2].hist(v / 1e6, bins=28, color=C["gold"])
            ax[2].axvline(1.0, ls="--", lw=1.3, color=C["red"])
            grid_axis(ax[2], "y")
            label(ax[2], "Median sequencing depth per study (M reads)", "Studies",
                  "Depth after quality control; dashed line, inclusion floor")

        if "n_subjects" in cb.columns:
            ax[3].grid(True)
            ax[3].scatter(cb["n_subjects"], cb["n_samples"], s=34, alpha=0.75,
                          color=C["blue"], edgecolors="white", linewidths=0.7)
            lim = [1, max(cb["n_samples"].max(), cb["n_subjects"].max()) * 1.3]
            ax[3].plot(lim, lim, ls="--", lw=1.2, color=GREY)
            ax[3].set_xscale("log"); ax[3].set_yscale("log")
            # label the two extremes only; four labels collide in the corner
            for k2, (_, r) in enumerate(cb.nlargest(2, "n_samples").iterrows()):
                ax[3].annotate(short_study(r["study"]), (r["n_subjects"], r["n_samples"]),
                               textcoords="offset points",
                               xytext=(8, 5) if k2 == 0 else (8, -12), fontsize=7)
            label(ax[3], "Subjects per study (log)", "Samples per study (log)",
                  "Studies above the line sampled subjects repeatedly")
        panel(fig, ax)
        save(fig, out, "SupplementaryFigure1_cohort_landscape")

    # ---- S2: per-study held-out performance ------------------------------
    hb = rd(T, "holdout_by_study_adult.csv")
    if hb is not None and len(hb):
        d = hb[hb["model"] != "mean"].dropna(subset=["median_spearman"])
        if len(d):
            fig, ax = plt.subplots(1, 2, figsize=(11.6, 4.6))
            fig.subplots_adjust(wspace=0.34)
            piv = d.pivot_table(index="study", columns="model",
                                values="median_spearman")
            order = piv.mean(axis=1).sort_values().index
            piv = piv.loc[order]
            y = np.arange(len(piv))
            for i, m in enumerate(piv.columns):
                ax[0].scatter(piv[m], y, s=46, color=SERIES[i % len(SERIES)],
                              label=m, alpha=0.85, edgecolors="white", linewidths=0.7)
            ax[0].set_yticks(y)
            ax[0].set_yticklabels([short_study(x) for x in piv.index], fontsize=7.5)
            ax[0].legend(ncol=2, fontsize=7)
            grid_axis(ax[0], "x"); despine(ax[0], left=True)
            label(ax[0], r"Median Spearman $\rho$", "",
                  "Held-out performance varies markedly between studies")

            r2 = d.pivot_table(index="study", columns="model",
                               values="mean_r2_centered").loc[order]
            for i, m in enumerate(r2.columns):
                ax[1].scatter(r2[m], np.arange(len(r2)), s=46,
                              color=SERIES[i % len(SERIES)], alpha=0.85,
                              edgecolors="white", linewidths=0.7)
            ax[1].axvline(0, lw=1.1, color=INK)
            ax[1].set_yticks(np.arange(len(r2)))
            ax[1].set_yticklabels([short_study(x) for x in r2.index], fontsize=7.5)
            grid_axis(ax[1], "x"); despine(ax[1], left=True)
            label(ax[1], r"Mean centred $R^2$", "",
                  "Calibration by study; below zero is worse than the cohort mean")
            panel(fig, ax)
            save(fig, out, "SupplementaryFigure2_holdout_by_study")

    # ---- attribution (exploratory, not part of the submitted set) --------
    # Removed from the paper because it derives from an earlier training run.
    # Still produced on request, into a separate directory, so the numbering of
    # the submitted supplementary figures matches the manuscript exactly.
    av = rd(T, "attribution_validation.csv") if exploratory else None
    ua = rd(T, "unexplained_associations.csv")
    if av is not None:
        fig, ax = plt.subplots(1, 2, figsize=(11.6, 4.6))
        fig.subplots_adjust(wspace=0.34)
        d = av.sort_values("n")
        bars = ax[0].barh(np.arange(len(d)), d["n"], height=0.62,
                          color=[C["green"] if c == "confirmed" else
                                 (C["red"] if c == "unexplained" else C["grey"])
                                 for c in d["category"]])
        for b, v in zip(bars, d["n"]):
            ax[0].annotate(f"{int(v):,}", (v, b.get_y() + b.get_height() / 2),
                           textcoords="offset points", xytext=(7, -3), fontsize=8)
        ax[0].set_yticks(np.arange(len(d)))
        ax[0].set_yticklabels([c.capitalize() for c in d["category"]])
        ax[0].set_xscale("log")
        grid_axis(ax[0], "x"); despine(ax[0], left=True)
        label(ax[0], "Species–gene pairs (log scale)", "",
              "Attributions against pangenome evidence")

        if ua is not None and "attribution" in ua.columns:
            t = ua.nlargest(15, "attribution").sort_values("attribution")
            ax[1].barh(np.arange(len(t)), t["attribution"], height=0.62,
                       color=C["red"])
            ax[1].set_yticks(np.arange(len(t)))
            ax[1].set_yticklabels([f"{s.replace('_',' ')} – {g}"
                                   for s, g in zip(t["species"], t["gene"])],
                                  fontsize=6.8)
            grid_axis(ax[1], "x"); despine(ax[1], left=True)
            label(ax[1], "Attribution", "",
                  "Strongest attributions without genomic support")
        panel(fig, ax)
        save(fig, out / "exploratory", "SupplementaryFigure_attribution_exploratory")

    # ---- S4: exposure-boundary detail ------------------------------------
    eb = rd(T, "exposure_boundary_summary_all.csv")
    if eb is not None and "intervention" in eb.columns:
        d = eb.dropna(subset=["intervention"])
        if len(d):
            fig, ax = plt.subplots(1, 2, figsize=(11.6, 4.6))
            fig.subplots_adjust(wspace=0.34)
            dd = d.sort_values("spearman")
            ax[0].barh(np.arange(len(dd)), dd["spearman"], height=0.6, color=C["blue"])
            for i2, (v, n2) in enumerate(zip(dd["spearman"], dd["n_subjects"])):
                ax[0].annotate(f"{v:.2f}  (n = {int(n2)})", (v, i2),
                               textcoords="offset points", xytext=(7, -3), fontsize=7.5)
            ax[0].set_yticks(np.arange(len(dd)))
            ax[0].set_yticklabels([str(i).capitalize() for i in dd["intervention"]])
            ax[0].set_xlim(0, dd["spearman"].max() * 1.45)
            grid_axis(ax[0], "x"); despine(ax[0], left=True)
            label(ax[0], r"Coupling across boundary, Spearman $\rho$", "",
                  "Intervention type assigned per sample")

            cp = eb[eb["comparison"].astype(str).str.startswith("coupling")]
            if len(cp):
                ax[1].bar(np.arange(len(cp)), cp["median_crossing"], width=0.55,
                          color=C["blue"])
                for i2, v in enumerate(cp["median_crossing"]):
                    ax[1].annotate(f"{v:.3f}", (i2, v), textcoords="offset points",
                                   xytext=(0, 6), ha="center", fontsize=8)
                ax[1].set_xticks(np.arange(len(cp)))
                ax[1].set_xticklabels([str(c).replace("coupling, ", "")
                                       .replace(" pairs", "") for c in cp["comparison"]])
                grid_axis(ax[1], "y"); despine(ax[1], bottom=True)
                label(ax[1], "", r"Coupling, Spearman $\rho$",
                      "Coupling is similar regardless of exposure boundary")
            panel(fig, ax)
            save(fig, out, "SupplementaryFigure3_exposure_detail")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", required=True)
    ap.add_argument("--habitats", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--work", default=None,
                    help="pipeline work directory; enables the workflow figure "
                         "to read join counts and matrix shapes")
    ap.add_argument("--exploratory", action="store_true",
                    help="also build exploratory figures excluded from the "
                         "paper, into an 'exploratory' subdirectory")
    ap.add_argument("--supplementary", action="store_true",
                    help="also produce the supplementary figure set")
    ap.add_argument("--font", default="serif", choices=("serif", "sans"),
                    help="Nature-family journals specify sans-serif for figures")
    a = ap.parse_args()

    T, H, O = Path(a.tables), Path(a.habitats), Path(a.outdir)
    style(a.font)
    print(f"tables   : {T}\nhabitats : {H}\noutput   : {O}\n")
    W = Path(a.work) if a.work else None
    figure_workflow(T, H, O, W)
    figure1(T, H, O, W)
    figure2(T, O)
    figure3(T, O)
    figure4(T, H, O)
    figure5(T, H, O)
    if a.supplementary:
        print("\nsupplementary:")
        supplementary(T, H, O, W, a.exploratory)
    print("\nAll panels use the adult stratum except the longitudinal, exposure "
          "and habitat analyses, which are stated as combined or per-habitat in "
          "their legends. Cross-habitat variance appears only in Figure 1.")


if __name__ == "__main__":
    main()
