"""
s21_workflow_figure.py
======================
The study workflow, as Figure 1.

Every count in the diagram is read from the pipeline's own outputs rather than
typed in, so the figure cannot drift out of step with the analysis. If a filter
threshold changes and the pipeline is re-run, the figure changes with it.

Layout: two data sources at the top, the join and filtering cascade down the
middle with sample counts at each step, the two derived matrices, and the
analysis branches below.

Outputs:
    figures/fig1_workflow.png / .pdf / .svg

Usage:  python src/s21_workflow_figure.py
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import plotstyle as ps
from common import LOG, load_config, work_path


# ------------------------------------------------------------------ data ---

def read_counts(cfg) -> dict:
    """Pull every number in the figure from the pipeline's own outputs."""
    c = {}
    rep = work_path(cfg, "join_report.txt")
    if rep.exists():
        txt = rep.read_text(encoding="utf-8")
        def grab(pat, default=None):
            m = re.search(pat, txt)
            return int(m.group(1).replace(",", "")) if m else default
        c["zenodo_runs"] = grab(r"Zenodo runs\s*:\s*([\d,]+)")
        c["bridged_runs"] = grab(r"Union of both bridges -> runs\s*:\s*([\d,]+)")
        c["bridged_samples"] = grab(r"Union of both bridges -> Metalog samples:\s*([\d,]+)")
        c["metalog_human"] = grab(r"Metalog human samples with metadata\s*:\s*([\d,]+)")
        c["joined"] = grab(r"Joined: resistome \+ curated metadata\s*:\s*([\d,]+)")
        c["faecal"] = grab(r"ENVO:00002003\]\s*:\s*([\d,]+)")
        c["with_status"] = grab(r"With known disease status\s*:\s*([\d,]+)")
        c["final"] = grab(r"FINAL USABLE SAMPLES:\s*([\d,]+)")

    st = work_path(cfg, "sample_table.parquet")
    if st.exists():
        s = pd.read_parquet(st)
        c.setdefault("final", len(s))
        c["studies"] = s["study"].nunique() if "study" in s.columns else None
        c["countries"] = s["country"].nunique() if "country" in s.columns else None
        # The repeat-subject count must match the longitudinal analysis, which
        # namespaces identifiers by study and operates on the ALIGNED dataset.
        # Counting raw subject_id values over the full sample table gives a
        # larger, different number, and the figure would then contradict the
        # Results.
        # The within-subject analysis reported in the manuscript uses the
        # COMBINED stratum, so the figure must quote that count. Reading the
        # current stratum's file instead gives the adult-only number and
        # silently contradicts the Results.
        lp = work_path(cfg, "longitudinal_pairs_all.parquet")
        if not lp.exists():
            lp = work_path(cfg, "longitudinal_pairs.parquet", per_stratum=True)
        if lp.exists():
            c["repeat_subjects"] = int(
                pd.read_parquet(lp)["subject_id"].nunique())
            c["repeat_subjects_source"] = lp.name
            LOG.info("Repeat-subject count read from %s: %d", lp.name,
                     c["repeat_subjects"])
        elif "subject_id" in s.columns and "study" in s.columns:
            # reproduce stage 11's namespacing and alignment if it has not run
            al = work_path(cfg, "taxa_clr.parquet")
            keep = set(pd.read_parquet(al).index) if al.exists() else set(s["sample"])
            t = s[s["sample"].astype(str).isin(keep)].copy()
            uid = t["study"].astype(str) + "::" + t["subject_id"].astype(str)
            c["repeat_subjects"] = int((uid.value_counts() >= 2).sum())
            c["repeat_subjects_source"] = "recomputed"

    for key, f in (("arg", "arg_abundance.parquet"), ("taxa", "taxa_clr.parquet")):
        p = work_path(cfg, f)
        if p.exists():
            d = pd.read_parquet(p)
            c[f"{key}_samples"], c[f"{key}_features"] = d.shape
    al = work_path(cfg, "aligned_samples.parquet")
    if al.exists():
        c["aligned"] = len(pd.read_parquet(al))
    hr = work_path(cfg, "arg_highrisk_carriage.parquet")
    if hr.exists():
        c["highrisk"] = pd.read_parquet(hr).shape[1]
    LOG.info("Counts read from pipeline outputs: %s",
             {k: v for k, v in c.items() if v is not None})
    return c


# ---------------------------------------------------------------- drawing --

def box(ax, x, y, w, h, text, face, edge=None, fontsize=7.4, weight="normal",
        text_colour=None):
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        facecolor=face, edgecolor=edge or face, linewidth=1.0, zorder=2))
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            weight=weight, zorder=3, linespacing=1.45,
            color=text_colour or ps.PALETTE["ink"])


def arrow(ax, x1, y1, x2, y2, colour=None, style="-|>", lw=1.2, rad=0.0):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle=style, mutation_scale=11,
        linewidth=lw, color=colour or ps.PALETTE["neutral"],
        connectionstyle=f"arc3,rad={rad}", zorder=1))


def n(c, k, fmt="{:,}"):
    v = c.get(k)
    return fmt.format(v) if isinstance(v, (int, np.integer)) else "—"


def main() -> None:
    cfg = load_config()
    c = read_counts(cfg)

    ps.apply_style()
    fig, ax = ps.multipanel(2, 1, panel_width=7.4, panel_height=4.6, labels=False)
    ax[1].remove()
    ax = ax[0]
    ax.set_xlim(0, 10); ax.set_ylim(0, 10.6)
    ax.axis("off")

    BLUE, RED, GREEN, GOLD = (ps.PALETTE["primary"], ps.PALETTE["secondary"],
                              ps.PALETTE["tertiary"], ps.PALETTE["accent"])
    LIGHT, GREY = ps.PALETTE["light"], "#EDEFF3"

    # ---- sources ---------------------------------------------------------
    ax.text(5, 10.3, "Public metagenome resources", ha="center",
            fontsize=8.5, weight="bold", color=ps.PALETTE["ink"])
    box(ax, 2.6, 9.4, 4.2, 0.92,
        f"Resistome resource (Zenodo)\n{n(c,'zenodo_runs')} sequencing runs\n"
        "KMA alignment to ResFinder",
        LIGHT, edge=BLUE)
    box(ax, 7.4, 9.4, 4.2, 0.92,
        f"Metalog metadata\n{n(c,'metalog_human')} human samples\n"
        "curated annotation, MetaPhlAn 4",
        LIGHT, edge=BLUE)

    # ---- bridging --------------------------------------------------------
    box(ax, 5, 8.05, 6.4, 0.78,
        f"Bridged by run and by sample accession\n"
        f"{n(c,'bridged_runs')} runs from {n(c,'bridged_samples')} samples",
        GREY, edge=ps.PALETTE["neutral"])
    arrow(ax, 2.6, 8.94, 4.0, 8.44)
    arrow(ax, 7.4, 8.94, 6.0, 8.44)

    # ---- filtering cascade ----------------------------------------------
    steps = [
        (f"joined with curated metadata", n(c, "joined")),
        ("faecal material [ENVO:00002003]", n(c, "faecal")),
        ("recorded health status", n(c, "with_status")),
        (r"$\geq$10$^6$ post-QC reads", n(c, "final")),
    ]
    y0, dy = 7.1, 0.62
    for i, (lab, val) in enumerate(steps):
        y = y0 - i * dy
        box(ax, 5, y, 6.4, 0.46, f"{lab}          {val}", GREY,
            edge=ps.PALETTE["neutral"], fontsize=7.0)
        if i:
            arrow(ax, 5, y + dy - 0.23, 5, y + 0.23, lw=1.0)
    arrow(ax, 5, 7.66, 5, 7.33, lw=1.0)

    box(ax, 5, 4.42, 6.4, 0.62,
        f"Analysed cohort: {n(c,'final')} samples · {n(c,'studies')} studies · "
        f"{n(c,'countries')} countries",
        BLUE, fontsize=7.8, weight="bold", text_colour="white")
    arrow(ax, 5, 4.87 - 0.22, 5, 4.73, lw=1.2)

    # ---- matrices --------------------------------------------------------
    box(ax, 2.5, 3.3, 4.0, 0.86,
        f"Resistome matrix\n{n(c,'arg_samples')} × {n(c,'arg_features')} gene families\n"
        "length- and rRNA-normalised",
        LIGHT, edge=RED)
    box(ax, 7.5, 3.3, 4.0, 0.86,
        f"Taxonomic matrix\n{n(c,'taxa_samples')} × {n(c,'taxa_features')} species\n"
        "centred log-ratio",
        LIGHT, edge=GREEN)
    arrow(ax, 4.2, 4.11, 2.9, 3.73, colour=RED)
    arrow(ax, 5.8, 4.11, 7.1, 3.73, colour=GREEN)

    box(ax, 5, 2.28, 3.5, 0.42,
        f"aligned: {n(c,'aligned')} samples", GREY,
        edge=ps.PALETTE["neutral"], fontsize=7.0)
    arrow(ax, 2.9, 2.87, 4.2, 2.49, colour=RED, rad=-0.12)
    arrow(ax, 7.1, 2.87, 5.8, 2.49, colour=GREEN, rad=0.12)

    # ---- analyses --------------------------------------------------------
    # placed at the margin: a centred heading here sits on top of the fan of
    # arrows leaving the aligned-samples box
    ax.text(0.15, 1.28, "Analyses", ha="left", va="center", fontsize=8.2,
            weight="bold", color=ps.PALETTE["ink"], rotation=90)
    panels = [
        ("Variance\npartitioning", "taxonomy vs study,\ngeography, host", GOLD),
        ("Cross-cohort\nprediction", "leave-one-study-out\n+ held-out studies", GOLD),
        ("Within-subject\n& intervention",
         f"{n(c,'repeat_subjects')} repeat subjects", GOLD),
        ("Genomic\nconservation", "within-species conservation\nvs predictability", RED),
    ]
    xs = np.linspace(1.75, 8.55, len(panels))
    for x, (title, sub, col) in zip(xs, panels):
        box(ax, x, 0.92, 2.05, 1.02, f"{title}\n\n{sub}", "white", edge=col,
            fontsize=6.8)
        arrow(ax, 5, 2.07, x, 1.47, colour=ps.PALETTE["neutral"], lw=0.9,
              rad=0.10 if x < 5 else -0.10)

    written = ps.save(fig, cfg["paths"]["fig_dir"], "fig1_workflow",
                      cfg["figures"]["formats"], cfg["figures"]["dpi"])
    LOG.info("fig1 workflow -> %s", written[0])
    LOG.info("Every count is read from the pipeline outputs, so the figure "
             "cannot drift out of step with the analysis. The existing cohort "
             "figure becomes Figure 2 in the manuscript.")


if __name__ == "__main__":
    main()
