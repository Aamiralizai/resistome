"""
s16_exposure_boundary.py
========================
Each subject as their own control, across a recorded exposure boundary.

Metalog records `intervention` per SAMPLE, not per subject, and inspection
shows most subjects in interventional studies have both annotated and
unannotated samples: 35 of 35 in Zmora 2018, 18 of 24 in Raymond 2016, 13 of
21 in Vaughn 2016. The field therefore marks WHEN a subject was exposed, not
WHETHER. That makes a treated-versus-control comparison impossible, and a
within-subject before/after comparison available - which is the better design
anyway, because between-person confounding is removed by construction rather
than by adjustment.

The analysis
------------
For each subject, consecutive sample pairs are split into two kinds:

    CROSSING  an exposure boundary: the earlier sample carries no intervention
              annotation and the later one does
    INTERNAL  no boundary: both samples are annotated the same way

Both are within the same person, over comparable intervals, in the same study
and sequencing run. The internal pairs are the control: they carry every
confounder that crossing pairs carry, except the exposure.

Three questions follow:

1. Do crossing pairs show larger compositional and resistome displacement than
   internal pairs from the same subjects?
2. Is compositional and resistome change more tightly coupled across an
   exposure boundary than within an unexposed interval?
3. Does this differ by intervention type - antibiotic, transplant, probiotic,
   diet - now that types are assigned per sample rather than per study?

What this does not establish: interventions act on resistance genes directly
as well as through the community, and elapsed time is not randomised with
respect to exposure. The design removes stable between-person confounding, not
time-varying confounding.

Outputs:
    tables/exposure_boundary_pairs_<stratum>.csv
    tables/exposure_boundary_summary_<stratum>.csv
    figures/fig9_exposure_<stratum>.png / .pdf / .svg

Usage:  python src/s16_exposure_boundary.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

import plotstyle as ps
from common import (LOG, apply_stratum, harmonise_disease, load_config,
                    namespace_subjects, table_path, tag, work_path)
from s12_perturbation import classify_samples

MIN_PAIRS = 30


def build_boundary_pairs(st, taxa, arg, kind) -> pd.DataFrame:
    """Consecutive within-subject pairs, labelled by exposure boundary."""
    rows = []
    for sid, g in st.groupby("subject_id"):
        if "t" in g.columns and g["t"].notna().all():
            g = g.sort_values("t")
        ids = [s for s in g["sample"] if s in taxa.index and s in arg.index]
        if len(ids) < 2:
            continue
        tv = dict(zip(g["sample"], g.get("t", pd.Series(index=g.index, dtype=float))))
        study = g["study"].iloc[0] if "study" in g.columns else "NA"
        for a, b in zip(ids[:-1], ids[1:]):
            ka, kb = kind.get(a), kind.get(b)
            exposed_a = isinstance(ka, str)
            exposed_b = isinstance(kb, str)
            if not exposed_a and exposed_b:
                label, itype = "crossing", kb
            elif exposed_a == exposed_b:
                label, itype = "internal", (kb if exposed_b else "none")
            else:
                label, itype = "recovery", ka      # exposed -> unexposed
            gap = (abs(tv.get(b, np.nan) - tv.get(a, np.nan))
                   if pd.notna(tv.get(a)) and pd.notna(tv.get(b)) else np.nan)
            rows.append({
                "subject_id": sid, "study": study, "s1": a, "s2": b,
                "pair_type": label, "intervention": itype, "time_gap": gap,
                "d_taxa": float(np.linalg.norm(taxa.loc[a].values - taxa.loc[b].values)),
                "d_arg": float(np.linalg.norm(arg.loc[a].values - arg.loc[b].values)),
            })
    return pd.DataFrame(rows)


def main() -> None:
    cfg = load_config()
    st = namespace_subjects(harmonise_disease(apply_stratum(cfg, pd.read_parquet(
        work_path(cfg, "sample_table.parquet")))))
    taxa = pd.read_parquet(work_path(cfg, "taxa_clr.parquet"))
    arg = pd.read_parquet(work_path(cfg, "arg_abundance.parquet"))
    shared = sorted(set(taxa.index) & set(arg.index) & set(st["sample"].astype(str)))
    taxa, arg = taxa.loc[shared], arg.loc[shared]
    st = st[st["sample"].isin(shared)].copy()
    st["t"] = pd.to_numeric(st.get("timepoint"), errors="coerce")

    kind = dict(zip(st["sample"], classify_samples(st)))
    n_annot = sum(isinstance(v, str) for v in kind.values())
    LOG.info("Samples with a per-sample intervention annotation: %d / %d",
             n_annot, len(st))
    if n_annot < 100:
        raise SystemExit("Too few annotated samples for a boundary analysis.")

    pairs = build_boundary_pairs(st, taxa, arg, kind)
    counts = pairs["pair_type"].value_counts().to_dict()
    LOG.info("Consecutive pairs: %s", counts)
    LOG.info("Subjects contributing: %d across %d studies",
             pairs["subject_id"].nunique(), pairs["study"].nunique())

    cross = pairs[pairs["pair_type"] == "crossing"]
    intern = pairs[pairs["pair_type"] == "internal"]
    if len(cross) < MIN_PAIRS or len(intern) < MIN_PAIRS:
        raise SystemExit(f"Too few pairs (crossing {len(cross)}, "
                         f"internal {len(intern)}) for a comparison.")

    # ---- restrict the control set to subjects who also cross a boundary ---
    # so the comparison is within the same people, not between exposed and
    # never-exposed individuals.
    shared_subj = set(cross["subject_id"]) & set(intern["subject_id"])
    LOG.info("Subjects contributing BOTH crossing and internal pairs: %d",
             len(shared_subj))
    cw = cross[cross["subject_id"].isin(shared_subj)]
    iw = intern[intern["subject_id"].isin(shared_subj)]

    rows = []
    LOG.info("=" * 66)
    LOG.info("1. DISPLACEMENT ACROSS AN EXPOSURE BOUNDARY vs WITHIN")
    for layer in ("d_taxa", "d_arg"):
        if len(cw) >= 10 and len(iw) >= 10:
            u = mannwhitneyu(cw[layer], iw[layer], alternative="two-sided")
            LOG.info("  %-7s crossing median=%.2f  internal median=%.2f  P=%.3g",
                     layer, float(cw[layer].median()), float(iw[layer].median()),
                     u.pvalue)
            rows.append({"comparison": f"{layer}: crossing vs internal",
                         "n_crossing": len(cw), "n_internal": len(iw),
                         "median_crossing": float(cw[layer].median()),
                         "median_internal": float(iw[layer].median()),
                         "p_value": float(u.pvalue)})

    LOG.info("2. COUPLING OF COMPOSITIONAL AND RESISTOME CHANGE")
    for name, g in (("crossing", cross), ("internal", intern),
                    ("recovery", pairs[pairs["pair_type"] == "recovery"])):
        if len(g) >= MIN_PAIRS:
            r = spearmanr(g["d_taxa"], g["d_arg"])
            LOG.info("  %-9s n=%4d  rho=%.3f (P=%.3g)", name, len(g),
                     r.statistic, r.pvalue)
            rows.append({"comparison": f"coupling, {name} pairs",
                         "n_crossing": len(g), "n_internal": np.nan,
                         "median_crossing": r.statistic, "median_internal": np.nan,
                         "p_value": float(r.pvalue)})

    LOG.info("3. BY INTERVENTION TYPE (assigned per sample)")
    by_type = []
    for itype, g in cross.groupby("intervention"):
        if len(g) < 20:
            continue
        r = spearmanr(g["d_taxa"], g["d_arg"])
        LOG.info("  %-12s n=%4d subjects=%3d  rho=%.3f  median d_arg=%.2f",
                 itype, len(g), g["subject_id"].nunique(), r.statistic,
                 float(g["d_arg"].median()))
        by_type.append({"intervention": itype, "n_pairs": len(g),
                        "n_subjects": g["subject_id"].nunique(),
                        "spearman": r.statistic, "p_value": r.pvalue,
                        "median_d_taxa": float(g["d_taxa"].median()),
                        "median_d_arg": float(g["d_arg"].median())})
    LOG.info("=" * 66)

    # time gap is the obvious confounder: exposure intervals may be longer
    if pairs["time_gap"].notna().sum() > 50:
        gc, gi = cw["time_gap"].dropna(), iw["time_gap"].dropna()
        if len(gc) > 10 and len(gi) > 10:
            LOG.info("Median elapsed time: crossing %.1f, internal %.1f. If these "
                     "differ substantially the displacement comparison is partly "
                     "a time effect and must be reported with that caveat.",
                     float(gc.median()), float(gi.median()))

    pairs.to_csv(table_path(cfg, "exposure_boundary_pairs.csv"), index=False)
    summary = pd.DataFrame(rows)
    if by_type:
        summary = pd.concat([summary, pd.DataFrame(by_type)], ignore_index=True)
    summary.to_csv(table_path(cfg, "exposure_boundary_summary.csv"), index=False)
    print("\n" + summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # ---- figure ----------------------------------------------------------
    ps.apply_style()
    fig, ax = ps.multipanel(2, 2, panel_height=2.9)

    from s13_mobility_predictability import _boxplot
    for k, layer, lab in ((0, "d_taxa", "Compositional displacement"),
                          (1, "d_arg", "Resistome displacement")):
        bp = _boxplot(ax[k], [iw[layer].values, cw[layer].values],
                      ["within\nexposure state", "across\nexposure boundary"],
                      showfliers=False, patch_artist=True, widths=0.55)
        for b, c in zip(bp["boxes"], [ps.PALETTE["neutral"], ps.PALETTE["secondary"]]):
            b.set(facecolor=c, alpha=0.45, linewidth=0)
        for m in bp["medians"]:
            m.set(color=ps.PALETTE["ink"], linewidth=1.6)
        ps.grid_axis(ax[k], "y"); ps.despine(ax[k], bottom=True)
        ax[k].tick_params(axis="x", length=0)
        ps.label_axes(ax[k], "Pair type", lab)

    ps.no_grid(ax[2])
    for g, c, lab in ((intern, ps.PALETTE["neutral"], "internal"),
                      (cross, ps.PALETTE["secondary"], "crossing")):
        if len(g):
            ax[2].scatter(g["d_taxa"], g["d_arg"], s=12, alpha=0.45, color=c,
                          edgecolors="none", label=lab)
    ax[2].legend(fontsize=7)
    ps.label_axes(ax[2], "Compositional displacement", "Resistome displacement")

    if by_type:
        bt = pd.DataFrame(by_type).sort_values("spearman")
        bars = ax[3].barh(np.arange(len(bt)), bt["spearman"], height=0.6,
                          color=ps.PALETTE["primary"], linewidth=0)
        ps.bar_values(ax[3], bars, "{:.2f}", horizontal=True)
        ax[3].set_yticks(np.arange(len(bt)))
        ax[3].set_yticklabels([f"{r.intervention} (n={r.n_subjects})"
                               for r in bt.itertuples()], fontsize=7)
        ps.grid_axis(ax[3], "x"); ps.despine(ax[3], left=True)
        ax[3].tick_params(axis="y", length=0)
        ps.label_axes(ax[3], r"Coupling $\rho$ across boundary", "Intervention")
    else:
        ax[3].axis("off")

    written = ps.save(fig, cfg["paths"]["fig_dir"], f"fig9_exposure{tag(cfg)}",
                      cfg["figures"]["formats"], cfg["figures"]["dpi"])
    LOG.info("fig9 -> %s", written[0])
    LOG.info("Caveat: interventions act on resistance genes directly as well as "
             "through the community, and elapsed time is not randomised with "
             "respect to exposure. This design removes stable between-person "
             "confounding, not time-varying confounding.")


if __name__ == "__main__":
    main()
