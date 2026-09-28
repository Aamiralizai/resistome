"""
compare_habitats.py
===================
Assembles the per-habitat results into a comparison, and guards against the
ways such a comparison can mislead.

Standalone: reads only the outputs written by habitat_analysis.py, plus the
human gut result if its directory is supplied. Nothing in the gut pipeline is
read from or written to.

Two cautions built in
---------------------
Adjusted R2 depends on sample size and on the number of terms in each block,
and habitats differ greatly in both. The comparison therefore reports the
estimate against habitat sample size, so a difference that is really a power
effect is visible rather than hidden.

Only blocks present in every habitat are compared. A block measured in one
habitat and absent elsewhere is dropped rather than shown as zero.

Usage:
    python habitat/compare_habitats.py \\
        --habitat-dirs /mnt/x/w1_resistome/habitats/* \\
        --gut-tables /mnt/x/w1_resistome/tables \\
        --gut-work /mnt/x/w1_resistome/work \\
        --outdir /mnt/x/w1_resistome/habitats
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))
import plotstyle as ps          # noqa: E402
from common import LOG          # noqa: E402


def load_habitat(d: Path) -> dict | None:
    vp = d / "variance_partition.csv"
    if not vp.exists():
        return None
    out = {"habitat": d.name, "variance": pd.read_csv(vp)}
    mp = d / "model_performance.csv"
    if mp.exists():
        out["models"] = pd.read_csv(mp)
    st = d / "sample_table.parquet"
    if st.exists():
        s = pd.read_parquet(st)
        out["n_samples"] = len(s)
        out["n_studies"] = s["study"].nunique() if "study" in s.columns else np.nan
    return out


def load_gut(tables: Path, work: Path, stratum: str = "adult") -> dict | None:
    """The human gut result, from the main pipeline's own outputs."""
    cand = [tables / "variance_partition_adult.csv",
            tables / "variance_partition_all.csv"]
    vp = next((c for c in cand if c.exists()), None)
    if vp is None:
        return None
    out = {"habitat": "human gut", "variance": pd.read_csv(vp)}
    ms = tables / "model_summary_adult.csv"
    if ms.exists():
        m = pd.read_csv(ms)
        m = m[m["scheme"].str.contains("study", na=False)] if "scheme" in m else m
        out["models"] = m.rename(columns={"median_spearman": "median_spearman"})
    st = work / "sample_table.parquet"
    if st.exists():
        s = pd.read_parquet(st)
        # The effect estimate comes from the adult analysis, so the sample
        # size plotted beside it must be the adult count. Pairing an adult
        # estimate with the full cohort size misplaces the point on any
        # sample-size axis and distorts the power check.
        if stratum and stratum != "all" and "age_category" in s.columns:
            s = s[s["age_category"].astype(str) == stratum]
        # Intersect with the samples that actually carry BOTH layers. The
        # variance estimate is computed on the aligned matrix, so quoting the
        # pre-alignment stratum count beside it overstates the analysis
        # population - 11,028 adults exist, but 8,873 have both resistome and
        # taxonomic data and enter the partition.
        taxa_p = work / "taxa_clr.parquet"
        arg_p = work / "arg_abundance.parquet"
        if taxa_p.exists() and arg_p.exists():
            aligned = (set(pd.read_parquet(taxa_p).index.astype(str))
                       & set(pd.read_parquet(arg_p).index.astype(str)))
            before = len(s)
            s = s[s["sample"].astype(str).isin(aligned)]
            if len(s) != before:
                print(f"    human gut: {len(s):,} of {before:,} {stratum} samples "
                      f"carry both layers; using the aligned count.")
        out["n_samples"] = len(s)
        out["n_studies"] = s["study"].nunique() if "study" in s.columns else np.nan
        out["stratum"] = stratum
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--habitat-dirs", nargs="+", required=True)
    ap.add_argument("--gut-tables", default=None)
    ap.add_argument("--gut-work", default=None)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--gut-stratum", default="adult",
                    help="life stage whose variance estimate is reported; the "
                         "plotted sample size is taken from the same subset")
    args = ap.parse_args()

    habs = [h for h in (load_habitat(Path(d)) for d in args.habitat_dirs) if h]
    if args.gut_tables:
        g = load_gut(Path(args.gut_tables),
                     Path(args.gut_work or args.gut_tables), args.gut_stratum)
        if g:
            habs.insert(0, g)
    if len(habs) < 2:
        raise SystemExit("Need at least two habitats with completed results.")
    LOG.info("Habitats: %s", [h["habitat"] for h in habs])

    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)

    # ---- assemble, keeping only blocks present everywhere -----------------
    rows = []
    for h in habs:
        v = h["variance"]
        cu = ("unique_total_fraction_adj" if "unique_total_fraction_adj" in v.columns
              else "unique_R2_adj")
        cm = ("marginal_R2_adj" if "marginal_R2_adj" in v.columns else "R2_adj")
        for _, r in v.iterrows():
            rows.append({"habitat": h["habitat"], "block": r["block"],
                         "marginal": r[cm], "unique": r[cu],
                         "n_samples": h.get("n_samples"),
                         "n_studies": h.get("n_studies")})
    vc = pd.DataFrame(rows)
    common_blocks = (vc.groupby("block")["habitat"].nunique()
                     == vc["habitat"].nunique())
    keep = list(common_blocks[common_blocks].index)
    dropped = sorted(set(vc["block"]) - set(keep))
    if dropped:
        LOG.info("Blocks not present in every habitat, dropped from the "
                 "comparison rather than shown as zero: %s", dropped)
    vc = vc[vc["block"].isin(keep)]
    vc.to_csv(out / "cross_habitat_variance.csv", index=False)

    wide = vc.pivot_table(index="block", columns="habitat", values="unique")
    order = [h["habitat"] for h in habs if h["habitat"] in wide.columns]
    wide = wide[order]
    print("\nUNIQUE VARIANCE EXPLAINED (share of total, adjusted)\n")
    print(wide.to_string(float_format=lambda v: f"{v:.4f}"))

    cohort = (vc.groupby("habitat")[["n_samples", "n_studies"]].first()
              .reindex(order))
    print("\nCOHORTS\n")
    print(cohort.to_string())

    # ---- the interpretation, with its confound stated --------------------
    if "taxonomy" in wide.index:
        t = wide.loc["taxonomy"].dropna()
        LOG.info("Taxonomy unique variance by habitat: %s",
                 ", ".join(f"{k} {v:.4f}" for k, v in t.items()))
        if len(t) >= 2 and t.min() > 0:
            fold = t.max() / t.min()
            LOG.info("Range %.4f to %.4f (%.1f-fold)", t.min(), t.max(), fold)
            if fold < 2:
                LOG.info("Similar magnitude across habitats - consistent with a "
                         "general property of microbial communities.")
            else:
                LOG.info("Substantial variation across habitats. Check the "
                         "sample-size relationship below before interpreting.")
        n = cohort["n_samples"].reindex(t.index)
        if n.notna().sum() >= 3:
            r = spearmanr(n.values, t.values)
            LOG.info("Taxonomy estimate against habitat sample size: rho=%.2f "
                     "across %d habitats. A strong positive value would mean "
                     "the comparison is partly a power effect and should be "
                     "reported as such.", r.statistic, len(n))

    # ---- models ----------------------------------------------------------
    mrows = []
    for h in habs:
        if "models" not in h:
            continue
        m = h["models"].copy(); m["habitat"] = h["habitat"]
        mrows.append(m)
    mc = pd.concat(mrows, ignore_index=True) if mrows else pd.DataFrame()
    piv = pd.DataFrame()
    if len(mc) and "median_spearman" in mc.columns:
        piv = mc.pivot_table(index="model", columns="habitat",
                             values="median_spearman")
        piv = piv[[c for c in order if c in piv.columns]]
        print("\nCROSS-COHORT PREDICTABILITY (leave-one-study-out)\n")
        print(piv.to_string(float_format=lambda v: f"{v:.3f}"))
        mc.to_csv(out / "cross_habitat_models.csv", index=False)

    # ---- figure ----------------------------------------------------------
    ps.apply_style()
    fig, ax = ps.multipanel(2, 2, panel_height=3.0)
    colours = {h: ps.SERIES[i % len(ps.SERIES)] for i, h in enumerate(order)}
    # Directories are named after the download; the manuscript says "marine".
    PRETTY = {"ocean": "marine"}
    lab = [PRETTY.get(h, h) for h in order]

    blocks = [b for b in ["taxonomy", "study", "geography", "technical"]
              if b in wide.index]
    y = np.arange(len(blocks)); hh = 0.8 / max(len(order), 1)
    for i, hab in enumerate(order):
        ax[0].barh(y + i * hh - 0.4 + hh / 2, wide.loc[blocks, hab].values,
                   height=hh, label=PRETTY.get(hab, hab), color=colours[hab], linewidth=0)
    ax[0].set_yticks(y); ax[0].set_yticklabels(blocks)
    ax[0].invert_yaxis(); ax[0].legend(fontsize=7)
    ps.grid_axis(ax[0], "x"); ps.despine(ax[0], left=True)
    ax[0].tick_params(axis="y", length=0)
    ps.label_axes(ax[0], "Unique variance explained", "Explanatory block")

    if "taxonomy" in wide.index:
        t = wide.loc["taxonomy"].reindex(order)
        bars = ax[1].bar(np.arange(len(order)), t.values, width=0.6,
                         color=[colours[h] for h in order], linewidth=0)
        ps.bar_values(ax[1], bars, "{:.3f}")
        ax[1].set_xticks(np.arange(len(order)))
        ax[1].set_xticklabels(lab, rotation=20, ha="right", fontsize=8)
        ps.grid_axis(ax[1], "y"); ps.despine(ax[1], bottom=True)
        ax[1].tick_params(axis="x", length=0)
        ps.label_axes(ax[1], "Habitat", "Taxonomy unique variance")

        if cohort["n_samples"].notna().any():
            ps.no_grid(ax[2])
            ax[2].scatter(cohort["n_samples"], t.values, s=90,
                          c=[colours[h] for h in order],
                          edgecolors="white", linewidths=1.2, zorder=3)
            for hab in order:
                ax[2].annotate(PRETTY.get(hab, hab), (cohort.loc[hab, "n_samples"], t[hab]),
                               textcoords="offset points", xytext=(7, 4),
                               fontsize=7)
            ax[2].set_xscale("log")
            ps.label_axes(ax[2], "Samples (log scale)",
                          "Taxonomy unique variance")
            ax[2].set_title("Is the difference a power effect?",
                            fontweight="bold", fontsize=8)

    if len(piv):
        x = np.arange(len(piv.columns)); w = 0.8 / max(len(piv.index), 1)
        for i, mdl in enumerate(piv.index):
            ax[3].bar(x + i * w - 0.4 + w / 2, piv.loc[mdl].values, width=w,
                      label=mdl, color=ps.SERIES[i % len(ps.SERIES)], linewidth=0)
        ax[3].set_xticks(x)
        ax[3].set_xticklabels([PRETTY.get(c, c) for c in piv.columns],
                              rotation=20, ha="right", fontsize=8)
        ax[3].legend(fontsize=7)
        ps.grid_axis(ax[3], "y"); ps.despine(ax[3], bottom=True)
        ax[3].tick_params(axis="x", length=0)
        ps.label_axes(ax[3], "Habitat", r"Cross-cohort $\rho$")
    else:
        ax[3].axis("off")

    written = ps.save(fig, out, "fig_cross_habitat", ["png", "pdf", "svg"], 600)
    LOG.info("figure -> %s", written[0])
    LOG.info("Report the sample sizes alongside the estimates: adjusted R2 "
             "depends on both sample size and block dimensionality, and these "
             "habitats differ in both.")


if __name__ == "__main__":
    main()
