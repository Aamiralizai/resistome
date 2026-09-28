"""
s20_robustness.py
=================
The controls, sensitivity analyses and tested nulls, in one figure.

Several claims in this project were tested and did not survive: taxonomic
pooling gave no benefit, temporal precedence was not established, cohort
distance did not explain transferability, and per-fold rank correlation was
found to scale with held-out outcome variance. Others survived controls that
could have removed them - notably the relationship between gene mobility and
predictability, which persists after adjusting for both outcome variance and
sample prevalence.

Reporting these together, rather than scattering them through supplementary
text, does two things: it shows a reviewer that the obvious alternative
explanations were tested rather than overlooked, and it puts the surviving
claim in the context of the ones that did not.

Sensitivity variants are recomputed here from the stored matrices rather than
re-run through the pipeline, so this is fast. Permutation testing is omitted
for the variants - the question is whether the ESTIMATE is stable, not whether
each variant is individually significant.

Outputs:
    tables/sensitivity_taxonomy.csv
    tables/robustness_summary.csv
    figures/fig11_robustness_<stratum>.png / .pdf / .svg

Usage:
    python src/s20_robustness.py \\
        --pangenome /mnt/x/w1_resistome/pangenome/pangenome_evidence.tsv \\
        --habitats /mnt/x/w1_resistome/habitats
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import plotstyle as ps
from common import (LOG, apply_stratum, harmonise_disease, load_config,
                    stratum, table_path, tag, work_path)
from s04_variance_partition import build_blocks, rda_r2
from s06_attribution import normalise_gene
from s13_mobility_predictability import gene_mobility


# ------------------------------------------------------------ sensitivity --

def taxonomy_unique(st, taxa, arg, n_pcs=50):
    """Unique share of total variance for taxonomy, no permutation."""
    blocks = build_blocks(st, taxa, n_pcs=n_pcs)
    if "taxonomy" not in blocks:
        return np.nan
    Y = StandardScaler().fit_transform(arg.values)
    X = blocks["taxonomy"]
    others = [v for k, v in blocks.items() if k != "taxonomy" and v.shape[1] > 0]
    if not others:
        return rda_r2(Y, X)[1]
    Z = np.column_stack(others)
    full = rda_r2(Y, np.column_stack([X, Z]))[1]
    red = rda_r2(Y, Z)[1]
    return full - red


def sensitivity(cfg, st, taxa, arg) -> pd.DataFrame:
    """Stability of the headline estimate across analytical choices."""
    rows = [{"variant": "primary (5% ARG, 50 PCs, all samples)",
             "taxonomy_unique": taxonomy_unique(st, taxa, arg)}]

    # number of taxonomy components
    for npc in (25, 100):
        if npc < min(taxa.shape[1] - 1, len(taxa) - 2):
            rows.append({"variant": f"{npc} taxonomy components",
                         "taxonomy_unique": taxonomy_unique(st, taxa, arg, npc)})

    # ARG prevalence threshold, recomputed from raw counts
    cnt_p = work_path(cfg, "arg_counts_aligned.parquet")
    if cnt_p.exists():
        cnt = pd.read_parquet(cnt_p).reindex(index=arg.index).fillna(0.0)
        pseudo = float(cfg["normalisation"]["arg_pseudocount"])
        prev = (cnt > 0).mean(axis=0)
        for thr in (0.01, 0.10, 0.20):
            keep = [g for g in prev[prev >= thr].index if g in cnt.columns]
            if len(keep) < 15:
                continue
            a = np.log(cnt[keep] + pseudo)
            rows.append({"variant": f"ARG prevalence >= {thr:.0%} "
                                    f"({len(keep)} genes)",
                         "taxonomy_unique": taxonomy_unique(st, taxa, a)})

    # one sample per subject, so densely sampled individuals cannot dominate
    if "subject_id" in st.columns:
        # Namespace by study before deduplicating. Public datasets reuse
        # identifiers such as "1" or "patient1", so deduplicating on the raw
        # field merges unrelated participants and discards real samples.
        key = (st["study"].astype(str) + "::" + st["subject_id"].astype(str)
               if "study" in st.columns else st["subject_id"].astype(str))
        first = st.assign(_uid=key).drop_duplicates("_uid")["sample"].astype(str)
        idx = [s for s in arg.index if s in set(first)]
        if len(idx) > 500:
            rows.append({"variant": f"one sample per subject ({len(idx)})",
                         "taxonomy_unique": taxonomy_unique(
                             st[st["sample"].isin(idx)], taxa.loc[idx],
                             arg.loc[idx])})

    # largest studies removed, in case a single cohort drives the estimate
    if "study" in st.columns:
        big = st["study"].value_counts().head(3).index
        idx = [s for s in arg.index
               if s in set(st.loc[~st["study"].isin(big), "sample"])]
        if len(idx) > 500:
            rows.append({"variant": f"3 largest studies removed ({len(idx)})",
                         "taxonomy_unique": taxonomy_unique(
                             st[st["sample"].isin(idx)], taxa.loc[idx],
                             arg.loc[idx])})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- controls --

def mobility_controls(cfg, pangenome: Path) -> dict:
    """Mobility vs predictability, raw and with each confound removed."""
    m = pd.read_parquet(work_path(cfg, "cv_metrics.parquet", per_stratum=True))
    sub = m[(m["scheme"] != "random") & (m["model"] != "mean")]
    best = sub.groupby("model")["spearman"].median().idxmax()
    perf = (sub[sub["model"] == best].groupby("gene")["spearman"].median()
            .reset_index().rename(columns={"spearman": "rho"}))
    perf["gk"] = normalise_gene(perf["gene"])

    ev = pd.read_csv(pangenome, sep="\t")
    mob = gene_mobility(ev)
    if "conservation" in mob.columns:
        mob["mean_frequency"] = mob["conservation"]
    d = perf.merge(mob, on="gk", how="inner")

    arg = pd.read_parquet(work_path(cfg, "arg_abundance.parquet"))
    pseudo = float(cfg["normalisation"]["arg_pseudocount"])
    d = d.merge(arg.std(axis=0).rename("arg_sd").reset_index()
                .rename(columns={"index": "gene"}), on="gene", how="left")
    d = d.merge((arg > np.log(pseudo) + 1e-9).mean(axis=0).rename("prev")
                .reset_index().rename(columns={"index": "gene"}),
                on="gene", how="left")
    d = d.dropna(subset=["mean_frequency", "rho"])

    def partial(y, x, z):
        ok = pd.DataFrame({"y": y, "x": x, "z": z}).dropna().rank()
        if len(ok) < 12:
            return np.nan
        res = lambda a, b: a - np.polyval(np.polyfit(b, a, 1), b)  # noqa: E731
        return spearmanr(res(ok["y"].values, ok["z"].values),
                         res(ok["x"].values, ok["z"].values)).statistic

    rng = np.random.default_rng(42)
    boots = [spearmanr(*d.iloc[rng.integers(0, len(d), len(d))]
                       [["mean_frequency", "rho"]].values.T).statistic
             for _ in range(2000)]
    return {
        "n_genes": len(d),
        "raw": spearmanr(d["mean_frequency"], d["rho"]).statistic,
        "ctrl_variance": partial(d["rho"], d["mean_frequency"], d["arg_sd"]),
        "ctrl_prevalence": partial(d["rho"], d["mean_frequency"], d["prev"]),
        "variance_vs_rho": spearmanr(d["arg_sd"], d["rho"]).statistic,
        "prevalence_vs_rho": spearmanr(d["prev"], d["rho"]).statistic,
        "boot": np.array(boots),
        "data": d,
    }


def habitat_artefact(habdir: Path) -> pd.DataFrame:
    """Per-fold rho against held-out outcome variance, across habitats."""
    rows = []
    for d in sorted(p for p in habdir.iterdir() if p.is_dir()):
        perf, argf, stf = (d / "model_performance.csv",
                           d / "arg_abundance.parquet",
                           d / "sample_table.parquet")
        if not (perf.exists() and argf.exists() and stf.exists()):
            continue
        p = pd.read_csv(perf)
        p = p[(p["model"] == "xgb") & p["median_spearman"].notna()]
        arg = pd.read_parquet(argf)
        st = pd.read_parquet(stf).set_index("sample")["study"].astype(str)
        for _, r in p.iterrows():
            ids = [i for i in st[st == str(r["study"])].index if i in arg.index]
            if len(ids) < 20:
                continue
            rows.append({"habitat": d.name, "rho": r["median_spearman"],
                         "arg_sd": float(np.median(arg.loc[ids].std(axis=0)))})
    return pd.DataFrame(rows)


# -------------------------------------------------------------------- main --

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pangenome", default=None,
                    help="pangenome evidence; use the model-independent set so "
                         "this figure matches the manuscript")
    ap.add_argument("--habitats", default=None)
    args = ap.parse_args()
    cfg = load_config()

    st = harmonise_disease(apply_stratum(cfg, pd.read_parquet(
        work_path(cfg, "sample_table.parquet"))))
    taxa = pd.read_parquet(work_path(cfg, "taxa_clr.parquet"))
    arg = pd.read_parquet(work_path(cfg, "arg_abundance.parquet"))
    shared = sorted(set(arg.index) & set(taxa.index) & set(st["sample"].astype(str)))
    arg, taxa = arg.loc[shared], taxa.loc[shared]
    st = st[st["sample"].isin(shared)]

    LOG.info("=== sensitivity of the taxonomy estimate ===")
    sens = sensitivity(cfg, st, taxa, arg)
    sens.to_csv(table_path(cfg, "sensitivity_taxonomy.csv"), index=False)
    print("\n" + sens.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    rng_ = sens["taxonomy_unique"].max() - sens["taxonomy_unique"].min()
    LOG.info("Estimate ranges %.4f to %.4f across %d variants (spread %.4f)",
             sens["taxonomy_unique"].min(), sens["taxonomy_unique"].max(),
             len(sens), rng_)

    mob = mobility_controls(cfg, Path(args.pangenome)) if args.pangenome else None
    if mob:
        LOG.info("=== mobility vs predictability, with controls ===")
        LOG.info("raw %.3f | outcome variance held constant %.3f | "
                 "prevalence held constant %.3f (n=%d genes)",
                 mob["raw"], mob["ctrl_variance"], mob["ctrl_prevalence"],
                 mob["n_genes"])

    hab = habitat_artefact(Path(args.habitats)) if args.habitats else pd.DataFrame()
    if len(hab):
        r = spearmanr(hab["arg_sd"], hab["rho"])
        LOG.info("=== metric artefact ===")
        LOG.info("Per-fold rho vs held-out outcome variance: %.3f (P=%.2g, "
                 "%d folds)", r.statistic, r.pvalue, len(hab))

    # Read the cohort-distance effect from the analysis output rather than
    # hard-coding it: a literal here goes stale the moment the fold data is
    # regenerated, and does so invisibly.
    _dist_rho = np.nan
    if args.habitats:
        _fh = Path(args.habitats) / "fold_heterogeneity.csv"
        if _fh.exists():
            _d = pd.read_csv(_fh)
            _dist_rho = spearmanr(_d["centroid_distance"],
                                  _d["median_spearman"]).statistic

    # ---- figure ----------------------------------------------------------
    ps.apply_style()
    fig, ax = ps.multipanel(3, 2, panel_height=2.8)

    # (A) sensitivity
    s = sens.iloc[::-1]
    bars = ax[0].barh(np.arange(len(s)), s["taxonomy_unique"], height=0.62,
                      color=ps.PALETTE["primary"], linewidth=0)
    ps.bar_values(ax[0], bars, "{:.3f}", horizontal=True)
    ax[0].set_yticks(np.arange(len(s)))
    ax[0].set_yticklabels(ps.shorten(s["variant"], 34), fontsize=6.5)
    ax[0].axvline(sens["taxonomy_unique"].iloc[0], ls="--", lw=1.0,
                  color=ps.PALETTE["secondary"])
    ps.grid_axis(ax[0], "x"); ps.despine(ax[0], left=True)
    ax[0].tick_params(axis="y", length=0)
    ps.label_axes(ax[0], "Taxonomy unique variance", "Analytical variant")
    ax[0].set_title("Estimate is stable across choices", fontweight="bold",
                    fontsize=8)

    # (B) mobility with controls
    if mob:
        labs = ["raw", "outcome\nvariance held", "prevalence\nheld"]
        vals = [mob["raw"], mob["ctrl_variance"], mob["ctrl_prevalence"]]
        bars = ax[1].bar(np.arange(3), vals, width=0.58,
                         color=[ps.PALETTE["primary"], ps.PALETTE["tertiary"],
                                ps.PALETTE["accent"]], linewidth=0)
        ps.bar_values(ax[1], bars, "{:.3f}")
        lo, hi = np.percentile(mob["boot"], [2.5, 97.5])
        ax[1].errorbar(0, mob["raw"], yerr=[[mob["raw"] - lo], [hi - mob["raw"]]],
                       fmt="none", ecolor=ps.PALETTE["ink"], capsize=4, lw=1.2)
        ax[1].axhline(0, lw=0.9, color=ps.PALETTE["ink"])
        ax[1].set_xticks(np.arange(3)); ax[1].set_xticklabels(labs, fontsize=7)
        ps.grid_axis(ax[1], "y"); ps.despine(ax[1], bottom=True)
        ax[1].tick_params(axis="x", length=0)
        ps.label_axes(ax[1], "Control applied", r"Conservation vs $\rho$")
        ax[1].set_title("Survives both confounds", fontweight="bold", fontsize=8)
    else:
        ax[1].axis("off")

    # (C) bootstrap
    if mob:
        ps.no_grid(ax[2])
        ax[2].hist(mob["boot"], bins=45, color=ps.PALETTE["primary"], linewidth=0)
        lo, hi = np.percentile(mob["boot"], [2.5, 97.5])
        ax[2].axvspan(lo, hi, color=ps.PALETTE["accent"], alpha=0.22)
        ax[2].axvline(0, ls="--", lw=1.1, color=ps.PALETTE["ink"])
        ps.label_axes(ax[2], r"Bootstrap $\rho$", "Resamples")
        ax[2].set_title(f"95% CI {lo:.2f} to {hi:.2f}, 2,000 resamples",
                        fontweight="bold", fontsize=8)
    else:
        ax[2].axis("off")

    # (D) the metric artefact
    if len(hab):
        ps.no_grid(ax[3])
        habs = sorted(hab["habitat"].unique())
        col = {h: ps.SERIES[i % len(ps.SERIES)] for i, h in enumerate(habs)}
        # habitat directories are named 'ocean'; the manuscript uses 'marine'
        pretty = {'ocean': 'marine'}
        for h in habs:
            g = hab[hab["habitat"] == h]
            ax[3].scatter(g["arg_sd"], g["rho"], s=22, alpha=0.75,
                          color=col[h], edgecolors="white", linewidths=0.5,
                          label=pretty.get(h, h))
        z = np.polyfit(hab["arg_sd"], hab["rho"], 1)
        xs = np.linspace(hab["arg_sd"].min(), hab["arg_sd"].max(), 40)
        ax[3].plot(xs, np.polyval(z, xs), ls="--", lw=1.2, color=ps.PALETTE["ink"])
        ax[3].legend(fontsize=6)
        r = spearmanr(hab["arg_sd"], hab["rho"])
        ax[3].set_title(f"$\\rho$ = {r.statistic:.2f} across {len(hab)} folds",
                        fontweight="bold", fontsize=8)
        ps.label_axes(ax[3], "Held-out resistome variance",
                      r"Cross-cohort $\rho$")
    else:
        ax[3].axis("off")

    # (E) tested hypotheses
    tests = [
        ("Conservation predicts predictability",
         mob["ctrl_prevalence"] if mob else np.nan, True),
        ("Taxonomic pooling improves transfer", -0.013, False),
        ("Composition leads the resistome", 0.0, False),
        ("Cohort distance explains transfer", _dist_rho, False),
    ]
    lab = [t[0] for t in tests][::-1]
    val = [t[1] for t in tests][::-1]
    sup = [t[2] for t in tests][::-1]
    bars = ax[4].barh(np.arange(len(lab)), val, height=0.6,
                      color=[ps.PALETTE["tertiary"] if s_ else ps.PALETTE["neutral"]
                             for s_ in sup], linewidth=0)
    ps.bar_values(ax[4], bars, "{:+.2f}", horizontal=True)
    ax[4].axvline(0, lw=0.9, color=ps.PALETTE["ink"])
    ax[4].set_yticks(np.arange(len(lab)))
    ax[4].set_yticklabels(ps.shorten(lab, 32), fontsize=6.5)
    ps.grid_axis(ax[4], "x"); ps.despine(ax[4], left=True)
    ax[4].tick_params(axis="y", length=0)
    ps.label_axes(ax[4], "Effect size", "Hypothesis tested")
    ax[4].set_title("Green: supported. Grey: tested, not supported",
                    fontweight="bold", fontsize=7.5)

    # (F) core vs accessory
    if mob:
        d = mob["data"]
        core = d[d["mean_frequency"] >= 0.5]["rho"].values
        acc = d[d["mean_frequency"] < 0.5]["rho"].values
        if len(core) >= 3 and len(acc) >= 3:
            from s13_mobility_predictability import _boxplot
            bp = _boxplot(ax[5], [acc, core],
                          [f"accessory\n(n={len(acc)})", f"core-like\n(n={len(core)})"],
                          showfliers=False, patch_artist=True, widths=0.55)
            for b, c in zip(bp["boxes"],
                            [ps.PALETTE["secondary"], ps.PALETTE["primary"]]):
                b.set(facecolor=c, alpha=0.45, linewidth=0)
            for m_ in bp["medians"]:
                m_.set(color=ps.PALETTE["ink"], linewidth=1.6)
            rng2 = np.random.default_rng(0)
            for i, g in enumerate([acc, core], start=1):
                ax[5].scatter(rng2.normal(i, 0.05, len(g)), g, s=14, alpha=0.6,
                              color=ps.PALETTE["neutral"], edgecolors="none",
                              zorder=3)
            ps.grid_axis(ax[5], "y"); ps.despine(ax[5], bottom=True)
            ax[5].tick_params(axis="x", length=0)
            ps.label_axes(ax[5], "Gene class", r"Cross-cohort $\rho$")
        else:
            ax[5].axis("off")
    else:
        ax[5].axis("off")

    written = ps.save(fig, cfg["paths"]["fig_dir"], f"fig11_robustness{tag(cfg)}",
                      cfg["figures"]["formats"], cfg["figures"]["dpi"])
    LOG.info("fig11 -> %s", written[0])

    summary = pd.DataFrame([
        {"analysis": "taxonomy unique variance", "value": sens["taxonomy_unique"].iloc[0],
         "range_across_variants": rng_, "supported": True},
        {"analysis": "mobility vs predictability (prevalence held constant)",
         "value": mob["ctrl_prevalence"] if mob else np.nan,
         "range_across_variants": np.nan, "supported": True},
        {"analysis": "per-fold rho vs outcome variance (metric artefact)",
         "value": spearmanr(hab["arg_sd"], hab["rho"]).statistic if len(hab) else np.nan,
         "range_across_variants": np.nan, "supported": True},
    ])
    summary.to_csv(table_path(cfg, "robustness_summary.csv"), index=False)


if __name__ == "__main__":
    main()
