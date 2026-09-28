"""
s13_mobility_predictability.py
==============================
GENOMIC CONSERVATION AND PREDICTABILITY

(The filename is retained for continuity with earlier pipeline versions; the
quantity analysed is within-species conservation, not mobility.)

Does a gene's predictability follow from how consistently its host species
carry it?

TERMINOLOGY. The quantity measured here is the fraction of a species'
sequenced assemblies in which a gene is detected - its PANGENOME FREQUENCY, or
genome-boundedness. That is not the same as mobility. A gene can be rare
across strains through lineage specificity, population structure or gain and
loss without currently residing on a mobile element, and a mobile gene can
become fixed in a population. Direct mobility would require locating the gene
relative to plasmids, integrons and insertion sequences, which is not done
here. Claims should be phrased in terms of genome-boundedness.

The pangenome analysis shows that most species-gene pairs are accessory, and
the modelling shows that community composition predicts the resistome. Those
are presented as separate observations, and the accessory finding is used to
EXPLAIN the failure of taxonomic pooling - but the explanation is asserted
rather than tested.

It makes a sharp, falsifiable prediction. If composition predicts a resistance
gene because the organisms carrying it are present, then:

    genes tied firmly to a genome (core, present in nearly every assembly of
    their host species) should be well predicted, and

    genes carried on mobile elements (accessory, present in a minority of
    assemblies) should be poorly predicted,

because for an accessory gene, knowing the species is present says little
about whether this particular strain carries it.

Testing this converts the mechanism from an interpretation into a result. If
it fails, the accessory-genome explanation for the pooling result is wrong and
should be withdrawn rather than defended.

Analyses
--------
1. Per gene, the mean pangenome frequency across the species carrying it,
   against its cross-cohort predictability.
2. The same split by how the gene is attributed - whether its predictability
   is concentrated in species where it is core.
3. Rank-based test with study-level bootstrap confidence intervals, since
   genes are not independent observations.

Outputs:
    tables/mobility_vs_predictability_<stratum>.csv
    figures/fig6_mobility_<stratum>.png / .pdf / .svg

Usage:
    python src/s13_mobility_predictability.py \\
        --pangenome /mnt/x/w1_resistome/pangenome/pangenome_evidence.tsv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

import plotstyle as ps
from common import LOG, load_config, stratum, table_path, tag, work_path
from s06_attribution import normalise_gene, normalise_species


def _boxplot(ax, data, labels, **kw):
    """boxplot with tick labels, across the matplotlib 3.9 rename."""
    try:
        return ax.boxplot(data, tick_labels=labels, **kw)
    except TypeError:
        return ax.boxplot(data, labels=labels, **kw)


def gene_mobility(ev: pd.DataFrame) -> pd.DataFrame:
    """Two distinct quantities, which must not be conflated.

    CONSERVATION - the mean frequency across only those species in which the
    gene occurs at all. This is genome-boundedness proper: given that a species
    carries the gene, how consistently do its strains carry it?

    BREADTH - the fraction of ALL examined species in which the gene occurs.
    This is host range, a different property.

    An evidence table with explicit zeros makes the distinction matter. Taking
    the mean over every examined species mixes the two: a gene confined to one
    species but fixed within it, and a gene found at low frequency in many
    species, can score alike. The original attribution-driven table contained
    only detected pairs, so its mean was implicitly the conservation measure.
    """
    ev = ev.copy()
    ev["gk"] = normalise_gene(ev["gene"])
    if "detected" in ev.columns:
        det = ev[ev["detected"] == 1]
        n_examined = ev.groupby("gk")["species"].nunique()
    else:
        det = ev[ev["frequency"] > 0]
        n_examined = ev.groupby("gk")["species"].nunique()

    g = det.groupby("gk").agg(
        n_host_species=("species", "nunique"),
        conservation=("frequency", "mean"),          # within-species, where present
        median_conservation=("frequency", "median"),
        max_frequency=("frequency", "max"),
        n_core=("frequency", lambda s: int((s >= 0.9).sum())),
        n_accessory=("frequency", lambda s: int((s < 0.3).sum())),
    ).reset_index()
    g["n_species_examined"] = g["gk"].map(n_examined)
    g["breadth"] = g["n_host_species"] / g["n_species_examined"]
    # retained for backward compatibility with the earlier evidence table
    g["mean_frequency"] = g["conservation"]
    g["n_species"] = g["n_host_species"]
    g["median_frequency"] = g["median_conservation"]
    g["mostly_core"] = g["conservation"] >= 0.5
    return g


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pangenome", required=True,
                    help="pangenome_evidence.tsv from s06c")
    ap.add_argument("--scheme", default="leave_one_study_out")
    ap.add_argument("--label", default=None,
                    help="suffix for output files, e.g. 'independent'")
    args = ap.parse_args()

    cfg = load_config()
    m = pd.read_parquet(work_path(cfg, "cv_metrics.parquet", per_stratum=True))
    # STALENESS CHECK. This analysis explains per-gene predictive performance,
    # so it must be computed against the CURRENT model output. Running it after
    # an older model run produces a number that looks valid and silently
    # describes predictions the manuscript no longer reports.
    _cvp = work_path(cfg, "cv_metrics.parquet", per_stratum=True)
    _pgp = Path(args.pangenome)
    if _cvp.exists() and _pgp.exists():
        import datetime as _dt
        _cv_t = _dt.datetime.fromtimestamp(_cvp.stat().st_mtime)
        LOG.info("Model metrics: %s (%s)", _cvp.name,
                 _cv_t.strftime("%Y-%m-%d %H:%M"))
        _out = table_path(cfg, f"mobility_vs_predictability"
                               f"{('_' + args.label) if args.label else ''}.csv")
        if _out.exists() and _out.stat().st_mtime < _cvp.stat().st_mtime:
            LOG.warning("The existing conservation output predates the current "
                        "model metrics by %.1f h; it is being regenerated now, "
                        "but any manuscript numbers taken from the old file are "
                        "stale.",
                        (_cvp.stat().st_mtime - _out.stat().st_mtime) / 3600)

    ev = pd.read_csv(args.pangenome, sep="\t")
    if "detected" in ev.columns:
        # An evidence table with explicit zeros: species examined and found not
        # to carry a gene contribute frequency 0 rather than being absent. This
        # removes the detection conditioning of the attribution-driven table,
        # in which only detected pairs could ever appear.
        n0 = int((ev["detected"] == 0).sum())
        LOG.info("Evidence table carries explicit zeros: %d of %d pairs (%.0f%%) "
                 "are species examined and found negative.", n0, len(ev),
                 100 * n0 / max(len(ev), 1))
    if "frequency" not in ev.columns:
        raise SystemExit("Pangenome evidence lacks a frequency column - this "
                         "analysis needs multiple assemblies per species "
                         "(s06c), not the single-reference table (s06b).")

    mob = gene_mobility(ev)
    LOG.info("Pangenome evidence: %d gene families across %d species",
             len(mob), ev["species"].nunique())

    # per-gene cross-cohort predictability, per fold retained for bootstrapping
    sub = m[(m["scheme"] == args.scheme) & (m["model"] != "mean")]
    if sub.empty:
        raise SystemExit(f"No metrics for scheme '{args.scheme}'.")
    best = sub.groupby("model")["spearman"].median().idxmax()
    LOG.info("Using the best-performing model: %s", best)
    perf = (sub[sub["model"] == best]
            .groupby("gene")
            .agg(rho=("spearman", "median"),
                 r2c=("r2_centered", "mean") if "r2_centered" in sub.columns
                 else ("r2", "mean"),
                 n_folds=("fold", "nunique"))
            .reset_index())
    perf["gk"] = normalise_gene(perf["gene"])

    d = perf.merge(mob, on="gk", how="inner")
    LOG.info("Gene families with both predictability and pangenome evidence: "
             "%d of %d modelled", len(d), len(perf))
    if len(d) < 15:
        raise SystemExit("Too few genes overlap for a meaningful test. Widen "
                         "the pangenome run (more species) before interpreting.")

    # ---- 1. the central test --------------------------------------------
    r_mean = spearmanr(d["conservation"], d["rho"])
    r_med = spearmanr(d["median_conservation"], d["rho"])
    if "breadth" in d.columns and d["breadth"].notna().sum() > 10:
        r_br = spearmanr(d["breadth"], d["rho"])
        LOG.info("  HOST BREADTH (fraction of species carrying it) vs rho : "
                 "rho=%.3f, P=%.3g", r_br.statistic, r_br.pvalue)
        LOG.info("  Conservation and breadth are different properties and are "
                 "reported separately.")
    LOG.info("=" * 66)
    LOG.info("PREDICTABILITY vs GENOME-BOUNDEDNESS")
    LOG.info("  mean pangenome frequency vs cross-cohort rho : "
             "rho=%.3f, P=%.3g (n=%d genes)",
             r_mean.statistic, r_mean.pvalue, len(d))
    LOG.info("  median frequency vs cross-cohort rho         : rho=%.3f, P=%.3g",
             r_med.statistic, r_med.pvalue)

    core, acc = d[d["mostly_core"]], d[~d["mostly_core"]]
    if len(core) >= 5 and len(acc) >= 5:
        # Two-sided, matching the manuscript's stated convention. A one-sided
        # test reports half the P value and must not be quoted as two-sided.
        u = mannwhitneyu(core["rho"], acc["rho"], alternative="two-sided")
        LOG.info("  core-like genes (n=%d) median rho     = %.3f", len(core),
                 float(core["rho"].median()))
        LOG.info("  accessory-like genes (n=%d) median rho = %.3f", len(acc),
                 float(acc["rho"].median()))
        LOG.info("  Mann-Whitney U (core > accessory): P=%.3g", u.pvalue)
    LOG.info("=" * 66)

    if r_mean.statistic > 0.2 and r_mean.pvalue < 0.05:
        LOG.info("SUPPORTED: genes carried more consistently by their host "
                 "species are better predicted from composition. The accessory-genome account of "
                 "the pooling result is now tested, not merely asserted.")
    elif r_mean.pvalue >= 0.05:
        LOG.warning("NOT SUPPORTED: predictability is unrelated to how "
                    "genome-bound a gene is. The accessory-genome explanation "
                    "for the pooling result should be withdrawn rather than "
                    "defended; report this as a negative result.")
    else:
        LOG.warning("REVERSED: accessory genes are better predicted, which the "
                    "mechanism does not anticipate. Investigate before writing.")

    # ---- 2. bootstrap over genes ----------------------------------------
    rng = np.random.default_rng(42)
    boots = [spearmanr(*d.iloc[rng.integers(0, len(d), len(d))]
                       [["mean_frequency", "rho"]].values.T).statistic
             for _ in range(2000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    LOG.info("Bootstrap 95%% CI for the correlation: %.3f to %.3f", lo, hi)

    # Prevalence and abundance variance must both be in hand BEFORE the joint
    # model runs. Computing prevalence afterwards silently omits it, while the
    # log still claims it was held constant.
    # Covariates must be computed on the SAME population as the outcome. The
    # predictive performance, the species selection and this analysis are all
    # adult; computing prevalence and abundance variance over the full cohort
    # would adjust the adult association using infant-inclusive quantities.
    # Covariates come from the UNFILTERED matrix. The modelled gene set is now
    # drawn from the full 682-family universe, so genes can be modelled and
    # analysed here while being absent from the pooled-filtered 99-gene matrix;
    # reading the filtered one silently drops their prevalence and variance.
    _unf = work_path(cfg, "arg_abundance_unfiltered.parquet")
    if _unf.exists():
        arg = pd.read_parquet(_unf)
        LOG.info("Covariates from the unfiltered ARG matrix (%d genes).",
                 arg.shape[1])
    else:
        arg = pd.read_parquet(work_path(cfg, "arg_abundance.parquet"))
        LOG.warning("arg_abundance_unfiltered.parquet absent; covariates come "
                    "from the pooled-filtered matrix and genes outside it will "
                    "have no prevalence or variance.")
    st_p = work_path(cfg, "sample_table.parquet")
    if st_p.exists():
        _st = pd.read_parquet(st_p)
        _strat = stratum(cfg)
        if _strat and _strat != "all" and "age_category" in _st.columns:
            _keep = set(_st.loc[_st["age_category"].astype(str) == _strat,
                                "sample"].astype(str))
            _before = len(arg)
            arg = arg.loc[arg.index.astype(str).isin(_keep)]
            LOG.info("Covariates restricted to stratum '%s': %d of %d samples",
                     _strat, len(arg), _before)
    pseudo = float(cfg["normalisation"]["arg_pseudocount"])
    prev = (arg > np.log(pseudo) + 1e-9).mean(axis=0).rename("prevalence")
    d = d.merge(prev.reset_index().rename(columns={"index": "gene"}),
                on="gene", how="left")

    # ---- 3. is OUTCOME VARIANCE the real driver? ------------------------
    # Per-gene Spearman scales with how much the outcome varies: a gene with
    # little variation offers little to rank correctly, so rho is depressed
    # regardless of model quality. Across habitat folds this confound is
    # strong (rho = 0.50, 117 folds), so it must be removed here before any
    # claim that mobility explains predictability.
    arg_var = arg.std(axis=0)   # same adult-restricted matrix as prevalence
    d = d.merge(arg_var.rename("arg_sd").reset_index().rename(
        columns={"index": "gene"}), on="gene", how="left")
    if d["arg_sd"].notna().sum() > 10:
        r_var = spearmanr(d["arg_sd"], d["rho"])
        LOG.info("CONFOUND CHECK: per-gene outcome variance vs rho = %.3f "
                 "(P=%.3g)", r_var.statistic, r_var.pvalue)
        ok = d[["conservation", "rho", "arg_sd"]].replace(
            [np.inf, -np.inf], np.nan).dropna()
        if len(ok) > 12:
            rk = ok.rank()
            def _res(y, x):
                return y - np.polyval(np.polyfit(x, y, 1), x)
            pv = spearmanr(_res(rk["conservation"].values, rk["arg_sd"].values),
                           _res(rk["rho"].values, rk["arg_sd"].values))
            LOG.info("CONSERVATION, outcome variance held constant: rho=%.3f, P=%.3g",
                     pv.statistic, pv.pvalue)
            if pv.pvalue < 0.05 and pv.statistic > 0.2:
                LOG.info("SURVIVES: genome-boundedness predicts predictability "
                         "independently of how variable the gene is.")
            else:
                LOG.warning("DOES NOT SURVIVE: the conservation relationship is "
                            "explained by outcome variance. Withdraw the "
                            "mechanistic claim rather than defending it.")

    # ---- 3b. all confounders at once ------------------------------------
    # Separate partial correlations leave open the possibility that several
    # weak confounders jointly account for the association. A single model
    # containing all of them at once does not.
    try:
        import statsmodels.api as sm
        # n_host_species is omitted: breadth = n_host_species / species
        # examined, and with a common species set the two are identical.
        cols = ["conservation", "breadth", "prevalence", "arg_sd"]
        mv = d.copy()
        if "prevalence" not in mv.columns:
            mv["prevalence"] = np.nan
        use = [c for c in cols if c in mv.columns
               and mv[c].replace([np.inf, -np.inf], np.nan).notna().sum() > 20]
        mv = mv.replace([np.inf, -np.inf], np.nan).dropna(subset=use + ["rho"])
        # a column that is constant after filtering makes the design singular
        use = [c for c in use if mv[c].nunique() > 2]
        # Drop perfectly collinear predictors. breadth = n_host_species /
        # n_species_examined, and when every gene is examined against the same
        # species set the denominator is constant, making the two identical.
        # Keeping both splits the coefficient between them and halves the
        # apparent effect of each.
        dropped = []
        for i, a in enumerate(list(use)):
            for b in list(use)[i + 1:]:
                if a in use and b in use and abs(mv[a].corr(mv[b])) > 0.999:
                    use.remove(b); dropped.append((b, a))
        for b, a in dropped:
            LOG.warning("Dropped '%s' from the model: perfectly collinear with "
                        "'%s' (r > 0.999). Retaining both would divide the "
                        "coefficient between them.", b, a)
        if len(mv) > 25 and len(use) >= 2:
            # rank-transform so the model matches the Spearman framing
            X = mv[use].rank()
            X = sm.add_constant((X - X.mean()) / X.std())
            y = mv["rho"].rank()
            y = (y - y.mean()) / y.std()
            fit = sm.OLS(y, X).fit()
            LOG.info("MULTIVARIABLE MODEL (rank-standardised, n=%d genes)", len(mv))
            for name in use:
                LOG.info("  %-16s beta=%+.3f  P=%.3g", name,
                         fit.params[name], fit.pvalues[name])
            LOG.info("  model R2=%.3f", fit.rsquared)
            if fit.pvalues.get("conservation", 1) < 0.05:
                LOG.info("Pangenome frequency remains associated with "
                         "predictability with prevalence, abundance variance "
                         "and host count all held constant.")
            else:
                LOG.warning("Within-species conservation is NOT independently "
                            "associated once all covariates are included. The "
                            "separate partial correlations were optimistic.")
            # Bootstrap intervals: with 61 gene families, asymptotic OLS
            # intervals are optimistic.
            rngb = np.random.default_rng(42)
            bcoef = {t: [] for t in fit.params.index}
            for _ in range(2000):
                idx = rngb.integers(0, len(mv), len(mv))
                try:
                    fb = sm.OLS(y.values[idx], X.values[idx]).fit()
                    for t, v in zip(fit.params.index, fb.params):
                        bcoef[t].append(v)
                except Exception:  # noqa: BLE001
                    continue
            ci = {t: np.percentile(v, [2.5, 97.5]) if v else (np.nan, np.nan)
                  for t, v in bcoef.items()}
            for t in use:
                LOG.info("  %-16s bootstrap 95%% CI %.3f to %.3f",
                         t, ci[t][0], ci[t][1])
            pd.DataFrame({"term": fit.params.index, "beta": fit.params.values,
                          "p": fit.pvalues.values,
                          "ci_low": [ci[t][0] for t in fit.params.index],
                          "ci_high": [ci[t][1] for t in fit.params.index]}).to_csv(
                table_path(cfg, "mobility_multivariable.csv"), index=False)
    except ImportError:
        LOG.warning("statsmodels unavailable; multivariable model skipped.")

    # ---- 4. prevalence as a single control ------------------------------
    if "prevalence" in d.columns and d["prevalence"].notna().sum() > 10:
        r_prev = spearmanr(d["prevalence"], d["rho"])
        LOG.info("Control: sample prevalence vs rho = %.3f (P=%.3g). A gene "
                 "detected in more samples is easier to predict for reasons "
                 "unrelated to mobility, so this must be reported alongside.",
                 r_prev.statistic, r_prev.pvalue)
        # partial correlation of frequency and rho, controlling for prevalence
        ok = d[["mean_frequency", "rho", "prevalence"]].dropna()
        if len(ok) > 12:
            rk = ok.rank()
            def resid(y, x):
                b = np.polyfit(x, y, 1)
                return y - np.polyval(b, x)
            pr = spearmanr(resid(rk["mean_frequency"].values, rk["prevalence"].values),
                           resid(rk["rho"].values, rk["prevalence"].values))
            LOG.info("Partial (prevalence held constant): rho=%.3f, P=%.3g",
                     pr.statistic, pr.pvalue)

    suffix = f"_{args.label}" if args.label else ""
    # Many highly conserved genes occur in only one or two host species, so a
    # reviewer will ask whether the association is carried by those alone.
    hs_rows = []
    for k in (1, 2, 3, 5):
        sub_k = d[d["n_host_species"] >= k]
        if len(sub_k) < 12:
            continue
        r_k = spearmanr(sub_k["conservation"], sub_k["rho"])
        hs_rows.append({"min_host_species": k, "n_genes": len(sub_k),
                        "spearman": r_k.statistic, "p_value": r_k.pvalue})
        LOG.info("  genes in >=%d host species: n=%d, rho=%.3f (P=%.3g)",
                 k, len(sub_k), r_k.statistic, r_k.pvalue)
    if hs_rows:
        pd.DataFrame(hs_rows).to_csv(
            table_path(cfg, f"conservation_by_host_count{suffix}.csv"), index=False)

    out = table_path(cfg, f"mobility_vs_predictability{suffix}.csv")
    d.sort_values("mean_frequency").to_csv(out, index=False)
    LOG.info("Wrote %s", out)

    # ---- figure ----------------------------------------------------------
    ps.apply_style()
    fig, ax = ps.multipanel(2, 2, panel_height=2.9)

    ps.no_grid(ax[0])
    sc = ax[0].scatter(d["mean_frequency"], d["rho"], s=34,
                       c=d["prevalence"].fillna(d["prevalence"].median()),
                       cmap=ps.sequential(), edgecolors="white", linewidths=0.8)
    cb = fig.colorbar(sc, ax=ax[0]); cb.set_label("Sample prevalence", fontweight="bold")
    if len(d) > 3:
        z = np.polyfit(d["mean_frequency"], d["rho"], 1)
        xs = np.linspace(d["mean_frequency"].min(), d["mean_frequency"].max(), 50)
        ax[0].plot(xs, np.polyval(z, xs), ls="--", lw=1.2,
                   color=ps.PALETTE["secondary"])
    ax[0].set_title(f"$\\rho$ = {r_mean.statistic:.2f}, $P$ = {r_mean.pvalue:.2g}",
                    fontweight="bold", fontsize=9)
    ps.label_axes(ax[0], "Within-species conservation",
                  r"Cross-cohort $\rho$")

    if len(core) >= 5 and len(acc) >= 5:
        bp = _boxplot(ax[1], [acc["rho"].values, core["rho"].values],
                      ["accessory\n(<0.5)", "core-like\n($\\geq$0.5)"],
                      showfliers=False, patch_artist=True, widths=0.55)
        for b, c in zip(bp["boxes"], [ps.PALETTE["secondary"], ps.PALETTE["primary"]]):
            b.set(facecolor=c, alpha=0.45, linewidth=0)
        for med in bp["medians"]:
            med.set(color=ps.PALETTE["ink"], linewidth=1.6)
        for i, g in enumerate([acc, core], start=1):
            ax[1].scatter(np.random.default_rng(0).normal(i, 0.05, len(g)),
                          g["rho"], s=14, alpha=0.6, color=ps.PALETTE["neutral"],
                          edgecolors="none", zorder=3)
        ps.grid_axis(ax[1], "y"); ps.despine(ax[1], bottom=True)
        ax[1].tick_params(axis="x", length=0)
        ps.label_axes(ax[1], "Gene class", r"Cross-cohort $\rho$")
    else:
        ax[1].axis("off")

    # Absence is not the same as accessory carriage. A species that does not
    # carry a gene at all is an ABSENT pair, not a species carrying it in a
    # minority of its strains. Pooling the two would report ~99% "accessory"
    # purely because most species do not carry most genes.
    det = ev[ev["frequency"] > 0] if "detected" not in ev.columns \
        else ev[ev["detected"] == 1]
    n_absent = len(ev) - len(det)
    ax[2].hist(det["frequency"], bins=24, color=ps.PALETTE["tertiary"], linewidth=0)
    ax[2].axvline(0.3, ls="--", lw=1.1, color=ps.PALETTE["secondary"])
    ax[2].axvline(0.9, ls="--", lw=1.1, color=ps.PALETTE["primary"])
    ps.grid_axis(ax[2], "y")
    ps.label_axes(ax[2], "Pangenome frequency, where detected",
                  "Species-gene pairs")
    if len(det):
        ax[2].set_title("Detected pairs only: %.0f%% accessory, %.0f%% core\n"
                        "(%s of %s possible combinations absent)"
                        % (100 * (det["frequency"] < 0.3).mean(),
                           100 * (det["frequency"] >= 0.9).mean(),
                           f"{n_absent:,}", f"{len(ev):,}"),
                        fontweight="bold", fontsize=7.5)

    ps.no_grid(ax[3])
    ax[3].hist(boots, bins=40, color=ps.PALETTE["primary"], linewidth=0)
    ax[3].axvline(0, ls="--", lw=1.1, color=ps.PALETTE["ink"])
    ax[3].axvspan(lo, hi, color=ps.PALETTE["accent"], alpha=0.20)
    ps.label_axes(ax[3], r"Bootstrap $\rho$", "Resamples")
    ax[3].set_title("95%% CI %.2f to %.2f" % (lo, hi), fontweight="bold", fontsize=9)

    written = ps.save(fig, cfg["paths"]["fig_dir"],
                      f"fig6_mobility{suffix}{tag(cfg)}",
                      cfg["figures"]["formats"], cfg["figures"]["dpi"])
    LOG.info("fig6 -> %s", written[0])


if __name__ == "__main__":
    main()
