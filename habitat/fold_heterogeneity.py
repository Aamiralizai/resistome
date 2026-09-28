"""
fold_heterogeneity.py
=====================
Does cross-cohort predictability depend on how different the held-out cohort
is from the training data?

The habitat comparison shows an 8.5-fold range in transferability (human gut
0.264, animal 0.181, environmental 0.087, ocean 0.031) while the underlying
association varies only 1.7-fold. Habitat is an unsatisfying explanation for
that: within the animal habitat alone, chicken caecum transfers at 0.28 and
honeybee at -0.01, so whatever governs transfer operates below the level of
habitat.

The obvious candidate is compositional dissimilarity. A model trained on other
cohorts should predict a held-out cohort well when that cohort resembles the
training data, and poorly when it does not. This script tests that directly,
using every leave-one-study-out fold from every habitat as an observation -
roughly 170 folds rather than four habitats, which is a properly powered test
of the mechanism rather than an eyeball comparison of group means.

For each fold:
    dissimilarity = distance from the held-out study's centroid to the
                    centroid of the training studies, in centred log-ratio
                    space (Aitchison distance)
    dispersion    = mean within-study distance to that study's own centroid,
                    which controls for a study simply being internally variable

The relationship is then tested overall, within habitat, and with habitat as a
covariate, because a pooled correlation across habitats could be produced
entirely by between-habitat differences (Simpson's paradox).

Outputs:
    <outdir>/fold_heterogeneity.csv
    <outdir>/fig_fold_heterogeneity.png / .pdf / .svg

Usage:
    python habitat/fold_heterogeneity.py \\
        --habitat-dirs /mnt/x/w1_resistome/habitats/ocean \\
                       /mnt/x/w1_resistome/habitats/animal \\
                       /mnt/x/w1_resistome/habitats/environmental \\
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
import plotstyle as ps      # noqa: E402
from common import LOG      # noqa: E402

MIN_TEST = 20


def fold_geometry(taxa: pd.DataFrame, studies: pd.Series) -> pd.DataFrame:
    """Per study: distance from its centroid to the centroid of all others,
    and its own internal dispersion."""
    rows = []
    X = taxa.values
    s = studies.reindex(taxa.index).astype(str).values
    for g in pd.unique(s):
        te = s == g
        if te.sum() < MIN_TEST or (~te).sum() < 50:
            continue
        c_te = X[te].mean(axis=0)
        c_tr = X[~te].mean(axis=0)
        rows.append({
            "study": g,
            "n_test": int(te.sum()),
            # Aitchison distance between centroids: how far the held-out
            # cohort sits from everything the model was trained on.
            "centroid_distance": float(np.linalg.norm(c_te - c_tr)),
            # how internally variable the held-out cohort is, which affects
            # predictability for reasons unrelated to novelty
            "dispersion": float(np.mean(np.linalg.norm(X[te] - c_te, axis=1))),
            # spread of the training data, for scale
            "train_dispersion": float(np.mean(np.linalg.norm(X[~te] - c_tr, axis=1))),
        })
    return pd.DataFrame(rows)


def load_habitat(d: Path) -> pd.DataFrame | None:
    perf = d / "model_performance.csv"
    taxa = d / "taxa_clr.parquet"
    st = d / "sample_table.parquet"
    if not (perf.exists() and taxa.exists() and st.exists()):
        LOG.warning("%s: missing outputs, skipped", d.name)
        return None
    p = pd.read_csv(perf)
    best = p.groupby("model")["median_spearman"].median().idxmax()
    p = p[p["model"] == best][["study", "median_spearman", "n_test"]]
    t = pd.read_parquet(taxa)
    s = pd.read_parquet(st).set_index("sample")["study"]
    geo = fold_geometry(t, s)
    out = p.merge(geo, on="study", how="inner", suffixes=("", "_geo"))
    out["habitat"] = d.name
    out["model"] = best
    LOG.info("%-14s %d folds (model %s)", d.name, len(out), best)
    return out


def load_gut(work: Path, stratum: str = "adult") -> pd.DataFrame | None:
    """The human gut folds, from the main pipeline's own outputs.

    Outcome and geometry must come from the SAME samples. Pairing adult
    prediction performance with centroids and dispersion computed over all
    life stages measures the distance between the wrong populations.
    """
    cm = next((work / f"cv_metrics_{s}.parquet" for s in ("adult", "all")
               if (work / f"cv_metrics_{s}.parquet").exists()), None)
    taxa_f, st_f = work / "taxa_clr.parquet", work / "sample_table.parquet"
    if cm is None or not (taxa_f.exists() and st_f.exists()):
        LOG.warning("gut outputs not found, skipped")
        return None
    m = pd.read_parquet(cm)
    m = m[(m["scheme"] != "random") & (m["model"] != "mean")]
    best = m.groupby("model")["spearman"].median().idxmax()
    p = (m[m["model"] == best].groupby("fold")["spearman"].median()
         .reset_index().rename(columns={"fold": "study",
                                        "spearman": "median_spearman"}))
    t = pd.read_parquet(taxa_f)
    stt = pd.read_parquet(st_f)
    if stratum and stratum != "all" and "age_category" in stt.columns:
        keep = set(stt.loc[stt["age_category"].astype(str) == stratum,
                           "sample"].astype(str))
        before = len(t)
        t = t.loc[t.index.astype(str).isin(keep)]
        LOG.info("Gut geometry restricted to stratum '%s': %d of %d samples",
                 stratum, len(t), before)
    s = stt.set_index("sample")["study"]
    geo = fold_geometry(t, s)
    out = p.merge(geo, on="study", how="inner")
    out["habitat"] = "human gut"
    out["model"] = best
    LOG.info("%-14s %d folds (model %s)", "human gut", len(out), best)
    return out


def partial_spearman(y, x, z, categorical=False):
    """Spearman partial correlation of y and x controlling for z.

    When z is CATEGORICAL it must be dummy-coded, not passed as integer codes.
    Fitting a straight line through category codes 0,1,2,3 treats an unordered
    label as a numeric scale, so only the linear trend across an arbitrary
    ordering is removed and most of the group structure survives into the
    residuals. On the habitat data that error gave rho = 0.120 (P = 0.12)
    where correct dummy coding gives rho = 0.263 (P = 5.4e-4) - the difference
    between an unsupported and a supported conclusion.
    """
    df = pd.DataFrame({"y": np.asarray(y), "x": np.asarray(x)})
    z = pd.Series(z).reset_index(drop=True)
    keep = df.notna().all(axis=1).values & z.notna().values
    df, z = df[keep].reset_index(drop=True), z[keep].reset_index(drop=True)
    if len(df) < 12:
        return np.nan, np.nan
    if categorical or z.dtype == object or str(z.dtype).startswith("category"):
        Z = pd.get_dummies(z, drop_first=True).astype(float)
    else:
        Z = pd.DataFrame({"z": z.rank().astype(float)})
    Z.insert(0, "const", 1.0)
    Zv = Z.values

    def resid(v):
        beta, *_ = np.linalg.lstsq(Zv, v, rcond=None)
        return v - Zv @ beta

    r = spearmanr(resid(df["y"].rank().values), resid(df["x"].rank().values))
    return r.statistic, r.pvalue


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--habitat-dirs", nargs="+", required=True)
    ap.add_argument("--gut-work", default=None)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--gut-tables", default=None,
                    help="gut tables directory, for the final modelled gene "
                         "list used to match the variance to the outcome")
    ap.add_argument("--gut-stratum", default="adult",
                    help="life stage for the gut geometry; must match the "
                         "stratum whose predictions are used")
    args = ap.parse_args()

    frames = [load_habitat(Path(d)) for d in args.habitat_dirs]
    if args.gut_work:
        frames.append(load_gut(Path(args.gut_work), args.gut_stratum))
    frames = [f for f in frames if f is not None and len(f)]
    if not frames:
        raise SystemExit("No habitat produced usable fold data.")
    d = pd.concat(frames, ignore_index=True)

    # Held-out RESISTOME variance per fold. This is the quantity behind the
    # metric caveat, and it is distinct from compositional dispersion; the
    # cohort-distance interpretation depends on adjusting for both, so it must
    # be a column in the output rather than computed ad hoc.
    var_rows = []
    for hab in d["habitat"].unique():
        if hab == "human gut":
            # Match the OUTCOME exactly: the same adults and the same final
            # modelled genes as the prediction fold. Selecting by study alone
            # admits the infants and other life stages present in mixed
            # cohorts, and the pooled-filtered matrix carries a different gene
            # set from the one the model actually predicts.
            gw = Path(args.gut_work) if args.gut_work else None
            argf = None
            if gw:
                argf = gw / "arg_abundance_unfiltered.parquet"
                if not argf.exists():
                    argf = gw / "arg_abundance.parquet"
            stf = (gw / "sample_table.parquet") if gw else None
        else:
            hd = next((Path(x) for x in args.habitat_dirs
                       if Path(x).name == hab or Path(x).name.rstrip("/") == hab), None)
            argf = (hd / "arg_abundance.parquet") if hd else None
            stf = (hd / "sample_table.parquet") if hd else None
        if not (argf and stf and argf.exists() and stf.exists()):
            LOG.warning("No abundance matrix for habitat '%s'; its folds will "
                        "have no resistome-variance value.", hab)
            continue
        arg = pd.read_parquet(argf)
        stt = pd.read_parquet(stf)
        if hab == "human gut":
            strat = getattr(args, "gut_stratum", "adult")
            if strat and strat != "all" and "age_category" in stt.columns:
                before = len(stt)
                stt = stt[stt["age_category"].astype(str) == strat]
                LOG.info("Gut resistome variance restricted to '%s': %d of %d "
                         "samples", strat, len(stt), before)
            # The pipeline writes this per stratum, so the adult run produces
            # modelled_genes_adult.csv. Looking only for the unsuffixed name
            # means the lookup silently fails and the variance falls back to
            # every column of the abundance matrix - the mismatch this code
            # exists to prevent.
            gl = None
            if getattr(args, "gut_tables", None):
                gt = Path(args.gut_tables)
                for cand in (f"modelled_genes_{strat}.csv", "modelled_genes.csv"):
                    if (gt / cand).exists():
                        gl = gt / cand
                        break
                if gl is None:
                    LOG.error("No modelled-gene list in %s (looked for "
                              "modelled_genes_%s.csv and modelled_genes.csv). "
                              "Held-out resistome variance would be computed "
                              "over a different gene set from the one the model "
                              "predicts.", gt, strat)
            if gl and gl.exists():
                keep = [g for g in pd.read_csv(gl)["gene"] if g in arg.columns]
                if len(keep) >= 10:
                    arg = arg[keep]
                    LOG.info("Gut resistome variance over the %d final "
                             "modelled genes from %s", len(keep), gl.name)
            else:
                LOG.warning("modelled_genes.csv not found; gut variance uses "
                            "all %d columns of the abundance matrix, which may "
                            "differ from the modelled targets.", arg.shape[1])
        st = stt.set_index("sample")["study"].astype(str)
        for study in d.loc[d["habitat"] == hab, "study"]:
            ids = [i for i in st[st == str(study)].index if i in arg.index]
            var_rows.append({"habitat": hab, "study": study,
                             "resistome_variance":
                                 float(np.median(arg.loc[ids].std(axis=0)))
                                 if len(ids) >= 20 else np.nan})
    if var_rows:
        d = d.merge(pd.DataFrame(var_rows).drop_duplicates(["habitat", "study"]),
                    on=["habitat", "study"], how="left")
        LOG.info("Held-out resistome variance available for %d of %d folds",
                 int(d["resistome_variance"].notna().sum()), len(d))
    else:
        d["resistome_variance"] = np.nan

    # A fold whose ARGs were all constant in the held-out set has an undefined
    # Spearman. Two such folds propagate NaN through every correlation and
    # silently void an entire habitat, so they are removed explicitly.
    bad = d["median_spearman"].isna() | ~np.isfinite(d["centroid_distance"])
    if bad.any():
        LOG.warning("Dropping %d fold(s) with an undefined outcome or distance: "
                    "%s", int(bad.sum()),
                    ", ".join(d.loc[bad, "study"].astype(str).head(5)))
        d = d[~bad].reset_index(drop=True)
    d = d.drop(columns=[c for c in d.columns if c.endswith("_geo")])

    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    d.to_csv(out / "fold_heterogeneity.csv", index=False)

    LOG.info("=" * 68)
    LOG.info("TRANSFERABILITY vs COHORT DISSIMILARITY  (%d folds, %d habitats)",
             len(d), d["habitat"].nunique())
    LOG.info("=" * 68)

    r_all = spearmanr(d["centroid_distance"], d["median_spearman"])
    LOG.info("Pooled across habitats : rho=%.3f, P=%.3g", r_all.statistic,
             r_all.pvalue)

    # A pooled correlation can be produced entirely by between-habitat
    # differences, so the within-habitat estimate is the one that matters.
    rows = []
    for h, g in d.groupby("habitat"):
        if len(g) < 8:
            continue
        r = spearmanr(g["centroid_distance"], g["median_spearman"])
        rd = spearmanr(g["dispersion"], g["median_spearman"])
        rows.append({"habitat": h, "n_folds": len(g),
                     "rho_distance": r.statistic, "p_distance": r.pvalue,
                     "rho_dispersion": rd.statistic,
                     "median_rho": float(g["median_spearman"].median())})
        LOG.info("  %-14s n=%3d  distance rho=%+.3f (P=%.3g)  dispersion rho=%+.3f",
                 h, len(g), r.statistic, r.pvalue, rd.statistic)
    per_hab = pd.DataFrame(rows)

    # habitat as a covariate: does distance explain transferability once
    # habitat identity is removed?
    pr, pp = partial_spearman(d["median_spearman"], d["centroid_distance"],
                              d["habitat"], categorical=True)
    LOG.info("Controlling for habitat: partial rho=%.3f, P=%.3g", pr, pp)

    # dispersion alongside habitat: the metric confound, controlled jointly.
    # Controlling for dispersion ALONE gives a different model and a different
    # answer, and must not be quoted as though habitat were also held constant.
    def _partial_multi(y, x, cat=None, nums=()):
        """Partial Spearman with a categorical control and continuous ones.

        Quartile-binning a continuous covariate and pasting it onto the
        categorical control is not the same model: it discards within-bin
        variation and cannot reproduce the reported statistic.
        """
        df = pd.DataFrame({"y": np.asarray(y), "x": np.asarray(x)})
        parts = [pd.Series(1.0, index=df.index, name="const")]
        if cat is not None:
            parts.append(pd.get_dummies(pd.Series(cat).reset_index(drop=True),
                                        drop_first=True).astype(float))
        for nm, v in nums:
            parts.append(pd.Series(pd.Series(v).reset_index(drop=True).rank(),
                                   name=nm).astype(float))
        Z = pd.concat(parts, axis=1)
        keep = df.notna().all(axis=1).values & Z.notna().all(axis=1).values
        df, Zv = df[keep], Z[keep].values
        if len(df) < 12:
            return np.nan, np.nan

        def resid(v):
            beta, *_ = np.linalg.lstsq(Zv, v, rcond=None)
            return v - Zv @ beta
        r = spearmanr(resid(df["y"].rank().values), resid(df["x"].rank().values))
        return r.statistic, r.pvalue

    if "dispersion" in d.columns:
        pr_d, pp_d = _partial_multi(d["median_spearman"], d["centroid_distance"],
                                    cat=d["habitat"],
                                    nums=[("dispersion", d["dispersion"])])
        LOG.info("Controlling for habitat AND dispersion: partial rho=%.3f, "
                 "P=%.3g", pr_d, pp_d)

    # The hypothesis failed in an unexpected direction, so the covariates that
    # explain the positive association must be reported, and reproducibly.
    adj_rows = [{"adjustment": "habitat", "rho": pr, "p_value": pp,
                 "n_folds": int(d["centroid_distance"].notna().sum())}]
    if d["resistome_variance"].notna().any():
        pr_v, pp_v = _partial_multi(
            d["median_spearman"], d["centroid_distance"], cat=d["habitat"],
            nums=[("resistome_variance", d["resistome_variance"])])
        n_v = int((d["resistome_variance"].notna()
                   & d["centroid_distance"].notna()).sum())
        LOG.info("Controlling for habitat AND held-out resistome variance: "
                 "partial rho=%.3f, P=%.3g (n=%d)", pr_v, pp_v, n_v)
        adj_rows.append({"adjustment": "habitat + resistome variance",
                         "rho": pr_v, "p_value": pp_v, "n_folds": n_v})
        pr_b, pp_b = _partial_multi(
            d["median_spearman"], d["centroid_distance"], cat=d["habitat"],
            nums=[("dispersion", d["dispersion"]),
                  ("resistome_variance", d["resistome_variance"])])
        LOG.info("Controlling for habitat, dispersion AND resistome variance: "
                 "partial rho=%.3f, P=%.3g (n=%d)", pr_b, pp_b, n_v)
        adj_rows.append({"adjustment": "habitat + dispersion + resistome variance",
                         "rho": pr_b, "p_value": pp_b, "n_folds": n_v})

    # sample size is the obvious alternative explanation
    r_n = spearmanr(d["n_test"], d["median_spearman"])
    LOG.info("Control - held-out cohort size vs rho: rho=%.3f (P=%.3g)",
             r_n.statistic, r_n.pvalue)
    # Habitat AND cohort size. Adjusting for size alone answers a different
    # question from the one the manuscript asks.
    pr_n, pp_n = _partial_multi(d["median_spearman"], d["centroid_distance"],
                                cat=d["habitat"], nums=[("n_test", d["n_test"])])
    LOG.info("Controlling for habitat AND cohort size: partial rho=%.3f, "
             "P=%.3g", pr_n, pp_n)

    # Persist the whole adjustment series. Logging it only would leave the
    # interpretation unreproducible from the deposited outputs.
    if "dispersion" in d.columns:
        adj_rows.insert(1, {"adjustment": "habitat + dispersion",
                            "rho": pr_d, "p_value": pp_d,
                            "n_folds": int(d["dispersion"].notna().sum())})
    adj_rows.append({"adjustment": "habitat + cohort size", "rho": pr_n,
                     "p_value": pp_n, "n_folds": len(d)})
    pd.DataFrame(adj_rows).to_csv(out / "cohort_distance_adjustments.csv",
                                  index=False)
    LOG.info("Adjustment series -> %s",
             (out / "cohort_distance_adjustments.csv").name)

    # Between-habitat check: if habitats whose cohorts are MORE similar
    # transfer WORSE, the distance hypothesis is contradicted at the group
    # level as well as being unsupported within habitats.
    hm = (d.groupby("habitat")
            .agg(dist=("centroid_distance", "median"),
                 rho=("median_spearman", "median")))
    if len(hm) >= 3:
        rh = spearmanr(hm["dist"], hm["rho"])
        LOG.info("Between habitats: median distance vs median rho, rho=%.3f "
                 "(n=%d habitats)", rh.statistic, len(hm))
        LOG.info("%s", hm.round(3).to_string())

    LOG.info("=" * 68)
    if np.isfinite(pr) and pr < -0.2 and pp < 0.05:
        LOG.info("SUPPORTED AS HYPOTHESISED: cohorts further from the training "
                 "data are predicted less well, holding within habitats and "
                 "after controlling for habitat identity.")
    elif np.isfinite(pr) and pr > 0.2 and pp < 0.05:
        LOG.warning("SIGNIFICANT BUT OPPOSITE IN SIGN: cohorts further from the "
                    "training data are predicted BETTER (rho=%.3f, P=%.3g). "
                    "This is not the hypothesised mechanism. Distant cohorts "
                    "also tend to have more variable resistomes, and per-fold "
                    "rank correlation rises with outcome variance, so this "
                    "association should be reported alongside that confound "
                    "rather than as support for a distance effect.", pr, pp)
    elif np.isfinite(pr) and pp >= 0.05:
        LOG.warning("NOT SUPPORTED: dissimilarity does not explain "
                    "transferability once habitat is accounted for. The "
                    "habitat differences require another explanation, and the "
                    "mechanistic claim should be withdrawn.")
    else:
        LOG.warning("UNEXPECTED DIRECTION: inspect before interpreting.")

    if len(per_hab):
        per_hab.to_csv(out / "fold_heterogeneity_by_habitat.csv", index=False)
        print("\n" + per_hab.to_string(index=False,
                                       float_format=lambda v: f"{v:.4f}"))

    # ---- figure ----------------------------------------------------------
    ps.apply_style()
    fig, ax = ps.multipanel(2, 2, panel_height=3.0)
    habs = sorted(d["habitat"].unique())
    col = {h: ps.SERIES[i % len(ps.SERIES)] for i, h in enumerate(habs)}
    PRETTY = {"ocean": "marine"}

    ps.no_grid(ax[0])
    for h in habs:
        g = d[d["habitat"] == h]
        ax[0].scatter(g["centroid_distance"], g["median_spearman"], s=26,
                      alpha=0.75, color=col[h], edgecolors="white",
                      linewidths=0.6, label=PRETTY.get(h, h))
    if len(d) > 5:
        z = np.polyfit(d["centroid_distance"], d["median_spearman"], 1)
        xs = np.linspace(d["centroid_distance"].min(),
                         d["centroid_distance"].max(), 50)
        ax[0].plot(xs, np.polyval(z, xs), ls="--", lw=1.2,
                   color=ps.PALETTE["ink"])
    ax[0].axhline(0, lw=0.8, color=ps.PALETTE["neutral"])
    ax[0].legend(fontsize=7)
    ax[0].set_title(f"pooled $\\rho$ = {r_all.statistic:.2f}, "
                    f"partial (habitat held constant) = {pr:.2f}",
                    fontweight="bold", fontsize=8)
    ps.label_axes(ax[0], "Distance from training cohorts",
                  r"Cross-cohort $\rho$")

    if len(per_hab):
        o = per_hab.sort_values("rho_distance")
        bars = ax[1].barh(np.arange(len(o)), o["rho_distance"], height=0.6,
                          color=[col[h] for h in o["habitat"]], linewidth=0)
        ps.bar_values(ax[1], bars, "{:+.2f}", horizontal=True)
        ax[1].axvline(0, lw=0.9, color=ps.PALETTE["ink"])
        ax[1].set_yticks(np.arange(len(o)))
        ax[1].set_yticklabels([f"{r.habitat} (n={r.n_folds})"
                               for r in o.itertuples()], fontsize=7)
        ps.grid_axis(ax[1], "x"); ps.despine(ax[1], left=True)
        ax[1].tick_params(axis="y", length=0)
        ps.label_axes(ax[1], r"Within-habitat $\rho$", "Habitat")
        ax[1].set_title("Negative means dissimilar cohorts predict worse",
                        fontweight="bold", fontsize=8)

    ps.no_grid(ax[2])
    for h in habs:
        g = d[d["habitat"] == h]
        ax[2].scatter(g["n_test"], g["median_spearman"], s=26, alpha=0.75,
                      color=col[h], edgecolors="white", linewidths=0.6)
    ax[2].set_xscale("log")
    ax[2].set_title(f"control: cohort size, $\\rho$ = {r_n.statistic:.2f}",
                    fontweight="bold", fontsize=8)
    ps.label_axes(ax[2], "Held-out cohort size (log)", r"Cross-cohort $\rho$")

    med = (d.groupby("habitat")
             .agg(rho=("median_spearman", "median"),
                  dist=("centroid_distance", "median")).reindex(habs))
    ps.no_grid(ax[3])
    ax[3].scatter(med["dist"], med["rho"], s=110,
                  c=[col[h] for h in habs], edgecolors="white",
                  linewidths=1.3, zorder=3)
    for h in habs:
        ax[3].annotate(PRETTY.get(h, h), (med.loc[h, "dist"], med.loc[h, "rho"]),
                       textcoords="offset points", xytext=(8, 4), fontsize=7)
    ps.label_axes(ax[3], "Median cohort distance", r"Median habitat $\rho$")
    ax[3].set_title("Habitat means", fontweight="bold", fontsize=8)

    written = ps.save(fig, out, "fig_fold_heterogeneity",
                      ["png", "pdf", "svg"], 600)
    LOG.info("figure -> %s", written[0])
    LOG.info("Note: distance and dispersion are measured in the same centred "
             "log-ratio space used for prediction, so a fold that is far from "
             "the training data is far in exactly the features the model uses.")


if __name__ == "__main__":
    main()
