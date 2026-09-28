"""
s07_figures.py
==============
Every figure for the manuscript, in house style:
multi-panel (2-3 rows), Times New Roman, bold axis labels on every axis.

Figures produced
----------------
fig1_cohort            study, geography, age, depth, disease, attrition
fig2_resistome         richness, load, class composition, depth confounding,
                       relative vs absolute burden
fig3_variance          variance partitioning, unique vs marginal, ordination
fig4_models            model comparison, LOSO-vs-random shift, calibration
fig5_attribution       species x ARG heatmap, genome validation, candidates

Panels whose inputs are missing are blanked rather than crashing the run, so
this is safe to execute at any point in the pipeline.

Usage:  python src/s07_figures.py [--only fig3_variance]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import plotstyle as ps
from common import LOG, load_config, stratum, tag, work_path

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _load(cfg, name: str):
    p = work_path(cfg, name, per_stratum=True)
    if not p.exists():
        p = work_path(cfg, name)
    if not p.exists():
        LOG.warning("Missing %s - dependent panels will be blank.", p.name)
        return None
    return pd.read_parquet(p)


def _boxplot(ax, data, labels, **kw):
    """boxplot with tick labels, across the matplotlib 3.9 rename."""
    try:
        return ax.boxplot(data, tick_labels=labels, **kw)
    except TypeError:
        return ax.boxplot(data, labels=labels, **kw)


def _blank(ax, msg: str) -> None:
    ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes,
            fontsize=9, color=ps.PALETTE["neutral"], style="italic")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def _bar(ax, labels, values, color, cat_label, val_label, rotate=45,
         horizontal=False, maxlen=20, values_fmt=None):
    """Bar chart. `cat_label` names the category axis, `val_label` the value
    axis, whichever orientation is used - the axes swap when horizontal, and
    the labels must swap with them."""
    labels = ps.shorten(labels, maxlen)
    if horizontal:
        y = np.arange(len(labels))
        bars = ax.barh(y, values, color=color, height=0.66, linewidth=0)
        ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=7)
        ax.invert_yaxis()
        ps.grid_axis(ax, "x")
        ps.despine(ax, left=True)
        ax.tick_params(axis="y", length=0)
        if values_fmt:
            ps.bar_values(ax, bars, values_fmt, horizontal=True)
        ps.label_axes(ax, val_label, cat_label)      # x = values, y = categories
    else:
        x = np.arange(len(labels))
        bars = ax.bar(x, values, color=color, width=0.66, linewidth=0)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=rotate,
                           ha="right" if rotate else "center", fontsize=7)
        ps.grid_axis(ax, "y")
        ps.despine(ax, bottom=True)
        ax.tick_params(axis="x", length=0)
        if values_fmt:
            ps.bar_values(ax, bars, values_fmt)
        ps.label_axes(ax, cat_label, val_label)


# ---------------------------------------------------------------------------
# Figure 1 - cohort composition
# ---------------------------------------------------------------------------

def fig1_cohort(cfg) -> None:
    st = _load(cfg, "sample_table.parquet")
    if st is None:
        LOG.info("fig1 skipped: run s02 first.")
        return
    fig, ax = ps.multipanel(3, 2, panel_height=2.6)

    if st is None:
        for a in ax:
            _blank(a, "sample_table.parquet not built")
    else:
        if "study" in st.columns:
            top = st["study"].value_counts().head(15)
            _bar(ax[0], [s[:26] for s in top.index], top.values,
                 ps.PALETTE["primary"], "Study", "Number of samples", horizontal=True)
            ax[0].set_title(f"{st['study'].nunique()} studies", fontweight="bold", fontsize=9)
        else:
            _blank(ax[0], "no study column")

        if "country" in st.columns:
            top = st["country"].value_counts().head(15)
            _bar(ax[1], [str(s)[:18] for s in top.index], top.values,
                 ps.PALETTE["tertiary"], "Country", "Number of samples", horizontal=True)
        else:
            _blank(ax[1], "no country column")

        if "age" in st.columns:
            a = pd.to_numeric(st["age"], errors="coerce").dropna()
            if len(a):
                ax[2].hist(a, bins=40, color=ps.PALETTE["accent"], linewidth=0)
                ps.label_axes(ax[2], "Host age (years)", "Number of samples")
            else:
                _blank(ax[2], "age column empty")
        else:
            _blank(ax[2], "no age column")

        if "reads_after_qc" in st.columns:
            d = pd.to_numeric(st["reads_after_qc"], errors="coerce").dropna()
            ax[3].hist(np.log10(d[d > 0]), bins=40, color=ps.PALETTE["secondary"], linewidth=0)
            ps.label_axes(ax[3], r"Post-QC reads (log$_{10}$)", "Number of samples")
        else:
            _blank(ax[3], "no depth column")

        if "disease" in st.columns:
            top = st["disease"].astype(str).value_counts().head(12)
            _bar(ax[4], [s[:20] for s in top.index], top.values,
                 ps.PALETTE["primary"], "Health status", "Number of samples",
                 horizontal=True)
        else:
            _blank(ax[4], "no disease column")

        rep = work_path(cfg, "join_report.txt")
        stages, counts = [], []
        if rep.exists():
            for line in rep.read_text(encoding="utf-8").splitlines():
                if line.startswith("[") and ":" in line:
                    lab, val = line.split(":", 1)
                    v = val.strip().replace(",", "").split()[0]
                    if v.isdigit():
                        stages.append(lab.split("]")[-1].strip()[:24])
                        counts.append(int(v))
        if counts:
            _bar(ax[5], stages, counts, ps.PALETTE["neutral"],
                 "Pipeline stage", "Samples retained", horizontal=True)
            ax[5].set_xscale("log")
            ax[5].xaxis.set_major_locator(__import__("matplotlib").ticker.LogLocator(numticks=4))
            ax[5].xaxis.set_minor_locator(__import__("matplotlib").ticker.NullLocator())
        else:
            _blank(ax[5], "join_report.txt not parsed")

    written = ps.save(fig, cfg["paths"]["fig_dir"], f"fig1_cohort{tag(cfg)}",
                      cfg["figures"]["formats"], cfg["figures"]["dpi"])
    LOG.info("fig1 -> %s", written[0])


# ---------------------------------------------------------------------------
# Figure 2 - resistome landscape
# ---------------------------------------------------------------------------

def fig2_resistome(cfg) -> None:
    arg = _load(cfg, "arg_abundance.parquet")
    if arg is None:
        LOG.info("fig2 skipped: run s03 first.")
        return
    st = _load(cfg, "sample_table.parquet")
    absolute = _load(cfg, "arg_absolute.parquet")
    fig, ax = ps.multipanel(3, 2, panel_height=2.6)

    if arg is None:
        for a in ax:
            _blank(a, "arg_abundance.parquet not built")
    else:
        pseudo = np.log(cfg["normalisation"]["arg_pseudocount"])
        detected = (arg > pseudo + 1e-9)
        richness = detected.sum(axis=1)
        load = np.log10(np.exp(arg).sum(axis=1) + 1e-12)

        ax[0].hist(richness, bins=40, color=ps.PALETTE["primary"], linewidth=0)
        ps.label_axes(ax[0], "ARG richness (genes detected)", "Number of samples")

        ax[1].hist(load, bins=40, color=ps.PALETTE["secondary"], linewidth=0)
        ps.label_axes(ax[1], r"Total ARG load (log$_{10}$, rRNA-normalised)",
                      "Number of samples")

        if st is not None and "country" in st.columns:
            s = st.set_index("sample").reindex(arg.index)
            df = pd.DataFrame({"country": s["country"].astype(str), "load": load.values})
            keep = df["country"].value_counts().head(10).index
            df = df[df["country"].isin(keep)]
            order = df.groupby("country")["load"].median().sort_values().index
            ax[2].boxplot([df.loc[df["country"] == c, "load"].values for c in order],
                          vert=False, widths=0.6, showfliers=False,
                          patch_artist=True,
                          boxprops=dict(facecolor=ps.PALETTE["light"], linewidth=0.7),
                          medianprops=dict(color=ps.PALETTE["secondary"], linewidth=1.4))
            ax[2].set_yticklabels([str(c)[:16] for c in order])
            ps.label_axes(ax[2], r"Total ARG load (log$_{10}$)", "Country")
        else:
            _blank(ax[2], "no country data")

        top = np.exp(arg).mean(axis=0).sort_values(ascending=False).head(20)
        _bar(ax[3], list(top.index), top.values, ps.PALETTE["tertiary"],
             "ARG gene", "Mean normalised abundance", horizontal=True)
        ax[3].set_xscale("log")
        ax[3].xaxis.set_major_locator(__import__("matplotlib").ticker.LogLocator(numticks=4))
        ax[3].xaxis.set_minor_locator(__import__("matplotlib").ticker.NullLocator())

        if st is not None and "reads_after_qc" in st.columns:
            s = st.set_index("sample").reindex(arg.index)
            d = pd.to_numeric(s["reads_after_qc"], errors="coerce")
            ok = d.notna() & (d > 0)
            ax[4].scatter(np.log10(d[ok]), richness[ok.values], s=3, alpha=0.25,
                          color=ps.PALETTE["neutral"], edgecolors="none")
            if ok.sum() > 10:
                r = np.corrcoef(np.log10(d[ok]), richness[ok.values])[0, 1]
                ax[4].set_title(f"Pearson r = {r:.2f}", fontweight="bold", fontsize=9)
            ps.label_axes(ax[4], r"Post-QC reads (log$_{10}$)", "ARG richness")
        else:
            _blank(ax[4], "no depth data")

        if absolute is not None:
            shared = absolute.index.intersection(arg.index)
            rel = load.loc[shared]
            ab = np.log10(np.exp(absolute.loc[shared]).sum(axis=1) + 1e-12)
            ax[5].scatter(rel, ab, s=3, alpha=0.25,
                          color=ps.PALETTE["primary"], edgecolors="none")
            ps.label_axes(ax[5], r"Relative ARG load (log$_{10}$)",
                          r"Absolute ARG burden (log$_{10}$)")
        else:
            _blank(ax[5], "microbial load not available")

    written = ps.save(fig, cfg["paths"]["fig_dir"], f"fig2_resistome{tag(cfg)}",
                      cfg["figures"]["formats"], cfg["figures"]["dpi"])
    LOG.info("fig2 -> %s", written[0])


# ---------------------------------------------------------------------------
# Figure 3 - variance partitioning
# ---------------------------------------------------------------------------

def fig3_variance(cfg) -> None:
    vp = _load(cfg, "vp_results.parquet")
    if vp is None:
        LOG.info("fig3 skipped for stratum '%s': run s04 first.", stratum(cfg))
        return
    arg = _load(cfg, "arg_abundance.parquet")
    taxa = _load(cfg, "taxa_clr.parquet")
    fig, ax = ps.multipanel(2, 2, panel_height=2.9)

    if vp is None:
        for a in ax:
            _blank(a, "vp_results.parquet not built")
    else:
        v = vp.sort_values("R2_adj", ascending=True)
        _bar(ax[0], list(v["block"]), 100 * v["R2_adj"], ps.PALETTE["primary"],
             "Explanatory block", "Adjusted variance explained (%)", horizontal=True)

        idx = np.arange(len(vp))
        w = 0.38
        ax[1].barh(idx - w / 2, 100 * vp["R2_adj"], height=w,
                   color=ps.PALETTE["light"], linewidth=0, label="Marginal")
        ax[1].barh(idx + w / 2, 100 * vp["unique_R2_adj"], height=w,
                   color=ps.PALETTE["secondary"], linewidth=0, label="Unique share of total")
        ax[1].set_yticks(idx); ax[1].set_yticklabels(vp["block"])
        ax[1].invert_yaxis()
        ax[1].legend(loc="lower right")
        ps.grid_axis(ax[1], "x"); ps.despine(ax[1], left=True)
        ax[1].tick_params(axis="y", length=0)
        ps.label_axes(ax[1], "Variance explained (%)", "Explanatory block")

        if arg is not None and taxa is not None:
            from sklearn.decomposition import PCA
            shared = sorted(set(arg.index) & set(taxa.index))
            if len(shared) > 20:
                a = PCA(n_components=2, random_state=0).fit_transform(
                    (arg.loc[shared] - arg.loc[shared].mean()).values)
                t = PCA(n_components=1, random_state=0).fit_transform(
                    (taxa.loc[shared] - taxa.loc[shared].mean()).values)
                sc = ax[2].scatter(a[:, 0], a[:, 1], c=t[:, 0], s=4, alpha=0.6,
                                   cmap=ps.diverging(), edgecolors="none")
                cb = fig.colorbar(sc, ax=ax[2]); cb.set_label("Microbiome PC1", fontweight="bold")
                ps.label_axes(ax[2], "Resistome PC1", "Resistome PC2")
            else:
                _blank(ax[2], "too few samples")
        else:
            _blank(ax[2], "matrices not built")

        ax[3].scatter(100 * vp["R2_adj"], 100 * vp["unique_R2_adj"],
                      s=64, color=ps.PALETTE["tertiary"], edgecolors="white", linewidths=1.0, zorder=3)
        for _, r in vp.iterrows():
            ax[3].annotate(r["block"], (100 * r["R2_adj"], 100 * r["unique_R2_adj"]),
                           textcoords="offset points", xytext=(5, 4), fontsize=8)
        lim = max(100 * vp["R2_adj"].max(), 1) * 1.15
        ax[3].plot([0, lim], [0, lim], ls="--", lw=0.9, color=ps.PALETTE["neutral"])
        ps.label_axes(ax[3], "Marginal variance explained (%)",
                      "Unique variance explained (%)")

    written = ps.save(fig, cfg["paths"]["fig_dir"], f"fig3_variance{tag(cfg)}",
                      cfg["figures"]["formats"], cfg["figures"]["dpi"])
    LOG.info("fig3 -> %s", written[0])


# ---------------------------------------------------------------------------
# Figure 4 - model performance
# ---------------------------------------------------------------------------

def fig4_models(cfg) -> None:
    """Model comparison. Rank correlation alone hides the result: the five
    models differ by <0.06 in Spearman but by >0.19 in calibration, and only
    the held-out studies give an estimate free of selection bias."""
    m = _load(cfg, "cv_metrics.parquet")
    if m is None:
        LOG.info("fig4 skipped for stratum '%s': run s05 first.", stratum(cfg))
        return
    p = _load(cfg, "cv_predictions.parquet")
    hold = _load(cfg, "holdout_metrics.parquet")
    fig, ax = ps.multipanel(3, 2, panel_height=2.7)

    if m is None:
        for a in ax:
            _blank(a, "cv_metrics.parquet not built")
    else:
        m_rho = m[m["model"] != "mean"]
        r2col = "r2_centered" if "r2_centered" in m.columns else "r2"

        def grouped_bars(a, piv, ylab, hline=None):
            x = np.arange(len(piv))
            w = 0.8 / max(len(piv.columns), 1)
            for i, sch in enumerate(piv.columns):
                a.bar(x + i * w - 0.4 + w / 2, piv[sch].values, width=w,
                      label=str(sch).replace("_", " "),
                      color=ps.SERIES[i % len(ps.SERIES)], linewidth=0)
            a.set_xticks(x)
            a.set_xticklabels(piv.index, rotation=20, ha="right")
            if hline is not None:
                a.axhline(hline, ls="--", lw=0.9, color=ps.PALETTE["neutral"])
            a.legend(fontsize=7)
            ps.grid_axis(a, "y"); ps.despine(a, bottom=True)
            a.tick_params(axis="x", length=0)
            ps.label_axes(a, "Model", ylab)

        # (A) rank correlation - where the models look alike
        grouped_bars(ax[0],
                     m_rho.groupby(["model", "scheme"])["spearman"].median()
                          .unstack("scheme"),
                     r"Median Spearman $\rho$")

        # (B) calibration - where they do not
        grouped_bars(ax[1],
                     m_rho.groupby(["model", "scheme"])[r2col].mean()
                          .unstack("scheme"),
                     "Mean centred $R^2$", hline=0.0)
        ax[1].set_title("Below zero: worse than the cohort mean",
                        fontweight="bold", fontsize=8)

        # (C) held-out studies, the unbiased estimate
        if hold is not None and len(hold):
            h = hold[hold["model"] != "mean"]
            hc = "r2_centered" if "r2_centered" in h.columns else "r2"
            g = (h.groupby("model")
                   .agg(rho=("spearman", "median"), r2=(hc, "mean"))
                   .sort_values("rho"))
            y = np.arange(len(g))
            ax[2].barh(y - 0.2, g["rho"], height=0.36, label=r"$\rho$",
                       color=ps.PALETTE["primary"], linewidth=0)
            ax[2].barh(y + 0.2, g["r2"], height=0.36, label="centred $R^2$",
                       color=ps.PALETTE["secondary"], linewidth=0)
            ax[2].axvline(0, lw=0.9, color=ps.PALETTE["neutral"])
            ax[2].set_yticks(y); ax[2].set_yticklabels(g.index)
            ax[2].legend(fontsize=7)
            ps.grid_axis(ax[2], "x"); ps.despine(ax[2], left=True)
            ax[2].tick_params(axis="y", length=0)
            ps.label_axes(ax[2], "Held-out performance", "Model")
            ax[2].set_title("Eight studies, scored once", fontweight="bold", fontsize=8)
        else:
            _blank(ax[2], "holdout_metrics.parquet not built")

        # (D) the generalisation gap, per gene, for the best model
        best = m_rho.groupby("model")["spearman"].median().idxmax()
        sub = m_rho[m_rho["model"] == best]
        schemes = list(sub["scheme"].unique())
        if len(schemes) >= 2 and "random" in schemes:
            other = [s_ for s_ in schemes if s_ != "random"][0]
            a_ = sub[sub["scheme"] == "random"].groupby("gene")["spearman"].median()
            b_ = sub[sub["scheme"] == other].groupby("gene")["spearman"].median()
            j = a_.index.intersection(b_.index)
            ps.no_grid(ax[3])
            ax[3].scatter(a_[j], b_[j], s=9, alpha=0.55,
                          color=ps.PALETTE["primary"], edgecolors="none")
            lo = float(min(a_[j].min(), b_[j].min(), 0))
            ax[3].plot([lo, 1], [lo, 1], ls="--", lw=0.9, color=ps.PALETTE["neutral"])
            ax[3].set_title(f"{best}: {(b_[j] < a_[j]).mean()*100:.0f}% of genes degrade",
                            fontweight="bold", fontsize=8)
            ps.label_axes(ax[3], r"Random-split $\rho$",
                          f"{other.replace('_',' ')} " + r"$\rho$")
        else:
            _blank(ax[3], "need two CV schemes")

        # (E) per-gene distributions under cohort shift
        loso = m_rho[m_rho["scheme"] != "random"]
        if len(loso):
            order = (loso.groupby("model")["spearman"].median()
                          .sort_values().index.tolist())
            data = [loso.loc[loso["model"] == mm, "spearman"].dropna().values
                    for mm in order]
            bp = ax[4].boxplot(data, showfliers=False, patch_artist=True, widths=0.6)
            for box in bp["boxes"]:
                box.set(facecolor=ps.PALETTE["light"], linewidth=0.7)
            for med in bp["medians"]:
                med.set(color=ps.PALETTE["secondary"], linewidth=1.4)
            ax[4].set_xticklabels(order, rotation=25, ha="right", fontsize=7)
            ps.grid_axis(ax[4], "y")
            ps.label_axes(ax[4], "Model", r"Per-gene $\rho$ (cohort shift)")

        # (F) calibration of the best model
        if p is not None:
            q = p[p["model"] == best]
            q = q[q["scheme"] != "random"] if (q["scheme"] != "random").any() else q
            floor = float(np.log(cfg["normalisation"]["arg_pseudocount"]))
            q = q[q["observed"] > floor + 1e-6]
            if len(q) > 50:
                ps.no_grid(ax[5])
                hb = ax[5].hexbin(q["observed"], q["predicted"], gridsize=45,
                                  cmap=ps.sequential(), mincnt=1, linewidths=0)
                cb = fig.colorbar(hb, ax=ax[5]); cb.set_label("Count", fontweight="bold")
                lo = float(min(q["observed"].min(), q["predicted"].min()))
                hi = float(max(q["observed"].max(), q["predicted"].max()))
                ax[5].plot([lo, hi], [lo, hi], ls="--", lw=0.9, color="white")
                ps.label_axes(ax[5], "Observed log ARG abundance",
                              "Predicted log ARG abundance")
            else:
                _blank(ax[5], "too few predictions")
        else:
            _blank(ax[5], "cv_predictions.parquet missing")

    written = ps.save(fig, cfg["paths"]["fig_dir"], f"fig4_models{tag(cfg)}",
                      cfg["figures"]["formats"], cfg["figures"]["dpi"])
    LOG.info("fig4 -> %s", written[0])


# ---------------------------------------------------------------------------
# Figure 5 - attribution and genome validation
# ---------------------------------------------------------------------------

def fig5_attribution(cfg) -> None:
    attr = _load(cfg, "attribution.parquet")
    if attr is None:
        # Writing a blank figure is worse than writing none: it looks like a
        # rendering fault and has to be re-diagnosed every time it is seen.
        LOG.info("fig5 skipped for stratum '%s': attribution has not been "
                 "computed. Run s06_attribution.py with --genome-evidence for "
                 "this stratum if the figure is wanted.", stratum(cfg))
        return
    tabdir = Path(cfg["paths"]["table_dir"])
    fig, ax = ps.multipanel(2, 2, panel_height=3.0)

    if True:
        sp = attr.abs().sum(axis=1).sort_values(ascending=False).head(25).index
        gn = attr.abs().sum(axis=0).sort_values(ascending=False).head(25).index
        sub = attr.loc[sp, gn]
        v = float(np.abs(sub.values).max())
        ps.no_grid(ax[0])
        im = ax[0].imshow(sub.values, aspect="auto", cmap=ps.diverging(),
                          vmin=-v, vmax=v)
        ax[0].set_xticks(range(len(gn)))
        ax[0].set_xticklabels(gn, rotation=90, fontsize=5)
        ax[0].set_yticks(range(len(sp)))
        ax[0].set_yticklabels([str(s)[:26] for s in sp], fontsize=5)
        cb = fig.colorbar(im, ax=ax[0]); cb.set_label("Attribution", fontweight="bold")
        ps.label_axes(ax[0], "ARG gene", "Species")

        vals = attr.values.ravel()
        ax[1].hist(vals, bins=80, color=ps.PALETTE["primary"],
                   edgecolor="none", log=True)
        ax[1].axvline(np.quantile(vals, 0.99), ls="--", lw=1.1,
                      color=ps.PALETTE["secondary"], label="99th percentile")
        ax[1].legend(fontsize=7)
        ps.label_axes(ax[1], "Attribution value", "Number of species-gene pairs")

        val = tabdir / "attribution_validation.csv"
        if val.exists():
            d = pd.read_csv(val)
            _bar(ax[2], list(d["category"]), d["n"], ps.PALETTE["tertiary"],
                 "Validation category", "Species-gene pairs", rotate=25)
            ax[2].set_yscale("log")
        else:
            _blank(ax[2], "run s06 with --genome-evidence")

        unex = tabdir / "unexplained_associations.csv"
        if unex.exists():
            d = pd.read_csv(unex).head(20)
            lab = [f"{s[:20]} - {g}" for s, g in zip(d["species"], d["gene"])]
            _bar(ax[3], lab, d["attribution"], ps.PALETTE["secondary"],
                 "Species - ARG pair", "Attribution", horizontal=True)
            ax[3].tick_params(axis="y", labelsize=6)
        else:
            _blank(ax[3], "no unexplained associations table")

    written = ps.save(fig, cfg["paths"]["fig_dir"], f"fig5_attribution{tag(cfg)}",
                      cfg["figures"]["formats"], cfg["figures"]["dpi"])
    LOG.info("fig5 -> %s", written[0])


FIGS = {
    "fig1_cohort": fig1_cohort,
    "fig2_resistome": fig2_resistome,
    "fig3_variance": fig3_variance,
    "fig4_models": fig4_models,
    "fig5_attribution": fig5_attribution,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=list(FIGS), default=None)
    args = ap.parse_args()
    cfg = load_config()
    ps.apply_style()
    for name, fn in FIGS.items():
        if args.only and name != args.only:
            continue
        LOG.info("building %s", name)
        fn(cfg)
    plt.close("all")
    LOG.info("Figures written to %s", cfg["paths"]["fig_dir"])


if __name__ == "__main__":
    main()
