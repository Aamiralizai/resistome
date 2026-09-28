"""
s10_compare_strata.py
=====================
Puts the adult, infant and combined analyses side by side.

Why stratify at all: infant gut communities are structurally different from
adult ones - low diversity, Enterobacteriaceae-dominated - and the preterm
NICU cohorts carry heavy antibiotic exposure. Analysed together, taxonomy
looks strongly predictive of the resistome partly because it separates infants
from adults. That is a real signal, but it is not the claim the paper makes,
and a reviewer will ask.

Analysed separately, the CONTRAST becomes a result in its own right. If
taxonomy explains substantially more of the infant resistome than the adult
one, "the infant gut resistome is compositionally determined in a way the
adult resistome is not" is a sharper claim than either stratum alone supports.

Run each stratum first, then this:
    for s in adult infant all; do
      sed -i "s/stratum: .*/stratum: $s/" config.yaml
      python src/s04_variance_partition.py
      python src/s05_models.py
    done
    python src/s10_compare_strata.py

Outputs:
    tables/stratum_comparison_variance.csv
    tables/stratum_comparison_models.csv
    figures/fig7_strata.png / .pdf

Usage:  python src/s10_compare_strata.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import plotstyle as ps
from common import LOG, load_config

STRATA = ["adult", "infant", "child", "adolescent", "all"]


def collect(cfg, stem: str) -> dict[str, pd.DataFrame]:
    work = Path(cfg["paths"]["work_dir"])
    out = {}
    for s in STRATA:
        p = work / f"{stem}_{s}.parquet"
        if p.exists():
            out[s] = pd.read_parquet(p)
    return out


def main() -> None:
    cfg = load_config()
    figdir = Path(cfg["paths"]["fig_dir"])
    tabdir = Path(cfg["paths"]["table_dir"])

    vp = collect(cfg, "vp_results")
    cv = collect(cfg, "cv_metrics")
    if not vp:
        raise SystemExit(
            "No per-stratum results found. Run s04/s05 with analysis.stratum set "
            "to adult, then infant, then all.")
    LOG.info("Strata with variance results: %s", list(vp))
    LOG.info("Strata with model results   : %s", list(cv))

    # ---- variance comparison --------------------------------------------
    rows = []
    for s, d in vp.items():
        for _, r in d.iterrows():
            rows.append({"stratum": s, "block": r["block"],
                         "marginal_R2adj": r["R2_adj"],
                         "unique_R2adj": r["unique_R2_adj"]})
    vcmp = pd.DataFrame(rows)
    wide = vcmp.pivot_table(index="block", columns="stratum", values="unique_R2adj")
    wide.to_csv(tabdir / "stratum_comparison_variance.csv")
    print("\nUNIQUE VARIANCE EXPLAINED (adjusted R2) BY STRATUM\n")
    print(wide.to_string(float_format=lambda v: f"{v:.4f}"))

    if {"adult", "infant"} <= set(wide.columns) and "taxonomy" in wide.index:
        a, i = wide.loc["taxonomy", "adult"], wide.loc["taxonomy", "infant"]
        LOG.info("Taxonomy unique R2adj: adult %.4f vs infant %.4f (ratio %.2f)",
                 a, i, i / a if a else np.nan)
        if i > 1.5 * a:
            LOG.info("Infant resistome is substantially more compositionally "
                     "determined - that contrast is a headline, not a nuisance.")
        elif a > 1.5 * i:
            LOG.info("Adult resistome is the more compositionally determined of "
                     "the two, which is the less expected direction - worth "
                     "checking depth and diversity differences before claiming it.")

    # ---- model comparison ------------------------------------------------
    mrows = []
    for s, d in cv.items():
        d = d[d["model"] != "mean"]
        g = (d.groupby(["scheme", "model"])
               .agg(median_spearman=("spearman", "median"),
                    mean_r2_centered=("r2_centered", "mean")
                    if "r2_centered" in d.columns else ("r2", "mean"),
                    n_genes=("gene", "nunique"))
               .reset_index())
        g["stratum"] = s
        mrows.append(g)
    mcmp = pd.concat(mrows, ignore_index=True) if mrows else pd.DataFrame()
    if len(mcmp):
        mcmp.to_csv(tabdir / "stratum_comparison_models.csv", index=False)
        print("\nMODEL PERFORMANCE BY STRATUM\n")
        print(mcmp.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # ---- figure ----------------------------------------------------------
    ps.apply_style()
    fig, ax = ps.multipanel(2, 2, panel_height=3.0)
    strata = [s for s in STRATA if s in vp]
    colours = {s: ps.SERIES[k % len(ps.SERIES)] for k, s in enumerate(strata)}

    blocks = [b for b in ["taxonomy", "study", "disease", "geography",
                          "host", "exposure", "technical", "life_stage"]
              if b in wide.index]
    y = np.arange(len(blocks))
    h = 0.8 / max(len(strata), 1)
    for k, s in enumerate(strata):
        if s not in wide.columns:
            continue
        ax[0].barh(y + k * h - 0.4 + h / 2, 100 * wide.loc[blocks, s].values,
                   height=h, label=s, color=colours[s], linewidth=0)
    ax[0].set_yticks(y); ax[0].set_yticklabels(blocks, fontsize=7)
    ax[0].invert_yaxis(); ax[0].legend(fontsize=7)
    ps.grid_axis(ax[0], "x"); ps.despine(ax[0], left=True)
    ax[0].tick_params(axis="y", length=0)
    ps.label_axes(ax[0], "Unique variance explained (%)", "Explanatory block")

    marg = vcmp.pivot_table(index="block", columns="stratum", values="marginal_R2adj")
    for k, s in enumerate(strata):
        if s not in marg.columns:
            continue
        ax[1].barh(y + k * h - 0.4 + h / 2, 100 * marg.loc[blocks, s].values,
                   height=h, label=s, color=colours[s], linewidth=0)
    ax[1].set_yticks(y); ax[1].set_yticklabels(blocks, fontsize=7)
    ax[1].invert_yaxis()
    ps.grid_axis(ax[1], "x"); ps.despine(ax[1], left=True)
    ax[1].tick_params(axis="y", length=0)
    ps.label_axes(ax[1], "Marginal variance explained (%)", "Explanatory block")

    if len(mcmp):
        sub = mcmp[mcmp["scheme"].str.contains("study")]
        if len(sub):
            piv = sub.pivot_table(index="stratum", columns="model",
                                  values="median_spearman")
            x = np.arange(len(piv))
            w = 0.8 / max(len(piv.columns), 1)
            for k, m in enumerate(piv.columns):
                ax[2].bar(x + k * w - 0.4 + w / 2, piv[m].values, width=w,
                          label=m, color=ps.SERIES[k % len(ps.SERIES)], linewidth=0)
            ax[2].set_xticks(x); ax[2].set_xticklabels(piv.index)
            ax[2].legend(fontsize=7, ncol=2)
            ps.grid_axis(ax[2], "y"); ps.despine(ax[2], bottom=True)
            ax[2].tick_params(axis="x", length=0)
            ps.label_axes(ax[2], "Stratum", r"Median Spearman $\rho$ (LOSO)")
            if piv.shape[1] < 3:
                ax[2].set_title("Only %d models available - rerun s05 with "
                                "--models all" % piv.shape[1],
                                fontweight="bold", fontsize=7,
                                color=ps.PALETTE["secondary"])
        else:
            ax[2].axis("off")

        piv2 = (mcmp.pivot_table(index="stratum", columns="scheme",
                                 values="median_spearman"))
        if piv2.shape[1] >= 2:
            cols = list(piv2.columns)
            ax[3].scatter(piv2[cols[0]], piv2[cols[1]], s=76,
                          color=ps.PALETTE["tertiary"], edgecolors="white",
                          linewidths=1.0, zorder=3)
            for s_, r in piv2.iterrows():
                ax[3].annotate(s_, (r[cols[0]], r[cols[1]]),
                               textcoords="offset points", xytext=(6, 4), fontsize=8)
            lim = float(np.nanmax(piv2.values)) * 1.15
            ax[3].plot([0, lim], [0, lim], ls="--", lw=0.9, color=ps.PALETTE["neutral"])
            ps.label_axes(ax[3], f"{cols[0].replace('_',' ')} " + r"$\rho$",
                          f"{cols[1].replace('_',' ')} " + r"$\rho$")
        else:
            ax[3].axis("off")
    else:
        ax[2].axis("off"); ax[3].axis("off")

    written = ps.save(fig, figdir, "fig7_strata",
                      cfg["figures"]["formats"], cfg["figures"]["dpi"])
    LOG.info("fig7 -> %s", written[0])


if __name__ == "__main__":
    main()
