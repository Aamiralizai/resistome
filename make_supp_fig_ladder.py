#!/usr/bin/env python3
"""Supplementary figure: the three-scheme validation ladder.

Panels
  (a) median per-gene correlation for each model under random, subject-grouped
      and leave-one-study-out cross-validation
  (b) the random-to-leave-one-study-out drop, split into the part explained by
      evaluating on unseen individuals and the part explained by evaluating on
      unseen cohorts
  (c) per-gene correlation under subject-grouped against leave-one-study-out
      for the best model, showing that the ordering holds gene by gene

Reads model_summary_<stratum>.csv and cv_metrics_<stratum>.parquet from the
configured table and work directories. House style throughout: multi-panel,
serif face, bold axis labels.

    python make_supp_fig_ladder.py --tables /path/tables --work /path/work \
        --outdir /path/figures [--stratum adult]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import plotstyle as ps  # noqa: E402

NICE = {"xgb": "XGBoost", "rf": "RF", "ridge": "Ridge", "taxa_nn": "Taxa NN",
        "mlp": "MLP", "ft": "FT-Transformer"}
ORDER = ["random", "subject_grouped", "leave_one_study_out"]
TITLES = {"random": "Random", "subject_grouped": "Subject-grouped",
          "leave_one_study_out": "Leave-one-study-out"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", required=True, type=Path)
    ap.add_argument("--work", required=True, type=Path)
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--stratum", default="adult")
    ap.add_argument("--stem", default="SupplementaryFigure_validation_ladder")
    args = ap.parse_args()

    ps.apply_style()
    st = args.stratum

    ms = pd.read_csv(args.tables / f"model_summary_{st}.csv")
    ms = ms[ms["model"] != "mean"]
    if "subject_grouped" not in set(ms["scheme"]):
        print("ERROR: no subject_grouped rows in model_summary. Run "
              "src/s05_models.py --models all after applying the subject "
              "patch.", file=sys.stderr)
        return 2

    wide = (ms.pivot(index="model", columns="scheme", values="median_spearman")
              .reindex(columns=ORDER))
    wide = wide.sort_values("leave_one_study_out", ascending=False)

    fig, ax = ps.multipanel(nrows=2, ncols=2, panel_width=3.5, panel_height=2.8)

    # (a) grouped bars, one group per model
    x = np.arange(len(wide))
    width = 0.26
    for k, scheme in enumerate(ORDER):
        ax[0].bar(x + (k - 1) * width, wide[scheme].values, width,
                  color=ps.SERIES[k], label=TITLES[scheme],
                  edgecolor="white", linewidth=0.5)
    ax[0].set_xticks(x)
    ax[0].set_xticklabels([NICE.get(m, m) for m in wide.index],
                          rotation=20, ha="right", fontsize=7.5)
    ps.grid_axis(ax[0], "y")
    ps.label_axes(ax[0], "Model", r"Median per-gene $\rho$")
    ax[0].legend(frameon=False, fontsize=7, loc="upper right")
    ax[0].set_title("Performance under three evaluation schemes",
                    fontweight="bold", fontsize=8.5)

    # (b) decomposition of the drop
    drop = wide["random"] - wide["leave_one_study_out"]
    indiv = (wide["random"] - wide["subject_grouped"]) / drop
    cohort = 1 - indiv
    ax[1].barh(x, indiv.values * 100, color=ps.PALETTE["secondary"],
               edgecolor="white", linewidth=0.5, label="Unseen individuals")
    ax[1].barh(x, cohort.values * 100, left=indiv.values * 100,
               color=ps.PALETTE["primary"], edgecolor="white", linewidth=0.5,
               label="Unseen cohorts")
    ax[1].set_yticks(x)
    ax[1].set_yticklabels([NICE.get(m, m) for m in wide.index], fontsize=7.5)
    ax[1].invert_yaxis()
    ax[1].set_xlim(0, 100)
    ps.grid_axis(ax[1], "x")
    ps.label_axes(ax[1], "Share of the drop (%)", "Model")
    ax[1].legend(frameon=False, fontsize=6.5, loc="upper center",
                 bbox_to_anchor=(0.5, -0.22), ncol=2)
    ax[1].set_title("Where the apparent performance goes",
                    fontweight="bold", fontsize=8.5)

    # (c) per-gene, best model
    cv = pd.read_parquet(args.work / f"cv_metrics_{st}.parquet")
    cv = cv[cv["model"] != "mean"]
    best = wide.index[0]
    sub = cv[cv["model"] == best]
    a = sub[sub["scheme"] == "subject_grouped"].groupby("gene")["spearman"].median()
    b = sub[sub["scheme"] == "leave_one_study_out"].groupby("gene")["spearman"].median()
    j = a.index.intersection(b.index)
    ps.no_grid(ax[2])
    ax[2].scatter(a[j], b[j], s=10, alpha=0.6, color=ps.PALETTE["primary"],
                  edgecolors="none")
    lo = float(min(a[j].min(), b[j].min(), 0))
    hi = float(max(a[j].max(), b[j].max()))
    ax[2].plot([lo, hi], [lo, hi], ls="--", lw=0.9, color=ps.PALETTE["neutral"])
    ps.label_axes(ax[2], r"Subject-grouped $\rho$", r"Leave-one-study-out $\rho$")
    ax[2].set_title(f"{NICE.get(best, best)}: "
                    f"{(b[j] < a[j]).mean() * 100:.0f}% of genes degrade further",
                    fontweight="bold", fontsize=8.5)

    # (d) distribution per scheme, best model
    data = [sub.loc[sub["scheme"] == s, "spearman"].dropna().values for s in ORDER]
    bp = ax[3].boxplot(data, showfliers=False, patch_artist=True, widths=0.55)
    for box, colour in zip(bp["boxes"], ps.SERIES):
        box.set(facecolor=colour, alpha=0.45, linewidth=0.7)
    for med in bp["medians"]:
        med.set(color=ps.PALETTE["ink"], linewidth=1.3)
    ax[3].set_xticklabels([TITLES[s] for s in ORDER], rotation=15, ha="right",
                          fontsize=7.5)
    ps.grid_axis(ax[3], "y")
    ps.label_axes(ax[3], "Evaluation scheme", r"Per-gene $\rho$")
    ax[3].set_title(f"Per-gene distribution, {NICE.get(best, best)}",
                    fontweight="bold", fontsize=8.5)

    args.outdir.mkdir(parents=True, exist_ok=True)
    out = ps.save(fig, args.outdir, args.stem, formats=("png", "pdf", "svg"))
    plt.close(fig)
    for p in out:
        print(f"wrote {p}")
    print("\nladder (median per-gene rho):")
    print(wide.round(4).to_string())
    print("\nshare of the drop attributable to unseen individuals:")
    print((indiv * 100).round(1).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
