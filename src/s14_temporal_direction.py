"""
s14_temporal_direction.py
=========================
Does community change precede resistome change, or the reverse?

The within-subject analysis correlates SIMULTANEOUS change, which establishes
coupling but says nothing about order. Ordered samples permit a stronger test:
compare how well composition at time t predicts the resistome at t+1 against
how well the resistome at t predicts composition at t+1.

If composition leads, the first should exceed the second. Symmetry would mean
the two simply co-vary, and the directional language in the manuscript would
have to be withdrawn.

This is cross-prediction, not Granger causality in the strict sense: both are
measured from the same sequencing library, so a shared measurement artefact
affects both directions equally - which is precisely why the comparison
between directions is more informative than either direction alone.

Design
------
Consecutive within-subject pairs (t, t+1) are split by subject, so no subject
contributes to both training and test. Two ridge models are fitted:

    forward   composition(t)  ->  resistome(t+1)
    reverse   resistome(t)    ->  composition(t+1)

Both are compared against the corresponding autoregressive baseline (the same
layer at time t predicting itself at t+1), because a layer that is simply
stable over time will appear predictable regardless of the other layer.

The reported quantity is INCREMENTAL skill over that baseline, which is what
the directional claim actually requires.

Outputs:
    tables/temporal_direction_<stratum>.csv
    figures/fig8_direction_<stratum>.png / .pdf / .svg

Usage:  python src/s14_temporal_direction.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

import plotstyle as ps
from common import (LOG, apply_stratum, harmonise_disease, load_config,
                    namespace_subjects,
                    table_path, tag, work_path)

N_SPLITS = 5
MIN_PAIRS = 200


def build_pairs(st, taxa, arg):
    """Consecutive within-subject sample pairs, ordered in time."""
    rows = []
    for sid, g in st.groupby("subject_id"):
        if "t" in g.columns and g["t"].notna().all():
            g = g.sort_values("t")
        ids = [s for s in g["sample"] if s in taxa.index and s in arg.index]
        for a, b in zip(ids[:-1], ids[1:]):
            rows.append({"subject_id": sid,
                         "study": g["study"].iloc[0] if "study" in g else "NA",
                         "t0": a, "t1": b})
    return pd.DataFrame(rows)


def nested_predict(base, extra, target, groups, seed=42):
    """Grouped CV skill for target ~ base against target ~ base + extra.

    The question is whether one layer adds information ABOUT THE SAME OUTCOME
    beyond that outcome's own past, so both models must predict the identical
    target on identical folds. Subtracting the medians of two models with
    different outcomes - 99 genes against 449 species - compares nothing.
    """
    rng = np.random.default_rng(seed)
    uniq = pd.unique(groups)
    folds = np.array_split(rng.permutation(uniq), N_SPLITS)
    rows_base, rows_full = [], []
    for f in folds:
        te = np.isin(groups, f)
        tr = ~te
        if te.sum() < 20 or tr.sum() < 50:
            continue
        ys = StandardScaler().fit(target[tr])
        Yt = ys.transform(target[tr])
        obs = target[te]
        for name, X, store in (("base", base, rows_base),
                               ("full", np.hstack([base, extra]), rows_full)):
            xs = StandardScaler().fit(X[tr])
            m = Ridge(alpha=10.0).fit(xs.transform(X[tr]), Yt)
            pred = ys.inverse_transform(m.predict(xs.transform(X[te])))
            store.append([
                spearmanr(obs[:, j], pred[:, j]).statistic
                if np.std(obs[:, j]) > 1e-9 and np.std(pred[:, j]) > 1e-9 else np.nan
                for j in range(obs.shape[1])])
    if not rows_base:
        return np.array([]), np.array([])
    return (np.nanmedian(np.array(rows_base), axis=0),
            np.nanmedian(np.array(rows_full), axis=0))


def cross_predict(Xt, Yt1, groups, seed=42):
    """Grouped cross-validated prediction, returning per-target Spearman."""
    rng = np.random.default_rng(seed)
    uniq = pd.unique(groups)
    folds = np.array_split(rng.permutation(uniq), N_SPLITS)
    per_target = []
    for f in folds:
        te = np.isin(groups, f)
        tr = ~te
        if te.sum() < 20 or tr.sum() < 50:
            continue
        xs = StandardScaler().fit(Xt[tr])
        ys = StandardScaler().fit(Yt1[tr])
        m = Ridge(alpha=10.0).fit(xs.transform(Xt[tr]), ys.transform(Yt1[tr]))
        pred = ys.inverse_transform(m.predict(xs.transform(Xt[te])))
        obs = Yt1[te]
        per_target.append([
            spearmanr(obs[:, j], pred[:, j]).statistic
            if np.std(obs[:, j]) > 1e-9 and np.std(pred[:, j]) > 1e-9 else np.nan
            for j in range(obs.shape[1])])
    return np.nanmedian(np.array(per_target), axis=0) if per_target else np.array([])


def main() -> None:
    cfg = load_config()
    st = namespace_subjects(harmonise_disease(apply_stratum(cfg, pd.read_parquet(
        work_path(cfg, "sample_table.parquet")))))
    taxa = pd.read_parquet(work_path(cfg, "taxa_clr.parquet"))
    arg = pd.read_parquet(work_path(cfg, "arg_abundance.parquet"))

    shared = sorted(set(taxa.index) & set(arg.index) & set(st["sample"].astype(str)))
    taxa, arg = taxa.loc[shared], arg.loc[shared]
    st = st[st["sample"].isin(shared)].copy()
    st["subject_id"] = st["subject_id"].astype(str)
    st["t"] = pd.to_numeric(st.get("timepoint"), errors="coerce")

    pairs = build_pairs(st, taxa, arg)
    LOG.info("Consecutive within-subject pairs: %d from %d subjects, %d studies",
             len(pairs), pairs["subject_id"].nunique(), pairs["study"].nunique())
    if len(pairs) < MIN_PAIRS:
        raise SystemExit(f"Only {len(pairs)} ordered pairs in stratum "
                         f"'{cfg['analysis']['stratum']}' - try stratum: all.")

    T0, T1 = pairs["t0"].values, pairs["t1"].values
    Xc0, Xc1 = taxa.loc[T0].values, taxa.loc[T1].values      # composition
    Xr0, Xr1 = arg.loc[T0].values, arg.loc[T1].values        # resistome
    grp = pairs["subject_id"].values

    LOG.info("=" * 66)
    LOG.info("CROSS-PREDICTION BETWEEN TIME POINTS (grouped by subject)")

    # Resistome at t+1: does composition add to the resistome's own past?
    r_base, r_full = nested_predict(Xr0, Xc0, Xr1, grp)
    # Composition at t+1: does the resistome add to composition's own past?
    c_base, c_full = nested_predict(Xc0, Xr0, Xc1, grp)

    def med(a):
        return float(np.nanmedian(a)) if len(a) else np.nan

    LOG.info("  resistome(t+1) ~ resistome(t)                : %.3f", med(r_base))
    LOG.info("  resistome(t+1) ~ resistome(t) + composition(t): %.3f", med(r_full))
    LOG.info("  composition(t+1) ~ composition(t)                : %.3f", med(c_base))
    LOG.info("  composition(t+1) ~ composition(t) + resistome(t) : %.3f", med(c_base))

    gain_r = med(r_full) - med(r_base)      # composition -> future resistome
    gain_c = med(c_full) - med(c_base)      # resistome  -> future composition
    LOG.info("  gain from adding composition to the resistome model : %+.4f", gain_r)
    LOG.info("  gain from adding the resistome to the composition model: %+.4f", gain_c)

    # Paired only WITHIN a direction, where base and full share the target.
    p_r = p_c = np.nan
    if len(r_base) and len(r_full):
        ok = ~(np.isnan(r_base) | np.isnan(r_full))
        if ok.sum() > 10:
            try:
                p_r = wilcoxon(r_full[ok], r_base[ok]).pvalue
            except ValueError:
                pass
    if len(c_base) and len(c_full):
        ok = ~(np.isnan(c_base) | np.isnan(c_full))
        if ok.sum() > 10:
            try:
                p_c = wilcoxon(c_full[ok], c_base[ok]).pvalue
            except ValueError:
                pass
    LOG.info("  paired within-direction tests: resistome model P=%.3g, "
             "composition model P=%.3g", p_r, p_c)

    LOG.info("=" * 66)
    if gain_r > 0.01 and (np.isnan(p_r) or p_r < 0.05) and gain_r > gain_c:
        LOG.info("DIRECTIONAL: composition adds information about the future "
                 "resistome beyond the resistome's own past, and more than the "
                 "converse. Supports, but does not prove, that community "
                 "change leads.")
    elif gain_r <= 0.01 and gain_c <= 0.01:
        LOG.warning("UNINFORMATIVE: neither layer adds meaningfully to the "
                    "other's own past. No directional claim is supported; "
                    "describe the relationship as contemporaneous co-variation.")
    else:
        LOG.warning("MIXED or REVERSED: inspect the per-direction gains before "
                    "making any claim.")

    res = pd.DataFrame([
        {"model": "resistome(t+1) ~ resistome(t)", "median_rho": med(r_base),
         "n_targets": int(np.sum(~np.isnan(r_base))), "gain": np.nan, "p": np.nan},
        {"model": "resistome(t+1) ~ resistome(t) + composition(t)",
         "median_rho": med(r_full), "n_targets": int(np.sum(~np.isnan(r_full))),
         "gain": gain_r, "p": p_r},
        {"model": "composition(t+1) ~ composition(t)", "median_rho": med(c_base),
         "n_targets": int(np.sum(~np.isnan(c_base))), "gain": np.nan, "p": np.nan},
        {"model": "composition(t+1) ~ composition(t) + resistome(t)",
         "median_rho": med(c_full), "n_targets": int(np.sum(~np.isnan(c_full))),
         "gain": gain_c, "p": p_c},
    ])
    res.to_csv(table_path(cfg, "temporal_direction.csv"), index=False)
    print("\n" + res.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # ---- figure ----------------------------------------------------------
    ps.apply_style()
    fig, ax = ps.multipanel(2, 2, panel_height=2.9)

    labels = ["resistome model", "composition model"]
    vals = [med(r_full), med(c_full)]
    bars = ax[0].bar(np.arange(2), vals, width=0.55,
                     color=[ps.PALETTE["primary"], ps.PALETTE["secondary"]],
                     linewidth=0)
    ps.bar_values(ax[0], bars, "{:.3f}")
    ax[0].set_xticks(np.arange(2)); ax[0].set_xticklabels(labels, fontsize=8)
    ps.grid_axis(ax[0], "y"); ps.despine(ax[0], bottom=True)
    ax[0].tick_params(axis="x", length=0)
    ps.label_axes(ax[0], "Direction", r"Median $\rho$ at $t{+}1$")

    bars = ax[1].bar(np.arange(2), [gain_r, gain_c], width=0.55,
                     color=[ps.PALETTE["primary"], ps.PALETTE["secondary"]],
                     linewidth=0)
    ps.bar_values(ax[1], bars, "{:+.3f}")
    ax[1].axhline(0, lw=0.9, color=ps.PALETTE["ink"])
    ax[1].set_xticks(np.arange(2)); ax[1].set_xticklabels(labels, fontsize=8)
    ps.grid_axis(ax[1], "y"); ps.despine(ax[1], bottom=True)
    ax[1].tick_params(axis="x", length=0)
    ps.label_axes(ax[1], "Direction", "Skill above persistence")
    ax[1].set_title("Incremental information", fontweight="bold", fontsize=9)

    if len(r_base) and len(r_full):
        ps.no_grid(ax[2])
        ax[2].scatter(r_base, r_full, s=26, alpha=0.65,
                      color=ps.PALETTE["tertiary"], edgecolors="white",
                      linewidths=0.7)
        lim = float(np.nanmax(np.abs(np.concatenate([r_base, r_full])))) * 1.1
        ax[2].plot([-lim, lim], [-lim, lim], ls="--", lw=1.0,
                   color=ps.PALETTE["neutral"])
        ps.label_axes(ax[2], r"ARG $\rho$: resistome past only",
                      r"ARG $\rho$: + composition")
        ax[2].set_title("Same 99 targets, same folds", fontweight="bold", fontsize=8)

    ax[3].hist(r_base[~np.isnan(r_base)], bins=24, alpha=0.65,
               color=ps.PALETTE["primary"], linewidth=0, label="resistome past")
    ax[3].hist(r_full[~np.isnan(r_full)], bins=24, alpha=0.65,
               color=ps.PALETTE["secondary"], linewidth=0, label="+ composition")
    ax[3].legend(fontsize=7)
    ps.grid_axis(ax[3], "y")
    ps.label_axes(ax[3], r"Per-target $\rho$", "Targets")

    written = ps.save(fig, cfg["paths"]["fig_dir"], f"fig8_direction{tag(cfg)}",
                      cfg["figures"]["formats"], cfg["figures"]["dpi"])
    LOG.info("fig8 -> %s", written[0])
    LOG.info("Caveat for the manuscript: both layers derive from the same "
             "sequencing library, so this is cross-prediction rather than "
             "causal inference. The comparison BETWEEN directions is the "
             "informative quantity, not either direction alone.")


if __name__ == "__main__":
    main()
