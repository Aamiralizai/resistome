"""
s11_longitudinal.py
===================
Does the resistome track composition WITHIN an individual over time?

Every result so far is cross-sectional, so every claim is associational: people
with different communities have different resistomes, but that could be driven
by anything that differs between people - genetics, diet, geography, the study
they were enrolled in.

Within-subject analysis removes all of it. Each person acts as their own
control, so any time-invariant confounder is differenced away. If community
change predicts resistome change inside individuals, the relationship is far
harder to explain as confounding.

Three analyses
--------------
1. WITHIN TRANSFORMATION. Subtract each subject's own mean from both the taxa
   and ARG matrices, then partition variance on the residuals. This is the
   econometric fixed-effects estimator: what remains is purely within-person
   variation. Compare with the between-subject partition to see how much of
   the cross-sectional signal survives.

2. PAIRED CHANGE. For consecutive samples from one subject, compute the
   compositional shift (Aitchison distance) and the resistome shift, and test
   whether they correlate - controlling for study and for the time gap, since
   both distances grow with elapsed time regardless of any real coupling.

3. TIME-GAP CONTROL. The correlation is recomputed within narrow time-gap
   bands. If it holds at fixed elapsed time, shared temporal drift is not the
   explanation.

Outputs:
    tables/longitudinal_variance_<stratum>.csv
    tables/longitudinal_paired_change_<stratum>.csv
    work_dir/longitudinal_pairs_<stratum>.parquet

Usage:  python src/s11_longitudinal.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr
from sklearn.preprocessing import StandardScaler

from common import (LOG, apply_stratum, harmonise_disease, load_config,
                    namespace_subjects,
                    stratum, table_path, work_path)
from s04_variance_partition import build_blocks, run_partition, rda_r2

MIN_SUBJECT_SAMPLES = 2
MIN_SUBJECTS = 30


def find_repeats(st: pd.DataFrame) -> pd.DataFrame:
    """Subjects sampled more than once, with a usable time variable."""
    if "subject_id" not in st.columns:
        raise SystemExit("No subject_id column - rerun s02 to retain it.")
    st = st.copy()
    st["subject_id"] = st["subject_id"].astype(str)

    n = st.groupby("subject_id").size()
    repeats = n[n >= MIN_SUBJECT_SAMPLES].index
    sub = st[st["subject_id"].isin(repeats)].copy()
    LOG.info("Subjects with >=%d samples: %d (%d samples of %d)",
             MIN_SUBJECT_SAMPLES, len(repeats), len(sub), len(st))
    if "study" in sub.columns:
        LOG.info("Longitudinal studies: %d", sub["study"].nunique())
        top = sub.groupby("study")["subject_id"].nunique().sort_values(ascending=False)
        LOG.info("Top contributing studies:\n%s", top.head(10).to_string())

    # a time axis: explicit timepoint if present, else collection date
    if "timepoint" in sub.columns and pd.to_numeric(sub["timepoint"],
                                                    errors="coerce").notna().any():
        sub["t"] = pd.to_numeric(sub["timepoint"], errors="coerce")
    elif "collection_date" in sub.columns:
        sub["t"] = pd.to_datetime(sub["collection_date"], errors="coerce").map(
            lambda d: d.toordinal() if pd.notna(d) else np.nan)
    else:
        sub["t"] = np.nan
    LOG.info("Samples with a usable time value: %d / %d",
             int(sub["t"].notna().sum()), len(sub))
    return sub


def within_transform(mat: pd.DataFrame, subject: pd.Series) -> pd.DataFrame:
    """Subtract each subject's own mean - the fixed-effects estimator."""
    g = mat.groupby(subject.reindex(mat.index).values)
    return mat - g.transform("mean")


def paired_changes(taxa: pd.DataFrame, arg: pd.DataFrame,
                   sub: pd.DataFrame) -> pd.DataFrame:
    """Consecutive within-subject pairs and the distance moved by each layer."""
    rows = []
    for sid, g in sub.groupby("subject_id"):
        g = g.sort_values("t") if g["t"].notna().all() else g
        ids = [s for s in g["sample"] if s in taxa.index and s in arg.index]
        if len(ids) < 2:
            continue
        tvals = dict(zip(g["sample"], g["t"]))
        study = g["study"].iloc[0] if "study" in g.columns else "NA"
        for a, b in zip(ids[:-1], ids[1:]):
            d_taxa = float(np.linalg.norm(taxa.loc[a].values - taxa.loc[b].values))
            d_arg = float(np.linalg.norm(arg.loc[a].values - arg.loc[b].values))
            gap = abs((tvals.get(b, np.nan) or np.nan) - (tvals.get(a, np.nan) or np.nan)) \
                if pd.notna(tvals.get(a)) and pd.notna(tvals.get(b)) else np.nan
            rows.append({"subject_id": sid, "study": study, "s1": a, "s2": b,
                         "d_taxa": d_taxa, "d_arg": d_arg, "time_gap": gap})
    return pd.DataFrame(rows)


def main() -> None:
    cfg = load_config()
    st = namespace_subjects(harmonise_disease(apply_stratum(cfg, pd.read_parquet(
        work_path(cfg, "sample_table.parquet")))))
    taxa = pd.read_parquet(work_path(cfg, "taxa_clr.parquet"))
    arg = pd.read_parquet(work_path(cfg, "arg_abundance.parquet"))

    shared = sorted(set(taxa.index) & set(arg.index) & set(st["sample"].astype(str)))
    taxa, arg = taxa.loc[shared], arg.loc[shared]
    st = st[st["sample"].isin(shared)]

    sub = find_repeats(st)
    sub = sub[sub["sample"].isin(shared)]
    if sub["subject_id"].nunique() < MIN_SUBJECTS:
        raise SystemExit(
            f"Only {sub['subject_id'].nunique()} subjects with repeats in stratum "
            f"'{stratum(cfg)}' - too few for a within-subject analysis.")

    idx = sorted(set(sub["sample"]))
    taxa_l, arg_l = taxa.loc[idx], arg.loc[idx]
    subject = sub.set_index("sample")["subject_id"].reindex(idx)

    # ---- 1. within vs between subject variance ---------------------------
    LOG.info("=== 1. within-subject variance partition (n=%d samples, %d subjects) ===",
             len(idx), subject.nunique())
    taxa_w = within_transform(taxa_l, subject)
    arg_w = within_transform(arg_l, subject)

    keep = arg_w.std(axis=0) > 1e-9
    arg_w = arg_w.loc[:, keep]
    LOG.info("ARG genes with within-subject variation: %d / %d",
             int(keep.sum()), len(keep))

    from sklearn.decomposition import PCA
    npc = int(min(50, taxa_w.shape[1] - 1, len(taxa_w) - 2))
    pcs = PCA(n_components=npc, random_state=0).fit_transform(
        StandardScaler().fit_transform(taxa_w.values))
    Yw = StandardScaler().fit_transform(arg_w.values)
    r2_w, r2adj_w, rank_w = rda_r2(Yw, pcs)

    # between-subject comparison on the same samples: subject means only
    taxa_b = taxa_l.groupby(subject.values).mean()
    arg_b = arg_l.groupby(subject.values).mean()
    npc_b = int(min(50, taxa_b.shape[1] - 1, len(taxa_b) - 2))
    pcs_b = PCA(n_components=npc_b, random_state=0).fit_transform(
        StandardScaler().fit_transform(taxa_b.values))
    r2_b, r2adj_b, _ = rda_r2(StandardScaler().fit_transform(arg_b.values), pcs_b)

    LOG.info("Taxonomy explains: WITHIN subjects R2adj=%.4f | BETWEEN subjects R2adj=%.4f",
             r2adj_w, r2adj_b)
    LOG.info("A substantial within-subject value means the relationship is not "
             "driven by stable between-person differences.")

    pd.DataFrame([
        {"level": "within_subject", "n": len(idx), "n_pcs": npc,
         "R2": r2_w, "R2_adj": r2adj_w},
        {"level": "between_subject", "n": len(taxa_b), "n_pcs": npc_b,
         "R2": r2_b, "R2_adj": r2adj_b},
    ]).to_csv(table_path(cfg, "longitudinal_variance.csv"), index=False)

    # ---- 2. paired change ------------------------------------------------
    LOG.info("=== 2. paired within-subject change ===")
    pairs = paired_changes(taxa_l, arg_l, sub)
    if len(pairs) < 50:
        LOG.warning("Only %d consecutive pairs; skipping change analysis.", len(pairs))
        return
    pairs.to_parquet(work_path(cfg, "longitudinal_pairs.parquet", per_stratum=True),
                     index=False)
    LOG.info("Consecutive pairs: %d from %d subjects across %d studies",
             len(pairs), pairs["subject_id"].nunique(), pairs["study"].nunique())

    rho_all = spearmanr(pairs["d_taxa"], pairs["d_arg"])
    LOG.info("Pooled (NOT a valid test): rho=%.3f", rho_all.statistic)
    LOG.warning("The pooled P value is meaningless here: consecutive pairs "
                "overlap, subjects contribute many pairs, and subjects nest "
                "within studies. Clustered estimates follow.")

    # Cluster bootstrap over SUBJECTS, resampled within study, which respects
    # both levels of the hierarchy.
    rng = np.random.default_rng(42)
    by_subj = {k: v for k, v in pairs.groupby("subject_id")}
    subj_study = pairs.groupby("subject_id")["study"].first()
    boots = []
    for _ in range(1000):
        picked = []
        for _stu, subs in subj_study.groupby(subj_study).groups.items():
            pass
        for _stu, g in subj_study.reset_index().groupby("study"):
            ids = g["subject_id"].values
            picked.extend(rng.choice(ids, len(ids), replace=True))
        d = pd.concat([by_subj[i] for i in picked], ignore_index=True)
        if len(d) > 20:
            boots.append(spearmanr(d["d_taxa"], d["d_arg"]).statistic)
    if boots:
        lo, hi = np.percentile(boots, [2.5, 97.5])
        LOG.info("Cluster bootstrap over subjects within studies: rho=%.3f "
                 "(95%% CI %.3f-%.3f, 1000 resamples, %d subjects)",
                 rho_all.statistic, lo, hi, pairs["subject_id"].nunique())

    # per study, so the pooled value is not a Simpson's-paradox artefact
    rows = []
    for study, g in pairs.groupby("study"):
        if len(g) < 20:
            continue
        r = spearmanr(g["d_taxa"], g["d_arg"])
        rows.append({"study": study, "n_pairs": len(g),
                     "spearman": r.statistic, "p_value": r.pvalue})
    per_study = pd.DataFrame(rows).sort_values("spearman", ascending=False)
    if len(per_study):
        LOG.info("Per-study correlations: median rho=%.3f, positive in %d/%d studies",
                 float(per_study["spearman"].median()),
                 int((per_study["spearman"] > 0).sum()), len(per_study))

        # Random-effects meta-analysis of Fisher-z transformed study estimates.
        # The study is the unit of replication; this is the defensible summary.
        z = np.arctanh(per_study["spearman"].clip(-0.999, 0.999).values)
        v = 1.0 / np.maximum(per_study["n_pairs"].values - 3, 1)
        w = 1 / v
        mu_f = np.sum(w * z) / np.sum(w)
        Q = float(np.sum(w * (z - mu_f) ** 2))
        df_ = len(z) - 1
        C = np.sum(w) - np.sum(w ** 2) / np.sum(w)
        tau2 = max(0.0, (Q - df_) / C) if C > 0 else 0.0
        w_r = 1 / (v + tau2)
        mu = np.sum(w_r * z) / np.sum(w_r)
        se = np.sqrt(1 / np.sum(w_r))
        lo_r, hi_r = np.tanh(mu - 1.96 * se), np.tanh(mu + 1.96 * se)
        I2 = max(0.0, (Q - df_) / Q) * 100 if Q > 0 else 0.0
        LOG.info("Random-effects meta-analysis across %d studies: rho=%.3f "
                 "(95%% CI %.3f-%.3f), tau2=%.4f, I2=%.0f%%, Q=%.1f on %d df",
                 len(z), np.tanh(mu), lo_r, hi_r, tau2, I2, Q, df_)
        LOG.info("Sign test: positive in %d of %d studies, P=%.2e",
                 int((per_study["spearman"] > 0).sum()), len(per_study),
                 float(binomtest(int((per_study["spearman"] > 0).sum()),
                                 len(per_study), 0.5).pvalue))
        per_study.attrs["meta_rho"] = float(np.tanh(mu))
        print("\nCOMPOSITIONAL SHIFT vs RESISTOME SHIFT, BY STUDY\n")
        print(per_study.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # ---- 3. time-gap control --------------------------------------------
    ok = pairs["time_gap"].notna() & (pairs["time_gap"] > 0)
    if ok.sum() >= 100:
        LOG.info("=== 3. within time-gap bands (n=%d pairs with gaps) ===", int(ok.sum()))
        p = pairs[ok].copy()
        p["band"] = pd.qcut(p["time_gap"], q=min(4, p["time_gap"].nunique()),
                            duplicates="drop")
        brows = []
        for band, g in p.groupby("band", observed=True):
            if len(g) < 20:
                continue
            r = spearmanr(g["d_taxa"], g["d_arg"])
            brows.append({"time_gap_band": str(band), "n_pairs": len(g),
                          "median_gap": float(g["time_gap"].median()),
                          "spearman": r.statistic, "p_value": r.pvalue})
        if brows:
            bdf = pd.DataFrame(brows)
            print("\nHOLDING ELAPSED TIME CONSTANT\n")
            print(bdf.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
            LOG.info("If the correlation persists within bands, shared temporal "
                     "drift does not explain it.")
            per_study = pd.concat([per_study, bdf], axis=0, ignore_index=True)
    else:
        LOG.warning("Too few pairs with usable time gaps for the band analysis.")

    per_study.to_csv(table_path(cfg, "longitudinal_paired_change.csv"), index=False)
    LOG.info("Wrote longitudinal tables to %s", cfg["paths"]["table_dir"])


if __name__ == "__main__":
    main()
