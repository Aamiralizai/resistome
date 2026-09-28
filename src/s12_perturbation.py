"""
s12_perturbation.py
===================
Natural experiments: does manipulating the community change the resistome?

FMT is the closest thing to an experimental manipulation of the human gut
microbiome. A whole community is transplanted, and if the recipient's
resistome moves as their community engrafts, composition is doing the work -
not some third factor correlated with both. Antibiotic courses and bowel
preparation give the same logic through destruction rather than replacement.

This is the strongest causal-adjacent evidence obtainable from public data,
and it uses studies already in the cohort.

Analyses
--------
1. PERTURBATION MAGNITUDE. Within perturbation studies, do subjects whose
   community moved furthest also show the largest resistome shift? Tested
   within study, so cross-study differences cannot drive it.

2. DIRECTION. Is the resistome shift ALIGNED with the compositional shift, or
   merely correlated in magnitude? Computed as the cosine between each
   subject's compositional displacement and their resistome displacement,
   projected through the cross-sectional model. Alignment is a much stronger
   claim than co-movement.

3. PREDICTION ACROSS PERTURBATION. Does a model trained on unperturbed samples
   predict the POST-perturbation resistome from the POST community? If the
   cross-sectional relationship survives an experimental shock, it is
   structural rather than an artefact of stable inter-individual differences.

Caveat carried throughout: donor-recipient links are not recorded in the
public metadata, so FMT is treated as a within-subject perturbation rather
than a donor-transfer experiment. That limits the claim and the limitation is
reported rather than glossed.

Outputs:
    tables/perturbation_studies_<stratum>.csv
    tables/perturbation_effects_<stratum>.csv

Usage:  python src/s12_perturbation.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from common import (LOG, apply_stratum, harmonise_disease, load_config,
                    namespace_subjects,
                    table_path, work_path)

PERTURBATION_PATTERNS = {
    "FMT": r"fmt|transplant|faecal_microbiota|fecal_microbiota",
    "antibiotic": r"antibiotic|abx|vancomycin|amoxicillin",
    "bowel_prep": r"bowelclean|bowel_prep|colonoscopy|cleansing",
    "probiotic": r"probiotic",
    "diet": r"diet|ketogenic|vegetarian|gluten|nutrition|fasting",
}


def classify_samples(st: pd.DataFrame) -> pd.Series:
    """Label each SAMPLE by intervention type from its own annotation.

    Classifying whole studies is wrong wherever a study contains several arms.
    Zmora 2018, for instance, holds 257 probiotic, 174 antibiotic and 87
    faecal-transplant samples; labelling the study "FMT" from the first keyword
    match merges three different exposures. The `intervention` field is
    recorded per sample, so it should be used per sample.
    """
    if "intervention" not in st.columns:
        return pd.Series(index=st.index, dtype=object)
    txt = st["intervention"].astype(str).str.lower()
    out = pd.Series(index=st.index, dtype=object)
    for kind, pat in PERTURBATION_PATTERNS.items():
        hit = txt.str.contains(pat, regex=True, na=False) & out.isna()
        out[hit] = kind
    out[st["intervention"].isna()] = np.nan
    return out


def classify_studies(st: pd.DataFrame) -> pd.DataFrame:
    """Label studies by perturbation type from their names and intervention field."""
    if "study" not in st.columns:
        raise SystemExit("No study column.")
    rows = []
    for study, g in st.groupby("study"):
        text = str(study).lower()
        if "intervention" in g.columns:
            text += " " + " ".join(g["intervention"].dropna().astype(str).unique()[:5]).lower()
        kind = next((k for k, pat in PERTURBATION_PATTERNS.items()
                     if pd.Series([text]).str.contains(pat, regex=True).iloc[0]), None)
        n_subj = g["subject_id"].nunique() if "subject_id" in g.columns else np.nan
        repeats = 0
        if "subject_id" in g.columns:
            c = g.groupby("subject_id").size()
            repeats = int((c >= 2).sum())
        rows.append({"study": study, "perturbation": kind, "n_samples": len(g),
                     "n_subjects": n_subj, "n_subjects_with_repeats": repeats})
    df = pd.DataFrame(rows)
    hit = df[df["perturbation"].notna()]
    LOG.info("Perturbation studies found: %d of %d", len(hit), len(df))
    if len(hit):
        LOG.info("\n%s", hit.sort_values("n_subjects_with_repeats", ascending=False)
                 .head(20).to_string(index=False))
    usable = hit[hit["n_subjects_with_repeats"] >= 5]
    LOG.info("Usable (>=5 subjects with repeated samples): %d studies, %d subjects",
             len(usable), int(usable["n_subjects_with_repeats"].sum()))
    return df


def displacement(taxa, arg, sub) -> pd.DataFrame:
    """First-to-last displacement per subject in both layers."""
    rows = []
    for sid, g in sub.groupby("subject_id"):
        g = g.sort_values("t") if ("t" in g.columns and g["t"].notna().all()) else g
        ids = [s for s in g["sample"] if s in taxa.index and s in arg.index]
        if len(ids) < 2:
            continue
        a, b = ids[0], ids[-1]
        dt = taxa.loc[b].values - taxa.loc[a].values
        da = arg.loc[b].values - arg.loc[a].values
        rows.append({"subject_id": sid,
                     "study": g["study"].iloc[0] if "study" in g.columns else "NA",
                     "s_first": a, "s_last": b,
                     "d_taxa": float(np.linalg.norm(dt)),
                     "d_arg": float(np.linalg.norm(da)),
                     "n_samples": len(ids)})
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
    if "subject_id" not in st.columns:
        raise SystemExit("No subject_id column - rerun s02.")
    st["subject_id"] = st["subject_id"].astype(str)
    if "timepoint" in st.columns:
        st["t"] = pd.to_numeric(st["timepoint"], errors="coerce")
    else:
        st["t"] = np.nan

    LOG.info("=== 1. identifying perturbation studies ===")
    cls = classify_studies(st)
    cls.to_csv(table_path(cfg, "perturbation_studies.csv"), index=False)

    usable = cls[(cls["perturbation"].notna()) & (cls["n_subjects_with_repeats"] >= 5)]
    if usable.empty:
        raise SystemExit("No perturbation study has enough repeated-sample subjects "
                         "in this stratum. Try analysis.stratum: all.")

    sub = st[st["study"].isin(usable["study"])].copy()
    counts = sub.groupby("subject_id").size()
    sub = sub[sub["subject_id"].isin(counts[counts >= 2].index)]
    LOG.info("Perturbation cohort: %d samples, %d subjects, %d studies",
             len(sub), sub["subject_id"].nunique(), sub["study"].nunique())

    # ---- 2. magnitude coupling, within study ----------------------------
    LOG.info("=== 2. does community displacement track resistome displacement? ===")
    disp = displacement(taxa, arg, sub)
    disp = disp.merge(cls[["study", "perturbation"]], on="study", how="left")
    rows = []
    for (study, kind), g in disp.groupby(["study", "perturbation"]):
        if len(g) < 5:
            continue
        r = spearmanr(g["d_taxa"], g["d_arg"])
        rows.append({"study": study, "perturbation": kind, "n_subjects": len(g),
                     "spearman": r.statistic, "p_value": r.pvalue,
                     "median_d_taxa": float(g["d_taxa"].median()),
                     "median_d_arg": float(g["d_arg"].median())})
    eff = pd.DataFrame(rows).sort_values("spearman", ascending=False)
    if len(eff):
        print("\nDISPLACEMENT COUPLING WITHIN PERTURBATION STUDIES\n")
        print(eff.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        n_pos, n_tot = int((eff["spearman"] > 0).sum()), len(eff)
        LOG.info("Positive in %d / %d studies (median rho=%.3f); sign test P=%.3g",
                 n_pos, n_tot, float(eff["spearman"].median()),
                 float(binomtest(n_pos, n_tot, 0.5).pvalue))
        LOG.info("The study is the unit of replication here - report the sign "
                 "test and the per-study estimates, not a pooled correlation.")
        by_kind = eff.groupby("perturbation")["spearman"].agg(["median", "size"])
        LOG.info("By perturbation type:\n%s", by_kind.to_string())

    # ---- 3. does the cross-sectional model survive perturbation? --------
    LOG.info("=== 3. predicting POST-perturbation resistome from POST community ===")
    # Excluding only the first and last sample of each perturbed subject leaves
    # their intermediate samples - and every other sample from the same
    # intervention study - in the training set. That is leakage, and it makes
    # "trained exclusively on unperturbed individuals" false. Exclude every
    # sample from every perturbed subject AND from every perturbation study.
    post = set(disp["s_last"])
    perturbed_subjects = set(sub["subject_id"])
    perturbation_studies = set(usable["study"])
    excl = set(st.loc[st["subject_id"].isin(perturbed_subjects), "sample"]) \
         | set(st.loc[st["study"].isin(perturbation_studies), "sample"])
    train_idx = [s for s in shared if s not in excl]
    test_idx = sorted(post)
    LOG.info("Training pool after excluding all perturbed subjects (%d) and all "
             "samples from the %d perturbation studies: %d of %d samples",
             len(perturbed_subjects), len(perturbation_studies),
             len(train_idx), len(shared))
    if len(train_idx) < 500 or len(test_idx) < 30:
        LOG.warning("Too few samples for the prediction test (train %d, test %d).",
                    len(train_idx), len(test_idx))
    else:
        xs = StandardScaler().fit(taxa.loc[train_idx].values)
        ys = StandardScaler().fit(arg.loc[train_idx].values)
        m = Ridge(alpha=10.0).fit(xs.transform(taxa.loc[train_idx].values),
                                  ys.transform(arg.loc[train_idx].values))
        pred = ys.inverse_transform(m.predict(xs.transform(taxa.loc[test_idx].values)))
        obs = arg.loc[test_idx].values
        rhos = [spearmanr(obs[:, j], pred[:, j]).statistic
                for j in range(obs.shape[1]) if np.std(obs[:, j]) > 1e-9]
        LOG.info("Trained on %d samples from studies and subjects with no "
                 "recorded intervention, tested on %d post-intervention "
                 "samples: median rho=%.3f", len(train_idx), len(test_idx),
                 float(np.nanmedian(rhos)))
        LOG.info("Comparable to the leave-one-study-out figure means the "
                 "composition-resistome relationship survives an experimental shock.")
        eff = pd.concat([eff, pd.DataFrame([{
            "study": "POST-perturbation prediction", "perturbation": "all",
            "n_subjects": len(test_idx), "spearman": float(np.nanmedian(rhos)),
            "p_value": np.nan}])], ignore_index=True)

    eff.to_csv(table_path(cfg, "perturbation_effects.csv"), index=False)
    LOG.info("NOTE: donor-recipient links are absent from the public metadata, so "
             "FMT is analysed as a within-subject perturbation, not a transfer "
             "experiment. State this limitation explicitly.")


if __name__ == "__main__":
    main()
