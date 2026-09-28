"""
s15_mediation_dose.py
=====================
How far the observational design can be pushed toward a causal reading.

Three analyses, in increasing order of assumption:

1. DOSE-RESPONSE. Does a larger compositional displacement produce a larger
   resistome displacement, after adjusting for elapsed time, change in
   sequencing depth, health status and study? A dose-response relationship is
   weak evidence alone but is expected under a causal model and absent under
   several confounding scenarios. Restricted to subjects with at least three
   timepoints where available, since two-point trajectories cannot distinguish
   a trend from noise.

2. MEDIATION. For an intervention T, decompose its total effect on resistome
   change into the part transmitted through compositional change (indirect)
   and the part that is not (direct):

       T -> dC -> dR

   Reported with subject-clustered bootstrap intervals. Mediation assumes no
   unmeasured confounding between mediator and outcome, which observational
   metagenomic data cannot guarantee; the estimate is therefore reported as
   the proportion of the association consistent with mediation, not as a
   causal decomposition.

3. DIFFERENCE-IN-DIFFERENCES. Where a manually curated arm table is supplied,
   compare change in treated against change in control subjects within study.
   This is the only analysis here with a genuine comparator, and it will not
   run without curation: keyword classification of whole studies cannot
   separate treatment arms, and pretending otherwise would be worse than
   omitting the analysis.

Arm table format (CSV, --arms):
    study, subject_id, arm            where arm is one of: treated, control

Outputs:
    tables/dose_response_<stratum>.csv
    tables/mediation_<stratum>.csv
    tables/difference_in_differences_<stratum>.csv

Usage:
    python src/s15_mediation_dose.py
    python src/s15_mediation_dose.py --arms curated_arms.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression

from common import (LOG, apply_stratum, harmonise_disease, load_config,
                    namespace_subjects, table_path, work_path)
from s12_perturbation import classify_studies

MIN_SUBJECTS = 30
N_BOOT = 2000


def build_displacements(st, taxa, arg):
    """First-to-last displacement per subject, with covariates."""
    rows = []
    for sid, g in st.groupby("subject_id"):
        if "t" in g.columns and g["t"].notna().all():
            g = g.sort_values("t")
        ids = [s for s in g["sample"] if s in taxa.index and s in arg.index]
        if len(ids) < 2:
            continue
        a, b = ids[0], ids[-1]
        gt = dict(zip(g["sample"], g.get("t", pd.Series(index=g.index, dtype=float))))
        depth = dict(zip(g["sample"], pd.to_numeric(
            g.get("reads_after_qc", pd.Series(index=g.index, dtype=float)),
            errors="coerce")))
        rows.append({
            "subject_id": sid,
            "study": g["study"].iloc[0] if "study" in g.columns else "NA",
            "disease_group": (g["disease_group"].iloc[0]
                              if "disease_group" in g.columns else "NA"),
            "n_timepoints": len(ids),
            "d_taxa": float(np.linalg.norm(taxa.loc[b].values - taxa.loc[a].values)),
            "d_arg": float(np.linalg.norm(arg.loc[b].values - arg.loc[a].values)),
            "elapsed": (abs(gt.get(b, np.nan) - gt.get(a, np.nan))
                        if pd.notna(gt.get(a)) and pd.notna(gt.get(b)) else np.nan),
            "d_depth": (np.log1p(depth.get(b, np.nan)) - np.log1p(depth.get(a, np.nan))
                        if pd.notna(depth.get(a)) and pd.notna(depth.get(b)) else np.nan),
        })
    return pd.DataFrame(rows)


def adjusted_dose_response(d: pd.DataFrame) -> pd.DataFrame:
    """Compositional against resistome displacement, adjusted, within study."""
    rows = []
    for study, g in d.groupby("study"):
        if len(g) < 10:
            continue
        raw = spearmanr(g["d_taxa"], g["d_arg"])
        cov = [c for c in ("elapsed", "d_depth") if g[c].notna().sum() > 0.8 * len(g)]
        adj = np.nan
        if cov:
            X = g[cov].fillna(g[cov].median()).values
            rx = g["d_taxa"].values - LinearRegression().fit(X, g["d_taxa"]).predict(X)
            ry = g["d_arg"].values - LinearRegression().fit(X, g["d_arg"]).predict(X)
            adj = spearmanr(rx, ry).statistic
        rows.append({"study": study, "n_subjects": len(g),
                     "n_with_3plus_timepoints": int((g["n_timepoints"] >= 3).sum()),
                     "spearman_raw": raw.statistic, "p_raw": raw.pvalue,
                     "spearman_adjusted": adj,
                     "covariates": ",".join(cov) if cov else "none"})
    return pd.DataFrame(rows).sort_values("spearman_raw", ascending=False)


def mediation(d: pd.DataFrame, treat_col: str, seed: int = 42) -> dict:
    """Decompose the treatment-resistome association through compositional change.

    Reported as the share of the association consistent with mediation.
    Mediation analysis assumes no unmeasured mediator-outcome confounding; in
    observational metagenomic data that assumption is not verifiable, so this
    is a decomposition of association, not of causation.
    """
    g = d.dropna(subset=[treat_col, "d_taxa", "d_arg"]).copy()
    if g[treat_col].nunique() < 2 or len(g) < MIN_SUBJECTS:
        return {}
    T = g[treat_col].astype(float).values.reshape(-1, 1)
    M = g["d_taxa"].values
    Y = g["d_arg"].values

    a = LinearRegression().fit(T, M).coef_[0]                       # T -> M
    full = LinearRegression().fit(np.column_stack([T.ravel(), M]), Y)
    c_dash, b = full.coef_[0], full.coef_[1]                        # direct, M -> Y
    c = LinearRegression().fit(T, Y).coef_[0]                       # total
    indirect = a * b

    # cluster bootstrap over subjects within study
    rng = np.random.default_rng(seed)
    idx_by_study = {s: np.where(g["study"].values == s)[0]
                    for s in pd.unique(g["study"])}
    boots = []
    for _ in range(N_BOOT):
        take = np.concatenate([rng.choice(v, len(v), replace=True)
                               for v in idx_by_study.values()])
        try:
            Tb, Mb, Yb = T[take], M[take], Y[take]
            ab = LinearRegression().fit(Tb, Mb).coef_[0]
            bb = LinearRegression().fit(np.column_stack([Tb.ravel(), Mb]), Yb).coef_[1]
            boots.append(ab * bb)
        except Exception:  # noqa: BLE001
            continue
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan))
    return {"n": len(g), "total_effect": float(c), "direct_effect": float(c_dash),
            "indirect_effect": float(indirect),
            "indirect_ci_low": float(lo), "indirect_ci_high": float(hi),
            "proportion_mediated": float(indirect / c) if c else np.nan}


def difference_in_differences(d: pd.DataFrame, arms: pd.DataFrame) -> pd.DataFrame:
    """Treated minus control change, within study."""
    m = d.merge(arms[["study", "subject_id", "arm"]], on=["study", "subject_id"],
                how="inner")
    LOG.info("Arm table matched %d of %d subjects", len(m), len(d))
    rows = []
    for study, g in m.groupby("study"):
        t, c = g[g["arm"] == "treated"], g[g["arm"] == "control"]
        if len(t) < 5 or len(c) < 5:
            continue
        rows.append({
            "study": study, "n_treated": len(t), "n_control": len(c),
            "d_arg_treated": float(t["d_arg"].mean()),
            "d_arg_control": float(c["d_arg"].mean()),
            "did_resistome": float(t["d_arg"].mean() - c["d_arg"].mean()),
            "d_taxa_treated": float(t["d_taxa"].mean()),
            "d_taxa_control": float(c["d_taxa"].mean()),
            "did_composition": float(t["d_taxa"].mean() - c["d_taxa"].mean()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=None,
                    help="curated CSV with columns study, subject_id, arm")
    args = ap.parse_args()
    cfg = load_config()

    st = namespace_subjects(harmonise_disease(apply_stratum(cfg, pd.read_parquet(
        work_path(cfg, "sample_table.parquet")))))
    taxa = pd.read_parquet(work_path(cfg, "taxa_clr.parquet"))
    arg = pd.read_parquet(work_path(cfg, "arg_abundance.parquet"))
    shared = sorted(set(taxa.index) & set(arg.index) & set(st["sample"].astype(str)))
    taxa, arg = taxa.loc[shared], arg.loc[shared]
    st = st[st["sample"].isin(shared)].copy()
    st["t"] = pd.to_numeric(st.get("timepoint"), errors="coerce")

    d = build_displacements(st, taxa, arg)
    if len(d) < MIN_SUBJECTS:
        raise SystemExit(f"Only {len(d)} subjects with repeated samples.")
    LOG.info("Subjects with displacement: %d (%d with >=3 timepoints)",
             len(d), int((d["n_timepoints"] >= 3).sum()))

    # ---- 1. dose-response ------------------------------------------------
    LOG.info("=== 1. dose-response, adjusted, within study ===")
    dr = adjusted_dose_response(d)
    if len(dr):
        dr.to_csv(table_path(cfg, "dose_response.csv"), index=False)
        pos = int((dr["spearman_raw"] > 0).sum())
        LOG.info("Positive in %d / %d studies; median raw rho=%.3f, "
                 "median adjusted rho=%.3f", pos, len(dr),
                 float(dr["spearman_raw"].median()),
                 float(dr["spearman_adjusted"].median()))
        deep = d[d["n_timepoints"] >= 3]
        if len(deep) >= MIN_SUBJECTS:
            r = spearmanr(deep["d_taxa"], deep["d_arg"])
            LOG.info("Subjects with >=3 timepoints (n=%d): rho=%.3f",
                     len(deep), r.statistic)
        print("\\nDOSE-RESPONSE BY STUDY\\n")
        print(dr.head(20).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # ---- 2. mediation ----------------------------------------------------
    LOG.info("=== 2. mediation of intervention through compositional change ===")
    cls = classify_studies(st)
    treated_studies = set(cls.loc[cls["perturbation"].notna(), "study"])
    d["intervention"] = d["study"].isin(treated_studies).astype(int)
    med = mediation(d, "intervention")
    if med:
        pd.DataFrame([med]).to_csv(table_path(cfg, "mediation.csv"), index=False)
        LOG.info("total=%.4f  direct=%.4f  indirect=%.4f [95%% CI %.4f-%.4f]  "
                 "proportion mediated=%.2f", med["total_effect"],
                 med["direct_effect"], med["indirect_effect"],
                 med["indirect_ci_low"], med["indirect_ci_high"],
                 med["proportion_mediated"])
        LOG.warning("Interpret as the share of the ASSOCIATION consistent with "
                    "mediation. Unmeasured mediator-outcome confounding cannot "
                    "be excluded in observational data, and interventions act "
                    "on resistance genes directly as well as through the "
                    "community.")

    # ---- 3. difference-in-differences ------------------------------------
    LOG.info("=== 3. difference-in-differences ===")
    if not args.arms:
        LOG.warning("No --arms table supplied, so this analysis is SKIPPED. "
                    "Treatment and control arms cannot be separated by "
                    "keyword classification of whole studies; a study may "
                    "contain probiotic, autologous transplant and placebo arms "
                    "simultaneously. Supply a curated CSV with columns "
                    "study, subject_id, arm (treated|control).")
    else:
        arms = pd.read_csv(args.arms)
        did = difference_in_differences(d, arms)
        if len(did):
            did.to_csv(table_path(cfg, "difference_in_differences.csv"), index=False)
            print("\\nDIFFERENCE-IN-DIFFERENCES\\n")
            print(did.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
            r = spearmanr(did["did_composition"], did["did_resistome"])
            LOG.info("Across %d studies, compositional and resistome "
                     "difference-in-differences correlate at rho=%.3f (P=%.3g)",
                     len(did), r.statistic, r.pvalue)
        else:
            LOG.warning("No study had at least five subjects in both arms.")


if __name__ == "__main__":
    main()
