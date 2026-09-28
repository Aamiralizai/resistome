"""
s09_carriage.py
===============
Are some gut community configurations permissive to high-risk mobile ARGs -
and can that be predicted in a cohort the model has never seen?

This is the clinically legible half of the project. Everything before it is a
variance decomposition; this asks a question a clinician would recognise.

Three analyses
--------------
1. ENTEROTYPE ASSOCIATION. For each high-risk gene carried by enough samples,
   test carriage against enterotype and the Enterotype Dysbiosis Score,
   controlling for study, country, disease and sequencing depth. Study is
   included as a fixed effect because carriage is strongly clustered by
   cohort; omitting it manufactures associations.

2. CROSS-COHORT PREDICTION. Logistic regression on CLR species abundances,
   evaluated leave-one-study-out. AUC on unseen cohorts is the number that
   matters - a model that only works within its training cohort is not a
   screening tool.

3. GEOGRAPHY ROBUSTNESS. Country and study are near-collinear in public data,
   so "geography is compositional" may really be "geography is inseparable
   from study". Re-runs the variance partition restricted to countries
   represented by several independent studies, where the two can be told
   apart. If geography still collapses there, the claim holds.

Outputs:
    tables/carriage_enterotype_associations.csv
    tables/carriage_prediction_auc.csv
    tables/variance_partition_multistudy_countries.csv

Usage:  python src/s09_carriage.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings

from scipy.linalg import qr
from scipy.stats import chi2
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm

from common import (LOG, apply_stratum, harmonise_disease, load_config,
                    stratum, table_path, work_path)

MIN_CARRIERS = 50
MIN_FOLD_CARRIERS = 5


def design(st: pd.DataFrame, cols: list[str], min_count: int = 20) -> pd.DataFrame:
    """Fixed-effect design from categorical and numeric covariates."""
    parts = []
    for c in cols:
        if c not in st.columns:
            continue
        v = st[c]
        if pd.api.types.is_numeric_dtype(v):
            x = pd.to_numeric(v, errors="coerce")
            if x.notna().sum() == 0:
                continue
            parts.append(((x - x.mean()) / (x.std() or 1)).fillna(0).rename(c))
        else:
            s = v.astype(str).fillna("missing")
            vc = s.value_counts()
            s = s.where(s.isin(vc[vc >= min_count].index), "other")
            if s.nunique() > 1:
                parts.append(pd.get_dummies(s, prefix=c, drop_first=True).astype(float))
    return pd.concat(parts, axis=1) if parts else pd.DataFrame(index=st.index)


def drop_collinear(X: pd.DataFrame, tol: float = 1e-10) -> pd.DataFrame:
    """Keep a maximal set of linearly independent columns, via pivoted QR.

    Unpenalised logistic regression needs an invertible Hessian, so any exact
    linear dependence in the design is fatal. Dummy blocks routinely create
    them - a study level that appears in only one disease group, for instance.
    """
    if X.shape[1] == 0:
        return X
    A = X.values.astype(float)
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    _, r, piv = qr(A, mode="economic", pivoting=True)
    diag = np.abs(np.diag(r))
    rank = int((diag > tol * max(diag.max(), 1.0) * max(A.shape)).sum())
    keep = sorted(piv[:rank])
    if len(keep) < X.shape[1]:
        LOG.debug("Dropped %d linearly dependent design columns.",
                  X.shape[1] - len(keep))
    return X.iloc[:, keep]


def enterotype_associations(panel: pd.DataFrame, st: pd.DataFrame) -> pd.DataFrame:
    ent_col = next((c for c in st.columns if c.endswith("enterotype")), None)
    dys_col = next((c for c in st.columns if "dysbiosis" in c.lower()), None)
    if ent_col is None and dys_col is None:
        LOG.warning("No enterotype or dysbiosis column; skipping association tests.")
        return pd.DataFrame()

    # Enterotypes are computed only for adult faecal samples, so a "missing"
    # level is data availability, not a community state. Including it lets the
    # model report availability as if it were biology - which it did.
    have = pd.Series(True, index=st.index)
    for var in (ent_col, dys_col):
        if var is not None:
            have &= st[var].notna() & ~st[var].astype(str).str.lower().isin(
                ["missing", "nan", "none", ""])
    LOG.info("Samples with an enterotype call: %d / %d", int(have.sum()), len(st))
    if have.sum() < 300:
        LOG.warning("Too few samples with enterotype calls for a stable test.")
        return pd.DataFrame()
    st = st[have].reset_index(drop=True)
    panel = panel.loc[have.values]

    # Study is nested within country - every study comes from one country - so
    # including both makes the design rank-deficient by construction. Study is
    # the finer level and absorbs country, so country is dropped.
    covar_cols = ["study", "disease_group", "reads_after_qc", "age"]
    covars = drop_collinear(design(st, covar_cols))
    LOG.info("Covariate design: %d independent columns from %s",
             covars.shape[1], covar_cols)
    rows = []
    for gene in panel.columns:
        y = panel[gene].values
        if y.sum() < MIN_CARRIERS or y.sum() > len(y) - MIN_CARRIERS:
            continue
        for var, label in ((ent_col, "enterotype"), (dys_col, "dysbiosis_score")):
            if var is None:
                continue
            X_var = design(st, [var], min_count=30)
            if X_var.shape[1] == 0:
                continue
            X_var = drop_collinear(X_var)
            if X_var.shape[1] == 0:
                continue
            X_full = drop_collinear(
                pd.concat([covars, X_var], axis=1).astype(float).fillna(0.0))
            # keep only the variable-of-interest columns that survived
            var_cols = [c for c in X_var.columns if c in X_full.columns]
            if not var_cols:
                continue
            X_var = X_full[var_cols]
            # Unpenalised MLE via statsmodels: the chi2 null for a likelihood-
            # ratio test assumes unpenalised likelihoods, and an L2 penalty also
            # shrinks the odds ratios toward 1. Falls back to the penalised fit
            # only if separation prevents convergence.
            penalised = False
            try:
                Xf = sm.add_constant(X_full.values, has_constant="add")
                Xn = sm.add_constant(covars.values.astype(float), has_constant="add")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m_full = sm.Logit(y, Xf).fit(disp=0, method="lbfgs", maxiter=500)
                    m_null = sm.Logit(y, Xn).fit(disp=0, method="lbfgs", maxiter=500)
                if not (np.isfinite(m_full.llf) and np.isfinite(m_null.llf)):
                    raise ValueError("non-finite log-likelihood")
                ll_full = -m_full.llf / len(y)
                ll_null = -m_null.llf / len(y)
                coef_tail = m_full.params[-X_var.shape[1]:]
            except Exception as e:  # noqa: BLE001
                penalised = True
                LOG.info("%s / %s: unpenalised fit failed (%s); using L2 fallback.",
                         gene, label, str(e)[:60])
                try:
                    full = LogisticRegression(max_iter=2000, C=1.0).fit(X_full.values, y)
                    null = LogisticRegression(max_iter=2000, C=1.0).fit(
                        covars.values.astype(float), y)
                    ll_full = -np.mean(np.log(np.clip(
                        full.predict_proba(X_full.values)[np.arange(len(y)), y], 1e-9, 1)))
                    ll_null = -np.mean(np.log(np.clip(
                        null.predict_proba(covars.values.astype(float))[np.arange(len(y)), y],
                        1e-9, 1)))
                    coef_tail = full.coef_[0][-X_var.shape[1]:]
                except Exception as e2:  # noqa: BLE001
                    LOG.warning("%s / %s failed entirely: %s", gene, label, e2)
                    continue
            # Likelihood-ratio test: 2n(ll_null - ll_full) ~ chi2 with df equal
            # to the extra parameters. Deviance reduction alone says nothing
            # about whether the improvement exceeds chance.
            dev = 2 * len(y) * (ll_null - ll_full)
            df_extra = max(Xf.shape[1] - Xn.shape[1], 1) if not penalised else X_var.shape[1]
            pval = float(chi2.sf(max(dev, 0.0), df_extra))

            coefs = dict(zip(X_var.columns, np.asarray(coef_tail)))
            top = max(coefs, key=lambda k: abs(coefs[k]))
            rows.append({"gene": gene, "predictor": label,
                         "n_carriers": int(y.sum()), "prevalence": float(y.mean()),
                         "strongest_level": top,
                         "log_odds": float(coefs[top]),
                         "odds_ratio": float(np.exp(coefs[top])),
                         "deviance_reduction": float(ll_null - ll_full),
                         "lrt_chi2": float(dev), "df": int(df_extra), "p_value": pval,
                         "penalised_fallback": penalised})
    out = pd.DataFrame(rows)
    if len(out):
        out["p_adj_BH"] = multipletests(out["p_value"], method="fdr_bh")[1]
        out = out.sort_values("p_value")
        n_sig = int((out["p_adj_BH"] < 0.05).sum())
        LOG.info("%d / %d gene-predictor tests significant at BH FDR < 0.05",
                 n_sig, len(out))
        LOG.info("Largest deviance reduction: %.4f. With n=%d, tiny effects reach "
                 "significance - report odds ratios, not p-values.",
                 float(out["deviance_reduction"].max()), int(out["n_carriers"].max()))
        if out["penalised_fallback"].any():
            LOG.warning("%d tests used the penalised fallback; their p-values are "
                        "approximate and odds ratios shrunk toward 1.",
                        int(out["penalised_fallback"].sum()))
    return out


def cross_cohort_auc(panel: pd.DataFrame, taxa: pd.DataFrame,
                     st: pd.DataFrame) -> pd.DataFrame:
    """Leave-one-study-out AUC for carriage from community composition."""
    if "study" not in st.columns:
        LOG.warning("No study column; cross-cohort evaluation skipped.")
        return pd.DataFrame()
    groups = st.set_index("sample").reindex(taxa.index)["study"].astype(str)
    X = taxa.values
    rows = []

    for gene in panel.columns:
        y = panel[gene].reindex(taxa.index).fillna(0).astype(int).values
        if y.sum() < MIN_CARRIERS:
            continue
        aucs, aps, ns = [], [], []
        for g in groups.value_counts().index:
            te = (groups == g).values
            tr = ~te
            if te.sum() < 20 or y[te].sum() < MIN_FOLD_CARRIERS or y[te].sum() == te.sum():
                continue
            if y[tr].sum() < MIN_CARRIERS:
                continue
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(max_iter=3000, C=0.1, class_weight="balanced")
            clf.fit(sc.transform(X[tr]), y[tr])
            p = clf.predict_proba(sc.transform(X[te]))[:, 1]
            aucs.append(roc_auc_score(y[te], p))
            aps.append(average_precision_score(y[te], p))
            ns.append(int(te.sum()))
        if len(aucs) >= 3:
            a = np.array(aucs)
            # A high median over few folds can come from one or two cohorts.
            # The IQR and the worst fold expose that; report them alongside.
            rows.append({"gene": gene, "n_carriers": int(y.sum()),
                         "prevalence": float(y.mean()), "n_folds": len(aucs),
                         "median_auc": float(np.median(a)),
                         "mean_auc": float(np.mean(a)),
                         "auc_q25": float(np.percentile(a, 25)),
                         "auc_q75": float(np.percentile(a, 75)),
                         "auc_min": float(a.min()),
                         "frac_folds_auc_gt_0_7": float(np.mean(a > 0.7)),
                         "median_avg_precision": float(np.median(aps)),
                         "well_powered": bool(y.sum() >= 200 and len(a) >= 10)})
            LOG.info("%-12s carriers=%5d folds=%2d  AUC med=%.3f "
                     "[IQR %.3f-%.3f, min %.3f]  AP=%.3f%s",
                     gene, int(y.sum()), len(a), float(np.median(a)),
                     float(np.percentile(a, 25)), float(np.percentile(a, 75)),
                     float(a.min()), float(np.median(aps)),
                     "" if (y.sum() >= 200 and len(a) >= 10) else "   <- underpowered")
    return pd.DataFrame(rows).sort_values("median_auc", ascending=False)


def geography_robustness(cfg, st, arg, taxa, tabdir) -> None:
    """Re-run the partition where country and study are actually separable."""
    from s04_variance_partition import build_blocks, run_partition
    if not {"country", "study"} <= set(st.columns):
        return
    per_country = st.groupby("country")["study"].nunique()
    multi = per_country[per_country >= 3].index
    LOG.info("Countries with >=3 independent studies: %d of %d (%s)",
             len(multi), len(per_country), list(multi)[:10])
    if len(multi) < 3:
        LOG.warning("Too few multi-study countries; geography claim cannot be "
                    "separated from study effects in this dataset. Report that "
                    "as a limitation rather than a mediation result.")
        return

    sub = st[st["country"].isin(multi)]
    idx = sorted(set(sub["sample"]) & set(arg.index) & set(taxa.index))
    LOG.info("Multi-study-country subset: %d samples", len(idx))
    if len(idx) < 500:
        LOG.warning("Subset too small for a stable partition.")
        return
    blocks = build_blocks(sub[sub["sample"].isin(idx)], taxa.loc[idx])
    Y = StandardScaler().fit_transform(arg.loc[idx].values)
    grp = (sub[sub["sample"].isin(idx)].set_index("sample")
           .reindex(idx)["study"].astype(str).values)
    res = run_partition(Y, blocks, "countries with >=3 studies",
                        groups=grp, n_perm=999)
    res.to_csv(table_path(cfg, "variance_partition_multistudy_countries.csv"), index=False)
    print("\nGEOGRAPHY ROBUSTNESS (countries with >=3 independent studies)\n")
    print(res.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    geo = res[res["block"] == "geography"]
    if len(geo):
        u = float(geo.iloc[0]["unique_R2_adj"])
        LOG.info("Geography unique R2adj in the separable subset: %.4f", u)
        LOG.info("If this is still near zero, the compositional-mediation claim "
                 "holds. If it rises, geography was confounded with study.")


def main() -> None:
    cfg = load_config()
    tabdir = Path(cfg["paths"]["table_dir"])

    pp = work_path(cfg, "arg_highrisk_carriage.parquet")
    if not pp.exists():
        raise SystemExit("arg_highrisk_carriage.parquet missing - rerun s03.")
    panel = pd.read_parquet(pp)
    taxa = pd.read_parquet(work_path(cfg, "taxa_clr.parquet"))
    arg = pd.read_parquet(work_path(cfg, "arg_abundance.parquet"))
    st = harmonise_disease(apply_stratum(cfg, pd.read_parquet(
        work_path(cfg, "sample_table.parquet"))))

    shared = sorted(set(panel.index) & set(taxa.index) & set(st["sample"].astype(str)))
    panel, taxa = panel.loc[shared], taxa.loc[shared]
    st = st[st["sample"].isin(shared)].copy()
    sti = st.set_index("sample").reindex(shared).reset_index()

    keep = panel.columns[(panel.sum(axis=0) >= MIN_CARRIERS)]
    panel = panel[keep]
    LOG.info("Carriage analysis: %d samples, %d genes with >=%d carriers: %s",
             len(shared), len(keep), MIN_CARRIERS, list(keep))
    if len(keep) == 0:
        raise SystemExit("No high-risk gene has enough carriers in this stratum.")

    LOG.info("=== 1. enterotype and dysbiosis associations ===")
    assoc = enterotype_associations(panel, sti)
    if len(assoc):
        assoc.to_csv(table_path(cfg, "carriage_enterotype_associations.csv"), index=False)
        print("\nCARRIAGE vs COMMUNITY STATE (study, country, disease, depth controlled)\n")
        print(assoc.head(20).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    LOG.info("=== 2. cross-cohort carriage prediction (leave-one-study-out) ===")
    auc = cross_cohort_auc(panel, taxa, sti)
    if len(auc):
        auc.to_csv(table_path(cfg, "carriage_prediction_auc.csv"), index=False)
        print("\nCROSS-COHORT CARRIAGE PREDICTION\n")
        print(auc.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        good = auc[(auc["median_auc"] > 0.7) & auc["well_powered"]]
        weak = auc[(auc["median_auc"] > 0.7) & ~auc["well_powered"]]
        LOG.info("Well-powered and predictable (AUC > 0.7): %s",
                 ", ".join(good["gene"]) if len(good) else "none")
        if len(weak):
            LOG.warning("AUC > 0.7 but underpowered (<200 carriers or <10 folds): "
                        "%s - report as suggestive, not as a result.",
                        ", ".join(weak["gene"]))

    LOG.info("=== 3. geography robustness ===")
    geography_robustness(cfg, st, arg, taxa, tabdir)


if __name__ == "__main__":
    main()
