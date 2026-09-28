"""
s04_variance_partition.py
=========================
How much of the resistome is explained by WHO IS THERE versus WHAT THEY WERE
EXPOSED TO?

Method: redundancy analysis (RDA). RDA is multivariate multiple regression of
the response matrix on a block of explanatory variables, followed by an
eigendecomposition of the fitted values. The constrained fraction is the
variance of the resistome explainable by that block.

Blocks tested:
    taxonomy    - CLR species abundances (reduced to leading PCs)
    host        - age, sex, disease status
    geography   - country
    technical   - post-QC sequencing depth
    study       - study identity (the batch ceiling)

Reported per block:
    R2            raw constrained fraction
    R2_adj        Ezekiel-adjusted, the honest number - blocks differ hugely in
                  df and raw R2 rewards the wide ones for nothing
    p             permutation test, freely permuting rows
    unique        variance explained after conditioning on all other blocks
                  (partial RDA), which is what you actually report

The gap between `R2_adj` for taxonomy and its `unique` value is the headline
diagnostic: if taxonomy has little unique signal once study identity is
partialled out, the microbiome-resistome link is largely batch structure and
the whole framing must change. Better to learn that now than at review.

Outputs:
    tables/variance_partition.csv
    work_dir/vp_results.parquet

Usage:  python src/s04_variance_partition.py
"""

from __future__ import annotations

from pathlib import Path

import hashlib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from common import (LOG, apply_stratum, harmonise_disease, load_config,
                    stratum, table_path, work_path)

RNG = np.random.default_rng(42)

# Per-test permutation streams. The module-level RNG above is retained so any
# other caller keeps working, but every permutation test now draws from a
# generator derived from its own key, making P values independent of the order
# in which blocks are tested. See patch_permutation_streams.py for why.
PERM_SEED = 42


def perm_rng(*key) -> np.random.Generator:
    """An independent generator for one permutation test.

    The key should identify the test uniquely and stably: partition label,
    block name, and which statistic is being tested. Two different keys give
    independent streams; the same key always gives the same stream, so a
    reported P value is reproducible in isolation and does not depend on
    whether any other block was tested, skipped or short-circuited.
    """
    text = "|".join(str(k) for k in key)
    # SeedSequence mixes the configured seed with a stable digest of the key.
    digest = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
    return np.random.default_rng(np.random.SeedSequence([PERM_SEED, digest]))


# ---------------------------------------------------------------------------
# Core RDA
# ---------------------------------------------------------------------------

def _center(Y: np.ndarray) -> np.ndarray:
    return Y - Y.mean(axis=0, keepdims=True)


def _clean(X: np.ndarray) -> np.ndarray:
    """Make a design block safe for decomposition.

    Non-finite entries become zero and zero-variance columns are dropped.
    Both arise routinely here: an all-NaN covariate in a stratum gives a NaN
    median fill, and a dummy level absent from a subset gives a constant
    column. Either one poisons the whole factorisation.
    """
    X = np.asarray(X, dtype=float)
    if X.size == 0:
        return X.reshape(len(X), 0)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    sd = X.std(axis=0)
    return X[:, sd > 1e-12]


def _basis(X: np.ndarray, tol: float = 1e-8):
    """Orthonormal basis for the column space of centred X, via SVD.

    RDA only needs the projection onto that column space, so an explicit
    least-squares solve is unnecessary and fragile. Study identity and country
    are near-collinear in this dataset, which leaves the partial design
    rank-deficient and makes LAPACK's default gelsd driver fail to converge;
    a rank-truncated SVD is stable on exactly those matrices.
    """
    Xc = _center(_clean(X))
    if Xc.shape[1] == 0:
        return np.zeros((len(Xc), 0)), 0
    try:
        U, sv, _ = np.linalg.svd(Xc, full_matrices=False)
    except np.linalg.LinAlgError:
        from scipy.linalg import svd as scipy_svd
        U, sv, _ = scipy_svd(Xc, full_matrices=False, lapack_driver="gesvd")
    if sv.size == 0 or sv[0] <= 0:
        return np.zeros((len(Xc), 0)), 0
    rank = int((sv > tol * sv[0] * max(Xc.shape)).sum())
    return U[:, :rank], rank


def rda_r2(Y: np.ndarray, X: np.ndarray, Z: np.ndarray | None = None) -> tuple[float, float, int]:
    """Constrained variance fraction of Y explained by X, optionally
    conditioned on covariate block Z (partial RDA).

    Returns (R2, R2_adj, rank_used).
    """
    Y = _center(np.asarray(Y, dtype=float))
    total_ss = float((Y ** 2).sum())
    if total_ss == 0:
        return 0.0, 0.0, 0

    if Z is not None and np.asarray(Z).size:
        Uz, _ = _basis(Z)
        if Uz.shape[1]:
            Y = Y - Uz @ (Uz.T @ Y)
            X = _center(_clean(X))
            X = X - Uz @ (Uz.T @ X)
            total_ss = float((Y ** 2).sum())
            if total_ss <= 0:
                return 0.0, 0.0, 0

    Ux, rank = _basis(X)
    if rank == 0:
        return 0.0, 0.0, 0
    fitted = Ux @ (Ux.T @ Y)
    r2 = float((fitted ** 2).sum() / total_ss)
    r2 = min(max(r2, 0.0), 1.0)

    n = len(Y)
    r2_adj = 1.0 - (1.0 - r2) * (n - 1) / max(n - rank - 1, 1)
    return r2, r2_adj, rank


# Set by main() when W1_PERM_UNIT=subject: the per-sample subject labels,
# aligned to the analysis rows. Empty means sample-level permutation within
# study, i.e. the original behaviour, unchanged.
PERM_SUBJECT: list = []


def _subject_block_permutation(n: int, blocks, rng) -> np.ndarray:
    """Permute whole subjects within each study block.

    Within a study, subjects are relabelled at random and their samples move
    as intact groups, so a subject with 81 samples contributes one
    exchangeable unit rather than 81. Study membership is preserved at every
    position, so between-study structure survives the permutation exactly as
    it does under the sample-level scheme.

    Subjects differ in sample count, so the dealt-out donor sequence can be
    shorter than the block; the remainder falls back to a within-study
    shuffle, which is no worse than current behaviour for those positions.
    About 8% of destination subjects draw from more than one source subject on
    the real adult data -- see the module docstring of the patch script.

    ``rng`` is passed in rather than taken from module state so that each
    test keeps the independent stream established by the seed patch.
    """
    subj = PERM_SUBJECT[0]
    idx = np.arange(n)
    for b in blocks:
        if len(b) < 2:
            continue
        s_here = subj[b]
        uniq = pd.unique(s_here)
        if len(uniq) < 2:
            continue
        order = rng.permutation(len(uniq))
        pos = {u: b[s_here == u] for u in uniq}
        donor = np.concatenate([pos[uniq[j]] for j in order])
        idx[b] = donor[:len(b)] if len(donor) >= len(b) else np.concatenate(
            [donor, rng.permutation(b)[:len(b) - len(donor)]])
    return idx


def permutation_p(Y: np.ndarray, X: np.ndarray, Z: np.ndarray | None,
                  observed: float, n_perm: int = 999,
                  groups: np.ndarray | None = None,
                  rng: np.random.Generator | None = None) -> float:
    """Permutation test for the constrained variance fraction.

    Samples are clustered by study, so unrestricted row permutation destroys
    that structure and yields anti-conservative P values: any block correlated
    with study appears significant because the null it is compared against has
    no study structure at all. Permutation is therefore RESTRICTED within
    study when a grouping is supplied, preserving between-study differences
    under the null and testing only the within-study association.

    With 199 permutations the smallest attainable P value is 0.005, which is
    why every block previously reported exactly that. The default is now 999.
    """
    hits = 1
    n = len(X)
    rng = RNG if rng is None else rng
    if groups is not None:
        blocks = [np.where(groups == g)[0] for g in pd.unique(groups)]
        # A block that is constant within every permutation group is invariant
        # under the permutation, so every permuted statistic equals the
        # observed one and P is 1 by construction. That is a degenerate test,
        # not evidence of no effect - study identity is the obvious case.
        Xc = np.nan_to_num(np.asarray(X, dtype=float))
        within_var = sum(float(np.nanmax(Xc[b].std(axis=0), initial=0.0))
                         for b in blocks if len(b) > 1)
        if within_var < 1e-9:
            LOG.warning("Block is constant within permutation groups; it cannot "
                        "be tested by restricted permutation. Reporting P as "
                        "not applicable rather than 1.0.")
            return float("nan")
    for _ in range(n_perm):
        if groups is None:
            idx = rng.permutation(n)
        else:
            if PERM_SUBJECT:
                idx = _subject_block_permutation(n, blocks, rng)
            else:
                idx = np.arange(n)
                for b in blocks:                   # shuffle within study only
                    idx[b] = rng.permutation(b)
        r2p, _, _ = rda_r2(Y, X[idx], Z)
        if r2p >= observed:
            hits += 1
    return hits / (n_perm + 1)


# ---------------------------------------------------------------------------
# Design matrices
# ---------------------------------------------------------------------------

def dummies(series: pd.Series, min_count: int = 20, prefix: str = "x") -> np.ndarray:
    s = series.astype(str).fillna("missing")
    vc = s.value_counts()
    s = s.where(s.isin(vc[vc >= min_count].index), "other")
    d = pd.get_dummies(s, prefix=prefix, drop_first=True)
    return d.values.astype(float)


def numeric_block(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    if not cols:
        return np.empty((len(df), 0))
    X = df[cols].apply(pd.to_numeric, errors="coerce")
    usable = [c for c in X.columns if X[c].notna().any()]
    if not usable:
        LOG.warning("Columns %s are entirely missing in this stratum; dropped.", cols)
        return np.empty((len(df), 0))
    X = X[usable].fillna(X[usable].median())
    return StandardScaler().fit_transform(X.values)


def build_blocks(st: pd.DataFrame, taxa_clr: pd.DataFrame, n_pcs: int = 50) -> dict:
    """Assemble the explanatory blocks, aligned to taxa_clr.index."""
    st = st.set_index("sample").reindex(taxa_clr.index)
    blocks: dict[str, np.ndarray] = {}

    # taxonomy: reduce to leading PCs so df is comparable to other blocks
    n_pcs = int(min(n_pcs, taxa_clr.shape[1] - 1, len(taxa_clr) - 2))
    pca = PCA(n_components=n_pcs, random_state=0)
    blocks["taxonomy"] = pca.fit_transform(StandardScaler().fit_transform(taxa_clr.values))
    LOG.info("Taxonomy block: %d PCs capturing %.1f%% of species variance",
             n_pcs, 100 * pca.explained_variance_ratio_.sum())

    host_cols = [c for c in ("age", "bmi") if c in st.columns]
    parts = [numeric_block(st, host_cols)]
    for c in ("sex",):
        if c in st.columns:
            parts.append(dummies(st[c], prefix=c))
    parts = [p for p in parts if p.shape[1] > 0]
    blocks["host"] = np.column_stack(parts) if parts else np.empty((len(st), 0))

    # exposure: the 'what were they exposed to' side of the question.
    # Only meaningful where medication was actually annotated - see
    # run_partition(..., annotated_only=True) below.
    exp_parts = []
    for c in ("abx_systemic", "abx_intestinal", "abx_any"):
        if c in st.columns:
            v = pd.to_numeric(st[c], errors="coerce")
            if v.notna().sum() > 0:
                exp_parts.append(v.fillna(v.mean()).values[:, None])
    if "n_medications" in st.columns:
        exp_parts.append(numeric_block(st, ["n_medications"]))
    for c in ("diet", "smoker"):
        if c in st.columns:
            exp_parts.append(dummies(st[c], prefix=c))
    exp_parts = [p for p in exp_parts if p.shape[1] > 0]
    blocks["exposure"] = np.column_stack(exp_parts) if exp_parts else np.empty((len(st), 0))

    dz_col = "disease_detail" if "disease_detail" in st.columns else "disease"
    blocks["disease"] = dummies(st[dz_col], prefix="dz") if dz_col in st.columns \
        else np.empty((len(st), 0))

    if "life_stage" in st.columns and st["life_stage"].nunique() > 1:
        blocks["life_stage"] = dummies(st["life_stage"], min_count=50, prefix="ls")

    blocks["geography"] = dummies(st["country"], prefix="country") if "country" in st.columns \
        else np.empty((len(st), 0))

    depth_col = "reads_after_qc" if "reads_after_qc" in st.columns else None
    blocks["technical"] = numeric_block(st, [depth_col]) if depth_col else np.empty((len(st), 0))
    if depth_col:
        blocks["technical"] = np.column_stack([
            blocks["technical"],
            np.log1p(pd.to_numeric(st[depth_col], errors="coerce").fillna(0)).values[:, None],
        ])

    blocks["study"] = dummies(st["study"], prefix="study") if "study" in st.columns \
        else np.empty((len(st), 0))

    for k, v in blocks.items():
        LOG.info("Block '%s': %d columns", k, v.shape[1])
    return blocks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def freedman_lane_p(Y, X, Z, observed, n_perm=999, groups=None, seed=42,
                    rng=None):
    """Permutation test for a PARTIAL contribution (Freedman-Lane).

    Permuting the raw response tests the MARGINAL association. To test whether
    a block adds anything beyond the others, the response is first residualised
    on those others, the RESIDUALS are permuted, the nuisance fit is added
    back, and the partial statistic is recomputed. Without this, a reported
    P value describes a different hypothesis from the reported effect.
    """
    rng = np.random.default_rng(seed) if rng is None else rng
    Y = _center(np.asarray(Y, dtype=float))
    Uz, _ = _basis(Z) if (Z is not None and np.asarray(Z).size) else (None, 0)
    if Uz is None or Uz.shape[1] == 0:
        return permutation_p(Y, X, None, observed, n_perm, groups, rng=rng)
    fit_z = Uz @ (Uz.T @ Y)
    res_z = Y - fit_z
    n = len(Y)
    if groups is not None:
        gblocks = [np.where(groups == g)[0] for g in pd.unique(groups)]
    hits = 1
    for _ in range(n_perm):
        if groups is None:
            idx = rng.permutation(n)
        else:
            if PERM_SUBJECT:
                idx = _subject_block_permutation(n, gblocks, rng)
            else:
                idx = np.arange(n)
                for b in gblocks:
                    idx[b] = rng.permutation(b)
        Yp = fit_z + res_z[idx]
        r2p, _, _ = rda_r2(Yp, X, Z)
        if r2p >= observed:
            hits += 1
    return hits / (n_perm + 1)


def run_partition(Y: np.ndarray, blocks: dict, label: str,
                  groups: np.ndarray | None = None,
                  n_perm: int = 999) -> pd.DataFrame:
    """Marginal and unique (partial) variance explained, per block.

    `groups` restricts permutation within study; passing None reverts to the
    unrestricted test, which is anti-conservative for clustered data.
    """
    LOG.info("--- variance partition: %s (%d permutations, %s) ---", label, n_perm,
             "restricted within study" if groups is not None else "UNRESTRICTED")
    rows = []
    for name, X in blocks.items():
        if X.shape[1] == 0:
            LOG.warning("Block '%s' is empty; skipped.", name)
            continue
        r2, r2adj, df = rda_r2(Y, X)
        p_marg = permutation_p(Y, X, None, r2, n_perm=n_perm, groups=groups,
                               rng=perm_rng(label, name, "marginal"))

        others = [v for k, v in blocks.items() if k != name and v.shape[1] > 0]
        Z = np.column_stack(others) if others else None

        # PARTIAL R2: fraction of the variance REMAINING after conditioning on
        # the other blocks. This is what rda_r2(Y, X, Z) returns.
        part_r2, part_adj, _ = rda_r2(Y, X, Z)

        # UNIQUE FRACTION OF TOTAL: the drop in explained variance when this
        # block is removed from the full model. These are different quantities
        # and reporting one as the other overstates the effect.
        if Z is not None:
            full = np.column_stack([X, Z])
            r2_full, adj_full, _ = rda_r2(Y, full)
            r2_red, adj_red, _ = rda_r2(Y, Z)
            uniq_total = r2_full - r2_red
            uniq_total_adj = adj_full - adj_red
        else:
            uniq_total, uniq_total_adj = r2, r2adj

        p_part = freedman_lane_p(Y, X, Z, part_r2, n_perm=n_perm, groups=groups,
                                 rng=perm_rng(label, name, "partial"))

        rows.append({"block": name, "n_terms": df,
                     "marginal_R2": r2, "marginal_R2_adj": r2adj,
                     "marginal_p": p_marg,
                     "partial_R2": part_r2, "partial_R2_adj": part_adj,
                     "partial_p": p_part,
                     "unique_total_fraction": uniq_total,
                     "unique_total_fraction_adj": uniq_total_adj,
                     "p_testable": bool(np.isfinite(p_marg)),
                     # legacy aliases so downstream code keeps working
                     "R2": r2, "R2_adj": r2adj, "p_perm": p_marg,
                     "unique_R2": part_r2, "unique_R2_adj": part_adj})
        LOG.info("%-10s marginal R2adj=%.4f (p=%s) | partial R2adj=%.4f (p=%s) | "
                 "unique share of total=%.4f", name, r2adj,
                 f"{p_marg:.3f}" if np.isfinite(p_marg) else "n/a",
                 part_adj, f"{p_part:.3f}" if np.isfinite(p_part) else "n/a",
                 uniq_total_adj)
    out = pd.DataFrame(rows).sort_values("partial_R2_adj", ascending=False)
    LOG.info("NOTE: partial R2 is the share of REMAINING variance after "
             "conditioning; unique_total_fraction is the share of TOTAL "
             "variance attributable to the block alone. Report whichever the "
             "text claims, and do not conflate them.")
    return out


def main() -> None:
    cfg = load_config()
    st = pd.read_parquet(work_path(cfg, "sample_table.parquet"))
    st = harmonise_disease(apply_stratum(cfg, st))
    arg = pd.read_parquet(work_path(cfg, "arg_abundance.parquet"))
    taxa = pd.read_parquet(work_path(cfg, "taxa_clr.parquet"))

    shared = sorted(set(arg.index) & set(taxa.index) & set(st["sample"].astype(str)))

    # Sensitivity option: retain one sample per subject, so that densely
    # sampled individuals cannot dominate the estimate.
    import os
    if os.environ.get("W1_ONE_PER_SUBJECT") and "subject_id" in st.columns:
        first = (st[st["sample"].isin(shared)]
                 .drop_duplicates("subject_id")["sample"].astype(str))
        shared = sorted(set(shared) & set(first))
        LOG.info("One sample per subject: %d retained", len(shared))

    arg, taxa = arg.loc[shared], taxa.loc[shared]
    LOG.info("Variance partitioning on %d samples, %d ARGs", len(shared), arg.shape[1])

    Y = StandardScaler().fit_transform(arg.values)
    blocks = build_blocks(st, taxa)

    grp = (st.set_index("sample").reindex(shared)["study"].astype(str).values
           if "study" in st.columns else None)

    # Permutation unit. Restricting to study preserves between-study structure
    # but still treats repeat samples from one person as exchangeable. With
    # W1_PERM_UNIT=subject, whole subjects are permuted within study instead.
    import os as _os
    if _os.environ.get("W1_PERM_UNIT", "").lower() == "subject":
        _sc = ("subject_uid" if "subject_uid" in st.columns
               else "subject_id" if "subject_id" in st.columns else None)
        if _sc is None:
            LOG.warning("W1_PERM_UNIT=subject requested but no subject column "
                        "is present; falling back to sample-level permutation "
                        "within study. Rerun s02 to retain subject_id.")
        else:
            _subj = st.set_index("sample").reindex(shared)[_sc].astype(str).values
            PERM_SUBJECT[:] = [_subj]
            _vc = pd.Series(_subj).value_counts()
            LOG.info("Permutation unit: SUBJECT within study. %d subjects over "
                     "%d samples; %.1f%% of samples are repeats; largest "
                     "subject contributes %d samples.",
                     len(_vc), len(_subj),
                     100.0 * pd.Series(_subj).map(_vc).gt(1).mean(),
                     int(_vc.max()))
    else:
        LOG.info("Permutation unit: SAMPLE within study (original behaviour).")
    n_perm = int((cfg.get("analysis", {}) or {}).get("n_permutations", 999))
    res = run_partition(Y, blocks, f"stratum: {stratum(cfg)}",
                        groups=grp, n_perm=n_perm)
    res.to_parquet(work_path(cfg, "vp_results.parquet", per_stratum=True), index=False)
    out = table_path(cfg, "variance_partition.csv")
    res.to_csv(out, index=False)
    print("\n" + res.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    LOG.info("Wrote %s", out)

    # ---- exposure analysis on the annotated subset only ------------------
    if "has_medication_data" in st.columns:
        ann = st.set_index("sample").reindex(shared)["has_medication_data"].fillna(False)
        ann = ann.astype(bool).values
        n_ann = int(ann.sum())
        LOG.info("Annotated-medication subset: %d / %d samples", n_ann, len(ann))
        if n_ann >= 500:
            Y2 = StandardScaler().fit_transform(arg.values[ann])
            blocks2 = build_blocks(st[st["sample"].isin(np.array(shared)[ann])],
                                   taxa.iloc[ann])
            g2 = (st[st["sample"].isin(np.array(shared)[ann])]
                  .set_index("sample").reindex(np.array(shared)[ann])["study"]
                  .astype(str).values if "study" in st.columns else None)
            res2 = run_partition(Y2, blocks2, "medication-annotated subset",
                                 groups=g2, n_perm=n_perm)
            res2.to_parquet(work_path(cfg, "vp_results_annotated.parquet", per_stratum=True), index=False)
            res2.to_csv(table_path(cfg, "variance_partition_annotated.csv"),
                        index=False)
            print("\nMEDICATION-ANNOTATED SUBSET (n=%d)" % n_ann)
            print(res2.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        else:
            LOG.warning("Only %d annotated samples; exposure partition skipped.", n_ann)

    tax = res[res["block"] == "taxonomy"]
    if len(tax):
        r = tax.iloc[0]
        if r["unique_R2_adj"] < 0.02:
            LOG.warning(
                "Taxonomy explains <2%% of resistome variance uniquely. The "
                "microbiome-resistome link may be mostly study structure. "
                "Inspect per-study effects before proceeding to modelling."
            )


if __name__ == "__main__":
    main()
