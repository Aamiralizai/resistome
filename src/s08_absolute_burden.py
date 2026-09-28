"""
s08_absolute_burden.py
======================
Relative versus ABSOLUTE resistome burden.

Almost the entire cross-cohort resistome literature is relative-abundance
only: ARG fragments over bacterial fragments. But two samples with the same
ratio carry very different actual reservoirs if one holds ten times the
bacterial load. Metalog supplies predicted microbial load for adult faecal
samples, so absolute burden is computable here and, as far as I can find, has
not been reported for the gut resistome across cohorts.

The question is not "is absolute different" - it must be, since it is the
relative value times load. The question is whether the CONCLUSIONS change:

  * does taxonomy still dominate the variance partition on the absolute scale?
  * do the same ARGs remain predictable?
  * do country and disease rankings reorder when load is accounted for?

A divergence is a genuine finding: it would mean published relative-scale
comparisons of resistome burden between populations are measuring something
other than what they claim.

Outputs:
    tables/variance_partition_absolute.csv
    tables/absolute_vs_relative_genes.csv
    tables/absolute_vs_relative_groups.csv
    work_dir/vp_results_absolute.parquet

Usage:  python src/s08_absolute_burden.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.preprocessing import StandardScaler

from common import (LOG, apply_stratum, harmonise_disease, load_config,
                    stratum, table_path, work_path)
from s04_variance_partition import build_blocks, run_partition


def group_medians(load: pd.Series, st: pd.DataFrame, key: str) -> pd.Series | None:
    if key not in st.columns:
        return None
    g = load.groupby(st[key].astype(str))
    keep = g.size()[g.size() >= 30].index
    return g.median()[keep].sort_values(ascending=False)


def main() -> None:
    cfg = load_config()
    tabdir = Path(cfg["paths"]["table_dir"])

    rel = pd.read_parquet(work_path(cfg, "arg_abundance.parquet"))
    ap = work_path(cfg, "arg_absolute.parquet")
    if not ap.exists():
        raise SystemExit("arg_absolute.parquet missing - rerun s03 with microbial "
                         "load predictions present.")
    absol = pd.read_parquet(ap)
    taxa = pd.read_parquet(work_path(cfg, "taxa_clr.parquet"))
    st = harmonise_disease(apply_stratum(cfg, pd.read_parquet(
        work_path(cfg, "sample_table.parquet"))))

    shared = sorted(set(rel.index) & set(absol.index) & set(taxa.index)
                    & set(st["sample"].astype(str)))
    genes = [g for g in rel.columns if g in absol.columns]
    rel, absol, taxa = rel.loc[shared, genes], absol.loc[shared, genes], taxa.loc[shared]
    st = st[st["sample"].isin(shared)]
    LOG.info("Load-adjusted analysis: %d samples, %d genes", len(shared), len(genes))
    if len(shared) < 200:
        LOG.warning("Only %d samples carry predicted microbial load in this "
                    "stratum, which is too few for a partition. Microbial load "
                    "is predicted for adult faecal samples only, so this is "
                    "expected outside the adult stratum. Skipping rather than "
                    "failing.", len(shared))
        return
    if len(shared) < 500:
        LOG.warning("Few samples with microbial load; results will be unstable.")

    # ---- 1. variance partition on the absolute scale ---------------------
    blocks = build_blocks(st, taxa)
    Y_abs = StandardScaler().fit_transform(absol.values)
    grp = (st.set_index("sample").reindex(shared)["study"].astype(str).values
           if "study" in st.columns else None)
    res_abs = run_partition(Y_abs, blocks, "load-adjusted burden", groups=grp)
    res_abs.to_parquet(work_path(cfg, "vp_results_absolute.parquet", per_stratum=True), index=False)
    res_abs.to_csv(table_path(cfg, "variance_partition_absolute.csv"), index=False)

    Y_rel = StandardScaler().fit_transform(rel.values)
    res_rel = run_partition(Y_rel, blocks, "relative burden (same samples)", groups=grp)

    cmp = (res_rel.set_index("block")[["R2_adj", "unique_R2_adj"]]
           .join(res_abs.set_index("block")[["R2_adj", "unique_R2_adj"]],
                 lsuffix="_relative", rsuffix="_absolute"))
    cmp["delta_unique"] = cmp["unique_R2_adj_absolute"] - cmp["unique_R2_adj_relative"]
    cmp = cmp.sort_values("unique_R2_adj_absolute", ascending=False)
    cmp.to_csv(table_path(cfg, "variance_relative_vs_absolute.csv"))
    print("\nVARIANCE EXPLAINED: relative vs absolute scale\n")
    print(cmp.to_string(float_format=lambda v: f"{v:.4f}"))

    # ---- 2. per-gene agreement between scales ----------------------------
    rows = []
    for g in genes:
        r, a = rel[g].values, absol[g].values
        rows.append({"gene": g,
                     "spearman_rel_vs_abs": spearmanr(r, a).statistic,
                     "median_relative": float(np.median(r)),
                     "median_absolute": float(np.median(a))})
    per_gene = pd.DataFrame(rows).sort_values("spearman_rel_vs_abs")
    per_gene.to_csv(table_path(cfg, "absolute_vs_relative_genes.csv"), index=False)
    LOG.info("Per-gene rank agreement between scales: median rho %.3f, "
             "%d/%d genes below 0.8",
             float(per_gene["spearman_rel_vs_abs"].median()),
             int((per_gene["spearman_rel_vs_abs"] < 0.8).sum()), len(per_gene))
    print("\nGenes whose ranking changes most between scales:")
    print(per_gene.head(10).to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # ---- 3. do population comparisons reorder? ---------------------------
    load_rel = pd.Series(np.log10(np.exp(rel).sum(axis=1)), index=rel.index)
    load_abs = pd.Series(np.log10(np.exp(absol).sum(axis=1)), index=absol.index)
    sti = st.set_index("sample").reindex(rel.index)

    out = []
    for key in ("country", "disease_group", "disease_detail", "study"):
        mr = group_medians(load_rel, sti, key)
        ma = group_medians(load_abs, sti, key)
        if mr is None or ma is None or len(mr) < 3:
            continue
        common_idx = mr.index.intersection(ma.index)
        rho = spearmanr(mr[common_idx], ma[common_idx]).statistic
        # how many pairs swap order between the two scales
        swaps, total = 0, 0
        idx = list(common_idx)
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                total += 1
                if np.sign(mr[idx[i]] - mr[idx[j]]) != np.sign(ma[idx[i]] - ma[idx[j]]):
                    swaps += 1
        LOG.info("%-14s: %d groups, rank agreement rho=%.3f, %d/%d pairs reorder",
                 key, len(idx), rho, swaps, total)
        out.append({"grouping": key, "n_groups": len(idx), "rank_rho": rho,
                    "pairs_reordered": swaps, "pairs_total": total,
                    "frac_reordered": swaps / max(total, 1)})
        if key == "country":
            print(f"\nCountry ranking by ARG burden ({len(idx)} countries):")
            print(pd.DataFrame({"relative": mr[common_idx].rank(ascending=False),
                                "absolute": ma[common_idx].rank(ascending=False)})
                  .sort_values("relative").to_string())
    if out:
        pd.DataFrame(out).to_csv(table_path(cfg, "absolute_vs_relative_groups.csv"), index=False)

    # ---- 4. is the difference systematic? --------------------------------
    try:
        stat, p = wilcoxon(load_rel.values, load_abs.values)
        LOG.info("Relative vs absolute total burden, paired Wilcoxon p=%.2e", p)
    except ValueError:
        pass
    LOG.info("Correlation of total burden between scales: rho=%.3f",
             spearmanr(load_rel, load_abs).statistic)
    LOG.info("Wrote absolute-scale tables to %s", tabdir)


if __name__ == "__main__":
    main()
