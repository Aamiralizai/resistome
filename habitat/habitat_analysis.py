"""
habitat_analysis.py
===================
Cross-habitat resistome analysis, standalone.

This script does NOT modify or depend on the state of the human gut pipeline.
It imports only pure functions from it - chunked HDF5 reading, the centred
log-ratio transform, redundancy analysis, the model fitters - and manages its
own configuration, filtering and outputs. The gut pipeline's scripts, config
and work directory are untouched, so a habitat run cannot perturb a validated
result.

What it does, for one habitat:

  1. joins the habitat's Metalog samples to the Zenodo resistome resource by
     the same two bridges the gut analysis uses
  2. builds ARG and species matrices with the same normalisation
  3. partitions variance across the blocks that mean the same thing in every
     habitat - taxonomy, study, geography, sequencing depth
  4. evaluates cross-cohort prediction under leave-one-study-out

Host characteristics, health status and antibiotic exposure are omitted rather
than reported as zero: they do not exist outside the human data, and a zero
would imply they had been measured and found unimportant.

Usage:
    python habitat/habitat_analysis.py \\
        --habitat-dir /mnt/x/w1_resistome/Metalog_ocean \\
        --name ocean \\
        --outdir /mnt/x/w1_resistome/habitats

    # optional: restrict to a biome within a habitat
    python habitat/habitat_analysis.py \\
        --habitat-dir /mnt/x/w1_resistome/Metalog_environmental \\
        --name soil --biome-pattern 'soil|terrestrial|rhizosphere'
"""

from __future__ import annotations

import argparse
import glob
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# read-only imports from the gut pipeline: pure functions, no shared state
SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))
import common                                     # noqa: E402
from common import LOG, clr, read_table           # noqa: E402
from s04_variance_partition import (              # noqa: E402
    dummies, freedman_lane_p, numeric_block, permutation_p, rda_r2)

RUN_RX = r"^[EDS]RR\d+$"
SAMPLE_RX = r"^(SAM[EDN]A?\d+|SAMN\d+)$"
COMPARABLE_BLOCKS = ["taxonomy", "study", "geography", "technical"]


# --------------------------------------------------------------- inputs ----

def find(d: Path, *pats):
    for p in pats:
        hit = sorted(glob.glob(str(d / f"*{p}*")))
        if hit:
            return Path(hit[0])
    return None


def load_zenodo(zdir: Path):
    """Run-level metadata and ARG counts, shared across all habitats."""
    meta = pd.concat(list(common.iter_h5_chunks(zdir / "metadata.h5")),
                     ignore_index=True)
    meta = meta.rename(columns={"run_accession": "ena_run",
                                "sample_accession": "ena_sample"})
    for c in ("ena_run", "ena_sample"):
        if c in meta.columns:
            meta[c] = meta[c].astype(str).str.strip()
    LOG.info("Zenodo metadata: %d runs", meta["ena_run"].nunique())
    return meta


def build_run_map(mapping_f: Path, meta: pd.DataFrame,
                  aliases: set) -> pd.DataFrame:
    mp = read_table(mapping_f)
    for c in ("sample_alias", "external_id", "kind"):
        mp[c] = mp[c].astype(str).str.strip()
    mp["kind"] = mp["kind"].str.lower()
    mp = mp[mp["sample_alias"].isin(aliases)]

    a = mp[(mp["kind"] == "run") & mp["external_id"].str.match(RUN_RX, na=False)]
    a = (a[["sample_alias", "external_id"]].rename(columns={"external_id": "ena_run"})
         .merge(meta[["ena_run"]].drop_duplicates(), on="ena_run", how="inner"))

    b = pd.DataFrame(columns=["sample_alias", "ena_run"])
    if "ena_sample" in meta.columns:
        bb = mp[(mp["kind"] == "sample")
                & mp["external_id"].str.match(SAMPLE_RX, na=False)]
        b = (bb[["sample_alias", "external_id"]]
             .rename(columns={"external_id": "ena_sample"})
             .merge(meta[["ena_sample", "ena_run"]].drop_duplicates(),
                    on="ena_sample", how="inner")[["sample_alias", "ena_run"]])

    rm = (pd.concat([a[["sample_alias", "ena_run"]], b], ignore_index=True)
          .drop_duplicates().drop_duplicates("ena_run")
          .rename(columns={"sample_alias": "sample"}))
    LOG.info("Bridged: %d runs from %d samples (run bridge %d, sample bridge %d)",
             rm["ena_run"].nunique(), rm["sample"].nunique(),
             a["ena_run"].nunique(), b["ena_run"].nunique())
    return rm


# ------------------------------------------------------------- matrices ----

GENE_STRIP = re.compile(r"-\d+[A-Za-z]?$")


def build_arg(zdir: Path, run_map: pd.DataFrame):
    """Sample x gene counts, gene lengths and the bacterial-fragment denominator."""
    run2s = dict(zip(run_map["ena_run"], run_map["sample"]))
    wanted = set(run2s)
    counts, glen, bact = {}, {}, {}
    cols = ["run_accession", "refSequence", "fragmentCountAln",
            "refSequence_length", "bacterial_fragment"]
    for i, chunk in enumerate(common.iter_table_chunks(zdir / "ARG.h5", usecols=cols)):
        c = chunk.rename(columns={"run_accession": "ena_run",
                                  "refSequence": "reference",
                                  "fragmentCountAln": "count"})
        c["ena_run"] = c["ena_run"].astype(str).str.strip()
        c = c[c["ena_run"].isin(wanted)]
        if c.empty:
            continue
        c["sample"] = c["ena_run"].map(run2s)
        c["gene"] = (c["reference"].astype(str).str.split("_").str[0]
                     .str.replace(GENE_STRIP, "", regex=True))
        c["count"] = pd.to_numeric(c["count"], errors="coerce").fillna(0.0)
        for k, v in c.groupby(["sample", "gene"], observed=True)["count"].sum().items():
            counts[k] = counts.get(k, 0.0) + float(v)
        for g, L in c.groupby("gene")["refSequence_length"].median().items():
            glen.setdefault(g, []).append(float(L))
        for r, v in c.groupby("ena_run")["bacterial_fragment"].max().items():
            v = float(v) if pd.notna(v) else 0.0
            bact[r] = max(bact.get(r, 0.0), v)
        if i and i % 20 == 0:
            LOG.info("  ARG chunks %d, cells %d", i, len(counts))
    if not counts:
        raise SystemExit("No ARG rows matched this habitat's runs.")
    s = pd.Series(counts)
    s.index = pd.MultiIndex.from_tuples(s.index, names=["sample", "gene"])
    mat = s.unstack(fill_value=0.0)
    lengths = pd.Series({g: float(np.median(v)) for g, v in glen.items()})
    bs = run_map.set_index("ena_run")["sample"]
    bact_s = pd.Series(bact).groupby(bs.reindex(pd.Series(bact).index)).sum()
    LOG.info("ARG counts: %d samples x %d genes", *mat.shape)
    return mat, lengths, bact_s


def load_profiles(prof_f: Path, samples: set) -> pd.DataFrame:
    df = read_table(prof_f)
    cols = {c.lower(): c for c in df.columns}
    idc = cols.get("sample_alias") or df.columns[0]
    spc = cols.get("species") or next(
        (c for c in df.columns if "species" in c.lower() or "clade" in c.lower()), None)
    valc = next((c for c in df.columns
                 if re.search(r"rel_?ab|abund|value", c.lower())), None)
    df[idc] = df[idc].astype(str).str.strip()
    df = df[df[idc].isin(samples)]
    mat = df.pivot_table(index=idc, columns=spc, values=valc,
                         aggfunc="sum", fill_value=0.0)
    mat.index.name = "sample"
    rs = mat.sum(axis=1).replace(0, np.nan)
    mat = mat.div(rs, axis=0).fillna(0.0)
    mat.columns = [str(c).replace("s__", "") for c in mat.columns]
    LOG.info("Species profiles: %d samples x %d species", *mat.shape)
    return mat


# -------------------------------------------------------------- analysis ---

def variance_partition(st, taxa, arg, n_perm=999, n_pcs=50):
    """Only blocks that mean the same thing in every habitat."""
    st = st.set_index("sample").reindex(taxa.index)
    blocks = {}
    npc = int(min(n_pcs, taxa.shape[1] - 1, len(taxa) - 2))
    blocks["taxonomy"] = PCA(n_components=npc, random_state=0).fit_transform(
        StandardScaler().fit_transform(taxa.values))
    if "study" in st.columns:
        blocks["study"] = dummies(st["study"], prefix="study")
    if "country" in st.columns:
        blocks["geography"] = dummies(st["country"], prefix="country")
    if "reads_after_qc" in st.columns:
        blocks["technical"] = numeric_block(st, ["reads_after_qc"])
    blocks = {k: v for k, v in blocks.items() if v.shape[1] > 0}
    LOG.info("Blocks: %s", {k: v.shape[1] for k, v in blocks.items()})

    Y = StandardScaler().fit_transform(arg.values)
    grp = st["study"].astype(str).values if "study" in st.columns else None
    rows = []
    for name, X in blocks.items():
        r2, r2adj, df = rda_r2(Y, X)
        p_m = permutation_p(Y, X, None, r2, n_perm=n_perm, groups=grp)
        others = [v for k, v in blocks.items() if k != name]
        Z = np.column_stack(others) if others else None
        part, part_adj, _ = rda_r2(Y, X, Z)
        if Z is not None:
            full, adj_f, _ = rda_r2(Y, np.column_stack([X, Z]))
            red, adj_r, _ = rda_r2(Y, Z)
            uniq, uniq_adj = full - red, adj_f - adj_r
        else:
            uniq, uniq_adj = r2, r2adj
        p_p = freedman_lane_p(Y, X, Z, part, n_perm=n_perm, groups=grp)
        rows.append({"block": name, "n_terms": df,
                     "marginal_R2_adj": r2adj, "marginal_p": p_m,
                     "partial_R2_adj": part_adj, "partial_p": p_p,
                     "unique_total_fraction_adj": uniq_adj})
        LOG.info("%-10s marginal=%.4f (p=%s) partial=%.4f unique=%.4f",
                 name, r2adj, f"{p_m:.3f}" if np.isfinite(p_m) else "n/a",
                 part_adj, uniq_adj)
    return pd.DataFrame(rows).sort_values("unique_total_fraction_adj",
                                          ascending=False)


def leave_one_study_out(st, taxa, arg, min_test=20):
    """Cross-cohort predictability, ridge and gradient boosting."""
    idx = taxa.index
    groups = st.set_index("sample").reindex(idx)["study"].astype(str)
    X, Y = taxa.values, arg.values
    try:
        from s05_models import fit_xgboost
        have_xgb = True
    except Exception:  # noqa: BLE001
        have_xgb = False
    rows = []
    for g in groups.value_counts().index:
        te = (groups == g).values
        tr = ~te
        if te.sum() < min_test or tr.sum() < 100:
            continue
        xs, ys = StandardScaler().fit(X[tr]), StandardScaler().fit(Y[tr])
        preds = {"ridge": ys.inverse_transform(
            Ridge(alpha=10.0).fit(xs.transform(X[tr]), ys.transform(Y[tr]))
            .predict(xs.transform(X[te])))}
        if have_xgb:
            preds["xgb"] = ys.inverse_transform(
                fit_xgboost(xs.transform(X[tr]), ys.transform(Y[tr]),
                            xs.transform(X[te])))
        for m, P in preds.items():
            rho = [spearmanr(Y[te][:, j], P[:, j]).statistic
                   for j in range(Y.shape[1]) if np.std(Y[te][:, j]) > 1e-9]
            rows.append({"model": m, "study": str(g), "n_test": int(te.sum()),
                         "median_spearman": float(np.nanmedian(rho))})
        LOG.info("  fold %-30s n=%4d  %s", str(g)[:30], te.sum(),
                 {m: f"{np.nanmedian([spearmanr(Y[te][:, j], P[:, j]).statistic for j in range(Y.shape[1]) if np.std(Y[te][:, j]) > 1e-9]):.3f}"
                  for m, P in preds.items()})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ main ---

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--habitat-dir", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--zenodo-dir", default="/mnt/x/w1_resistome/zenodo")
    ap.add_argument("--mapping", default=None,
                    help="Metalog sequencing_db_mapping file (defaults to the "
                         "one beside the gut Metalog data)")
    ap.add_argument("--outdir", default="/mnt/x/w1_resistome/habitats")
    ap.add_argument("--biome-pattern", default=None,
                    help="regex on environment_biome to select a sub-habitat")
    ap.add_argument("--min-arg-prevalence", type=float, default=0.05)
    ap.add_argument("--min-species-prevalence", type=float, default=0.05)
    ap.add_argument("--n-perm", type=int, default=999)
    ap.add_argument("--skip-models", action="store_true")
    args = ap.parse_args()

    hd, zdir = Path(args.habitat_dir), Path(args.zenodo_dir)
    out = Path(args.outdir) / args.name
    out.mkdir(parents=True, exist_ok=True)
    LOG.info("=" * 62)
    LOG.info("HABITAT: %s   ->  %s", args.name, out)
    LOG.info("=" * 62)

    mapping = Path(args.mapping) if args.mapping else find(
        hd.parent / "Metalog", "sequencing_db_mapping")
    if mapping is None or not mapping.exists():
        mapping = find(hd, "sequencing_db_mapping") or find(
            hd.parent, "sequencing_db_mapping")
    if mapping is None:
        raise SystemExit("Could not locate sequencing_db_mapping; pass --mapping.")
    LOG.info("mapping file: %s", mapping)

    meta_f = find(hd, "extended_wide", "core_wide", "wide")
    prof_f = find(hd, "metaphlan4_species", "metaphlan4")
    reads_f = find(hd, "read_counts")
    if not (meta_f and prof_f):
        raise SystemExit(f"{hd}: metadata or profiles missing.")

    md = read_table(meta_f)
    md["sample_alias"] = md["sample_alias"].astype(str).str.strip()
    if "artificial" in md.columns:
        n0 = len(md); md = md[md["artificial"].isna()]
        LOG.info("Excluded %d artificial samples", n0 - len(md))
    if args.biome_pattern and "environment_biome" in md.columns:
        n0 = len(md)
        md = md[md["environment_biome"].astype(str)
                .str.contains(args.biome_pattern, case=False, regex=True, na=False)]
        LOG.info("Biome filter '%s': %d of %d retained",
                 args.biome_pattern, len(md), n0)
    LOG.info("Habitat metadata: %d samples", len(md))

    zmeta = load_zenodo(zdir)
    run_map = build_run_map(mapping, zmeta, set(md["sample_alias"]))
    if run_map.empty:
        raise SystemExit("No samples bridged to the resistome resource.")

    counts, lengths, bact = build_arg(zdir, run_map)
    L = lengths.reindex(counts.columns).fillna(lengths.median())
    per_kb = counts.div(L.values / 1000.0, axis=1)
    d = bact.reindex(counts.index)
    ok = d.notna() & (d >= 100)
    arg = np.log(per_kb.loc[ok].div(d[ok].values / 1e6, axis=0) + 0.001)
    prev = (counts.reindex(index=arg.index) > 0).mean(axis=0)
    keep = prev[prev >= args.min_arg_prevalence].index
    arg = arg[keep]
    LOG.info("ARG matrix: %d samples x %d genes (>=%.0f%% prevalence)",
             *arg.shape, 100 * args.min_arg_prevalence)

    relab = load_profiles(prof_f, set(arg.index))
    relab = relab.loc[relab.index.isin(arg.index)]
    pres = (relab > 0.0001).mean(axis=0)
    relab = relab[pres[pres >= args.min_species_prevalence].index]
    taxa = pd.DataFrame(clr(relab.values), index=relab.index, columns=relab.columns)

    st = md.rename(columns={"sample_alias": "sample", "study_code": "study",
                            "geographic_location": "country"})
    keepc = [c for c in ("sample", "study", "country", "environment_biome",
                         "environment_material") if c in st.columns]
    st = st[keepc]
    if reads_f:
        rc = read_table(reads_f)
        rc = rc.rename(columns={rc.columns[0]: "sample"})
        rc["sample"] = rc["sample"].astype(str).str.strip()
        num = [c for c in rc.columns if c != "sample"
               and pd.api.types.is_numeric_dtype(rc[c])]
        if num:
            rc = rc[["sample", num[0]]].rename(columns={num[0]: "reads_after_qc"})
            st = st.merge(rc.drop_duplicates("sample"), on="sample", how="left")

    shared = sorted(set(arg.index) & set(taxa.index) & set(st["sample"]))
    arg, taxa = arg.loc[shared], taxa.loc[shared]
    st = st[st["sample"].isin(shared)]
    LOG.info("=" * 62)
    LOG.info("ANALYSED: %d samples | %d ARG genes | %d species | %d studies",
             len(shared), arg.shape[1], taxa.shape[1],
             st["study"].nunique() if "study" in st.columns else -1)
    LOG.info("=" * 62)
    if len(shared) < 300:
        LOG.warning("Fewer than 300 samples; a variance partition here will be "
                    "unstable and should not be compared with larger habitats.")

    arg.to_parquet(out / "arg_abundance.parquet")
    taxa.to_parquet(out / "taxa_clr.parquet")
    st.to_parquet(out / "sample_table.parquet", index=False)

    vp = variance_partition(st, taxa, arg, n_perm=args.n_perm)
    vp.insert(0, "habitat", args.name)
    vp.insert(1, "n_samples", len(shared))
    vp.insert(2, "n_studies", st["study"].nunique() if "study" in st else np.nan)
    vp.to_csv(out / "variance_partition.csv", index=False)
    print("\n" + vp.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    if not args.skip_models:
        LOG.info("--- leave-one-study-out prediction ---")
        cv = leave_one_study_out(st, taxa, arg)
        if len(cv):
            cv.insert(0, "habitat", args.name)
            cv.to_csv(out / "model_performance.csv", index=False)
            summ = cv.groupby("model")["median_spearman"].agg(["median", "size"])
            print("\n" + summ.to_string(float_format=lambda v: f"{v:.4f}"))

    LOG.info("Outputs in %s", out)


if __name__ == "__main__":
    main()
