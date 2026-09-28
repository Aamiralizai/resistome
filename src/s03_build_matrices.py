"""
s03_build_matrices.py
=====================
Builds the two modelling matrices.

ARG matrix (response)
---------------------
ARG.h5 is a single table `/ARG` with one row per run x ResFinder reference,
and it already carries everything needed:
    fragmentCountAln     aligned fragment count
    refSequence_length   reference length, for length normalisation
    bacterial_fragment   rRNA-derived bacterial fragment count for that run
The last column is the normalisation denominator, so rRNA.h5 is NOT required.
This matches the authors' own analysis code, which computes
    log( sum(fragmentCountAln) / (bacterial_fragment / 1e6) )
taking the max bacterial_fragment per run.

Steps: filter to mapped runs -> collapse ResFinder references to gene level
(many are allelic variants of one gene) -> sum counts across the runs of a
biological sample -> length-normalise -> divide by bacterial fragments per
million -> log.

Taxa matrix (predictor)
-----------------------
The Metalog MetaPhlAn 4 export is LONG: sample_alias / species / rel_abund,
with bare species labels like `s__Bacteroides_ovatus` rather than full
pipe-delimited lineages. Genus is therefore derived from the species label,
and family comes from the optional clade->lineage mapping file if present.

Outputs (parquet, in work_dir):
    arg_counts_sample.parquet   raw summed counts, sample x gene
    arg_abundance.parquet       log normalised abundance, sample x gene
    arg_absolute.parquet        optional, absolute-scale burden
    bacterial_fragments.parquet per-sample denominator
    taxa_relab.parquet / taxa_clr.parquet / taxa_lineage.parquet

Usage:  python src/s03_build_matrices.py
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

import common
from common import (LOG, clr, load_config, prevalence_filter, read_table,
                    resolve_roles, work_path)

GENE_STRIP = re.compile(r"-\d+[A-Za-z]?$")


def collapse_gene(ref: pd.Series) -> pd.Series:
    """blaACT-4_2_AJ311172 -> blaACT ;  tet(X4)_1_ABC -> tet(X)"""
    head = ref.astype(str).str.split("_").str[0]
    return head.str.replace(GENE_STRIP, "", regex=True)


def build_arg(cfg, run_map: pd.DataFrame):
    """Stream ARG.h5 -> (sample x gene counts, gene lengths, bacterial fragments)."""
    p = Path(cfg["paths"]["arg_h5"])
    run2sample = dict(zip(run_map["ena_run"], run_map["sample"]))
    wanted = set(run2sample)
    LOG.info("Streaming %s for %d runs", p, len(wanted))

    counts: dict[tuple[str, str], float] = {}
    breadth: dict[tuple[str, str], float] = {}
    glen: dict[str, list[float]] = {}
    bact_run: dict[str, float] = {}
    roles, n = None, 0

    ARG_COLS = ["run_accession", "refSequence", "fragmentCountAln",
                "refSequence_length", "refCoveredPositions", "bacterial_fragment"]
    for chunk in common.iter_table_chunks(p, usecols=ARG_COLS):
        n += 1
        if roles is None:
            roles = resolve_roles(chunk, {"run": True, "gene": True, "count": True}, "arg", cfg)
        c = chunk.rename(columns={roles["run"]: "ena_run",
                                  roles["gene"]: "reference",
                                  roles["count"]: "count"})
        c["ena_run"] = c["ena_run"].astype(str).str.strip()
        c = c[c["ena_run"].isin(wanted)]
        if c.empty:
            continue
        c["sample"] = c["ena_run"].map(run2sample)
        c["gene"] = collapse_gene(c["reference"])
        c["count"] = pd.to_numeric(c["count"], errors="coerce").fillna(0.0)

        for k, v in c.groupby(["sample", "gene"], observed=True)["count"].sum().items():
            counts[k] = counts.get(k, 0.0) + float(v)

        # breadth of coverage: fraction of the reference actually covered.
        # Take the best reference within a gene group, and the best run within
        # a sample - one well-covered observation is evidence of presence.
        if {"refCoveredPositions", "refSequence_length"} <= set(c.columns):
            cov = (pd.to_numeric(c["refCoveredPositions"], errors="coerce")
                   / pd.to_numeric(c["refSequence_length"], errors="coerce"))
            c = c.assign(_breadth=cov.clip(0, 1).fillna(0.0))
            for k, v in c.groupby(["sample", "gene"], observed=True)["_breadth"].max().items():
                breadth[k] = max(breadth.get(k, 0.0), float(v))

        if "refSequence_length" in c.columns:
            for g, L in c.groupby("gene")["refSequence_length"].median().items():
                glen.setdefault(g, []).append(float(L))

        if "bacterial_fragment" in c.columns:
            # one value per run; take max within the chunk, max across chunks
            for r, v in c.groupby("ena_run")["bacterial_fragment"].max().items():
                v = float(v) if pd.notna(v) else 0.0
                bact_run[r] = max(bact_run.get(r, 0.0), v)

        if n % 20 == 0:
            LOG.info("  chunk %d | cells=%d | runs seen=%d", n, len(counts), len(bact_run))

    if not counts:
        raise SystemExit("No ARG rows matched the run map. Check s02 output.")

    s = pd.Series(counts)
    s.index = pd.MultiIndex.from_tuples(s.index, names=["sample", "gene"])
    mat = s.unstack(fill_value=0.0)

    lengths = pd.Series({g: float(np.median(v)) for g, v in glen.items()}) if glen else None

    bact = pd.Series(bact_run, name="bacterial_fragment")
    bact.index.name = "ena_run"
    bs = (run_map.set_index("ena_run")["sample"].reindex(bact.index))
    bact_sample = bact.groupby(bs).sum()
    bact_sample.index.name = "sample"

    bmat = None
    if breadth:
        bs = pd.Series(breadth)
        bs.index = pd.MultiIndex.from_tuples(bs.index, names=["sample", "gene"])
        bmat = bs.unstack(fill_value=0.0)
        LOG.info("Coverage-breadth matrix: %d x %d (median non-zero breadth %.2f)",
                 *bmat.shape, float(bs[bs > 0].median()))

    LOG.info("ARG counts: %d samples x %d genes | denominator for %d samples",
             *mat.shape, len(bact_sample))
    return mat, lengths, bact_sample, bmat


def normalise_arg(counts, lengths, bact, pseudo, min_bact) -> pd.DataFrame:
    if lengths is None:
        LOG.warning("No refSequence_length column; skipping length normalisation.")
        per_kb = counts.copy()
    else:
        L = lengths.reindex(counts.columns).fillna(lengths.median())
        per_kb = counts.div(L.values / 1000.0, axis=1)

    d = bact.reindex(counts.index)
    bad = d.isna() | (d < min_bact)
    if bad.any():
        LOG.warning("%d samples dropped: missing or < %s bacterial fragments",
                    int(bad.sum()), f"{min_bact:,}")
    per_kb, d = per_kb.loc[~bad], d[~bad]
    return np.log(per_kb.div(d.values / 1e6, axis=0) + pseudo)


def load_mpa4(cfg, samples: set[str]):
    p = Path(cfg["paths"]["metalog_mpa4"])
    LOG.info("Loading MetaPhlAn 4 profiles: %s", p)
    df = read_table(p)
    cols = {c.lower(): c for c in df.columns}

    idcol = cols.get("sample_alias") or common.detect_column(df.columns, "sample", None)
    spcol = cols.get("species") or common.detect_column(df.columns, "taxon", None)
    valcol = next((c for c in df.columns
                   if re.search(r"rel_?ab|abund|value|count", c.lower())), None)
    if valcol is None:
        raise KeyError(f"No abundance column found in {p.name}. Columns: {list(df.columns)}")
    LOG.info("Long layout: id=%s clade=%s value=%s", idcol, spcol, valcol)

    df[idcol] = df[idcol].astype(str).str.strip()
    df = df[df[idcol].isin(samples)]
    LOG.info("Rows for our samples: %d (%d distinct samples)", len(df), df[idcol].nunique())
    mat = df.pivot_table(index=idcol, columns=spcol, values=valcol,
                         aggfunc="sum", fill_value=0.0)
    mat.index.name = "sample"

    rowsum = mat.sum(axis=1).replace(0, np.nan)
    mat = mat.div(rowsum, axis=0).fillna(0.0)

    lineage = build_lineage(cfg, mat.columns)
    return mat, lineage


def build_lineage(cfg, clades) -> pd.DataFrame:
    """Species -> genus (from the label) and family (from the optional map)."""
    sp = pd.Series([str(c) for c in clades], name="clade")
    bare = sp.str.replace(r"^s__", "", regex=True)
    genus = bare.str.split("_").str[0]
    lin = pd.DataFrame({"clade": sp, "species": bare, "genus": genus})
    lin["family"] = "unclassified_family"
    lin["phylum"] = "unclassified_phylum"

    lp = Path(cfg["paths"].get("metalog_lineage", ""))
    if lp.exists():
        try:
            m = read_table(lp)
            key = next((c for c in m.columns if "clade" in c.lower()), None)
            lcol = next((c for c in m.columns if "lineage" in c.lower()), None)
            if key and lcol:
                m = m[[key, lcol]].rename(columns={key: "clade", lcol: "lineage"})
                m["clade"] = m["clade"].astype(str).str.strip()
                m = m.drop_duplicates("clade")
                lin = lin.merge(m, on="clade", how="left")

                matched = lin["lineage"].notna()
                LOG.info("Lineage map: %d / %d clades matched (%.1f%%)",
                         int(matched.sum()), len(lin), 100 * matched.mean())
                if matched.mean() < 0.5:
                    LOG.warning("Under half the clades matched the lineage map. "
                                "Check that profile labels and clade_name use the "
                                "same convention (bare 's__X' vs full pipe path).")

                # Take EVERY rank from the map, genus included. Splitting the
                # species label fails on MetaPhlAn's unnamed GGB/SGB clades,
                # which is why genus came out near one-to-one with species.
                lg = lin["lineage"].astype(str)
                for rank, prefix in (("genus", "g__"), ("family", "f__"),
                                     ("order", "o__"), ("klass", "c__"),
                                     ("phylum", "p__")):
                    got = lg.str.extract(rf"{prefix}([^|;]+)", expand=False)
                    if rank == "genus":
                        # keep the label-derived genus only where the map is silent
                        lin["genus"] = got.fillna(lin["genus"])
                    else:
                        lin[rank] = got.fillna(f"unclassified_{rank}")
                LOG.info("From lineage map: %d genera, %d families, %d phyla",
                         lin["genus"].nunique(), lin["family"].nunique(),
                         lin["phylum"].nunique())
        except Exception as e:  # noqa: BLE001
            LOG.warning("Could not parse lineage map (%s); genus-only pooling.", e)
    else:
        LOG.warning("No clade->lineage map. Family pooling disabled; genus pooling "
                    "derived from species labels. Download the lineage file from "
                    "the Metalog Downloads page to enable family-level pooling.")
    n_fam = lin["family"].nunique()
    LOG.info("Taxonomy: %d species -> %d genera -> %d families",
             len(lin), lin["genus"].nunique(), n_fam)
    if n_fam <= 1:
        LOG.warning(
            "FAMILY POOLING IS DEGENERATE (%d family). The phylogeny-aware model "
            "will run with one of its two pooling levels replaced by a constant, "
            "and any conclusion about the architecture will be invalid. Download "
            "the clade->lineage map from the Metalog Downloads page (link in the "
            "taxonomic-profiles intro) and set paths.metalog_lineage.", n_fam)
    return lin


HIGH_RISK_PATTERNS = [
    r"^bla(NDM|KPC|OXA-?48|OXA-?23|OXA-?58|VIM|IMP|GES|CTX-?M)",
    r"^mcr",              # mobile colistin resistance
    r"^van[ABDGM]",       # vancomycin
    r"^cfr",              # phenicol-oxazolidinone
    r"^tet\(X",           # tigecycline
    r"^(optrA|poxtA)",    # oxazolidinone, mobile
    r"^(armA|rmt[A-H]|npmA)",  # 16S methyltransferases, pan-aminoglycoside
]


def high_risk_panel(counts: pd.DataFrame, breadth: pd.DataFrame | None = None,
                    min_breadth: float = 0.8, min_fragments: float = 2.0,
                    patterns=None) -> pd.DataFrame:
    """Presence/absence of clinically high-risk mobile ARGs.

    These genes are rare by construction - mcr, NDM, KPC, OXA-48, vanA,
    tet(X) - so a prevalence filter tuned for regression throws them away.
    They are precisely the genes the 'permissive community state' question is
    about, so they get their own binary panel.

    CRITICAL: presence is NOT `count > 0`. Short-read mapping against an ARG
    database yields abundant partial hits - a handful of reads aligning to a
    conserved fragment of a gene that is not actually present. Calling those
    positive produces nonsense like cfr(C) in 73% of human guts. Presence here
    requires the reference to be covered across at least `min_breadth` of its
    length AND to carry at least `min_fragments` aligned fragments.
    """
    pats = patterns or HIGH_RISK_PATTERNS
    rx = re.compile("|".join(pats), flags=re.IGNORECASE)
    hits = [g for g in counts.columns if rx.search(str(g))]
    if not hits:
        LOG.warning("No high-risk ARGs matched. Check gene naming after collapse.")
        return pd.DataFrame(index=counts.index)

    enough_reads = counts[hits] >= min_fragments
    if breadth is None:
        LOG.warning("No coverage-breadth matrix supplied; presence calls rest on "
                    "read count alone and WILL be inflated by partial hits.")
        panel = enough_reads.astype(int)
    else:
        b = breadth.reindex(index=counts.index, columns=hits).fillna(0.0)
        panel = (enough_reads & (b >= min_breadth)).astype(int)

    prev = panel.mean(axis=0).sort_values(ascending=False)
    keep = prev[prev > 0].index
    LOG.info("High-risk panel: %d genes matched, %d detected at >=%.0f%% breadth "
             "and >=%.0f fragments.", len(hits), len(keep), 100 * min_breadth,
             min_fragments)
    LOG.info("Carriage prevalence, top 15:\n%s", prev.head(15).to_string())
    LOG.info("Samples carrying >=1 high-risk ARG: %d / %d (%.1f%%)",
             int((panel.sum(axis=1) > 0).sum()), len(panel),
             100 * (panel.sum(axis=1) > 0).mean())
    modelable = prev[(prev * len(panel)) >= 50]
    LOG.info("Genes carried by >=50 samples (modelable): %d -> %s",
             len(modelable), list(modelable.index[:20]))
    return panel


def main() -> None:
    cfg = load_config()
    norm, filt = cfg["normalisation"], cfg["filters"]

    st = pd.read_parquet(work_path(cfg, "sample_table.parquet"))
    run_map = pd.read_parquet(work_path(cfg, "run_map.parquet"))
    samples = set(st["sample"].astype(str))
    LOG.info("Building matrices for %d samples / %d runs", len(samples), len(run_map))

    counts, lengths, bact, breadth = build_arg(cfg, run_map)
    counts.to_parquet(work_path(cfg, "arg_counts_sample.parquet"))
    if breadth is not None:
        breadth.to_parquet(work_path(cfg, "arg_breadth.parquet"))
    bact.to_frame().to_parquet(work_path(cfg, "bacterial_fragments.parquet"))

    arg = normalise_arg(counts, lengths, bact, norm["arg_pseudocount"],
                        filt.get("min_bacterial_fragments", 100))

    # Prevalence must be judged on RAW COUNTS. Inverting the log transform
    # leaves floating-point residue where a zero was, so every gene looks
    # "detected" everywhere and the filter silently becomes a no-op.
    counts.reindex(index=arg.index).to_parquet(
        work_path(cfg, "arg_counts_aligned.parquet"))   # for dev-only filtering
    raw = counts.reindex(index=arg.index, columns=arg.columns).fillna(0.0)
    prev = (raw > 0).mean(axis=0)
    # Save the normalised matrix BEFORE the pooled prevalence filter, so that
    # downstream feature selection can start from the full gene universe.
    # Filtering here and then re-filtering on development samples only leaves
    # the universe itself chosen with the held-out studies in view, which is
    # unsupervised leakage and contradicts a claim that all selection was
    # development-only.
    _unf = work_path(cfg, "arg_abundance_unfiltered.parquet")
    arg.to_parquet(_unf)
    LOG.info("Saved unfiltered ARG matrix (%d x %d) to %s for development-only "
             "feature selection downstream.", *arg.shape, _unf.name)

    keep = prev[prev >= filt["arg_min_prevalence"]].index
    LOG.info("ARG prevalence filter: kept %d / %d genes at >=%.0f%% of samples "
             "(median prevalence %.3f)", len(keep), arg.shape[1],
             100 * filt["arg_min_prevalence"], float(prev.median()))
    arg = arg[keep]
    arg.to_parquet(work_path(cfg, "arg_abundance.parquet"))
    LOG.info("ARG abundance matrix: %d x %d", *arg.shape)

    # Secondary, more permissive panel for sensitivity analysis.
    sec_thr = filt.get("arg_min_prevalence_secondary", 0.01)
    keep2 = prev[prev >= sec_thr].index
    LOG.info("Secondary ARG panel at >=%.0f%%: %d genes", 100 * sec_thr, len(keep2))
    (np.log(raw[keep2] + norm["arg_pseudocount"])
     .to_parquet(work_path(cfg, "arg_abundance_secondary.parquet")))

    # High-risk mobile ARGs: binary carriage, no prevalence filter.
    hr = cfg.get("high_risk", {}) or {}
    panel = high_risk_panel(
        counts.reindex(index=arg.index).fillna(0.0),
        breadth=None if breadth is None else breadth.reindex(index=arg.index),
        min_breadth=hr.get("min_breadth", 0.8),
        min_fragments=hr.get("min_fragments", 2))
    if panel.shape[1]:
        panel.to_parquet(work_path(cfg, "arg_highrisk_carriage.parquet"))

    load_cols = [c for c in st.columns if c.startswith("load_")
                 and pd.api.types.is_numeric_dtype(st[c])]
    if load_cols:
        load = st.set_index("sample")[load_cols[0]].reindex(arg.index)
        if load.notna().any():
            absolute = arg.add(np.log(load.clip(lower=1e-12)), axis=0).loc[load.notna()]
            absolute.to_parquet(work_path(cfg, "arg_absolute.parquet"))
            LOG.info("Absolute ARG burden: %d x %d (via %s)", *absolute.shape, load_cols[0])

    relab, lineage = load_mpa4(cfg, samples)
    relab = relab.loc[relab.index.isin(arg.index)]
    # The unfiltered copy must carry the SAME column names as the filtered
    # matrix. Stripping the s__ prefix from one and not the other means a model
    # trained on the unfiltered features cannot be matched back to taxa_clr or
    # to the lineage table, and taxonomic pooling silently degenerates.
    relab_raw = relab.copy()
    relab_raw.columns = [str(c).replace("s__", "") for c in relab_raw.columns]
    relab = prevalence_filter(relab, filt["taxa_min_prevalence"],
                              min_abund=filt["taxa_min_abundance"])
    relab.to_parquet(work_path(cfg, "taxa_relab.parquet"))
    # An UNFILTERED, untransformed copy is retained so that stage 5 can learn
    # prevalence thresholds and the CLR pseudocount from development studies
    # only. Filtering and transforming before the holdout is defined lets the
    # held-out studies influence which features exist.
    relab_raw.to_parquet(work_path(cfg, "taxa_relab_unfiltered.parquet"))

    lineage = lineage[lineage["clade"].isin(relab.columns)].copy()
    relab.columns = [str(c).replace("s__", "") for c in relab.columns]
    lineage["species"] = lineage["clade"].astype(str).str.replace("^s__", "", regex=True)
    lineage.drop_duplicates("species").to_parquet(
        work_path(cfg, "taxa_lineage.parquet"), index=False)

    taxa_clr = pd.DataFrame(clr(relab.values), index=relab.index, columns=relab.columns)
    taxa_clr.to_parquet(work_path(cfg, "taxa_clr.parquet"))

    shared = sorted(set(arg.index) & set(taxa_clr.index))
    LOG.info("=" * 60)
    LOG.info("FINAL ALIGNED DATASET: %d samples", len(shared))
    LOG.info("  ARG genes : %d", arg.shape[1])
    LOG.info("  species   : %d", taxa_clr.shape[1])
    LOG.info("=" * 60)
    pd.DataFrame({"sample": shared}).to_parquet(
        work_path(cfg, "aligned_samples.parquet"), index=False)


if __name__ == "__main__":
    main()
