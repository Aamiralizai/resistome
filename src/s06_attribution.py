"""
s06_attribution.py
==================
The step that turns a prediction paper into a discovery paper.

1. Integrated gradients over the trained network give a species x ARG
   attribution matrix: which taxa drive predicted abundance of which genes.

2. That matrix is then confronted with an INDEPENDENT source of truth - the
   ARG content actually called in reference genomes of those same species.
   Three outcomes, and the third is the finding:

   confirmed   strong attribution AND the species genuinely carries the gene
               -> sanity check, the model learned real biology
   absent      weak attribution, gene absent from genomes
               -> uninteresting
   UNEXPLAINED strong attribution but the gene is NOT in that species' known
               genome content -> candidate horizontal transfer, mobile-element
               carriage, co-occurrence through a shared reservoir, or an
               unrecognised host. These are the candidates worth writing up.

Preparing the genome evidence
-----------------------------
This script does not download genomes. Produce the evidence table separately:

  a) take the species list written to tables/species_for_genome_check.csv
  b) fetch representative RefSeq genomes per species (NCBI datasets CLI)
  c) run AMRFinderPlus on each:
        amrfinder -n genome.fna --plus -o out.tsv
  d) concatenate into one TSV with at least the columns:
        species, gene
     and point --genome-evidence at it

Without that file the script still runs and writes the attribution matrix,
skipping the validation half.

Outputs:
    work_dir/attribution.parquet             species x ARG signed attribution
    tables/species_for_genome_check.csv
    tables/unexplained_associations.csv      the candidate list
    tables/attribution_validation.csv        confusion summary

Usage:
    python src/s06_attribution.py
    python src/s06_attribution.py --genome-evidence X:/w1_resistome/amrfinder_all.tsv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import LOG, load_config, work_path

try:
    import torch
    import torch.nn as nn
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


def load_model(cfg):
    from s05_models import RankPool, TaxaNet  # noqa: F401
    ckpt = torch.load(work_path(cfg, "model_taxa_nn.pt", per_stratum=True), map_location="cpu",
                      weights_only=False)
    ga, n_g, fa, n_f = ckpt["assign"]
    model = TaxaNet(len(ckpt["species"]), torch.as_tensor(ga, dtype=torch.long), n_g,
                    torch.as_tensor(fa, dtype=torch.long), n_f,
                    ckpt["n_cov"], len(ckpt["genes"]),
                    hidden=tuple(cfg["model"]["hidden"]),
                    dropout=cfg["model"]["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def integrated_gradients(model, X: np.ndarray, cov: np.ndarray, n_steps: int = 16,
                         batch: int = 256, max_samples: int = 2000,
                         seed: int = 0) -> np.ndarray:
    """Attribution of every output to every input feature, averaged over samples.

    Baseline is the all-zero (post-standardisation, i.e. mean) composition,
    which is the right null here: attribution answers "relative to an average
    community, what does having more of this taxon do to this gene?"
    """
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(dev)
    n_out = model.head.out_features

    # Cost is n_samples x n_steps x n_outputs backward passes. Attribution is
    # an average over samples, so a random subsample of a couple of thousand
    # gives the same matrix to three decimals at a fraction of the cost.
    if max_samples and len(X) > max_samples:
        idx = np.random.default_rng(seed).choice(len(X), max_samples, replace=False)
        X, cov = X[idx], cov[idx]
        LOG.info("Attribution on a random subsample of %d samples.", max_samples)

    n_back = int(np.ceil(len(X) / batch)) * n_steps * n_out
    LOG.info("Integrated gradients: %d samples x %d steps x %d outputs "
             "= %d backward passes", len(X), n_steps, n_out, n_back)
    total = np.zeros((X.shape[1], n_out), dtype=np.float64)

    for start in range(0, len(X), batch):
        xb = torch.as_tensor(X[start:start + batch], dtype=torch.float32, device=dev)
        cb = torch.as_tensor(cov[start:start + batch], dtype=torch.float32, device=dev)
        base = torch.zeros_like(xb)
        grads = torch.zeros_like(xb).unsqueeze(-1).repeat(1, 1, n_out)
        for a in np.linspace(1.0 / n_steps, 1.0, n_steps):
            xi = (base + a * (xb - base)).clone().requires_grad_(True)
            out = model(xi, cb)
            for j in range(n_out):
                g = torch.autograd.grad(out[:, j].sum(), xi, retain_graph=(j < n_out - 1))[0]
                grads[:, :, j] += g.detach()
        ig = ((xb - base).unsqueeze(-1) * grads / n_steps).sum(dim=0)
        total += ig.detach().cpu().numpy()
        LOG.info("  IG: %d/%d samples", min(start + batch, len(X)), len(X))
    return total / len(X)


def normalise_species(s: pd.Series) -> pd.Series:
    """Harmonise species strings so MetaPhlAn and AMRFinder names can meet."""
    return (s.astype(str)
             .str.replace("_", " ", regex=False)
             .str.replace(r"\[|\]", "", regex=True)
             .str.strip()
             .str.lower()
             .str.replace(r"\s+", " ", regex=True))


def normalise_gene(s: pd.Series) -> pd.Series:
    """Reduce a gene name to the family level used by the ARG matrix.

    The ARG matrix holds ResFinder families collapsed from allelic variants
    (blaSHV-1, blaSHV-11 -> blaSHV), but AMRFinder reports the allele. Without
    stripping the suffix, blaSHV never matches blaSHV-1, and canonical
    chromosomal genes - blaSHV and oqxAB in Klebsiella pneumoniae, mdf(A) in
    Escherichia coli - are scored as 'unexplained' when they are in fact the
    strongest possible confirmations.
    """
    x = s.astype(str).str.strip()
    # blaSHV-11 -> blaSHV ; aac(6')-Ib-cr stays distinct from aac(6')-Ib
    x = x.str.replace(r"-\d+[A-Za-z]?$", "", regex=True)
    x = x.str.replace(r"_\d+$", "", regex=True)
    return (x.str.lower()
             .str.replace(r"[-_]", "", regex=True)
             .str.replace(r"\s+", "", regex=True))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--genome-evidence", default=None,
                    help="TSV with columns species,gene from AMRFinderPlus on RefSeq")
    ap.add_argument("--max-samples", type=int, default=2000,
                    help="random subsample for attribution (0 = all)")
    ap.add_argument("--n-steps", type=int, default=16,
                    help="integrated-gradients interpolation steps")
    ap.add_argument("--min-genome-frequency", type=float, default=0.0,
                    help="for pangenome evidence: minimum fraction of assemblies "
                         "carrying the gene for it to count as known")
    ap.add_argument("--top-quantile", type=float, default=0.99,
                    help="attribution quantile defining a 'strong' association")
    args = ap.parse_args()

    cfg = load_config()
    tabdir = Path(cfg["paths"]["table_dir"])
    if not HAVE_TORCH:
        raise SystemExit("PyTorch required for attribution.")

    model, ckpt = load_model(cfg)
    want = [str(c) for c in ckpt["species"]]
    tt = ckpt.get("taxa_transform") or {}

    def _rebuild_from_unfiltered(pc: float | None):
        """Reproduce the training matrix.

        The centred log-ratio is computed over a species SET, so the same
        species has different values under a different subset. Attribution
        must therefore rebuild the matrix from the untransformed abundances
        using the model's own species list, not read the filtered matrix.
        """
        rp = work_path(cfg, "taxa_relab_unfiltered.parquet")
        if not rp.exists():
            return None
        rel = pd.read_parquet(rp)
        cols = want
        if not set(cols) <= set(rel.columns):
            alt = [c.replace("s__", "") for c in cols]
            if set(alt) <= set(rel.columns):
                cols = alt
            else:
                return None
        sub = rel[cols]
        if pc is None:
            nz = sub.values[sub.values > 0]
            pc = float(nz.min() / 2.0) if nz.size else 1e-6
            LOG.warning("Checkpoint has no stored pseudocount; estimating %.3g "
                        "from all samples. Values may differ slightly from "
                        "training.", pc)
        lx = np.log(sub.values + pc)
        return pd.DataFrame(lx - lx.mean(axis=1, keepdims=True),
                            index=sub.index, columns=sub.columns)

    taxa = None
    if tt.get("species"):
        taxa = _rebuild_from_unfiltered(tt.get("pseudocount"))
        if taxa is not None:
            LOG.info("Rebuilt the training matrix from %s using the stored "
                     "pseudocount (%.3g) and %d species.",
                     tt.get("source", "unfiltered abundances"),
                     tt.get("pseudocount", float("nan")), taxa.shape[1])
    if taxa is None:
        filt = pd.read_parquet(work_path(cfg, "taxa_clr.parquet"))
        cols = want if set(want) <= set(filt.columns) else \
            [w.replace("s__", "") for w in want]
        if set(cols) <= set(filt.columns):
            taxa = filt[cols]
            LOG.info("Using the filtered matrix; species sets match.")
        else:
            missing = len(set(cols) - set(filt.columns))
            raise SystemExit(
                f"{missing} of {len(cols)} model species are absent from both "
                "taxa_clr.parquet and taxa_relab_unfiltered.parquet. Rerun s05 "
                "so the checkpoint records its transform, or rerun s03 first.")
    st = pd.read_parquet(work_path(cfg, "sample_table.parquet")).set_index("sample")
    st = st.reindex(taxa.index)

    X = (taxa.values - ckpt["x_mean"]) / ckpt["x_scale"]
    cov = np.zeros((len(X), ckpt["n_cov"]))
    if ckpt["n_cov"] and "reads_after_qc" in st.columns:
        d = pd.to_numeric(st["reads_after_qc"], errors="coerce")
        cov[:, 0] = np.log1p(d.fillna(d.median())).values

    A = integrated_gradients(model, X, cov, n_steps=args.n_steps,
                             max_samples=args.max_samples)
    attr = pd.DataFrame(A, index=ckpt["species"], columns=ckpt["genes"])
    attr.to_parquet(work_path(cfg, "attribution.parquet", per_stratum=True))
    LOG.info("Attribution matrix: %d species x %d genes", *attr.shape)

    pd.Series(ckpt["species"], name="species").to_frame().to_csv(
        tabdir / "species_for_genome_check.csv", index=False)

    long = (attr.stack().rename("attribution").reset_index()
            .rename(columns={"level_0": "species", "level_1": "gene"}))
    thr = long["attribution"].quantile(args.top_quantile)
    long["strong"] = long["attribution"] >= thr
    LOG.info("Strong associations (>= q%.2f = %.4f): %d",
             args.top_quantile, thr, int(long["strong"].sum()))

    if not args.genome_evidence:
        long.sort_values("attribution", ascending=False).head(2000).to_csv(
            tabdir / "top_associations.csv", index=False)
        LOG.warning("No --genome-evidence supplied; validation skipped. "
                    "Top associations written for manual inspection.")
        return

    ev = pd.read_csv(args.genome_evidence, sep="\t")

    # Pangenome evidence carries a frequency column. Presence in one assembly
    # out of twenty means accessory and mobile - exactly the interesting case -
    # so the default keeps every hit, but the threshold is available.
    if "frequency" in ev.columns:
        n_tot = len(ev)
        if args.min_genome_frequency > 0:
            ev = ev[ev["frequency"] >= args.min_genome_frequency]
        core = (ev["frequency"] >= 0.9).sum()
        acc = (ev["frequency"] < 0.5).sum()
        LOG.info("Pangenome evidence: %d pairs (%d retained), %d core (>=90%% of "
                 "assemblies), %d accessory (<50%%)", n_tot, len(ev), core, acc)
    scol = next(c for c in ev.columns if "species" in c.lower() or "organism" in c.lower())
    gcol = next(c for c in ev.columns if c.lower() in ("gene", "gene_symbol", "element symbol")
                or "symbol" in c.lower())
    ev = ev[[scol, gcol]].rename(columns={scol: "species", gcol: "gene"}).drop_duplicates()
    ev["sk"], ev["gk"] = normalise_species(ev["species"]), normalise_gene(ev["gene"])
    known = set(zip(ev["sk"], ev["gk"]))
    LOG.info("Genome evidence: %d species-gene pairs across %d species",
             len(known), ev["sk"].nunique())

    # Sanity check the name harmonisation before trusting any category counts.
    # If few evidence genes map onto modelled ARG families, the categories are
    # measuring a string mismatch rather than biology.
    model_genes = set(normalise_gene(pd.Series(ckpt["genes"])))
    ev_genes = set(ev["gk"])
    overlap = model_genes & ev_genes
    LOG.info("Gene-name harmonisation: %d / %d modelled ARG families also appear "
             "in the genome evidence (%d distinct evidence genes)",
             len(overlap), len(model_genes), len(ev_genes))
    if len(overlap) < 0.2 * len(model_genes):
        LOG.warning("Fewer than 20%% of modelled genes match any evidence gene. "
                    "The confirmed/unexplained split is probably a naming "
                    "artefact - inspect a few rows of the evidence file.")

    long["sk"] = normalise_species(long["species"])
    long["gk"] = normalise_gene(long["gene"])
    covered = set(ev["sk"])
    long["species_has_genomes"] = long["sk"].isin(covered)
    long["in_genome"] = [tuple(x) in known for x in zip(long["sk"], long["gk"])]

    testable = long[long["species_has_genomes"]].copy()
    testable["category"] = np.select(
        [testable["strong"] & testable["in_genome"],
         testable["strong"] & ~testable["in_genome"],
         ~testable["strong"] & testable["in_genome"]],
        ["confirmed", "unexplained", "missed"], default="absent")

    summary = testable["category"].value_counts().rename_axis("category").reset_index(name="n")
    summary.to_csv(tabdir / "attribution_validation.csv", index=False)

    unexplained = (testable[testable["category"] == "unexplained"]
                   .sort_values("attribution", ascending=False)
                   [["species", "gene", "attribution"]])
    unexplained.to_csv(tabdir / "unexplained_associations.csv", index=False)

    conf = testable[testable["category"] == "confirmed"].shape[0]
    strong = testable["strong"].sum()
    print("\n" + summary.to_string(index=False))
    LOG.info("Precision against genome evidence: %.1f%% of strong associations confirmed",
             100 * conf / max(strong, 1))
    LOG.info("%d unexplained associations written to %s",
             len(unexplained), tabdir / "unexplained_associations.csv")
    LOG.info("Species without any genome evidence are excluded from this table - "
             "check that exclusion is not systematically dropping the "
             "understudied taxa your finding depends on.")


if __name__ == "__main__":
    main()
