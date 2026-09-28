#!/usr/bin/env python3
"""Stage 22: build the supplementary tables that summarise the analysis.

Four tables are assembled from the result tables the earlier stages wrote,
together with the pangenome evidence:

    S27  validation ladder: random / subject-grouped / leave-one-study-out,
         with the random-to-LOSO drop apportioned between unseen individuals
         and unseen cohorts
    S28  conservation-predictability association under minimum-assembly
         thresholds, and allowing for measurement error in conservation
    S29  conservation and predictability within and across drug classes,
         including leave-one-class-out
    S30  family-level mapping between the ARG matrix (ResFinder families) and
         AMRFinderPlus symbols, with the genome support behind each
         conservation estimate

Nothing here is hard-coded. S27 is read from ``tables/model_summary.csv``,
which must contain the ``subject_grouped`` scheme, so s05 must have been run
with that scheme enabled. If it is absent the script stops rather than
silently reporting a two-rung ladder.

Usage
-----
    python src/s22_supplementary_tables.py \\
        --tables /path/to/results/tables \\
        --pangenome /path/to/pangenome_evidence_independent.tsv

    # also write the sheets into the supplementary workbook
    python src/s22_supplementary_tables.py --tables ... --pangenome ... \\
        --xlsx Supplementary_Tables.xlsx

Outputs, written into --out (default: the --tables directory):
    validation_ladder.csv
    conservation_min_assemblies.csv
    conservation_errors_in_variables.csv
    conservation_by_drug_class.csv
    conservation_leave_one_class_out.csv
    conservation_within_class_rank.csv
    conservation_partial_associations.csv
    family_mapping_65.csv

The errors-in-variables block is a parametric bootstrap and is therefore
seeded (default 42, 2000 replicates). Changing --seed or --n-boot changes the
interval; the manuscript quotes the defaults.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s06_attribution import normalise_gene  # noqa: E402

MODELS = ["xgb", "rf", "ft", "ridge", "taxa_nn", "mlp"]
SCHEMES = ["random", "subject_grouped", "leave_one_study_out"]


# --------------------------------------------------------------------- inputs
def load_conservation(tables: Path) -> pd.DataFrame:
    """Per-gene cross-cohort rho joined to within-species conservation."""
    for name in ("mobility_vs_predictability_adult_adult.csv",
                 "mobility_vs_predictability_adult.csv"):
        p = tables / name
        if p.exists():
            d = pd.read_csv(p)
            need = {"gene", "gk", "rho", "conservation", "n_species"}
            missing = need - set(d.columns)
            if missing:
                sys.exit(f"{p} lacks columns: {sorted(missing)}")
            return d
    sys.exit(f"no mobility_vs_predictability table in {tables}")


def gene_conservation(ev: pd.DataFrame) -> pd.DataFrame:
    """Family-level conservation from the raw evidence table.

    Kept deliberately identical to gene_mobility() in s13: group by normalised
    key, average the pangenome frequency over the species in which the family
    was DETECTED, and count species in which it was searched separately.
    Averaging over all species instead would confuse a gene absent from a
    species with a gene present in a minority of its strains.
    """
    ev = ev.copy()
    ev["gk"] = normalise_gene(ev["gene"])
    det = ev[ev["detected"] == 1] if "detected" in ev.columns \
        else ev[ev["frequency"] > 0]
    n_examined = ev.groupby("gk")["species"].nunique()
    g = det.groupby("gk").agg(
        n_host_species=("species", "nunique"),
        conservation=("frequency", "mean"),
        median_conservation=("frequency", "median"),
        max_frequency=("frequency", "max"),
    ).reset_index()
    g["n_species_examined"] = g["gk"].map(n_examined)
    return g


# ------------------------------------------------------------------- S27
def validation_ladder(tables: Path) -> pd.DataFrame:
    p = tables / "model_summary.csv"
    if not p.exists():
        sys.exit(f"{p} not found; run src/s05_models.py first")
    s = pd.read_csv(p)
    have = set(s["scheme"].unique())
    if "subject_grouped" not in have:
        sys.exit("model_summary.csv has no 'subject_grouped' scheme "
                 f"(found {sorted(have)}). Rerun s05 with the subject-grouped "
                 "scheme enabled, or the ladder cannot be built.")
    w = (s.pivot(index="model", columns="scheme", values="median_spearman")
          .reindex(columns=SCHEMES))
    w = w.dropna(how="any")
    if w.empty:
        sys.exit("no model has a value for all three schemes")
    w = w.reindex([m for m in MODELS if m in w.index] +
                  [m for m in w.index if m not in MODELS])
    w = w.reset_index().rename(columns={"index": "model"})
    w["drop_random_to_loso"] = (w["random"] - w["leave_one_study_out"]).round(4)
    w["share_from_unseen_individuals"] = (
        (w["random"] - w["subject_grouped"]) / w["drop_random_to_loso"]).round(3)
    w["share_from_unseen_cohorts"] = (
        1 - w["share_from_unseen_individuals"]).round(3)
    for c in SCHEMES:
        w[c] = w[c].round(4)
    return w


# ------------------------------------------------------------------- S28
def min_assembly_sweep(d: pd.DataFrame, ev: pd.DataFrame,
                       thresholds=(1, 3, 5, 10, 12)) -> pd.DataFrame:
    rows = []
    for k in thresholds:
        sub = ev[ev["n_genomes_total"] >= k].copy()
        g = gene_conservation(sub)
        m = d[["gk", "rho"]].merge(g[["gk", "conservation"]], on="gk")
        r = spearmanr(m["conservation"], m["rho"])
        rows.append({"min_assemblies_per_species": k,
                     "n_species_retained": int(sub["species"].nunique()),
                     "n_gene_families": len(m),
                     "spearman": round(float(r.statistic), 5),
                     "p_value": float(f"{r.pvalue:.3g}")})
    return pd.DataFrame(rows)


def errors_in_variables(d: pd.DataFrame, seed: int, n_boot: int) -> pd.DataFrame:
    """Conservation is a binomial proportion over at most twelve assemblies.

    Treating it as measured without error overstates the association. Each
    family's detection count is resampled from its own sampling distribution
    and the correlation recomputed, which gives an interval that includes the
    measurement error in the predictor rather than only sampling of families.
    """
    rng = np.random.default_rng(seed)
    n_trials = np.maximum(d["n_species"].values, 1) * 12
    p = d["conservation"].values.clip(1e-3, 1 - 1e-3)
    boot = np.array([spearmanr(rng.binomial(n_trials, p) / n_trials,
                               d["rho"]).statistic for _ in range(n_boot)])
    return pd.DataFrame({
        "quantity": ["point estimate (as reported)",
                     "errors-in-variables mean",
                     "errors-in-variables 2.5th percentile",
                     "errors-in-variables 97.5th percentile"],
        "spearman": [round(float(spearmanr(d["conservation"],
                                           d["rho"]).statistic), 5),
                     round(float(boot.mean()), 5),
                     round(float(np.percentile(boot, 2.5)), 5),
                     round(float(np.percentile(boot, 97.5)), 5)]})


# ------------------------------------------------- univariate and partial rho
def partial_associations(d: pd.DataFrame) -> pd.DataFrame:
    """The alternative-explanation controls quoted in the Results.

    s13 logs these but does not table them, which left several manuscript
    numbers with no deposited source. The residualisation is the same as in
    s13: rank-transform, remove a linear fit on the control, correlate the
    residuals.
    """
    def resid(y, x):
        return y - np.polyval(np.polyfit(x, y, 1), x)

    rows = []
    for ctrl, label in [("arg_sd", "abundance variance"),
                        ("prevalence", "adult prevalence")]:
        if ctrl not in d.columns:
            continue
        ok = d[["conservation", "rho", ctrl]].dropna()
        rk = ok.rank()
        r = spearmanr(resid(rk["conservation"].values, rk[ctrl].values),
                      resid(rk["rho"].values, rk[ctrl].values))
        rows.append({"quantity": f"conservation vs rho, controlling for {label}",
                     "spearman": round(float(r.statistic), 5),
                     "p_value": float(f"{r.pvalue:.3g}"), "n": len(ok)})
    for col, label in [("arg_sd", "abundance variance vs rho"),
                       ("prevalence", "adult prevalence vs rho"),
                       ("n_host_species", "conservation vs host species count")]:
        if col not in d.columns:
            continue
        x = d["conservation"] if col == "n_host_species" else d[col]
        y = d[col] if col == "n_host_species" else d["rho"]
        ok = pd.concat([x, y], axis=1).dropna()
        r = spearmanr(ok.iloc[:, 0], ok.iloc[:, 1])
        rows.append({"quantity": label,
                     "spearman": round(float(r.statistic), 5),
                     "p_value": float(f"{r.pvalue:.3g}"), "n": len(ok)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- S29
def drug_class(gene: str) -> str:
    g = str(gene).lower()
    if re.match(r"(aac|aad|aph|ant|str|rmt|arma)", g):
        return "Aminoglycoside"
    if re.match(r"(bla|cfx|ampc|cmy|oxa|tem|shv|ctx)", g):
        return "Beta-lactam"
    if re.match(r"tet", g):
        return "Tetracycline"
    if re.match(r"(erm|mef|msr|lnu|lsa|vat|vga|mph)", g):
        return "MLS"
    if re.match(r"van", g):
        return "Glycopeptide"
    if re.match(r"(sul|dfr|fol)", g):
        return "Folate pathway"
    if re.match(r"(cat|cml|flo|fex)", g):
        return "Phenicol"
    if re.match(r"(qnr|qep|oqx)", g):
        return "Quinolone"
    return "Other"


def by_drug_class(d: pd.DataFrame, min_members: int = 5):
    d = d.copy()
    d["drug_class"] = d["gene"].map(drug_class)
    counts = d["drug_class"].value_counts()

    summary = (d.groupby("drug_class")
               .agg(n_families=("gene", "size"),
                    median_conservation=("conservation", "median"),
                    median_rho=("rho", "median")).round(4)
               .sort_values("n_families", ascending=False).reset_index())

    loco = []
    for c in counts.index:
        s = d[d["drug_class"] != c]
        r = spearmanr(s["conservation"], s["rho"])
        loco.append({"class_excluded": c,
                     "n_families_excluded": int((d["drug_class"] == c).sum()),
                     "n_families_retained": len(s),
                     "spearman": round(float(r.statistic), 5),
                     "p_value": float(f"{r.pvalue:.3g}")})
    loco = pd.DataFrame(loco).sort_values("spearman").reset_index(drop=True)

    big = d[d["drug_class"].map(counts) >= min_members].copy()
    big["cons_rank_in_class"] = big.groupby("drug_class")["conservation"].rank()
    big["rho_rank_in_class"] = big.groupby("drug_class")["rho"].rank()
    wc = spearmanr(big["cons_rank_in_class"], big["rho_rank_in_class"])
    within = pd.DataFrame({
        "quantity": [f"within-class rank association (classes with >= "
                     f"{min_members} families)", "n gene families", "P value"],
        "value": [round(float(wc.statistic), 5), len(big),
                  float(f"{wc.pvalue:.3g}")]})
    return summary, loco, within


# ------------------------------------------------------------------- S30
def family_mapping(d: pd.DataFrame, ev: pd.DataFrame) -> pd.DataFrame:
    ev = ev.copy()
    ev["gk"] = normalise_gene(ev["gene"])
    det = ev[ev["detected"] == 1] if "detected" in ev.columns \
        else ev[ev["frequency"] > 0]
    asm = ev.groupby("species")["n_genomes_total"].max()

    rows = []
    for _, r in d.iterrows():
        g = det[det["gk"] == r["gk"]]
        hosts = sorted(g["species"].unique())
        rows.append({
            "resfinder_family": r["gene"],
            "normalised_key": r["gk"],
            "amrfinderplus_symbols": ", ".join(sorted(g["gene"].unique())),
            "n_host_species_detected": int(r.get("n_host_species", len(hosts))),
            "n_species_examined": int(r.get("n_species_examined", 0)),
            "n_species_allele_pairs": len(g),
            "n_assemblies_searched_in_hosts": int(asm.reindex(hosts).sum()),
            "n_assemblies_carrying_family":
                int(g.groupby("species")["n_genomes_with"].max().sum())
                if len(g) else 0,
            "conservation": round(float(r["conservation"]), 4),
            "median_conservation": round(float(r.get("median_conservation",
                                                     np.nan)), 4),
            "max_frequency": round(float(r.get("max_frequency", np.nan)), 4),
            "breadth": round(float(r.get("breadth", np.nan)), 4),
            "prevalence": round(float(r.get("prevalence", np.nan)), 4),
            "cross_cohort_rho": round(float(r["rho"]), 4),
            "_recomputed": round(float(g["frequency"].mean()), 4) if len(g) else None,
        })
    t = pd.DataFrame(rows)

    # The table is only worth shipping if it reproduces the published values.
    diff = (t["_recomputed"] - t["conservation"]).abs()
    n_bad = int((diff > 5e-5).sum()) + int(t["_recomputed"].isna().sum())
    if n_bad:
        print(t[diff > 5e-5].to_string(), file=sys.stderr)
        sys.exit(f"family mapping does not reproduce conservation for {n_bad} "
                 f"families; not writing")
    return t.drop(columns=["_recomputed"]).sort_values("conservation",
                                                       ascending=False)


# -------------------------------------------------------------------- output
def append_sheets(xlsx: Path, blocks: dict) -> None:
    import openpyxl
    from openpyxl.styles import Alignment, Font
    wb = openpyxl.load_workbook(xlsx)
    for name, (title, caption, frames) in blocks.items():
        if name in wb.sheetnames:
            del wb[name]
        ws = wb.create_sheet(name)
        ws["A1"] = title
        ws["A1"].font = Font(bold=True)
        ws["A2"] = caption
        ws["A2"].alignment = Alignment(wrap_text=True, vertical="top")
        r = 5
        for sub, frame in frames:
            if sub:
                ws.cell(r, 1, sub).font = Font(bold=True)
                r += 1
            for j, col in enumerate(frame.columns, 1):
                ws.cell(r, j, col).font = Font(bold=True)
            r += 1
            for _, row in frame.iterrows():
                for j, v in enumerate(row, 1):
                    ws.cell(r, j, v.item() if hasattr(v, "item") else v)
                r += 1
            r += 1
        ws.column_dimensions["A"].width = 34
        for c in "BCDEFGHIJKLMN":
            ws.column_dimensions[c].width = 17
    # keep the Contents sheet honest about what the workbook holds
    if "Contents" in wb.sheetnames:
        ws = wb["Contents"]
        listed = {str(c.value) for c in ws["A"] if c.value}
        row = ws.max_row
        for n in sorted(blocks, key=lambda s: int(s.lstrip("S"))):
            label = f"Supplementary Table {n.lstrip('S')}"
            if not any(label == s or s.startswith(label + " ") for s in listed):
                row += 1
                ws.cell(row, 1, label)
                ws.cell(row, 2, blocks[n][0].split(". ", 1)[-1])
    wb.save(xlsx)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", required=True, type=Path,
                    help="results tables directory")
    ap.add_argument("--pangenome", required=True, type=Path,
                    help="pangenome_evidence_independent.tsv (166 species)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--xlsx", type=Path, default=None,
                    help="supplementary workbook to refresh S27-S30 in")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--skip-ladder", action="store_true",
                    help="build S28-S30 only, when s05 has not been rerun")
    a = ap.parse_args()

    out = a.out or a.tables
    out.mkdir(parents=True, exist_ok=True)
    d = load_conservation(a.tables)
    ev = pd.read_csv(a.pangenome, sep="\t")
    print(f"gene families: {len(d)}   species in evidence: "
          f"{ev['species'].nunique()}   assemblies: "
          f"{int(ev.groupby('species')['n_genomes_total'].max().sum())}")

    written = []
    ladder = None
    if not a.skip_ladder:
        ladder = validation_ladder(a.tables)
        ladder.to_csv(out / "validation_ladder.csv", index=False)
        written.append("validation_ladder.csv")
        print("\n" + ladder.to_string(index=False))

    sweep = min_assembly_sweep(d, ev)
    eiv = errors_in_variables(d, a.seed, a.n_boot)
    summary, loco, within = by_drug_class(d)
    partials = partial_associations(d)
    mapping = family_mapping(d, ev)

    for frame, name in [(sweep, "conservation_min_assemblies.csv"),
                        (eiv, "conservation_errors_in_variables.csv"),
                        (summary, "conservation_by_drug_class.csv"),
                        (loco, "conservation_leave_one_class_out.csv"),
                        (within, "conservation_within_class_rank.csv"),
                        (partials, "conservation_partial_associations.csv"),
                        (mapping, "family_mapping_65.csv")]:
        frame.to_csv(out / name, index=False)
        written.append(name)

    print("\n" + eiv.to_string(index=False))
    print(f"\nwrote {len(written)} tables to {out}")
    for w in written:
        print("  " + w)

    if a.xlsx:
        blocks = {}
        if ladder is not None:
            blocks["S27"] = (
                "Supplementary Table 27. Validation ladder: random, "
                "subject-grouped and leave-one-study-out cross-validation",
                "Median per-gene Spearman correlation for each model family "
                "under three evaluation schemes. Random five-fold "
                "cross-validation splits on samples, so repeat samples from one "
                "individual appear in both training and test. Subject-grouped "
                "cross-validation assigns whole subjects to folds. "
                "Leave-one-study-out withholds each study in turn. The last two "
                "columns apportion the random-to-leave-one-study-out drop "
                "between evaluating on unseen individuals and on unseen "
                "cohorts.",
                [("", ladder)])
        blocks["S28"] = (
            "Supplementary Table 28. Conservation-predictability association "
            "under minimum-assembly thresholds and allowing for measurement "
            "error",
            "Upper block: the association recomputed after restricting the "
            "pangenome evidence to species with at least the stated number of "
            "sequenced assemblies. Lower block: conservation is a binomial "
            "proportion over at most twelve assemblies and is therefore "
            f"measured with error; each family's detection count was resampled "
            f"from its sampling distribution over {a.n_boot:,} replicates "
            f"(seed {a.seed}).",
            [("Minimum assemblies per species", sweep),
             ("Errors-in-variables resampling", eiv)])
        blocks["S29"] = (
            "Supplementary Table 29. Conservation and predictability within "
            "and across drug classes",
            "Conservation covaries with drug class, so the association could "
            "in principle reflect class membership alone. Upper block: "
            "conservation and predictability by class. Middle block: the "
            "association after ranking both quantities within class. Lower "
            "block: the association with each class excluded in turn.",
            [("Conservation and predictability by drug class", summary),
             ("Within-class rank association", within),
             ("Leave-one-class-out", loco),
             ("Alternative explanations, univariate and partial", partials)])
        blocks["S30"] = (
            "Supplementary Table 30. Family-level mapping between the ARG "
            "matrix and AMRFinderPlus, with genome support",
            "For each gene family in the conservation analysis: the ResFinder "
            "family as it appears in the ARG matrix, the normalised key used "
            "for matching, the AMRFinderPlus symbols collapsed into that key, "
            "and the species and assemblies behind the conservation estimate. "
            "Species in which a family was not detected are explicit negatives: "
            "they are excluded from the conservation mean and counted in "
            "'n_species_examined'.",
            [("", mapping)])
        append_sheets(a.xlsx, blocks)
        print(f"\nrefreshed {', '.join(sorted(blocks))} in {a.xlsx}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
