# Genomic conservation and gut resistome predictability

Analysis code for the manuscript *"Genomic conservation constrains prediction
of the human gut resistome from microbial community composition"*
(Khan, Fahira et al., under review).

The pipeline joins a uniformly processed resistome resource to curated
metadata and MetaPhlAn 4 taxonomic profiles, partitions resistome variance
across explanatory blocks, trains six model families under three
cross-validation schemes, and relates per-gene cross-cohort predictability to
within-species genomic conservation measured from RefSeq pangenomes.

Everything here runs on a single workstation. The full pipeline takes roughly
a day, most of it in stage 5.

---

## Data

No data is included in this repository. The inputs are public:

| Input | Source | Note |
|---|---|---|
| Resistome alignment counts | Zenodo [10.5281/zenodo.6919377](https://doi.org/10.5281/zenodo.6919377) | CC-BY 4.0 |
| Metadata and MetaPhlAn 4 profiles | [Metalog](https://metalog.embl.de) | ODbL. Version used: 7 July 2026. Metalog does not archive previous releases, so record your download date. |
| Reference assemblies | NCBI RefSeq, via `datasets` | Retrieved with `src/s06e_independent_pangenome.sh` |

Derived matrices, per-fold outputs and the pangenome evidence tables are
deposited on Zenodo at
[10.5281/zenodo.23010784](https://doi.org/10.5281/zenodo.23010784):

| Archive | Contents |
|---|---|
| `matrices.zip` | the standardised resistome and taxonomic matrices, and the sample table |
| `model_outputs.zip` | per-fold cross-validation metrics and predictions, and the held-out evaluation |
| `pangenome.zip` | the pangenome evidence tables, including the 166-species independent set |
| `tables.zip` | every result table the pipeline writes, including those from `s22` |
| `provenance.zip` | the freeze manifests recording which code produced which results |
| `Zenodo_README.md` | what each archive holds and how it maps to the manuscript |

Downloading `pangenome.zip` and `tables.zip` is enough to rerun
`src/s22_supplementary_tables.py` without repeating the pipeline.

## Setup

```bash
conda env create -f environment.yml && conda activate w1-resistome
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
```

PyTorch is installed separately because neither the conda nor the default PyPI
build carries sm_120 kernels; on Blackwell hardware (RTX 50-series) they
install cleanly and then fail at the first CUDA launch.

> **Version note.** The results in the manuscript were produced with
> **xgboost 3.4.0**, scikit-learn 1.9.0 and torch 2.11.0+cu128. `environment.yml`
> pins `py-xgboost-gpu=2.1.4`, which is an older major version and may not
> reproduce the gradient-boosting numbers exactly. Install xgboost 3.4.0 to
> match the published run.

Then point the pipeline at your copies of the data:

```bash
cp config.yaml config.local.yaml     # edit paths; config.local.yaml wins if present
```

Several scripts carry default paths under `/mnt/x/w1_resistome/`, the layout of
the machine the analysis was run on. They are all overridable by argument or
environment variable, and are left unedited so that this code is byte-identical
to the code that produced the published results.

## Reproducing the analysis

```bash
export PYTHONPATH="$PWD/src"
bash run_everything.sh                      # the whole pipeline, stage by stage
```

Four invocations are easy to get wrong and are worth running explicitly:

```bash
# 1. Models. WITHOUT --models all only ridge, taxa_nn and the mean baseline
#    are fitted; the published comparison needs all six families.
python src/s05_models.py --models all

# 2. Variance partition, three permutation schemes. Unset variables reproduce
#    the sample-level scheme. Each run overwrites the same table, so copy it
#    aside between runs.
python src/s04_variance_partition.py                          # sample-level
W1_PERM_UNIT=subject python src/s04_variance_partition.py     # subject-blocked
W1_ONE_PER_SUBJECT=1 python src/s04_variance_partition.py     # exact

# 3. Conservation. --label sets the output suffix; the manuscript table is
#    mobility_vs_predictability_adult_adult.csv, which needs --label adult.
python src/s13_mobility_predictability.py \
    --pangenome /path/to/pangenome_evidence_independent.tsv --label adult

# 4. Pangenome evidence, if rebuilding it from scratch (6-10 hours).
bash src/s06e_independent_pangenome.sh

# 5. Supplementary tables. Reads the tables written above; --xlsx also
#    writes the sheets into the workbook. Needs s05 to have been run with
#    the subject-grouped scheme, or it stops rather than report a partial
#    validation ladder.
python src/s22_supplementary_tables.py \
    --tables /path/to/results/tables \
    --pangenome /path/to/pangenome_evidence_independent.tsv \
    --xlsx Supplementary_Tables.xlsx
```

`src/s00_smoke_test.py` runs the whole chain on synthetic data in a few
minutes and is the fastest way to check an installation.

## Where each result comes from

| Manuscript result | Script | Output |
|---|---|---|
| Cohort assembly, Fig. 1 | `s02_build_sample_table.py`, `s21_workflow_figure.py` | `cohort_by_study.csv` |
| Variance partition, taxonomy 9.4% | `s04_variance_partition.py` | `variance_partition_adult.csv` |
| Model comparison, validation ladder | `s05_models.py --models all` | `model_summary_adult.csv`, `cv_metrics_adult.parquet` |
| Held-out evaluation, ρ = 0.288 | `s05_models.py` | `holdout_summary_adult.csv` |
| Conservation vs predictability, ρ = 0.590 | `s06e_*.sh` then `s13_*.py --label adult` | `mobility_vs_predictability_adult_adult.csv` |
| Multivariable model of predictability | `s13_mobility_predictability.py` | `mobility_multivariable_adult.csv` |
| High-risk gene carriage | `s09_carriage.py` | `carriage_prediction_auc_adult.csv` |
| Within-subject coupling | `s11_longitudinal.py` | `longitudinal_*_adult.csv` |
| Negative results (pooling, temporal, mediation, exposure, distance) | `s05`, `s14`, `s15`, `s16`, `habitat/fold_heterogeneity.py` | see `docs/PIPELINE_NOTES.md` |
| Sensitivity of the variance estimate | `run_sensitivity.sh`, `s20_robustness.py` | `sensitivity_taxonomy_adult.csv` |
| Cross-habitat analyses | `habitat/run_habitats.sh` | `habitats/` |
| Validation ladder, leakage vs cohort shift | `s22_supplementary_tables.py` | `validation_ladder.csv` |
| Conservation under minimum-assembly thresholds and measurement error | `s22_supplementary_tables.py` | `conservation_min_assemblies.csv`, `conservation_errors_in_variables.csv` |
| Conservation by drug class, leave-one-class-out | `s22_supplementary_tables.py` | `conservation_by_drug_class.csv`, `conservation_leave_one_class_out.csv` |
| Alternative explanations: prevalence, abundance variance, host breadth | `s22_supplementary_tables.py` | `conservation_partial_associations.csv` |
| ResFinder-to-AMRFinderPlus family mapping | `s22_supplementary_tables.py` | `family_mapping_65.csv` |

## Keeping results and code in step

A results directory and the code that produced it can drift apart without
anything failing, which is how a manuscript comes to cite numbers no current
script generates. Two tools in `tools/` guard against that:

```bash
# Record which code produced which results
python tools/freeze_results.py --results /path/to/results --src src \
       --label v86_frozen --out frozen --copy

# Refuse to build a deposit if source, results or manuscript have drifted
python tools/verify_package.py --manifest frozen/v86_frozen.json \
       --manuscript Manuscript.docx --tables /path/to/results/tables
```

`verify_package.py` hashes both trees and, given a manuscript, checks that
every number printed in it appears in a result table at the precision printed.
It reports one unmatched number, `0.999`: that is the collinearity threshold
above which predictors are dropped as redundant, a method parameter rather
than a result, and it correctly appears in no table.

## Ambiguous run-to-sample mappings

268 of the 70,937 bridged runs (0.4%) reach more than one Metalog sample alias.
`s02` resolves these by keeping the first, which is deterministic and
reproducible from the mapping table but is not an argument. To drop them
instead and see whether anything moves:

```bash
python tools/patch_ambiguous_runs.py --report   # what the current rule is
python tools/patch_ambiguous_runs.py --apply    # exclude them, list them
python src/s02_build_sample_table.py            # then rerun s02, s03, s04
python tools/patch_ambiguous_runs.py --revert   # restore the original rule
```

The excluded runs are written to `work/ambiguous_runs.tsv`.

## Design decisions that affect reported numbers

Three choices in the evaluation and permutation design are easy to get wrong
and change the numbers a rerun produces. They are recorded here so that a
rerun can be compared against the published values:

1. **Independent permutation streams** (`s04`). Each permutation test derives
   its own stream from the configured seed rather than drawing from one
   module-level generator consumed in block order, so a P value does not shift
   when an unrelated block changes. R² values are unaffected; mid-range P
   values move by roughly their Monte Carlo error.
2. **Subject-grouped cross-validation** (`s05`). A third scheme assigns whole
   subjects to folds. Without it, the random-to-leave-one-study-out gap
   conflates repeated sampling of individuals with cohort shift.
3. **Subject-blocked permutation** (`s04`). `W1_PERM_UNIT=subject` permutes
   whole subjects within study. Both permutation sites are covered: the
   marginal test in `permutation_p` and the Freedman–Lane partial test.

Subjects contribute between 1 and 81 samples, so equal-size block swaps are
not always possible and about 8% of destination subjects draw from more than
one source subject. `W1_ONE_PER_SUBJECT=1` is the exact alternative and is
reported alongside it.

## Repository layout

```
src/            pipeline stages s00-s22
habitat/        cross-habitat analyses
tools/          freeze, verification and sensitivity utilities
docs/           detailed stage-by-stage notes
config.yaml     template configuration (copy to config.local.yaml)
environment.yml pinned environment
```

## Licence and citation

Released under the MIT Licence (see `LICENSE`). If you use this code, please
cite the manuscript; citation metadata is in `CITATION.cff`.

Data sources carry their own terms: the resistome resource is CC-BY 4.0
(Martiny et al., *PLOS Biology* 2022) and Metalog is ODbL, which is
share-alike and must be acknowledged in any redistribution.
