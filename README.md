# Genomic conservation and gut resistome predictability

Analysis code for the manuscript *"Genomic conservation shapes predictability
of the human gut resistome from microbial community composition"*
(Khan, Fahira et al.).

The pipeline joins a uniformly processed resistome resource to curated
metadata and MetaPhlAn 4 taxonomic profiles, partitions resistome variance
across explanatory blocks, trains six model families under three
cross-validation schemes, and relates per-gene cross-cohort predictability to
within-species genomic conservation measured from RefSeq pangenomes.

Everything runs on a single workstation. A full run takes roughly a day, most
of it in stage 5.

---

## Quick start

```bash
conda env create -f environment.yml && conda activate w1-resistome
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
cp config.yaml config.local.yaml          # edit the paths inside

python src/s00_smoke_test.py              # synthetic data, a few minutes
bash run_everything.sh                    # the whole pipeline
```

`s00_smoke_test.py` exercises the whole chain on synthetic data and is the
fastest way to confirm an installation before committing a day of compute.

## Requirements

PyTorch is installed separately because neither the conda nor the default PyPI
build carries sm_120 kernels; on Blackwell hardware (RTX 50-series) they
install cleanly and then fail at the first CUDA launch.

The published results were produced with **xgboost 3.4.0**, scikit-learn 1.9.0
and torch 2.11.0+cu128. `environment.yml` pins `py-xgboost-gpu=2.1.4`, an older
major version that may not reproduce the gradient-boosting numbers exactly;
install xgboost 3.4.0 to match the published run.

Several scripts carry default paths under `/mnt/x/w1_resistome/`, the layout of
the machine the analysis ran on. All are overridable by argument or environment
variable, and are left unedited so this code is byte-identical to the code that
produced the published results. `config.local.yaml` takes precedence over
`config.yaml` when present.

## Input data

No data is included in this repository. The inputs are public:

| Input | Source | Terms |
|---|---|---|
| Resistome alignment counts | Zenodo [10.5281/zenodo.6919377](https://doi.org/10.5281/zenodo.6919377) | CC-BY 4.0 |
| Metadata and MetaPhlAn 4 profiles | [Metalog](https://metalog.embl.de) | ODbL. Version used: 7 July 2026. Metalog does not archive previous releases, so record your download date. |
| Reference assemblies | NCBI RefSeq, via `datasets` | Retrieved in stage 6c/6e |

Derived files from the published run are deposited at
[10.5281/zenodo.23010784](https://doi.org/10.5281/zenodo.23010784):

| Archive | Contents |
|---|---|
| `matrices.zip` | standardised resistome and taxonomic matrices, sample table |
| `model_outputs.zip` | per-fold cross-validation metrics and predictions, held-out evaluation |
| `pangenome.zip` | pangenome evidence tables, including the 166-species independent set |
| `tables.zip` | every result table the pipeline writes |
| `provenance.zip` | manifests recording which code produced which results |
| `Zenodo_README.md` | what each archive holds and how it maps to the manuscript |

Downloading `pangenome.zip` and `tables.zip` is enough to rerun stage 22
without repeating the pipeline.

## The pipeline

Stages run in order. `run_everything.sh` runs all of them and records a marker
per stage, so a rerun resumes rather than repeating completed work.

| # | Stage | What it does | Main output |
|---|---|---|---|
| 00 | `s00_smoke_test.py` | End-to-end check on synthetic data | console report |
| 00b | `s00b_discover_files.py` | Matches downloaded files to the roles the pipeline expects | `file_roles.csv` |
| 01 | `s01_inspect_inputs.py` | Reads the raw inputs and reports their shape and columns | `input_report.txt` |
| 02 | `s02_build_sample_table.py` | Joins the two resources by run and sample accession, applies the sample filters | `sample_table.parquet`, `join_report.txt` |
| 03 | `s03_build_matrices.py` | Builds the ARG and taxonomic matrices | `arg_matrix.parquet`, `taxa_matrix.parquet` |
| 04 | `s04_variance_partition.py` | Partitions resistome variance across explanatory blocks by redundancy analysis | `variance_partition_adult.csv` |
| 05 | `s05_models.py` | Trains six model families under three cross-validation schemes, plus the held-out studies | `model_summary.csv`, `cv_metrics.parquet`, `holdout_summary.csv` |
| 06 | `s06_attribution.py` | Attributes predictions to contributing species | `attribution_*.csv` |
| 06b–06e | `s06b`, `s06c`, `s06e` (shell), `s06d_pangenome_status.py` | Retrieves RefSeq assemblies and builds the pangenome evidence tables | `pangenome_evidence_independent.tsv` |
| 07 | `s07_figures.py` | Manuscript figures | `figures/` |
| 08 | `s08_absolute_burden.py` | Relative versus microbial-load-adjusted burden | `burden_*.csv` |
| 09 | `s09_carriage.py` | Carriage of high-risk mobile ARGs by community state | `carriage_prediction_auc_adult.csv` |
| 10 | `s10_compare_strata.py` | Adult, infant and combined strata side by side | `strata_comparison.csv` |
| 11 | `s11_longitudinal.py` | Within-subject coupling over time | `longitudinal_*_adult.csv` |
| 12 | `s12_perturbation.py` | Natural experiments that manipulate the community | `perturbation_*.csv` |
| 13 | `s13_mobility_predictability.py` | Within-species conservation against cross-cohort predictability | `mobility_vs_predictability_adult_adult.csv`, `mobility_multivariable_adult.csv` |
| 14 | `s14_temporal_direction.py` | Whether community change precedes resistome change | `temporal_direction_*.csv` |
| 15 | `s15_mediation_dose.py` | Dose-response and mediation | `mediation_*.csv` |
| 16 | `s16_exposure_boundary.py` | Each subject as their own control across an exposure boundary | `exposure_boundary_*.csv` |
| 17 | `s17_habitat_join_check.py`, `habitat/run_habitats.sh` | Cross-habitat extension | `habitats/` |
| 20 | `s20_robustness.py`, `run_sensitivity.sh` | Controls, sensitivity analyses and tested nulls | `sensitivity_taxonomy_adult.csv` |
| 21 | `s21_workflow_figure.py` | Study workflow, Figure 1 | `figures/Figure1_*` |
| 22 | `s22_supplementary_tables.py` | Assembles the supplementary tables from the result tables above | `validation_ladder.csv`, `conservation_*.csv`, `family_mapping_65.csv` |

### Stages that need explicit arguments

Four invocations do not do what you want on their defaults.

```bash
# Stage 5. Without --models all, only ridge, taxa_nn and the mean baseline are
# fitted; the published comparison needs all six families.
python src/s05_models.py --models all

# Stage 4. Three permutation schemes. Unset variables give the sample-level
# scheme. Each run overwrites the same table, so copy it aside between runs.
python src/s04_variance_partition.py                          # sample-level
W1_PERM_UNIT=subject python src/s04_variance_partition.py     # subject-blocked
W1_ONE_PER_SUBJECT=1 python src/s04_variance_partition.py     # one sample per subject

# Stage 6e. Pangenome evidence, if rebuilding from scratch (6-10 hours).
bash src/s06e_independent_pangenome.sh

# Stage 13. --label sets the output suffix; the manuscript table is
# mobility_vs_predictability_adult_adult.csv, which needs --label adult.
python src/s13_mobility_predictability.py \
    --pangenome /path/to/pangenome_evidence_independent.tsv --label adult
```

Stage 22 reads the tables the earlier stages wrote and needs the pangenome
evidence. It requires stage 5 to have been run with the subject-grouped
scheme, and stops rather than report a partial validation ladder if it was not:

```bash
python src/s22_supplementary_tables.py \
    --tables /path/to/results/tables \
    --pangenome /path/to/pangenome_evidence_independent.tsv \
    --xlsx Supplementary_Tables.xlsx
```

## Outputs

Three directories, set in `config.yaml`:

```
work_dir/     intermediate matrices and per-fold outputs
table_dir/    every result table, one CSV per analysis
fig_dir/      manuscript and supplementary figures
```

`bash make_all_figures.sh` regenerates the figures from existing tables without
rerunning the analyses.

## Utilities

`tools/` holds four scripts that are not part of the analysis:

| Script | Purpose |
|---|---|
| `freeze_results.py` | Records a hash of the source tree and the results tree as a manifest |
| `verify_package.py` | Checks a manifest, and that every number printed in a manuscript appears in a result table at the precision printed |
| `patch_scheme_selection.py` | Makes the cross-validation scheme explicit at every site that reads it, so subject-grouped folds are never pooled with leave-one-study-out |
| `patch_ambiguous_runs.py` | Switches stage 2 from keeping the first alias for a run that maps to several, to excluding those runs, and lists them |

Typical use before preparing a deposit:

```bash
python tools/freeze_results.py --results /path/to/results --src src \
       --label v1.0.0 --out frozen --copy
python tools/verify_package.py --manifest frozen/v1.0.0.json \
       --manuscript Manuscript.docx --tables /path/to/results/tables
```

`verify_package.py` reports one unmatched number, `0.999`. That is the
collinearity threshold above which predictors are dropped as redundant, a
method parameter rather than a result, and it correctly appears in no table.

## Notes on the analysis design

Three choices change the numbers a rerun produces and are recorded so a rerun
can be compared against the published values.

1. **Independent permutation streams** (stage 4). Each permutation test derives
   its own stream from the configured seed rather than drawing from one
   module-level generator consumed in block order, so a P value does not shift
   when an unrelated block changes.
2. **Subject-grouped cross-validation** (stage 5). Whole subjects are assigned
   to folds. Without it, the random-to-leave-one-study-out gap conflates
   repeated sampling of individuals with cohort shift.
3. **Subject-blocked permutation** (stage 4). `W1_PERM_UNIT=subject` permutes
   whole subjects within study, at both permutation sites: the marginal test in
   `permutation_p` and the Freedman–Lane partial test. Subjects contribute
   between 1 and 81 samples, so equal-size block swaps are not always possible;
   `W1_ONE_PER_SUBJECT=1` is the exact alternative.

In stage 2, 268 of the 70,937 bridged runs map to more than one sample alias.
The first listed in the mapping table is kept, which is deterministic and
reproducible from that table; `tools/patch_ambiguous_runs.py` excludes them
instead if you want to check the effect.

## Repository layout

```
src/            pipeline stages s00-s22
habitat/        cross-habitat analyses
tools/          freeze, verification and sensitivity utilities
docs/           stage-by-stage notes
config.yaml     template configuration (copy to config.local.yaml)
environment.yml pinned environment
```

`docs/PIPELINE_NOTES.md` documents each stage in detail.

## Licence and citation

Released under the MIT Licence (see `LICENSE`). If you use this code, please
cite the manuscript; citation metadata is in `CITATION.cff`.

Data sources carry their own terms: the resistome resource is CC-BY 4.0
(Martiny et al., *PLOS Biology* 2022) and Metalog is ODbL, which is share-alike
and must be acknowledged in any redistribution.
