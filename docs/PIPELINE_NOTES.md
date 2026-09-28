# Pipeline notes (detailed)

Stage-by-stage design notes kept during development. For setup and
reproduction, start with the top-level README.

# W1 — Gut resistome × microbiome across public cohorts

End-to-end pipeline joining the Zenodo 214K-metagenome resistome resource to
Metalog's curated metadata and MetaPhlAn 4 profiles, then asking how much of
an individual's gut resistome is explained by *who is there* versus *what they
were exposed to* — and whether particular community configurations are
permissive to high-risk ARGs.

Designed to run on a single workstation (RTX 5080, 16 GB) inside WSL2.

---

## 0. Current status of this dataset

Verified on the real downloads (Zenodo record 6919377, Metalog 2026-07-07):

| Stage | Result |
|---|---|
| Accession join | **19,427 usable samples**, 112 studies, 34 countries |
| ARG matrix | 19,185 samples × 682 genes (before the corrected prevalence filter) |
| Absolute burden | 9,233 samples (adult faecal only — microbial load coverage) |
| Taxa matrix | 449 species at 5% prevalence, from 5,344 |
| **Aligned dataset** | **17,028 samples** |

Composition that drives the analysis design:

- **Life stage is bimodal**: 11,028 adult, 6,025 infant, 710 child, 572
  adolescent, 908 unknown. Infant gut communities are structurally different
  and preterm NICU cohorts carry extreme resistomes, so `analysis.stratum` in
  config.yaml defaults to `adult`. Run `infant` separately; the adult/infant
  contrast is a result in its own right, not a nuisance.
- **Medication coverage is thin**: only 3,043 / 19,427 samples carry any
  medication annotation, of which 2,061 show antibiotic exposure. That leaves
  under a thousand annotated-but-unexposed samples. The exposure arm therefore
  cannot carry equal weight — report it as a scoped sub-analysis and state the
  coverage number explicitly in the results.
- **`subject_disease_status` is administrative**: COHORT (10,272) and CTR
  (2,764) are Metalog bookkeeping, not diagnoses. `harmonise_disease()`
  collapses them into `disease_group` / `disease_detail`.

The clade-to-lineage map is supplied by Metalog and configured in config.yaml.
from 5,344 species because MetaPhlAn 4's unnamed `GGB…_SGB…` clades have no
parseable genus. That guts the phylogeny-aware pooling, which is the main
reason the phylogeny-aware pooling has little to work with. In the final
analysis the pooling network did not outperform the plain network.

---

## 1. Setup

A pinned environment is provided in `environment.yml`:

```bash
conda env create -f environment.yml && conda activate w1-resistome
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
```

PyTorch is installed separately because neither the conda nor the default PyPI
build carries sm_120 kernels; on Blackwell hardware they install cleanly and
then fail at the first CUDA launch.

### Reporting the variance partition

Stage 4 reports three distinct quantities, and they must not be conflated:

| column | meaning |
|---|---|
| `marginal_R2_adj` | variance explained by the block alone |
| `partial_R2_adj` | share of the variance REMAINING after conditioning on all other blocks |
| `unique_total_fraction_adj` | drop in explained variance when the block is removed from the full model |

Partial R² is typically much larger than the unique share of total, because its
denominator has already had the other blocks removed. A claim of the form "X
explains N% of resistome variance uniquely" refers to
`unique_total_fraction_adj`. `partial_p` comes from a Freedman-Lane test of the
partial contribution; `marginal_p` tests only the marginal association.

### Original setup notes

```bash
# in WSL2 (Ubuntu). Do NOT run this on native Windows — bioconda has no
# Windows channel and the Blackwell wheels are unreliable there.
cd /mnt/x/w1_resistome
python -m venv .venv && source .venv/bin/activate

# PyTorch first, from the cu128 index — the default wheel has no sm_120
# kernels and dies at the first CUDA launch on a 5080.
pip install torch --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.cuda.get_device_capability())"   # (12, 0)

pip install -r requirements.txt

# real Times New Roman for the figures
sudo apt-get install -y ttf-mscorefonts-installer fonts-liberation
rm -rf ~/.cache/matplotlib

python src/s00_smoke_test.py      # must pass before you touch real data
```

Then edit `config.yaml` so the paths point at your downloads. Windows drives
appear under `/mnt/x/...` in WSL.

---

## 2. Run everything

```bash
export PANGENOME=/mnt/x/w1_resistome/pangenome/pangenome_evidence.tsv   # optional
bash run_everything.sh
```

One command, all stages, in dependency order. Each stage writes a marker on
success, so a rerun resumes rather than repeating completed work.

| flag | effect |
|---|---|
| `--list` | show the stage list and exit |
| `--from 04` | resume from a stage |
| `--only 13,14,16` | run selected stages only |
| `--quick` | limit the model comparison to ridge and gradient boosting |
| `--force` | repeat stages that already completed |

Optional inputs, as environment variables: `PANGENOME` enables attribution and
the conservation analysis, `ARMS` enables difference-in-differences, `HABITATS`
enables the cross-habitat viability check.

Per-stage logs are written to `logs/`, and failures are collected and reported
at the end rather than aborting the run - except that the cross-stratum
comparison is skipped if any stratum failed, since it would otherwise combine
fresh output with stale results.

`run_strata.sh`, `finish_run.sh` and `run_all.sh` remain for partial reruns.

## 2c. Manuscript figures

The six consolidated main figures are produced by a single script that reads
only the supplementary tables, so it runs unchanged for anyone with the data:

```bash
python make_manuscript_figures.py \
    --tables /mnt/x/w1_resistome/tables \
    --habitats /mnt/x/w1_resistome/habitats \
    --outdir /mnt/x/w1_resistome/manuscript_figures \
    --font sans        # Nature-family journals specify sans-serif for figures
```

| Figure | Argument it carries |
|---|---|
| 1 | variance structure: what composition explains, and how robustly |
| 2 | prediction degrades under cohort shift, and conservation explains which genes survive |
| 3 | within-subject coupling and high-risk gene carriage |
| 4 | hypotheses tested and not supported |
| 5 | habitat performance and the metric caveat |

Two conventions are enforced deliberately. **One result appears in one place**:
cross-habitat variance is in Figure 1 and habitat performance in Figure 5, so
no panel is duplicated. And **panels quote the same model and stratum as the
text**: habitat performance uses gradient boosting because the manuscript does,
and every carriage panel uses the adult stratum, since the combined-stratum
carriage figures differ substantially and are not interchangeable.

## 3. Stage order

| Stage | Script | What it does | Runtime |
|---|---|---|---|
| 0 | `s00_smoke_test.py` | synthetic data → figures; checks fonts + CUDA | seconds |
| 0b | `s00b_discover_files.py` | finds your downloads, writes `config.local.yaml` | seconds |
| 1 | `s01_inspect_inputs.py` | prints every schema, auto-detects columns | minutes |
| 2 | `s02_build_sample_table.py` | **the accession join — go/no-go** | ~15 min |
| 3 | `s03_build_matrices.py` | ARG + taxa matrices, normalised | ~1 min |
| 4 | `s04_variance_partition.py` | RDA variance partitioning | ~30 min |
| 5 | `s05_models.py` | baselines + phylogeny-aware net, LOSO CV | 1–4 h |
| 6 | `s06_attribution.py` | integrated gradients + genome validation | ~1 h |
| 7 | `s07_figures.py` | all five manuscript figures | minutes |
| 8 | `s08_absolute_burden.py` | relative vs ABSOLUTE resistome burden | ~5 min |
| 9 | `s09_carriage.py` | permissive community states, cross-cohort carriage AUC, geography robustness | ~15 min |
| 10 | `s10_compare_strata.py` | adult / infant / combined comparison | ~1 min |
| 11 | `s11_longitudinal.py` | **within-subject** analysis: each person as their own control | ~5 min |
| 12 | `s12_perturbation.py` | FMT, antibiotic and bowel-prep studies as natural experiments | ~5 min |
| 13 | `s13_mobility_predictability.py` | tests whether genomic conservation explains predictability | ~2 min |
| 14 | `s14_temporal_direction.py` | does composition lead the resistome, or the reverse? | ~10 min |
| 15 | `s15_mediation_dose.py` | dose-response, mediation, difference-in-differences | ~10 min |
| 16 | `s16_exposure_boundary.py` | within-subject before/after across a recorded exposure | ~10 min |
| 20 | `s20_robustness.py` | sensitivity, confound controls and tested nulls, in one figure | ~5 min |
| 21 | `s21_workflow_figure.py` | the study workflow, drawn from the pipeline's own counts | seconds |

Or `bash run_all.sh`, which pauses after stage 1 for you to confirm the
column mapping.

---

## 4. Decision points — read these before running

**After stage 1.** The Zenodo HDF5 schemas and Metalog TSV headers are not
guaranteed stable, so nothing is hardcoded. `s01` writes
`work/inspection_report.txt` with every column name and its auto-detected
role. Any line reading `** REQUIRED role NOT DETECTED` must be fixed by
setting the column explicitly under `columns:` in `config.yaml`. Everything
downstream depends on this being right.

**After stage 2.** This is the go/no-go. `work/join_report.txt` gives the full
attrition table. Expect losses at four points: the pre-2020 upload window
(Zenodo covers ENA deposits 2010-01-01 to 2020-01-01 only), MetaPhlAn 4
coverage (~80k of 113k Metalog human samples), the gut/faecal filter, and the
depth floor. Above ~5,000 samples with disease labels, proceed. Below ~1,000,
check the mapping file parsed correctly *before* redesigning — a silent
failure to explode multi-run cells is the most likely cause.

**Reading stage 4's P values.** Permutation is restricted within study, so a
block that is constant inside every study — study identity itself — is
invariant under the permutation and cannot be tested. Those report `n/a`
rather than P = 1.0, and `p_testable` is False in the output table. Describe
them in the manuscript as not testable under the restricted scheme, quoting
the variance explained without a P value; reporting P = 1.0 would read as
evidence of no effect, which it is not.

**After stage 4.** If taxonomy's *unique* adjusted R² is under ~2% once study
identity is partialled out, the microbiome–resistome link is largely batch
structure and the framing has to change. Better to find that out here than at
review.

---

## 4. Join semantics (verified against the real schemas)

The Metalog mapping file is **long**: one row per external identifier, with a
`kind` column taking `run`, `sample` or `experiment`, and the identifier in
`external_id`. The canonical Metalog key is **`sample_alias`**, not
`sample_id` — every Metalog profile table joins on `sample_alias`.

Zenodo carries both `run_accession` and `sample_accession`, so two bridges
exist and both are used:

| Bridge | Path | Runs matched |
|---|---|---|
| A | `kind='run'` → Zenodo `run_accession` | 64,968 |
| B | `kind='sample'` → Zenodo `sample_accession` → its runs | 69,678 |
| | union | 70,937 runs / 39,875 samples |

`experiment` rows (SRX/ERX) have no Zenodo counterpart and are dropped.
ARG counts are **summed** across the runs of one biological sample — they are
technical replicates of one library pool, so summing, not averaging, is right.

**No upload-date filter is applied.** Zenodo only contains runs deposited
before 2020, so the join itself enforces the coverage window. Metalog's
`collection_date` is the *sampling* date and filtering on it would wrongly
discard old samples that were deposited late.

Two filters come from Metalog's own reference usage example and are not
optional: rows where `artificial` is non-null are mock communities and must be
excluded, and the faecal selector is the exact ENVO term
`fecal material [ENVO:00002003]`, not a keyword match.

## 5. Normalisation

ARG.h5 already carries everything needed: `fragmentCountAln`,
`refSequence_length`, and `bacterial_fragment` (the rRNA-derived denominator
for that run). **rRNA.h5 is NOT required** and neither is ResFinder_anno.h5.
This matches the authors' own analysis code, which computes
`log(sum(fragmentCountAln) / (bacterial_fragment / 1e6))` taking the max
`bacterial_fragment` per run.

Counts are length-normalised per kb, divided by bacterial fragments per
million, then log-transformed. ResFinder references are collapsed to gene level
first, since many are allelic variants of one gene.

Prevalence filtering is done on **raw counts**, never on an inverted log
matrix — floating-point residue where a zero was makes every gene look
detected everywhere and silently turns the filter into a no-op.

Species abundances are re-closed to sum 1 after species-level subsetting,
prevalence-filtered, and CLR-transformed. Everything here is compositional
twice over — treat it accordingly and never run raw-proportion correlations.

If microbial load predictions are available, `arg_absolute.parquet` gives
**absolute** ARG burden. Relative resistome × predicted absolute load is the
panel almost nobody in the cross-cohort literature has produced; treat it as a
headline result rather than an afterthought.

---

## 6. The model

Multi-output structured regression, not classification: CLR species in, the
full ARG abundance vector out. The network pools species → genus → family with
learned weights, so it can borrow strength across related species never seen
together in the same cohort — precisely where flat models fail under
leave-one-study-out.

Six models are available via `--models` (default `mean,ridge,taxa_nn`):

| model | what it tests |
|---|---|
| `mean` | floor: per-gene training mean |
| `ridge` | regularised linear on CLR species |
| `taxa_nn` | phylogeny-aware pooling network |
| `mlp` | same capacity, NO pooling - isolates the architecture's contribution |
| `rf` | multi-output random forest, the field's standard baseline |
| `xgb` | gradient boosting, multi-output trees where available |

`mlp` is the important ablation: any difference between `taxa_nn` and `mlp` is
attributable to taxonomic pooling rather than to nonlinearity or capacity.
`rf` and `xgb` add substantial runtime across 60+ leave-one-study-out folds -
budget several hours for `--models all` (mean, ridge, rf, xgb, mlp, taxa_nn, ft).

Both LOSO and random splits are always evaluated. The **gap between them** is
itself a result: it quantifies how much published single-cohort performance is
batch and lineage leakage.

### Held-out studies

`analysis.holdout_studies` (default 8) locks away whole studies before any
model is fitted. They appear in no fold of any scheme and are scored exactly
once, at the end, in `tables/holdout_summary_<stratum>.csv`.

This matters as soon as more than one model is compared. Cross-validation
estimates become optimistic the moment a winner is CHOSEN on them - reporting
the best of six models' CV score is a selection estimate, not an unbiased one.
Studies are sampled across the size distribution rather than at random, so the
held-out set is not composed entirely of small cohorts.

**Report the CV table for model comparison and the held-out table for the
headline performance number.** Do not report the CV score of the selected
model as if it were unbiased.

---

## 6b. Stages 8 and 9 — the clinically legible half

Everything up to stage 7 is a variance decomposition. These two ask questions a
clinician would recognise, and they are what lift the work above a competent
meta-analysis.

**s08 — absolute burden.** The cross-cohort resistome literature is almost
entirely relative-abundance only. Two samples with the same ARG ratio carry
very different reservoirs if one holds ten times the bacterial load. This
re-runs the variance partition on the absolute scale and checks whether
country and disease rankings *reorder*. If they do, published relative-scale
population comparisons are measuring something other than what they claim.

**s09 — permissive community states.** For each high-risk gene with enough
carriers: association with enterotype and dysbiosis score controlling for
study, country, disease and depth; then leave-one-study-out AUC for carriage
predicted from composition alone. A gene predictable at AUC > 0.7 in an unseen
cohort is a screening-relevant result.

s09 also runs the **geography robustness check**. Country and study are
near-collinear in public data, so "geographic effects are compositional" may
really be "geography is inseparable from study". The partition is re-run on
countries represented by at least three independent studies, where the two can
be told apart. If geography still collapses there the claim holds; if it does
not, report the limitation instead.

## 6c. Stages 11 and 12 — from association to causation

Everything up to stage 10 is cross-sectional, so every claim is associational.
These two change the design rather than adding evidence.

**s11 — within-subject.** Subtracting each subject's own mean removes every
time-invariant confounder: genetics, diet, geography, study. If community
change predicts resistome change *inside* individuals, the relationship is
much harder to attribute to confounding. Reports within- against
between-subject variance explained, correlates paired compositional and
resistome shifts, and repeats the correlation within narrow elapsed-time bands
so that shared temporal drift is excluded.

**s12 — natural experiments.** FMT transplants a whole community; antibiotics
and bowel preparation destroy one. If the resistome moves as the community
moves under an experimental shock, composition is doing the work. Also tests
whether a model trained on unperturbed samples still predicts POST-shock
resistomes.

Caveat to state in the manuscript: donor-recipient links are absent from
public metadata, so FMT is analysed as a within-subject perturbation rather
than a transfer experiment.

## 6d. Stages 13 and 14 — testing the mechanism

**s13 turns the genomic-conservation interpretation into a falsifiable test.** The
manuscript uses the predominance of accessory genes to EXPLAIN why taxonomic
pooling fails, but asserts rather than tests it. The account predicts that
genes bound firmly to genomes should be well predicted from composition and
mobile genes poorly, because for an accessory gene the presence of a species
says little about whether that strain carries it. s13 correlates per-gene
cross-cohort performance against mean pangenome frequency, controls for sample
prevalence (a gene seen in more samples is easier to predict for unrelated
reasons), and reports a bootstrap interval. **If the correlation is absent, the
accessory-genome explanation should be withdrawn, not defended.**

**s14 asks whether composition leads.** The within-subject analysis correlates
simultaneous change, which cannot speak to order. s14 compares how well
composition at *t* predicts the resistome at *t+1* against the reverse, with
subjects held out entirely, and reports skill INCREMENTAL to each layer's own
persistence — a layer that is merely stable predicts itself well, which is not
evidence about direction. Both layers come from the same sequencing library, so
this is cross-prediction rather than causal inference; the comparison between
directions is the informative quantity, not either direction alone.

## 6e. Stage 15 — how far observational data can be pushed

**Dose-response.** Larger compositional displacement should produce larger
resistome displacement, adjusted for elapsed time, change in depth and study.
Reported separately for subjects with three or more timepoints, since two-point
trajectories cannot distinguish a trend from noise.

**Mediation.** Decomposes an intervention's association with resistome change
into a part transmitted through compositional change and a part that is not,
with subject-clustered bootstrap intervals. Mediation assumes no unmeasured
mediator-outcome confounding, which observational metagenomic data cannot
guarantee, so this is reported as the share of the ASSOCIATION consistent with
mediation - not as a causal decomposition. Interventions such as antibiotics
act on resistance genes directly as well as through the community, and that
direct path is not separable here.

**Difference-in-differences.** Requires a manually curated arm table
(`--arms`), because keyword classification of whole studies cannot separate
treatment arms: one study may contain probiotic, autologous transplant and
placebo groups simultaneously. Without curation the analysis is skipped rather
than approximated. A template is provided in `curated_arms_TEMPLATE.csv`.

**What none of this establishes.** No amount of observational analysis removes
unmeasured time-varying confounding. The defensible claim after these analyses
is that the evidence is *consistent with a causal contribution* of community
composition to resistome dynamics, not that composition causes it.

## 6f. Stage 16 — each subject as their own control

Metalog records `intervention` per SAMPLE, and most subjects in interventional
studies carry both annotated and unannotated samples (35 of 35 in Zmora 2018,
18 of 24 in Raymond 2016). The field marks WHEN a subject was exposed, not
WHETHER. Treated-versus-control comparison is therefore unavailable, and a
within-subject before/after comparison is - which is the stronger design,
because between-person confounding is removed by construction rather than
adjusted away.

Consecutive pairs are split into those CROSSING an exposure boundary
(unannotated then annotated) and INTERNAL pairs with no boundary. The control
set is restricted to subjects who also contribute crossing pairs, so the
comparison is within the same people. Internal pairs carry every confounder
that crossing pairs carry, except the exposure.

Stage 12's classification is now per sample rather than per study, which fixes
the misassignment of Zmora 2018 - a study containing 257 probiotic, 174
antibiotic and 87 transplant samples was previously labelled "FMT" from the
first keyword match.

Difference-in-differences remains unavailable and should be reported as such:
arm assignment exists only in the source publications, and the sample
identifiers cannot reliably be mapped to them.

## 6g. Stage 20 — what was tested and what survived

Several claims in this project were tested and did not hold: taxonomic pooling
gave no benefit, temporal precedence was not established, cohort distance did not predict poorer transferability (an unexpected positive
association was observed, substantially attenuated after accounting for
properties of the held-out cohort), and per-fold rank correlation was found to scale
with held-out outcome variance - which means cross-habitat comparisons of that
statistic cannot be read as differences in transferability.

One claim survived controls that could have removed it: genomic conservation predicts
predictability, and the relationship persists after adjusting for outcome
variance and for sample prevalence separately.

Stage 20 puts these in one figure alongside the sensitivity analyses -
prevalence thresholds, component counts, one sample per subject, largest
studies removed. Reporting them together shows that the alternative
explanations were tested rather than overlooked, and places the surviving
claim in the context of the ones that did not survive.

Sensitivity variants are recomputed from the stored matrices rather than
re-run through the pipeline, so this stage takes minutes.

## 6h. Stage 21 — the workflow figure

Every count in the workflow schematic is read from `join_report.txt`, the
sample table and the matrices rather than typed in. If a filter threshold
changes and the pipeline is re-run, the figure changes with it - a hand-drawn
schematic silently goes stale, and a reader has no way to tell.

This is Figure 1 in the manuscript; the cohort composition figure produced by
stage 7 becomes Figure 2.

## 6i. Independent pangenome evidence (stage 6e)

The original pangenome evidence used the 60 species with the strongest model
attributions. That makes the genomic quantity partly a product of the model
whose performance it is later used to explain, and it is the most likely
referee objection to the genome-boundedness result.

`s06e_independent_pangenome.sh` selects species by PREVALENCE alone - a
property of the data, computed before any model is fitted - and records
**explicit zeros**: species examined and found not to carry a gene contribute
frequency 0 rather than being absent from the table. The original evidence
could only ever contain detected pairs, which truncates the frequency
distribution.

```bash
bash src/s06e_independent_pangenome.sh /mnt/x/w1_resistome/pangenome_independent 12 200
python src/s13_mobility_predictability.py \
    --pangenome /mnt/x/w1_resistome/pangenome_independent/pangenome_evidence_independent.tsv \
    --label independent
```

Budget 6-10 hours for 200 species. If the relationship holds on this evidence
set, the selection and detection objections are answered. If it does not, the
original result was partly an artefact of how the species were chosen, and
should be reported as such.

## 7. Why stage 6 matters

Integrated gradients give a species × ARG attribution matrix. That matrix is
then confronted with ARG content independently called in RefSeq genomes of the
same species:

- **confirmed** — strong attribution, gene genuinely present → the model
  learned real biology (sanity check)
- **unexplained** — strong attribution, gene *absent* from that species'
  genomes → candidate horizontal transfer, mobile-element carriage, or an
  unrecognised reservoir

The unexplained set is the finding. This is what converts a prediction paper
into a discovery paper, and it's the panel reviewers will judge the work on.

To build the evidence table, run stage 6 once without evidence (it writes
`tables/species_for_genome_check.csv`), then:

```bash
bash src/s06b_fetch_genome_evidence.sh
python src/s06_attribution.py \
  --genome-evidence /mnt/x/w1_resistome/genomes/amrfinder_all.tsv
```

**One reference genome is not enough.** Acquired ARGs live in the accessory
genome, so a single reference - usually a susceptible strain - lacks them. The
E. coli reference assembly carries no blaTEM, blaCTX-M, sul1 or tet(A), all of
which are textbook E. coli resistance genes. Validating against one assembly
measures which strain NCBI designated "reference", not whether an association
is real. Use `s06c_pangenome_evidence.sh` for the defensible version: it takes
up to a dozen assemblies per species and records the FREQUENCY of each gene, so
core (present in nearly all) and accessory (present in a few) can be told
apart. The single-reference script below is only a quick first pass.

The script installs `ncbi-datasets-cli` and `ncbi-amrfinderplus` from bioconda,
downloads one reference genome per named species, runs AMRFinderPlus on each,
and concatenates the calls. It is resumable - already-processed species are
skipped - so Ctrl-C is safe. Budget 2-4 hours.

Unnamed `GGB…/SGB…` clades are skipped: they have no NCBI counterpart, so
roughly 60% of your species list resolves. **That exclusion is not random** -
it removes exactly the understudied taxa a discovery claim might rest on, so
report how many species were testable and check that your candidate list is
not concentrated among the untestable ones.

---

## 8. Figure standard

All figures are built through `src/plotstyle.py` and are, by construction:

- **multi-panel, 2 or 3 rows** — `multipanel()` raises on anything else
- **Times New Roman** (warns and substitutes Liberation/Nimbus if absent)
- **bold x and y axis labels on every panel**, via `label_axes()`
- bold `(A) (B) (C)` panel letters
- saved as **PNG (600 dpi) + PDF + SVG**
- SVG uses `svg.fonttype: none`, so labels stay as TEXT rather than being
  converted to outlines - they remain selectable and re-typeable in Illustrator
  or Inkscape. Converted paths cannot be corrected later, which is why this
  setting matters more than any styling choice
- gridlines sit behind the data on value axes only, and are suppressed on
  heatmaps and hexbins where they would overlay it
- palette is colour-blind safe with monotonic luminance across the first four
  series, so a greyscale print still separates them

Never call `plt.subplots` directly or the style drifts between figures.

| Figure | Content |
|---|---|
| `fig1_cohort` | studies, geography, age, depth, disease, attrition |
| `fig2_resistome` | richness, load, class composition, depth confounding, absolute vs relative |
| `fig3_variance` | variance partitioning, unique vs marginal, ordination |
| `fig4_models` | model comparison, LOSO-vs-random shift, calibration |
| `fig5_attribution` | species × ARG heatmap, genome validation, candidates |

---

## 9. Known limitations

- ResFinder read mapping captures **acquired** ARGs only — no chromosomal
  resistance mutations. State this as scope, early.
- The Zenodo coverage window ends 2020-01-01; nothing deposited later is
  represented, so recent resistance trends are out of reach.
- Antibiotic exposure metadata is sparse; mediation analysis only on the
  subset with real exposure data, and no causal language beyond it.
- Species without RefSeq genome evidence are excluded from the validation
  table — check that this isn't systematically dropping the understudied taxa
  your finding depends on.
- Metalog publishes metadata updates most weekends and does not archive prior
  versions. Record your download date and keep the raw files, or the analysis
  is not reproducible.

---

## 10. Data sources to cite

Cite: Martiny et al. *PLOS Biology* 2022 (resistome resource, CC-BY 4.0);
Kuhn et al. *Nucleic Acids Research* 2025 (Metalog, ODbL — share-alike, must be
noted in your data availability statement).
