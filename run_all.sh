#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# W1 resistome x microbiome - full pipeline driver
#
# Run from the project root inside WSL2:
#     bash run_all.sh
#
# Stages are independent; each writes parquet intermediates, so you can rerun
# any single stage without repeating the ones before it.
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

log() { printf '\n\033[1m==== %s ====\033[0m\n' "$1"; }

log "0/7  environment smoke test"
python src/s00_smoke_test.py

log "0b/7  discover downloaded files"
python src/s00b_discover_files.py

log "1/7  inspect inputs"
python src/s01_inspect_inputs.py
echo
read -rp "Review work_dir/inspection_report.txt. Column mapping correct? [y/N] " ok
[[ "$ok" == "y" ]] || { echo "Fix columns: in config.yaml, then rerun."; exit 1; }

log "2/7  accession join  (GO / NO-GO)"
python src/s02_build_sample_table.py

log "3/7  build ARG and taxa matrices"
python src/s03_build_matrices.py

log "4/7  variance partitioning"
python src/s04_variance_partition.py

log "5/7  models  (leave-one-study-out + random)"
python src/s05_models.py

log "6/7  attribution"
if [[ -n "${GENOME_EVIDENCE:-}" ]]; then
  python src/s06_attribution.py --genome-evidence "$GENOME_EVIDENCE"
else
  echo "GENOME_EVIDENCE not set - running attribution without genome validation."
  python src/s06_attribution.py
fi

log "7/12  figures"
python src/s07_figures.py

log "8/12  relative vs microbial-load-adjusted burden"
python src/s08_absolute_burden.py

log "9/12  carriage, community state, geography robustness"
python src/s09_carriage.py

log "10/12 within-subject longitudinal analysis"
python src/s11_longitudinal.py

log "11/12 perturbation studies"
python src/s12_perturbation.py

log "12/14 cross-stratum comparison"
python src/s10_compare_strata.py

log "13/14 does mobility explain predictability?"
if [[ -n "${PANGENOME:-}" ]]; then
  python src/s13_mobility_predictability.py --pangenome "$PANGENOME"
else
  echo "PANGENOME not set - skipping. Run s06c first, then:"
  echo "  PANGENOME=<path>/pangenome_evidence.tsv bash run_all.sh"
fi

log "14/15 temporal direction"
python src/s14_temporal_direction.py

log "15/16 dose-response and mediation"
if [[ -n "${ARMS:-}" ]]; then
  python src/s15_mediation_dose.py --arms "$ARMS"
else
  python src/s15_mediation_dose.py
fi

log "16/17 within-subject exposure boundary"
python src/s16_exposure_boundary.py

log "17/17 robustness, sensitivity and tested nulls"
if [[ -n "${PANGENOME:-}" ]]; then
  python src/s20_robustness.py --pangenome "$PANGENOME" \
      ${HABITAT_OUT:+--habitats "$HABITAT_OUT"}
else
  python src/s20_robustness.py ${HABITAT_OUT:+--habitats "$HABITAT_OUT"}
fi

log "done"
echo "Figures : $(python -c "import sys;sys.path.insert(0,'src');from common import load_config;print(load_config()['paths']['fig_dir'])")"
echo "Tables  : $(python -c "import sys;sys.path.insert(0,'src');from common import load_config;print(load_config()['paths']['table_dir'])")"
