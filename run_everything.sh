#!/usr/bin/env bash
# =============================================================================
# run_everything.sh - the whole analysis, start to finish.
#
#   bash run_everything.sh                    # everything that is available
#   bash run_everything.sh --from 04          # resume from a stage
#   bash run_everything.sh --only 13,14,16    # selected stages only
#   bash run_everything.sh --quick            # skip the multi-hour model runs
#   bash run_everything.sh --list             # show stages and exit
#
# Design notes
# ------------
# Stages 2 and 3 build the sample table and matrices ONCE for everyone.
# Stages 4, 5, 7, 8 and 9 are stratified and run per life stage. Stages 11-16
# need repeated sampling and run on the combined stratum, where the
# longitudinal cohorts live.
#
# Every stage writes a marker on success, so a rerun skips completed work
# unless --force is given. Failures are collected and reported at the end
# rather than aborting silently, EXCEPT where a later stage would otherwise
# combine fresh output with stale results - the cross-stratum comparison is
# skipped in that case rather than producing a misleading table.
#
# Optional inputs, set as environment variables:
#   PANGENOME=<path>/pangenome_evidence.tsv   enables stages 6, 6d, 13
#   ARMS=<path>/curated_arms.csv              enables difference-in-differences
#   HABITATS="dir1 dir2"                      enables the cross-habitat check
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

CFG="config.local.yaml"; [[ -f "$CFG" ]] || CFG="config.yaml"
STRATA=(adult infant all)
LONGITUDINAL_STRATUM="all"
MARKERS=".run_markers"; mkdir -p "$MARKERS"
LOGDIR="logs"; mkdir -p "$LOGDIR"
FAILURES=()
SKIPPED=()

FROM=""; ONLY=""; QUICK=0; FORCE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --from)  FROM="$2"; shift 2 ;;
    --only)  ONLY="$2"; shift 2 ;;
    --quick) QUICK=1; shift ;;
    --force) FORCE=1; shift ;;
    --list)  sed -n '/^# STAGE LIST/,/^# END STAGE LIST/p' "$0" | sed 's/^# //'; exit 0 ;;
    *) echo "unknown option: $1"; exit 1 ;;
  esac
done

# STAGE LIST
#  00  environment smoke test
#  00b discover downloaded files
#  01  inspect input schemas
#  02  accession join            -> sample_table
#  03  build ARG and taxa matrices
#  04  variance partitioning         [per stratum]
#  05  model comparison              [per stratum, slow]
#  07  manuscript figures            [per stratum]
#  08  load-adjusted burden          [per stratum, adult only in practice]
#  09  carriage and geography        [per stratum]
#  10  cross-stratum comparison
#  06  attribution                   [needs PANGENOME]
#  06d pangenome retrieval status    [needs PANGENOME]
#  13  mobility vs predictability    [needs PANGENOME]
#  11  within-subject longitudinal   [combined stratum]
#  12  intervention studies          [combined stratum]
#  14  temporal cross-prediction     [combined stratum]
#  15  dose-response and mediation   [combined stratum]
#  16  exposure boundary             [combined stratum]
#  17  cross-habitat viability       [needs HABITATS]
# END STAGE LIST

blue()  { printf '\n\033[1;34m%s\033[0m\n' "$*"; }
green() { printf '\033[0;32m%s\033[0m\n' "$*"; }
red()   { printf '\033[0;31m%s\033[0m\n' "$*"; }

set_stratum() {
  python - "$CFG" "$1" <<'PY'
import sys, yaml, pathlib
p, s = pathlib.Path(sys.argv[1]), sys.argv[2]
c = yaml.safe_load(p.read_text()); c.setdefault("analysis", {})["stratum"] = s
p.write_text(yaml.safe_dump(c, sort_keys=False))
PY
}

want() {                      # want <stage-id>
  local id="$1"
  if [[ -n "$ONLY" ]]; then [[ ",$ONLY," == *",$id,"* ]] && return 0 || return 1; fi
  if [[ -n "$FROM" ]]; then
    # numeric-ish comparison on the leading digits
    local a="${id%%[a-z]*}" b="${FROM%%[a-z]*}"
    (( 10#$a >= 10#$b )) && return 0 || return 1
  fi
  return 0
}

run() {                       # run <stage-id> <label> <command...>
  local id="$1" label="$2"; shift 2
  want "$id" || return 0
  local marker="$MARKERS/stage_${id}.done"
  if [[ $FORCE -eq 0 && -f "$marker" && -z "$ONLY" ]]; then
    echo "  [$id] $label - already done, skipping (use --force to repeat)"
    return 0
  fi
  local log="$LOGDIR/stage_${id}.log"
  echo "  [$id] $label"
  if "$@" > "$log" 2>&1; then
    green "       ok  ($log)"
    touch "$marker"
  else
    red   "       FAILED  (see $log)"
    tail -5 "$log" | sed 's/^/       | /'
    FAILURES+=("$id $label")
  fi
}

START=$(date +%s)
blue "================ W1 RESISTOME PIPELINE ================"
echo "config: $CFG"
echo "strata: ${STRATA[*]}"
[[ -n "${PANGENOME:-}" ]] && echo "pangenome evidence: $PANGENOME" \
  || SKIPPED+=("stages 06/06d/13 - set PANGENOME to enable")
[[ -n "${ARMS:-}" ]] && echo "curated arms: $ARMS" \
  || SKIPPED+=("difference-in-differences - set ARMS to enable")
[[ -n "${HABITATS:-}" ]] && echo "habitats: $HABITATS" \
  || SKIPPED+=("stage 17 - set HABITATS to enable")
[[ $QUICK -eq 1 ]] && echo "QUICK mode: model comparison limited to ridge and xgb"

# ---------------------------------------------------------------- setup ----
blue "-- setup and inputs --"
run 00  "environment smoke test"      python src/s00_smoke_test.py
run 00b "discover downloaded files"   python src/s00b_discover_files.py
run 01  "inspect input schemas"       python src/s01_inspect_inputs.py

# ------------------------------------------------------------- build once --
blue "-- build the dataset (once for all strata) --"
run 02 "accession join"               python src/s02_build_sample_table.py
run 03 "ARG and taxa matrices"        python src/s03_build_matrices.py

# --------------------------------------------------------- per stratum -----
MODELS="all"; [[ $QUICK -eq 1 ]] && MODELS="mean,ridge,xgb"
STRATUM_FAILED=0
for s in "${STRATA[@]}"; do
  blue "-- stratum: $s --"
  set_stratum "$s"
  before=${#FAILURES[@]}
  run "04_$s" "variance partitioning"  python src/s04_variance_partition.py
  run "05_$s" "model comparison"       python src/s05_models.py --models "$MODELS"
  run "08_$s" "load-adjusted burden"   python src/s08_absolute_burden.py
  run "09_$s" "carriage and geography" python src/s09_carriage.py
  # attribution needs the stratum's own trained model, so it belongs inside
  # the per-stratum loop rather than being run once for adults.
  if [[ -n "${PANGENOME:-}" ]]; then
    run "06_$s" "attribution vs pangenome" \
        python src/s06_attribution.py --genome-evidence "$PANGENOME"
  fi
  run "07_$s" "figures"                python src/s07_figures.py
  (( ${#FAILURES[@]} > before )) && STRATUM_FAILED=1
done

# ------------------------------------------------------ cross stratum ------
blue "-- cross-stratum comparison --"
if [[ $STRATUM_FAILED -eq 1 ]]; then
  red "  skipped: a stratum failed, and comparing now would mix fresh output"
  red "  with stale results from an earlier run."
  SKIPPED+=("stage 10 - a stratum failed")
else
  run 10 "compare strata" python src/s10_compare_strata.py
fi

# ------------------------------------------------ attribution & mechanism --
if [[ -n "${PANGENOME:-}" ]]; then
  blue "-- mechanism (adult) --"
  set_stratum adult
  run 06d "pangenome retrieval status" python src/s06d_pangenome_status.py \
        --pangenome-dir "$(dirname "$PANGENOME")"
  run 13  "mobility vs predictability" python src/s13_mobility_predictability.py \
        --pangenome "$PANGENOME"
fi

# ------------------------------------------------- longitudinal & causal ---
blue "-- longitudinal and quasi-causal (stratum: $LONGITUDINAL_STRATUM) --"
set_stratum "$LONGITUDINAL_STRATUM"
run 11 "within-subject longitudinal" python src/s11_longitudinal.py
run 12 "intervention studies"        python src/s12_perturbation.py
run 14 "temporal cross-prediction"   python src/s14_temporal_direction.py
if [[ -n "${ARMS:-}" ]]; then
  run 15 "dose-response and mediation" python src/s15_mediation_dose.py --arms "$ARMS"
else
  run 15 "dose-response and mediation" python src/s15_mediation_dose.py
fi
run 16 "exposure boundary"           python src/s16_exposure_boundary.py

# ------------------------------------------------------- cross habitat -----
if [[ -n "${HABITATS:-}" ]]; then
  blue "-- cross-habitat viability --"
  run 17 "habitat join check" python src/s17_habitat_join_check.py --habitats ${HABITATS}
fi

set_stratum adult

# ------------------------------------------------------------- summary -----
ELAPSED=$(( $(date +%s) - START ))
blue "======================= SUMMARY ======================="
printf 'elapsed: %dh %dm\n' $((ELAPSED/3600)) $(((ELAPSED%3600)/60))
if [[ ${#FAILURES[@]} -eq 0 ]]; then
  green "all attempted stages completed"
else
  red "${#FAILURES[@]} stage(s) failed:"
  for f in "${FAILURES[@]}"; do red "   - $f  (logs/stage_${f%% *}.log)"; done
fi
if [[ ${#SKIPPED[@]} -gt 0 ]]; then
  echo
  echo "not attempted:"
  for s in "${SKIPPED[@]}"; do echo "   - $s"; done
fi
echo
echo "outputs:"
python - "$CFG" <<'PY'
import sys, yaml, pathlib
c = yaml.safe_load(pathlib.Path(sys.argv[1]).read_text())["paths"]
for k in ("table_dir", "fig_dir", "work_dir"):
    d = pathlib.Path(c[k])
    n = len(list(d.glob("*"))) if d.exists() else 0
    print(f"   {k:<10} {c[k]}  ({n} files)")
PY
echo
echo "next: check that every file in the table directory carries a timestamp"
echo "from this run before using any of it in the manuscript."
[[ ${#FAILURES[@]} -eq 0 ]] || exit 1
