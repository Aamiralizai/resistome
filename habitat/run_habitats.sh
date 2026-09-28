#!/usr/bin/env bash
# =============================================================================
# run_habitats.sh - cross-habitat analysis, self-contained.
#
# Runs entirely outside the human gut pipeline: its own scripts, its own output
# directory, no shared configuration or state. The gut pipeline's files are
# imported read-only for pure functions and are never modified, so a habitat
# run cannot disturb a validated gut result.
#
#   bash habitat/run_habitats.sh                        # animal, ocean, environmental
#   bash habitat/run_habitats.sh ocean                  # one habitat
#   SPLIT_BIOMES=1 bash habitat/run_habitats.sh         # split environmental
#
# Outputs: /mnt/x/w1_resistome/habitats/<name>/
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="${METALOG_ROOT:-/mnt/x/w1_resistome}"
OUT="${HABITAT_OUT:-$ROOT/habitats}"
ZEN="${ZENODO_DIR:-$ROOT/zenodo}"
MAP="${MAPPING:-$(ls $ROOT/Metalog/sequencing_db_mapping*.tsv.gz 2>/dev/null | head -1)}"
LOGS="$OUT/logs"; mkdir -p "$LOGS"

HABS=("$@"); [[ ${#HABS[@]} -eq 0 ]] && HABS=(ocean animal environmental)
FAILED=0

echo "=============================================="
echo "habitats : ${HABS[*]}"
echo "mapping  : $MAP"
echo "outputs  : $OUT"
echo "=============================================="
[[ -f "$MAP" ]] || { echo "mapping file not found - set MAPPING"; exit 1; }

go() {   # go <name> <habitat-dir> [biome-regex]
  local name="$1" dir="$2" biome="${3:-}"
  printf '\n>>> %s\n' "$name"
  local extra=()
  [[ -n "$biome" ]] && extra=(--biome-pattern "$biome")
  if python habitat/habitat_analysis.py --habitat-dir "$dir" --name "$name" \
       --zenodo-dir "$ZEN" --mapping "$MAP" --outdir "$OUT" "${extra[@]}" \
       > "$LOGS/$name.log" 2>&1; then
    echo "    ok   ($LOGS/$name.log)"
    grep -E "ANALYSED|taxonomy " "$LOGS/$name.log" | tail -3 | sed 's/^/    /'
  else
    echo "    FAILED ($LOGS/$name.log)"; tail -4 "$LOGS/$name.log" | sed 's/^/    | /'
    FAILED=1
  fi
}

for h in "${HABS[@]}"; do
  case "$h" in
    ocean)  go ocean  "$ROOT/Metalog_ocean" ;;
    animal) go animal "$ROOT/Metalog_animal" ;;
    environmental)
      if [[ -n "${SPLIT_BIOMES:-}" ]]; then
        go soil       "$ROOT/Metalog_environmental" 'soil|terrestrial|rhizosphere'
        go wastewater "$ROOT/Metalog_environmental" 'wastewater|sewage|sludge|effluent'
        go freshwater "$ROOT/Metalog_environmental" 'freshwater|river|lake|groundwater'
      else
        go environmental "$ROOT/Metalog_environmental"
      fi ;;
    *) echo "unknown habitat: $h" ;;
  esac
done

printf '\n>>> comparison\n'
python habitat/compare_habitats.py \
  --habitat-dirs "$OUT"/*/ \
  --gut-tables "$ROOT/tables" --gut-work "$ROOT/work" \
  --outdir "$OUT" 2>&1 | tail -40

printf '\n>>> fold-level heterogeneity\n'
python habitat/fold_heterogeneity.py \
  --habitat-dirs "$OUT"/*/ \
  --gut-work "$ROOT/work" \
  --outdir "$OUT" 2>&1 | tail -30

[[ $FAILED -eq 0 ]] && echo -e "\nall habitats complete" || echo -e "\nsome habitats failed"
exit $FAILED
