#!/usr/bin/env bash
# =============================================================================
# run_strata.sh
#
# Runs the analysis separately for each life stage, then compares them.
#
# Stages 2 and 3 are NOT repeated - the sample table and matrices are built
# once for everyone, and only the analysis stages are stratified. So this is
# minutes per stratum, not hours.
#
# Usage:
#     bash run_strata.sh                 # adult, infant, all
#     bash run_strata.sh adult infant    # a chosen subset
# =============================================================================
set -uo pipefail
FAILED=0
cd "$(dirname "$0")"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

STRATA=("$@")
[[ ${#STRATA[@]} -eq 0 ]] && STRATA=(adult infant all)

CFG="config.local.yaml"
[[ -f "$CFG" ]] || CFG="config.yaml"
cp "$CFG" "${CFG}.bak"

for s in "${STRATA[@]}"; do
  printf '\n\033[1m======== stratum: %s ========\033[0m\n' "$s"
  python - "$CFG" "$s" <<'PY'
import sys, yaml, pathlib
p, s = pathlib.Path(sys.argv[1]), sys.argv[2]
c = yaml.safe_load(p.read_text())
c.setdefault("analysis", {})["stratum"] = s
p.write_text(yaml.safe_dump(c, sort_keys=False))
PY
  python src/s04_variance_partition.py || { echo "  s04 FAILED for $s"; FAILED=1; }
  python src/s05_models.py --models all || { echo "  s05 FAILED for $s"; FAILED=1; }
  python src/s08_absolute_burden.py    || echo "  s08 failed for $s (needs microbial load)"
  python src/s09_carriage.py           || echo "  s09 failed for $s"
  python src/s07_figures.py            || echo "  figures failed for $s"
done

if [[ $FAILED -ne 0 ]]; then
  echo "ABORTING before the cross-stratum comparison: a stratum failed, and"
  echo "comparing now would mix fresh output with stale results."
  mv "${CFG}.bak" "$CFG"; exit 1
fi

printf '\n\033[1m======== cross-stratum comparison ========\033[0m\n'
python src/s10_compare_strata.py

mv "${CFG}.bak" "$CFG"
echo
echo "Config restored. Per-stratum outputs carry a _<stratum> suffix."
