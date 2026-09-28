#!/usr/bin/env bash
# =============================================================================
# run_sensitivity.sh
#
# Stability of the headline variance estimate across analytical choices.
# Reviewers will ask whether the result depends on the 5% prevalence cut or the
# 50-component taxonomy representation; this answers the question rather than
# asserting robustness.
#
# Writes each variant's variance partition to tables/ with a descriptive
# suffix, then restores the original configuration and matrices.
#
# Usage:  bash run_sensitivity.sh
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

CFG="config.local.yaml"
[[ -f "$CFG" ]] || CFG="config.yaml"
cp "$CFG" "${CFG}.sens.bak"

set_cfg() {   # set_cfg <dotted.key> <value>
  python - "$CFG" "$1" "$2" <<'PY'
import sys, yaml, pathlib
p, key, val = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
c = yaml.safe_load(p.read_text())
node = c
parts = key.split(".")
for k in parts[:-1]:
    node = node.setdefault(k, {})
try:
    val = float(val) if "." in val else int(val)
except ValueError:
    pass
node[parts[-1]] = val
p.write_text(yaml.safe_dump(c, sort_keys=False))
PY
}

echo "=============================================================="
echo "1. ARG prevalence threshold"
echo "=============================================================="
for prev in 0.01 0.05 0.10; do
  echo "-- threshold $prev --"
  set_cfg filters.arg_min_prevalence "$prev"
  python src/s03_build_matrices.py 2>&1 | grep -E "ARG prevalence filter|FINAL ALIGNED"
  python src/s04_variance_partition.py 2>&1 \
    | grep -E "Variance partitioning on|^.*taxonomy marginal"
done
set_cfg filters.arg_min_prevalence 0.05

echo
echo "=============================================================="
echo "2. Number of taxonomy components"
echo "=============================================================="
python src/s03_build_matrices.py >/dev/null 2>&1
for npc in 25 50 100; do
  echo "-- $npc components --"
  W1_TAXA_PCS="$npc" python src/s04_variance_partition.py 2>&1 \
    | grep -E "Taxonomy block|taxonomy marginal"
done

echo
echo "=============================================================="
echo "3. One sample per subject"
echo "=============================================================="
W1_ONE_PER_SUBJECT=1 python src/s04_variance_partition.py 2>&1 \
  | grep -E "Variance partitioning on|taxonomy marginal"

mv "${CFG}.sens.bak" "$CFG"
python src/s03_build_matrices.py >/dev/null 2>&1
echo
echo "Configuration and matrices restored to the primary settings."
echo "Collect the taxonomy rows above into a sensitivity table for the"
echo "supplementary material; the estimate should be stable across all of them."
