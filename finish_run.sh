#!/usr/bin/env bash
# =============================================================================
# finish_run.sh
#
# Completes a run whose cross-validation succeeded but whose held-out block
# failed. Reuses the existing cv_metrics/cv_predictions for each stratum and
# runs only the held-out evaluation, the paired tests, the final model fit,
# and the downstream stages.
#
# Saves roughly six hours compared with repeating cross-validation.
#
# Usage: bash finish_run.sh
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
CFG="config.local.yaml"; [[ -f "$CFG" ]] || CFG="config.yaml"
cp "$CFG" "${CFG}.finish.bak"

set_stratum() {
  python - "$CFG" "$1" <<'PY'
import sys, yaml, pathlib
p, s = pathlib.Path(sys.argv[1]), sys.argv[2]
c = yaml.safe_load(p.read_text()); c.setdefault("analysis", {})["stratum"] = s
p.write_text(yaml.safe_dump(c, sort_keys=False))
PY
}

for s in adult infant all; do
  printf '\n\033[1m======== stratum: %s ========\033[0m\n' "$s"
  set_stratum "$s"
  python src/s05_models.py --models all --skip-cv || echo "  s05 holdout failed for $s"
  python src/s08_absolute_burden.py || echo "  s08 skipped for $s"
  python src/s09_carriage.py        || echo "  s09 failed for $s"
  python src/s07_figures.py         || echo "  figures failed for $s"
done

printf '\n\033[1m======== cross-stratum comparison ========\033[0m\n'
python src/s10_compare_strata.py

mv "${CFG}.finish.bak" "$CFG"
echo "Config restored."
