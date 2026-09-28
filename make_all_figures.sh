#!/usr/bin/env bash
# =============================================================================
# make_all_figures.sh
#
# Regenerates every figure from the data currently on disk, so that no figure
# is left over from an earlier run. Figures accumulated across many runs are a
# real hazard: a stale panel looks identical to a current one, and there is no
# way to tell from the file which analysis produced it.
#
# Existing figures are moved to figures/_superseded_<timestamp>/ rather than
# deleted, so nothing is lost if a regeneration fails.
#
#   bash make_all_figures.sh                       # gut figures, all strata
#   PANGENOME=<path> bash make_all_figures.sh      # adds attribution + mobility
#   HABITATS=<dir>   bash make_all_figures.sh      # adds cross-habitat figures
#
# Stages whose inputs are missing are SKIPPED and listed at the end - the
# figure code no longer writes blank panels, so an absent figure means an
# absent analysis, not a rendering fault.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

CFG="config.local.yaml"; [[ -f "$CFG" ]] || CFG="config.yaml"
STRATA=(adult infant all)
STAMP=$(date +%Y%m%d_%H%M)
MISSING=()

FIGDIR=$(python - "$CFG" <<'PY'
import sys, yaml, pathlib
print(yaml.safe_load(pathlib.Path(sys.argv[1]).read_text())["paths"]["fig_dir"])
PY
)

blue() { printf '\n\033[1;34m%s\033[0m\n' "$*"; }
ok()   { printf '\033[0;32m    ok\033[0m\n'; }
skip() { printf '\033[0;33m    skipped (%s)\033[0m\n' "$1"; MISSING+=("$2"); }

set_stratum() {
  python - "$CFG" "$1" <<'PY'
import sys, yaml, pathlib
p, s = pathlib.Path(sys.argv[1]), sys.argv[2]
c = yaml.safe_load(p.read_text()); c.setdefault("analysis", {})["stratum"] = s
p.write_text(yaml.safe_dump(c, sort_keys=False))
PY
}

run() {   # run <label> <marker-file-or-empty> <command...>
  local label="$1" need="$2"; shift 2
  printf '  %-34s' "$label"
  if [[ -n "$need" && ! -e "$need" ]]; then
    skip "$(basename "$need") not built" "$label"; return
  fi
  if "$@" > /tmp/figgen_last.log 2>&1; then ok
  else skip "see /tmp/figgen_last.log" "$label"; fi
}

# ---- archive whatever is there now --------------------------------------
blue "archiving existing figures"
OLD="$FIGDIR/_superseded_$STAMP"
mkdir -p "$OLD"
found=$(find "$FIGDIR" -maxdepth 1 -type f \( -name '*.png' -o -name '*.pdf' -o -name '*.svg' \) | wc -l)
if [[ "$found" -gt 0 ]]; then
  find "$FIGDIR" -maxdepth 1 -type f \( -name '*.png' -o -name '*.pdf' -o -name '*.svg' \) \
    -exec mv {} "$OLD"/ \;
  echo "  moved $found file(s) to $(basename "$OLD")"
else
  echo "  none to archive"; rmdir "$OLD" 2>/dev/null
fi

WORK=$(python - "$CFG" <<'PY'
import sys, yaml, pathlib
print(yaml.safe_load(pathlib.Path(sys.argv[1]).read_text())["paths"]["work_dir"])
PY
)

# ---- workflow schematic (Figure 1) --------------------------------------
blue "workflow"
set_stratum adult
run "workflow (fig1)" "$WORK/sample_table.parquet" python src/s21_workflow_figure.py

# ---- per stratum ---------------------------------------------------------
for s in "${STRATA[@]}"; do
  blue "stratum: $s"
  set_stratum "$s"
  run "figures 1-5"  "$WORK/arg_abundance.parquet"  python src/s07_figures.py
  if [[ -n "${PANGENOME:-}" ]]; then
    run "mobility (fig6)" "$WORK/cv_metrics_$s.parquet" \
        python src/s13_mobility_predictability.py --pangenome "$PANGENOME"
  fi
done

# ---- cross stratum and the rest -----------------------------------------
blue "cross-stratum and derived"
set_stratum all
run "life stages (fig7)"   "$WORK/vp_results_all.parquet"  python src/s10_compare_strata.py
run "temporal (fig8)"      "$WORK/taxa_clr.parquet"        python src/s14_temporal_direction.py
run "exposure (fig9)"      "$WORK/taxa_clr.parquet"        python src/s16_exposure_boundary.py

set_stratum adult
if [[ -n "${PANGENOME:-}" ]]; then
  run "robustness (fig11)" "$WORK/cv_metrics_adult.parquet" \
      python src/s20_robustness.py --pangenome "$PANGENOME" \
        ${HABITATS:+--habitats "$HABITATS"}
else
  run "robustness (fig11)" "$WORK/cv_metrics_adult.parquet" \
      python src/s20_robustness.py ${HABITATS:+--habitats "$HABITATS"}
fi

# ---- habitats ------------------------------------------------------------
if [[ -n "${HABITATS:-}" ]]; then
  blue "cross-habitat"
  run "habitat comparison" "$HABITATS" \
      python habitat/compare_habitats.py --habitat-dirs "$HABITATS"/*/ \
        --gut-tables "$(dirname "$WORK")/tables" --gut-work "$WORK" \
        --outdir "$HABITATS"
  run "fold heterogeneity" "$HABITATS" \
      python habitat/fold_heterogeneity.py --habitat-dirs "$HABITATS"/*/ \
        --gut-work "$WORK" --gut-tables "$(dirname "$WORK")/tables" \
        --gut-stratum "${STRATUM:-adult}" --outdir "$HABITATS"
fi

# ---- inventory -----------------------------------------------------------
blue "figures now present"
python - "$FIGDIR" "${HABITATS:-}" <<'PY'
import sys, pathlib, datetime, collections
figdir = pathlib.Path(sys.argv[1])
dirs = [figdir] + ([pathlib.Path(sys.argv[2])] if len(sys.argv) > 2 and sys.argv[2] else [])
rows = []
for d in dirs:
    if not d.exists():
        continue
    for f in sorted(d.glob("*.png")):
        t = datetime.datetime.fromtimestamp(f.stat().st_mtime)
        rows.append((f.name, t.strftime("%H:%M"), f.parent.name))
if not rows:
    print("  none")
else:
    stamps = collections.Counter(r[1] for r in rows)
    for name, t, where in rows:
        print(f"  {t}  {name:<42} {where}")
    print(f"\n  {len(rows)} figures across {len(stamps)} distinct minutes")
    if len(stamps) > 4:
        print("  NOTE: timestamps span several minutes - normal for one pass,")
        print("        but check none predate this run.")
PY

if [[ ${#MISSING[@]} -gt 0 ]]; then
  blue "not generated"
  for m in "${MISSING[@]}"; do echo "  - $m"; done
  echo
  echo "A missing figure means the analysis behind it has not been run,"
  echo "not that rendering failed - blank panels are no longer written."
fi
set_stratum adult
echo
echo "superseded figures kept in $OLD"
