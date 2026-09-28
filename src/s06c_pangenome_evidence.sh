#!/usr/bin/env bash
# =============================================================================
# s06c_pangenome_evidence.sh
#
# Builds PANGENOME-level evidence: which ARGs appear in ANY assembly of a
# species, not just in one reference.
#
# Why this exists: acquired ARGs live in the accessory genome. A single
# reference assembly is one strain, usually a susceptible one, so blaTEM,
# blaCTX-M, sul1 and tet(A) are absent from the E. coli reference even though
# they are textbook E. coli resistance genes. Validating against one genome
# therefore measures which assembly NCBI happens to call "reference", not
# whether a species-gene association is real.
#
# Strategy: for the species that actually carry the model's attributions,
# download up to N assemblies each, run AMRFinder on all of them, and record a
# gene as present if it appears in at least one. Also records the FREQUENCY,
# which is more informative than presence - a gene in 18/20 assemblies is core,
# a gene in 1/20 is accessory and genuinely mobile.
#
# Usage:
#     bash src/s06c_pangenome_evidence.sh [OUTDIR] [N_PER_SPECIES] [N_SPECIES]
# Defaults: ~/pangenome 12 60
#
# Resumable. Budget 2-4 hours for 60 species x 12 assemblies.
# =============================================================================
set -uo pipefail

OUTDIR="${1:-$HOME/pangenome}"
N_PER="${2:-12}"
N_SPECIES="${3:-60}"
DB="${AMRFINDER_DB:-/mnt/x/w1_resistome/amrfinder_db}"
THREADS="${THREADS:-8}"
TABLES="${TABLES:-/mnt/x/w1_resistome/tables}"
EVIDENCE="$OUTDIR/pangenome_evidence.tsv"

mkdir -p "$OUTDIR"/{fna,amr,tmp}

command -v datasets >/dev/null || { echo "datasets CLI not found"; exit 1; }
amrfinder -d "$DB" --database_version >/dev/null 2>&1 \
  || { echo "AMRFinder database not usable at $DB"; exit 1; }

# ------------------------------------------------------ pick the species ----
# Prioritise species carrying the model's strongest attributions - validating
# species the model never implicates would be wasted effort.
python - "$TABLES" "$N_SPECIES" > "$OUTDIR/species.txt" <<'PY'
import sys, pandas as pd, pathlib
tables, n = pathlib.Path(sys.argv[1]), int(sys.argv[2])
cands = []
for name in ("unexplained_associations.csv", "top_associations.csv"):
    p = tables / name
    if p.exists():
        d = pd.read_csv(p)
        if {"species", "attribution"} <= set(d.columns):
            cands.append(d[["species", "attribution"]])
if not cands:
    sys.exit("No association table found - run s06_attribution.py first.")
d = pd.concat(cands)
rank = (d.assign(a=d["attribution"].abs())
          .groupby("species")["a"].sum().sort_values(ascending=False))
rank = rank[~rank.index.str.contains("GGB|SGB", case=False, na=False)]
for s in rank.head(n).index:
    print(str(s).replace("_", " ").strip())
PY

mapfile -t SPECIES < "$OUTDIR/species.txt"
TOTAL=${#SPECIES[@]}
echo "=========================================================="
echo "Species            : $TOTAL   (top by attribution weight)"
echo "Assemblies each    : up to $N_PER"
echo "Output             : $EVIDENCE"
echo "=========================================================="

i=0
for sp in "${SPECIES[@]}"; do
  i=$((i + 1))
  slug=$(echo "$sp" | tr ' /' '__')
  done_marker="$OUTDIR/amr/${slug}.done"
  [[ -f "$done_marker" ]] && { echo "[$i/$TOTAL] $sp  (already done)"; continue; }

  printf '[%3d/%3d] %-40s ' "$i" "$TOTAL" "$sp"

  # ---- list accessions, preferring the most complete assemblies ----------
  acc_file="$OUTDIR/tmp/${slug}.acc"
  datasets summary genome taxon "$sp" --assembly-source RefSeq --as-json-lines \
    > "$OUTDIR/tmp/${slug}.jsonl" 2>/dev/null
  python - "$OUTDIR/tmp/${slug}.jsonl" "$N_PER" > "$acc_file" <<'PY'
import sys, json
path, n = sys.argv[1], int(sys.argv[2])
order = {"Complete Genome": 0, "Chromosome": 1, "Scaffold": 2, "Contig": 3}
rows = []
try:
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        acc = r.get("accession")
        lvl = (r.get("assembly_info") or {}).get("assembly_level", "Contig")
        if acc:
            rows.append((order.get(lvl, 4), acc))
except FileNotFoundError:
    pass
for _, acc in sorted(rows)[:n]:
    print(acc)
PY

  n_acc=$(wc -l < "$acc_file" 2>/dev/null || echo 0)
  if [[ "$n_acc" -eq 0 ]]; then
    echo "no assemblies"; touch "$done_marker"; continue
  fi

  # ---- download them in one request -------------------------------------
  zip="$OUTDIR/tmp/${slug}.zip"
  rm -f "$zip"
  # shellcheck disable=SC2046
  if ! datasets download genome accession $(tr '\n' ' ' < "$acc_file") \
       --include genome --filename "$zip" >/dev/null 2>&1 \
     || ! unzip -t "$zip" >/dev/null 2>&1; then
    echo "download failed"; rm -f "$zip"; touch "$done_marker"; continue
  fi

  rm -rf "$OUTDIR/tmp/${slug}"; mkdir -p "$OUTDIR/tmp/${slug}"
  unzip -qo "$zip" -d "$OUTDIR/tmp/${slug}" 2>/dev/null

  # ---- AMRFinder on each assembly ---------------------------------------
  k=0
  while IFS= read -r fna; do
    k=$((k + 1))
    out="$OUTDIR/amr/${slug}__$(printf '%02d' "$k").tsv"
    [[ -s "$out" ]] && continue
    amrfinder -d "$DB" -n "$fna" --plus --threads "$THREADS" -o "$out" >/dev/null 2>&1 \
      || rm -f "$out"
  done < <(find "$OUTDIR/tmp/${slug}" -name "*.fna" 2>/dev/null)

  rm -rf "$OUTDIR/tmp/${slug}" "$zip" "$OUTDIR/tmp/${slug}.jsonl"
  touch "$done_marker"
  echo "${k} assemblies"
done

# ------------------------------------------------------------ aggregate ----
echo
echo "Aggregating to $EVIDENCE ..."
python - "$OUTDIR/amr" "$EVIDENCE" <<'PY'
import sys, glob, os, csv, re
from collections import defaultdict
amr_dir, out = sys.argv[1], sys.argv[2]
per_species_genomes = defaultdict(set)
hits = defaultdict(set)          # (species, gene) -> {genome ids}

for f in sorted(glob.glob(os.path.join(amr_dir, "*.tsv"))):
    base = os.path.basename(f)[:-4]
    m = re.match(r"^(.*)__(\d+)$", base)
    if not m:
        continue
    species, gid = m.group(1).replace("_", " "), m.group(2)
    per_species_genomes[species].add(gid)
    with open(f, newline="") as fh:
        rd = csv.DictReader(fh, delimiter="\t")
        gcol = next((c for c in (rd.fieldnames or [])
                     if c.strip().lower() in ("element symbol", "gene symbol", "gene")), None)
        if gcol is None:
            continue
        for r in rd:
            g = (r.get(gcol) or "").strip()
            if g:
                hits[(species, g)].add(gid)

with open(out, "w", newline="") as fh:
    w = csv.writer(fh, delimiter="\t")
    w.writerow(["species", "gene", "n_genomes_with", "n_genomes_total", "frequency"])
    for (species, gene), gids in sorted(hits.items()):
        tot = len(per_species_genomes[species])
        w.writerow([species, gene, len(gids), tot, round(len(gids) / max(tot, 1), 3)])

n_sp = len(per_species_genomes)
n_gen = sum(len(v) for v in per_species_genomes.values())
print(f"{n_sp} species, {n_gen} assemblies, {len(hits)} species-gene pairs")
acc = [(s, len(v)) for s, v in sorted(per_species_genomes.items())]
print("assemblies per species: min %d, median %d, max %d" % (
    min(n for _, n in acc), sorted(n for _, n in acc)[len(acc) // 2],
    max(n for _, n in acc)))
PY

echo
echo "Next:"
echo "  python src/s06_attribution.py --genome-evidence $EVIDENCE"
echo
echo "The evidence file carries a 'frequency' column: a gene in most assemblies"
echo "is core, one in a few is accessory and genuinely mobile. Use"
echo "--min-genome-frequency to require more than a single hit."
