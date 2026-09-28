#!/usr/bin/env bash
# =============================================================================
# s06e_independent_pangenome.sh
#
# Rebuilds the pangenome evidence with species chosen INDEPENDENTLY of the
# prediction model.
#
# Why this exists
# ---------------
# The original evidence set used the 60 species carrying the strongest model
# attributions. That makes the resulting genomic quantity partly a product of
# the model whose performance it is later used to explain - the single most
# likely referee objection to the genome-boundedness result. Selecting species
# by abundance and prevalence alone removes the circularity: the model has no
# hand in deciding which species are examined.
#
# The evidence table also records explicit ZEROS - species-gene combinations
# examined and not found - which the attribution-driven table could not, since
# it only ever contained detected pairs.
#
# Usage:
#     bash src/s06e_independent_pangenome.sh [OUTDIR] [N_PER_SPECIES] [N_SPECIES]
# Defaults: /mnt/x/w1_resistome/pangenome_independent 12 200
#
# Budget 6-10 hours for 200 species x 12 assemblies. Resumable per species.
# =============================================================================
set -uo pipefail

OUTDIR="${1:-/mnt/x/w1_resistome/pangenome_independent}"
N_PER="${2:-12}"
N_SPECIES="${3:-200}"
DB="${AMRFINDER_DB:-/mnt/x/w1_resistome/amrfinder_db}"
STRATUM="${STRATUM:-adult}"
THREADS="${THREADS:-8}"
WORK="${WORK_DIR:-/mnt/x/w1_resistome/work}"
EVIDENCE="$OUTDIR/pangenome_evidence_independent.tsv"

mkdir -p "$OUTDIR"/{amr,tmp}
command -v datasets >/dev/null || { echo "datasets CLI not found"; exit 1; }
amrfinder -d "$DB" --database_version >/dev/null 2>&1 \
  || { echo "AMRFinder database not usable at $DB"; exit 1; }

# ---- species chosen by PREVALENCE, with no reference to the model ---------
python - "$WORK" "$N_SPECIES" "$STRATUM" > "$OUTDIR/species.txt" <<'PY'
import sys, re, pathlib, pandas as pd
work, n = pathlib.Path(sys.argv[1]), int(sys.argv[2])
stratum = sys.argv[3] if len(sys.argv) > 3 else "adult"
# Rank from the UNFILTERED taxonomic universe. taxa_relab.parquet has already
# passed a pooled prevalence filter across the whole cohort, so ranking within
# it would give "the most prevalent adult species among taxa that first passed
# a whole-cohort filter" - not what the Methods claim.
rel = work / "taxa_relab_unfiltered.parquet"
if not rel.exists():
    rel = work / "taxa_relab.parquet"
    sys.stderr.write("WARNING: unfiltered taxonomy matrix absent; ranking "
                     "within the pooled-filtered set\n")
d = pd.read_parquet(rel)

# Prevalence must be computed on the SAME population the predictive analysis
# uses. Ranking over all life stages and then reporting the result as an adult
# analysis is a population mismatch: infant cohorts have a very different
# species distribution, so the two rankings differ.
st_p = work / "sample_table.parquet"
if st_p.exists() and stratum and stratum != "all":
    st = pd.read_parquet(st_p)
    col = "age_category" if "age_category" in st.columns else None
    if col is not None:
        keep_s = set(st.loc[st[col].astype(str) == stratum, "sample"].astype(str))
        before = len(d)
        d = d.loc[d.index.astype(str).isin(keep_s)]
        sys.stderr.write(f"restricted to stratum '{stratum}': {len(d)} of "
                         f"{before} samples\n")
    else:
        sys.stderr.write("WARNING: no age_category column; prevalence computed "
                         "on the full aligned cohort, which does not match an "
                         "adult-only analysis\n")
# Prevalence above a detection floor - a property of the data alone. No
# attribution, no model output, nothing downstream of prediction.
prev = (d > 0.0001).mean(axis=0).sort_values(ascending=False)
sys.stderr.write(f"ranking {len(prev)} clades by prevalence\n")


def to_taxon(name: str) -> str:
    """MetaPhlAn clade label -> NCBI taxon name.

    The rank prefix must be stripped FIRST. Replacing underscores before
    removing "s__" turns it into "s  ", and a later attempt to restore "sp."
    then produces "s. Blautia wexlerae", which matches nothing in NCBI.
    """
    x = str(name)
    for pre in ("s__", "t__", "g__"):
        if x.startswith(pre):
            x = x[len(pre):]
    x = x.replace("_", " ").strip()
    # restore the abbreviation in unnamed species, e.g. "Blautia sp MCC283"
    x = re.sub(r"\bsp\b(?=\s)", "sp.", x)
    return " ".join(x.split())


# Distinct clade labels can normalise to the same NCBI taxon name - a double
# underscore, trailing whitespace or a "_group" suffix all collapse. Emitting
# both wastes a slot: the species is downloaded and annotated twice, but the
# aggregation groups by name, so the evidence matrix gains nothing and the
# requested count silently overstates the species examined. Deduplicate after
# normalisation and top up from the ranking to reach n.
seen, out = set(), []
for s in prev.index:
    if any(t in str(s).upper() for t in ("GGB", "SGB", "_BACTERIUM", "CAG_")):
        continue
    t = to_taxon(s)
    if t and t not in seen:
        seen.add(t)
        out.append(t)
    if len(out) >= n:
        break
if len(out) < n:
    sys.stderr.write(f"WARNING: only {len(out)} unique named species available, "
                     f"{n} requested\n")
sys.stderr.write(f"{len(out)} unique taxa after name normalisation\n")
for t in out:
    print(t)
PY

mapfile -t SPECIES < "$OUTDIR/species.txt"
TOTAL=${#SPECIES[@]}
echo "=========================================================="
echo "Stratum for prevalence ranking        : $STRATUM"
echo "Species (by prevalence, model-independent): $TOTAL"
echo "Assemblies each                           : up to $N_PER"
echo "Output                                    : $EVIDENCE"
echo "=========================================================="

# A name-mangling fault silently yields "no assemblies" for every species and
# wastes the whole run. Probe the first few before committing to all of them.
probe_ok=0
for sp in "${SPECIES[@]:0:5}"; do
  # Captured, not piped: with `set -o pipefail`, head closing the pipe kills
  # datasets with SIGPIPE and the pipeline reports failure even when the query
  # succeeded and the name was fine.
  probe_out=$(datasets summary genome taxon "$sp" --assembly-source RefSeq \
              --as-json-lines 2>/dev/null | head -c 2000) || true
  if [[ "$probe_out" == *'"accession"'* ]]; then
    probe_ok=$((probe_ok + 1))
  fi
done
if [[ $probe_ok -eq 0 ]]; then
  echo "ABORTING: none of the first five species resolved against NCBI."
  echo "The species names are probably mangled. First five were:"
  printf '  %s\n' "${SPECIES[@]:0:5}"
  exit 1
fi
echo "Name check: $probe_ok of the first 5 species resolved. Proceeding."
echo

i=0
for sp in "${SPECIES[@]}"; do
  i=$((i + 1))
  slug=$(echo "$sp" | tr ' /' '__')
  done_marker="$OUTDIR/amr/${slug}.done"
  [[ -f "$done_marker" ]] && { printf '[%3d/%3d] %-42s done\n' "$i" "$TOTAL" "$sp"; continue; }
  printf '[%3d/%3d] %-42s ' "$i" "$TOTAL" "$sp"

  datasets summary genome taxon "$sp" --assembly-source RefSeq --as-json-lines \
    > "$OUTDIR/tmp/${slug}.jsonl" 2>/dev/null
  python - "$OUTDIR/tmp/${slug}.jsonl" "$N_PER" > "$OUTDIR/tmp/${slug}.acc" <<'PY'
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

  n_acc=$(wc -l < "$OUTDIR/tmp/${slug}.acc" 2>/dev/null || echo 0)
  if [[ "$n_acc" -eq 0 ]]; then
    echo "no assemblies"; touch "$done_marker"; continue
  fi

  zip="$OUTDIR/tmp/${slug}.zip"; rm -f "$zip"
  # shellcheck disable=SC2046
  if ! datasets download genome accession $(tr '\n' ' ' < "$OUTDIR/tmp/${slug}.acc") \
       --include genome --filename "$zip" >/dev/null 2>&1 \
     || ! unzip -t "$zip" >/dev/null 2>&1; then
    echo "download failed"; rm -f "$zip"; touch "$done_marker"; continue
  fi
  rm -rf "$OUTDIR/tmp/${slug}"; mkdir -p "$OUTDIR/tmp/${slug}"
  unzip -qo "$zip" -d "$OUTDIR/tmp/${slug}" 2>/dev/null

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

# ---- aggregate, INCLUDING explicit zeros ---------------------------------
echo
echo "Aggregating to $EVIDENCE ..."
python - "$OUTDIR/amr" "$EVIDENCE" "$WORK" <<'PY'
import sys, glob, os, csv, re, collections, pathlib
import pandas as pd

amr_dir, out, work = sys.argv[1], sys.argv[2], pathlib.Path(sys.argv[3])
per_species = collections.defaultdict(set)
hits = collections.defaultdict(set)

for f in sorted(glob.glob(os.path.join(amr_dir, "*.tsv"))):
    m = re.match(r"^(.*)__(\d+)$", os.path.basename(f)[:-4])
    if not m:
        continue
    sp, gid = m.group(1).replace("_", " "), m.group(2)
    per_species[sp].add(gid)
    with open(f, newline="") as fh:
        rd = csv.DictReader(fh, delimiter="\t")
        gcol = next((c for c in (rd.fieldnames or [])
                     if c.strip().lower() in ("element symbol", "gene symbol", "gene")), None)
        if gcol:
            for r in rd:
                g = (r.get(gcol) or "").strip()
                if g:
                    hits[(sp, g)].add(gid)

# Every gene seen anywhere becomes a candidate for every examined species, so
# that a species examined and found NOT to carry a gene contributes a zero
# rather than being silently absent. Without this the evidence is conditioned
# on detection and the frequency distribution is truncated.
all_genes = sorted({g for _, g in hits})
with open(out, "w", newline="") as fh:
    w = csv.writer(fh, delimiter="\t")
    w.writerow(["species", "gene", "n_genomes_with", "n_genomes_total",
                "frequency", "detected"])
    for sp, gids in sorted(per_species.items()):
        tot = len(gids)
        if not tot:
            continue
        for g in all_genes:
            k = len(hits.get((sp, g), ()))
            w.writerow([sp, g, k, tot, round(k / tot, 4), int(k > 0)])

n_sp, n_gen = len(per_species), sum(len(v) for v in per_species.values())
n_det = sum(1 for v in hits.values() if v)
# Attrition must be reported at every step. Species requested, species with
# retrievable assemblies, and species surviving annotation are three different
# numbers, and quoting the wrong one in a manuscript is a reproducibility
# failure a reader can detect from the matrix dimensions alone.
req = len([l for l in open(pathlib.Path(amr_dir).parent / "species.txt")
           if l.strip()]) if (pathlib.Path(amr_dir).parent / "species.txt").exists() else -1
print(f"species requested                 : {req}")
print(f"species with annotated assemblies : {n_sp}")
print(f"assemblies annotated              : {n_gen}")
print(f"distinct gene families detected   : {len(all_genes)}")
print(f"matrix                            : {n_sp} x {len(all_genes)} = "
      f"{n_sp * len(all_genes)} combinations")
print(f"detected pairs                    : {n_det}")
print(f"explicit zeros                    : {n_sp * len(all_genes) - n_det}")
PY

echo
echo "Next:"
echo "  python src/s13_mobility_predictability.py \\"
echo "      --pangenome $EVIDENCE --detected-only-frequency false"
echo
echo "This evidence set was built without reference to model attributions and"
echo "records explicit zeros, so the genome-boundedness relationship can be"
echo "re-tested free of the selection and detection conditioning that applied"
echo "to the original set."
