#!/usr/bin/env bash
# =============================================================================
# s06b_fetch_genome_evidence.sh
#
# Builds the independent evidence table stage 6 needs: which ARGs are ACTUALLY
# present in reference genomes of the species in your dataset.
#
# The point of this file is not to confirm what the model found. It is to
# identify the associations the model found that genomes CANNOT explain -
# candidate horizontal transfer, mobile-element carriage, or unrecognised
# hosts. Those are the discovery half of the paper.
#
# Usage:
#     bash src/s06b_fetch_genome_evidence.sh [OUTDIR]
#
# Default OUTDIR: /mnt/x/w1_resistome/genomes
# Resumable: already-processed species are skipped, so it is safe to Ctrl-C
# and restart. Expect 2-4 hours for ~250 resolvable species.
# =============================================================================
set -uo pipefail

OUTDIR="${1:-/mnt/x/w1_resistome/genomes}"
SPECIES_CSV="${SPECIES_CSV:-/mnt/x/w1_resistome/tables/species_for_genome_check.csv}"
EVIDENCE="${OUTDIR}/amrfinder_all.tsv"
THREADS="${THREADS:-8}"

mkdir -p "$OUTDIR/fna" "$OUTDIR/amr" "$OUTDIR/tmp"

# ---------------------------------------------------------------- tooling ---
need() { command -v "$1" >/dev/null 2>&1; }

if ! need datasets || ! need amrfinder; then
  echo "Installing tools from bioconda (one-off, a few minutes)..."
  conda install -y -c conda-forge -c bioconda ncbi-datasets-cli ncbi-amrfinderplus \
    || { echo "conda install failed. Try: mamba install -c conda-forge -c bioconda ncbi-datasets-cli ncbi-amrfinderplus"; exit 1; }
fi

# AMRFinder needs its reference database before first use.
if ! amrfinder --database_version >/dev/null 2>&1; then
  echo "Fetching the AMRFinderPlus database (one-off, ~200 MB)..."
  if ! amrfinder -u; then
    echo
    echo "The AMRFinderPlus database did not download. Without it every genome"
    echo "will fail, so stopping here."
    echo "A TLS error (curl code 56) usually means a VPN or proxy is breaking"
    echo "HTTPS to ftp.ncbi.nlm.nih.gov - disable it and retry."
    exit 1
  fi
fi

if [[ ! -f "$SPECIES_CSV" ]]; then
  echo "Species list not found: $SPECIES_CSV"
  echo "Run 'python src/s06_attribution.py' first - it writes that file."
  exit 1
fi

# ------------------------------------------------------------- input prep ---
# MetaPhlAn labels are underscore-separated; NCBI wants spaces. Unnamed
# GGB/SGB clades have no NCBI counterpart and are skipped rather than queried.
mapfile -t SPECIES < <(
  tail -n +2 "$SPECIES_CSV" \
  | sed 's/^s__//; s/"//g' \
  | grep -viE '^(GGB|SGB)' \
  | grep -viE '_(SGB|GGB)[0-9]+$' \
  | tr '_' ' ' \
  | sed 's/  */ /g; s/^ *//; s/ *$//' \
  | sort -u
)

# Fail fast: if the first few all fail, the problem is the network or the
# tools, not genome availability, and 279 attempts will waste an hour.
consecutive_fail=0

TOTAL=${#SPECIES[@]}
echo "=========================================================="
echo "Species to resolve : $TOTAL"
echo "Output directory   : $OUTDIR"
echo "Evidence table     : $EVIDENCE"
echo "=========================================================="

# ------------------------------------------------------------- main loop ----
i=0; ok=0; nogenome=0; failed=0
for sp in "${SPECIES[@]}"; do
  i=$((i + 1))
  slug=$(echo "$sp" | tr ' /' '__')
  amr_out="$OUTDIR/amr/${slug}.tsv"
  fna="$OUTDIR/fna/${slug}.fna"

  if [[ -s "$amr_out" ]]; then
    ok=$((ok + 1)); continue                      # resume
  fi
  if [[ -f "$OUTDIR/amr/${slug}.none" ]]; then
    nogenome=$((nogenome + 1)); continue          # known to have no genome
  fi

  printf '[%4d/%4d] %-45s ' "$i" "$TOTAL" "$sp"

  if [[ ! -s "$fna" ]]; then
    zip="$OUTDIR/tmp/${slug}.zip"
    err="$OUTDIR/tmp/${slug}.err"
    rm -f "$zip"

    # Many gut commensals have no official RefSeq *reference* assembly, only a
    # representative or GenBank-only one. Try strict first, then relax.
    got=0
    for flags in "--reference" "--assembly-source RefSeq --assembly-level complete"; do
      if datasets download genome taxon "$sp" $flags --include genome \
           --filename "$zip" >"$err" 2>&1 && [[ -s "$zip" ]]; then
        got=1; break
      fi
    done

    # A failed download often still writes a small HTML or JSON error body,
    # which unzips to nothing. Validate the archive before trusting it.
    if [[ $got -eq 0 ]] || ! unzip -t "$zip" >/dev/null 2>&1; then
      sz=$( [[ -f "$zip" ]] && stat -c%s "$zip" || echo 0 )
      msg=$(head -c 200 "$err" 2>/dev/null | tr '\n' ' ')
      touch "$OUTDIR/amr/${slug}.none"; nogenome=$((nogenome + 1))
      echo "download failed (${sz} bytes): ${msg:0:90}"
      rm -f "$zip"
      consecutive_fail=$((consecutive_fail + 1))
      if [[ $consecutive_fail -ge 8 && $ok -eq 0 ]]; then
        echo
        echo "ABORTING: 8 consecutive failures and not one success."
        echo "That means the network or tooling is broken, not that these"
        echo "particular species lack genomes."
        echo "Check: (1) VPN or proxy interfering with TLS to NCBI,"
        echo "       (2) 'datasets download genome taxon \"Escherichia coli\" --include genome --filename /tmp/t.zip'"
        echo "Then remove $OUTDIR/amr/*.none before rerunning."
        exit 1
      fi
      continue
    fi

    rm -rf "$OUTDIR/tmp/${slug}"
    mkdir -p "$OUTDIR/tmp/${slug}"
    unzip -qo "$zip" -d "$OUTDIR/tmp/${slug}" 2>/dev/null
    found=$(find "$OUTDIR/tmp/${slug}" -name "*.fna" 2>/dev/null | head -1)
    if [[ -z "$found" ]]; then
      touch "$OUTDIR/amr/${slug}.none"; nogenome=$((nogenome + 1))
      echo "archive held no .fna"; continue
    fi
    mv "$found" "$fna"
    rm -rf "$OUTDIR/tmp/${slug}" "$zip" "$err"
  fi

  consecutive_fail=0
  if amrfinder -n "$fna" --plus --threads "$THREADS" -o "$amr_out" 2>"$OUTDIR/tmp/${slug}.amrerr"; then
    n=$(($(wc -l < "$amr_out") - 1))
    ok=$((ok + 1)); echo "${n} ARG hits"
  else
    failed=$((failed + 1)); rm -f "$amr_out"
    echo "amrfinder failed: $(head -c 120 "$OUTDIR/tmp/${slug}.amrerr" | tr '\n' ' ')"
    if [[ $failed -ge 5 && $ok -eq 0 ]]; then
      echo "ABORTING: amrfinder failing on every genome. Run 'amrfinder -u' "
      echo "successfully first - the database download failed earlier."
      exit 1
    fi
  fi
done

# ------------------------------------------------------------ concatenate ---
echo
echo "Building $EVIDENCE ..."
python - "$OUTDIR/amr" "$EVIDENCE" <<'PY'
import sys, glob, os, csv
amr_dir, out = sys.argv[1], sys.argv[2]
rows, files = [], sorted(glob.glob(os.path.join(amr_dir, "*.tsv")))
for f in files:
    species = os.path.basename(f)[:-4].replace("_", " ")
    with open(f, newline="") as fh:
        rd = csv.DictReader(fh, delimiter="\t")
        gcol = next((c for c in (rd.fieldnames or [])
                     if c.strip().lower() in ("element symbol", "gene symbol", "gene")), None)
        if gcol is None:
            continue
        for r in rd:
            g = (r.get(gcol) or "").strip()
            if g:
                rows.append((species, g))
rows = sorted(set(rows))
with open(out, "w", newline="") as fh:
    w = csv.writer(fh, delimiter="\t")
    w.writerow(["species", "gene"])
    w.writerows(rows)
print(f"{len(files)} genomes -> {len(rows)} unique species-gene pairs")
PY

echo
echo "=========================================================="
echo "genomes with ARG calls : $ok"
echo "  (species returning 0 ARG hits are a normal result for gut commensals"
echo "   and still count as successfully tested)"
echo "no reference genome    : $nogenome"
echo "amrfinder failures     : $failed"
echo "=========================================================="
echo
echo "Next:"
echo "  python src/s06_attribution.py --genome-evidence $EVIDENCE"
