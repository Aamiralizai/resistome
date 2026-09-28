"""
s06d_pangenome_status.py
========================
Records what the pangenome retrieval actually attempted, and why species
dropped out.

The supplementary table describing pangenome validation should list the
species that were SELECTED and their fate, not the full set of modelled
species. Reporting only the successes hides a non-random exclusion: species
without RefSeq assemblies are disproportionately the poorly characterised
ones, which is precisely where a novel host association would be most
interesting.

Usage:
    python src/s06d_pangenome_status.py --pangenome-dir /mnt/x/w1_resistome/pangenome
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from common import LOG, load_config, table_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pangenome-dir", required=True)
    args = ap.parse_args()
    cfg = load_config()
    d = Path(args.pangenome_dir)

    sp_file = d / "species.txt"
    if not sp_file.exists():
        raise SystemExit(f"{sp_file} not found - run s06c first.")
    wanted = [l.strip() for l in sp_file.read_text().splitlines() if l.strip()]

    n_assemblies: Counter = Counter()
    n_calls: Counter = Counter()
    for f in sorted(glob.glob(str(d / "amr" / "*.tsv"))):
        m = re.match(r"^(.*)__(\d+)$", os.path.basename(f)[:-4])
        if not m:
            continue
        sp = m.group(1).replace("_", " ")
        n_assemblies[sp] += 1
        with open(f, newline="") as fh:
            rd = csv.DictReader(fh, delimiter="\t")
            gcol = next((c for c in (rd.fieldnames or [])
                         if c.strip().lower() in ("element symbol", "gene symbol",
                                                  "gene")), None)
            if gcol:
                n_calls[sp] += sum(1 for r in rd if (r.get(gcol) or "").strip())

    rows = []
    for rank, sp in enumerate(wanted, start=1):
        n = n_assemblies.get(sp, 0)
        rows.append({
            "attribution_rank": rank,
            "species": sp,
            "assemblies_requested": 12,
            "assemblies_annotated": n,
            "arg_calls_total": n_calls.get(sp, 0),
            "included_in_evidence": "yes" if n else "no",
            "exclusion_reason": "" if n else "no RefSeq assemblies retrieved",
        })
    df = pd.DataFrame(rows)
    out = table_path(cfg, "pangenome_species_status.csv", per_stratum=False)
    df.to_csv(out, index=False)

    inc = int((df["included_in_evidence"] == "yes").sum())
    LOG.info("Species selected by attribution rank : %d", len(df))
    LOG.info("Species with usable assemblies       : %d", inc)
    LOG.info("Species excluded, no assemblies      : %d", len(df) - inc)
    LOG.info("Median assemblies per included species: %.0f",
             df.loc[df["assemblies_annotated"] > 0, "assemblies_annotated"].median())
    LOG.info("Wrote %s", out)
    LOG.info("Report the exclusions explicitly: species without reference "
             "assemblies are disproportionately the poorly characterised taxa, "
             "so this is not a random subset.")


if __name__ == "__main__":
    main()
