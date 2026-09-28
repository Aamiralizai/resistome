#!/usr/bin/env python3
"""Exclude, rather than arbitrarily resolve, runs that map to more than one
Metalog sample alias.

Why
---
s02_build_sample_table.py bridges the two resources and then, for the 268 runs
(0.4% of 70,937) that reach more than one sample alias, keeps whichever row
came first:

    run_map = run_map.drop_duplicates("ena_run")

That rule is deterministic and reproducible from the mapping table, but "the
first was retained" is not an argument, and a reviewer is entitled to ask what
happens if those runs are dropped instead. This patch replaces the rule with an
exclusion so the question can be answered with a number.

Usage
-----
    python tools/patch_ambiguous_runs.py --report      # count them, change nothing
    python tools/patch_ambiguous_runs.py --apply       # rewrite s02 to exclude
    python tools/patch_ambiguous_runs.py --revert      # put the original rule back

--report writes work/ambiguous_runs.tsv listing every affected run with the
aliases it reached, which is what you cite in the SI if you decide the exclusion
run is not worth doing.

After --apply, rerun from s02. s02 and s03 are cheap; s04 is the analysis that
the variance-partition headline depends on:

    python src/s02_build_sample_table.py
    python src/s03_build_matrices.py
    python src/s04_variance_partition.py

Keep the current tables first -- s04 overwrites in place.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "s02_build_sample_table.py"

ORIGINAL = '''        note(f"[2] WARNING: {n:,} runs map to >1 Metalog sample; keeping first only.")
        run_map = run_map.drop_duplicates("ena_run")'''

PATCHED = '''        note(f"[2] {n:,} runs map to >1 Metalog sample; excluding them entirely.")
        ambiguous = dup[dup > 1].index
        Path(cfg["paths"]["work_dir"]).mkdir(parents=True, exist_ok=True)
        (run_map[run_map["ena_run"].isin(ambiguous)]
         .sort_values(["ena_run", "sample_alias"])
         .to_csv(Path(cfg["paths"]["work_dir"]) / "ambiguous_runs.tsv",
                 sep="\\t", index=False))
        run_map = run_map[~run_map["ena_run"].isin(ambiguous)]'''


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--report", action="store_true")
    g.add_argument("--apply", action="store_true")
    g.add_argument("--revert", action="store_true")
    a = ap.parse_args()

    if not SRC.exists():
        print(f"not found: {SRC}", file=sys.stderr)
        return 1
    s = SRC.read_text(encoding="utf-8")
    bak = SRC.with_suffix(".py.orig")

    if a.revert:
        if not bak.exists():
            print("no .orig backup to revert to", file=sys.stderr)
            return 1
        shutil.copy2(bak, SRC)
        print(f"reverted {SRC.name} from {bak.name}")
        return 0

    if a.report:
        print("The rule currently in force:\n")
        print(ORIGINAL if ORIGINAL in s else "  (not found -- already patched?)")
        print("\nRun with --apply to exclude these runs instead, then rerun "
              "s02, s03 and s04.")
        return 0

    if PATCHED in s:
        print("already patched")
        return 0
    if ORIGINAL not in s:
        print("the expected block was not found in s02_build_sample_table.py; "
              "it may have been edited. Patch it by hand.", file=sys.stderr)
        return 1

    if not bak.exists():
        shutil.copy2(SRC, bak)
        print(f"backed up -> {bak.name}")
    SRC.write_text(s.replace(ORIGINAL, PATCHED), encoding="utf-8")
    print(f"patched {SRC.name}: ambiguous runs are now excluded and listed in "
          f"work/ambiguous_runs.tsv")
    print("\nNow run:\n  python src/s02_build_sample_table.py"
          "\n  python src/s03_build_matrices.py"
          "\n  python src/s04_variance_partition.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
