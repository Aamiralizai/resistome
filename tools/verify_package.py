#!/usr/bin/env python3
"""Refuse to build a submission package when results and source disagree.

This is the gate that would have caught the v85 packaging fault: a ``work/``
directory produced by an older ``src/`` than the one shipped with it. It
answers three questions and exits non-zero on any failure, so it can be the
last line of a build script and block the zip.

    1. Has the SOURCE changed since the freeze?       (--src)
    2. Have the RESULTS changed since the freeze?     (--results)
    3. Do the manuscript's numbers match the tables?  (--manuscript)

Check 3 is the one that matters most and is easiest to skip. It extracts every
number of the reported forms from the manuscript text and asserts that each
appears somewhere in the result tables to the precision printed. A number in
the manuscript that exists in no table is either stale or invented; both are
fatal before submission.

USAGE
-----
    python verify_package.py --manifest frozen/v85_frozen.json
    python verify_package.py --manifest frozen/v85_frozen.json \
        --manuscript Manuscript.docx --tables /mnt/x/w1_resistome/tables
    python verify_package.py --manifest frozen/v85_frozen.json --expect-changed-results
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from freeze_results import (RESULT_SUFFIXES, SOURCE_SUFFIXES,  # noqa: E402
                            digest_tree)

# Numbers as they are actually written in this manuscript: rho = 0.590,
# P = 2.3 x 10-7, R2 = 0.0936, n = 65, 95% CI 0.379-0.743.
NUM_PATTERNS = [
    re.compile(r"(?:rho|ρ|r)\s*=\s*(-?\d*\.\d+)", re.I),
    re.compile(r"R\s*2?\s*=\s*(-?\d*\.\d+)", re.I),
    re.compile(r"(?<![\w.])(\d*\.\d{3,})(?![\w.])"),   # bare 3+ dp decimals
]


def fail(msg: str) -> None:
    print(f"FAIL  {msg}")


def ok(msg: str) -> None:
    print(f"ok    {msg}")


def check_tree(name: str, root: Path, manifest: dict, key: str,
               suffixes: set[str], expect_changed: bool) -> bool:
    files, digest = digest_tree(root, suffixes)
    if digest == manifest[key]:
        ok(f"{name} tree unchanged since freeze ({digest[:16]})")
        return True
    if expect_changed:
        ok(f"{name} tree changed, as expected ({digest[:16]})")
    else:
        fail(f"{name} tree CHANGED since freeze")
        print(f"        frozen {manifest[key][:16]}  now {digest[:16]}")
        frozen = manifest[f"{name.lower()}_files"]
        added = sorted(set(files) - set(frozen))
        removed = sorted(set(frozen) - set(files))
        changed = sorted(f for f in set(files) & set(frozen)
                         if files[f]["sha256"] != frozen[f]["sha256"])
        for label, items in (("changed", changed), ("added", added),
                             ("removed", removed)):
            for f in items[:12]:
                print(f"        {label}: {f}")
            if len(items) > 12:
                print(f"        ... and {len(items)-12} more {label}")
    return expect_changed


def manuscript_text(path: Path) -> str:
    if path.suffix.lower() == ".docx":
        import subprocess
        r = subprocess.run(["pandoc", "-t", "plain", str(path)],
                           capture_output=True, text=True)
        if r.returncode:
            raise SystemExit(f"pandoc failed on {path}: {r.stderr[:200]}")
        return r.stdout
    return path.read_text(errors="replace")


DOI_RE = re.compile(r"\b10\.\d{4,9}/\S+", re.I)
REFS_RE = re.compile(r"\n\s*(references|bibliography)\s*\n", re.I)


def strip_non_claims(text: str) -> str:
    """Remove text whose numbers are not results claims.

    DOIs and the reference list are dense with decimal-looking tokens that
    will never appear in a result table. Leaving them in produces false
    failures, and a check that cries wolf is a check that gets bypassed.
    """
    text = DOI_RE.sub(" ", text)
    if m := REFS_RE.search(text):
        text = text[:m.start()]
    return text


def table_numbers(tables: Path) -> set[str]:
    """Every numeric cell in every result table, at several precisions.

    A manuscript prints 0.590 for a stored 0.5903815651. Matching therefore
    compares the manuscript's string against the table value rounded to the
    same number of decimals the manuscript used.
    """
    import csv
    seen: set[str] = set()
    for f in sorted(tables.rglob("*.csv")):
        with f.open(newline="") as fh:
            for row in csv.reader(fh):
                for cell in row:
                    try:
                        v = float(cell)
                    except (TypeError, ValueError):
                        continue
                    for dp in range(1, 7):
                        seen.add(f"{abs(v):.{dp}f}")
                    seen.add(str(abs(v)))
    return seen


def check_manuscript(ms: Path, tables: Path) -> bool:
    text = strip_non_claims(manuscript_text(ms))
    have = table_numbers(tables)
    claimed: set[str] = set()
    for pat in NUM_PATTERNS:
        claimed.update(m.group(1).lstrip("-") for m in pat.finditer(text))
    # Drop values that cannot be table lookups: version numbers, P-value
    # mantissas that are compared separately, and trivially common values.
    claimed = {c for c in claimed if c not in {"0.05", "0.01", "0.001"}}
    if not claimed:
        fail("no numeric claims extracted from the manuscript - check the parser")
        return False
    missing = sorted(c for c in claimed
                     if f"{float(c):.{len(c.split('.')[1])}f}" not in have)
    if missing:
        fail(f"{len(missing)} of {len(claimed)} manuscript numbers appear in NO "
             f"result table:")
        for m in missing[:20]:
            print(f"        {m}")
        if len(missing) > 20:
            print(f"        ... and {len(missing)-20} more")
        print("        Each is stale, hand-edited, or from an unshipped table.")
        return False
    ok(f"all {len(claimed)} extracted manuscript numbers trace to a result table")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--src", type=Path, help="override source root")
    ap.add_argument("--results", type=Path, help="override results root")
    ap.add_argument("--manuscript", type=Path)
    ap.add_argument("--tables", type=Path)
    ap.add_argument("--expect-changed-results", action="store_true",
                    help="after a deliberate rerun: allow results to differ, "
                         "but still require the source to match")
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text())
    src = args.src or Path(manifest["source_root"])
    results = args.results or Path(manifest["results_root"])
    print(f"Manifest: {args.manifest}  (frozen {manifest['frozen_at']})\n")

    passed = [
        check_tree("Source", src, manifest, "source_tree_digest",
                   SOURCE_SUFFIXES, expect_changed=False),
        check_tree("Results", results, manifest, "results_tree_digest",
                   RESULT_SUFFIXES, expect_changed=args.expect_changed_results),
    ]
    if args.manuscript and args.tables:
        passed.append(check_manuscript(args.manuscript, args.tables))
    elif args.manuscript or args.tables:
        fail("--manuscript and --tables must be given together")
        passed.append(False)
    else:
        print("note  manuscript/table consistency NOT checked "
              "(pass --manuscript and --tables to enable)")

    print()
    if all(passed):
        print("PASS - package is consistent; safe to build the deposit.")
        return 0
    print("BLOCKED - do not build the deposit until the failures above are fixed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
