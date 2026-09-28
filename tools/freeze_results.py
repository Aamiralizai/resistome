#!/usr/bin/env python3
"""Freeze a results set and bind it to the source tree that produced it.

WHY THIS EXISTS
---------------
The v85 submission package shipped a ``work/`` directory produced by an
EARLIER source tree than the ``src/`` directory shipped alongside it. The
results in it reported a held-out XGBoost median rho of 0.303 over 87 genes
with five model families; the manuscript reported 0.288 over 95 genes with
six. Both were internally valid. Neither matched the other, and nothing in
the pipeline noticed.

A copied folder does not prevent this, because a copy records no claim about
which code made it. This tool records that claim: it hashes every result file
AND every source file, and writes both into one manifest. ``verify_package.py``
later re-derives both and refuses to build a deposit when they disagree.

USAGE
-----
    python freeze_results.py --results /mnt/x/w1_resistome \
                             --src     /path/to/w1_resistome_pipeline/src \
                             --label   v85_frozen_pre_subject_rerun

    # later, before building the deposit:
    python verify_package.py --manifest frozen/v85_frozen_pre_subject_rerun.json

The freeze is a manifest plus, with --copy, a physical snapshot. The manifest
alone is enough to DETECT drift; --copy is what lets you go back afterwards.
Freeze before starting new analysis, not after.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Result artefacts worth hashing. Model weights (.pt) are included because a
# rerun that changes them changes predictions; figures are included because a
# stale figure in a submission is as damaging as a stale number.
RESULT_SUFFIXES = {".parquet", ".csv", ".tsv", ".json", ".yaml", ".yml",
                   ".pt", ".png", ".pdf", ".svg", ".xlsx", ".docx"}
SOURCE_SUFFIXES = {".py", ".sh", ".yaml", ".yml", ".txt"}

# Directories never worth hashing: caches, smoke-test scratch, VCS internals.
SKIP_DIRS = {"__pycache__", ".git", ".ipynb_checkpoints", "_smoke",
             ".run_markers", "node_modules"}


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def walk(root: Path, suffixes: set[str]) -> list[Path]:
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            p = Path(dirpath) / fn
            if p.suffix.lower() in suffixes:
                out.append(p)
    return sorted(out)


def digest_tree(root: Path, suffixes: set[str]) -> tuple[dict, str]:
    """Hash every file, then hash the sorted (relpath, hash) list.

    The tree digest is what makes drift detectable in one comparison: any
    edit to any source file changes it, so the manifest carries a single
    value that answers 'is this the same code?'.
    """
    files = {}
    for p in walk(root, suffixes):
        rel = str(p.relative_to(root))
        files[rel] = {"sha256": sha256(p), "bytes": p.stat().st_size,
                      "mtime": datetime.fromtimestamp(
                          p.stat().st_mtime, timezone.utc).isoformat()}
    joined = "\n".join(f"{k}:{v['sha256']}" for k, v in sorted(files.items()))
    return files, hashlib.sha256(joined.encode()).hexdigest()


def git_state(src: Path) -> dict:
    """Record the commit if there is one. Absence is recorded, not fatal."""
    def run(*args):
        try:
            return subprocess.run(args, cwd=src, capture_output=True,
                                  text=True, timeout=10).stdout.strip() or None
        except Exception:
            return None
    commit = run("git", "rev-parse", "HEAD")
    if commit is None:
        return {"available": False,
                "note": "no git repository; source tree digest is the only provenance"}
    return {"available": True, "commit": commit,
            "branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(run("git", "status", "--porcelain"))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True, type=Path,
                    help="results root (the directory holding tables/, work/, figures/)")
    ap.add_argument("--src", required=True, type=Path,
                    help="source tree that produced these results")
    ap.add_argument("--label", required=True,
                    help="name for this freeze, e.g. v85_frozen_pre_subject_rerun")
    ap.add_argument("--out", type=Path, default=Path("frozen"),
                    help="where manifests (and snapshots) are written")
    ap.add_argument("--copy", action="store_true",
                    help="also copy the result files, so the freeze is restorable")
    ap.add_argument("--note", default="",
                    help="free text: why this freeze was taken")
    args = ap.parse_args()

    for p in (args.results, args.src):
        if not p.is_dir():
            print(f"ERROR: not a directory: {p}", file=sys.stderr)
            return 2

    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / f"{args.label}.json"
    if manifest_path.exists():
        print(f"ERROR: {manifest_path} already exists. A freeze is immutable by "
              f"design; choose a new --label rather than overwriting one.",
              file=sys.stderr)
        return 2

    print(f"Hashing results under {args.results} ...")
    res_files, res_digest = digest_tree(args.results, RESULT_SUFFIXES)
    print(f"  {len(res_files)} result files, tree digest {res_digest[:16]}")

    print(f"Hashing source under {args.src} ...")
    src_files, src_digest = digest_tree(args.src, SOURCE_SUFFIXES)
    print(f"  {len(src_files)} source files, tree digest {src_digest[:16]}")

    manifest = {
        "label": args.label,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "note": args.note,
        "results_root": str(args.results.resolve()),
        "source_root": str(args.src.resolve()),
        "results_tree_digest": res_digest,
        "source_tree_digest": src_digest,
        "git": git_state(args.src),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "results_files": res_files,
        "source_files": src_files,
    }

    if args.copy:
        snap = args.out / args.label
        if snap.exists():
            print(f"ERROR: snapshot dir {snap} exists", file=sys.stderr)
            return 2
        print(f"Copying results to {snap} ...")
        total = 0
        for rel in res_files:
            s, d = args.results / rel, snap / rel
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(s, d)
            total += res_files[rel]["bytes"]
        manifest["snapshot_dir"] = str(snap.resolve())
        manifest["snapshot_bytes"] = total
        print(f"  copied {total/1e6:.1f} MB")

    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nFrozen: {manifest_path}")
    print(f"  results digest {res_digest}")
    print(f"  source  digest {src_digest}")
    print("\nThis pair is now the reference. Before building any deposit, run:")
    print(f"  python verify_package.py --manifest {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
