"""
s00b_discover_files.py
======================
Scans your Zenodo and Metalog download folders, matches every file to the role
the pipeline expects, and writes `config.local.yaml` with the real paths
filled in. All later stages read config.local.yaml if it exists, otherwise
config.yaml.

You only have to set two things by hand: `zenodo_dir` and `metalog_dir`.

Metalog filenames vary depending on which buttons you clicked and when, so
matching is done on filename patterns rather than exact names. Anything the
script cannot place is listed as UNMATCHED at the end — check that list, it is
usually a file you downloaded in the wrong format (long instead of wide, or
mOTUs instead of MetaPhlAn).

Usage:
    python src/s00b_discover_files.py
    python src/s00b_discover_files.py --zenodo /mnt/x/w1_resistome/zenodo \
                                      --metalog /mnt/x/w1_resistome/Metalog
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

from common import LOG, load_config

# role -> (ordered regex patterns, description). First match wins.
ZENODO_PATTERNS = {
    "arg_h5":         ([r"^arg\.h5$", r"^arg\.tsv$"], "ARG alignment counts"),
    "rrna_h5":        ([r"^rrna\.h5$", r"^rrna\.tsv$"], "rRNA counts (normalisation denominator)"),
    "metadata_h5":    ([r"^metadata\.h5$", r"^metadata\.tsv$"], "run-level metadata"),
    "diversity_h5":   ([r"^diversity\.h5$", r"^diversity\.tsv$"], "precomputed diversity"),
    "resfinder_anno": ([r"^resfinder_anno\.h5$", r"^resfinder_anno\.tsv$"], "ARG annotation"),
}

METALOG_PATTERNS = {
    # wide human metadata, harmonised preferred over core
    # NOTE: prefer the harmonised export over core - harmonisation is what
    # standardises sex, diet, smoker and subject_disease_status across studies.
    "metalog_meta":    ([r"human.*harmon.*wide", r"human.*wide.*harmon",
                         r"human.*wide", r"wide.*human",
                         r"human.*harmon", r"human.*metadata"],
                        "human metadata, WIDE format"),
    "metalog_mapping": ([r"mapping", r"accession", r"sample.*map"],
                        "Metalog id <-> ENA accession mapping"),
    "metalog_mpa4":    ([r"human.*mpa4?.*spec", r"human.*metaphlan.*spec",
                         r"human.*mpa4", r"human.*metaphlan"],
                        "MetaPhlAn 4 species profiles"),
    "metalog_lineage": ([r"metaphlan4_clades", r"clades?\.tsv", r"lineage",
                         r"clade.*ncbi", r"ncbi.*tax", r"clade.*name"],
                        "clade name -> lineage / taxid map"),
    "metalog_reads":   ([r"human.*read.*count", r"read.*count.*human", r"readcount"],
                        "post-QC read counts"),
    "metalog_entero":  ([r"entero"], "enterotype predictions"),
    "metalog_load":    ([r"mlp", r"microbial.*load", r"load.*pred"],
                        "microbial load predictions"),
}

BAD_HINTS = {
    "long format instead of wide": r"long",
    "mOTUs instead of MetaPhlAn": r"motus",
    "non-human habitat": r"^(animal|ocean|environmental)",
}


def scan(directory: Path) -> list[Path]:
    if not directory.exists():
        raise SystemExit(f"Directory does not exist: {directory}\n"
                         "If you are in WSL, remember X:\\ is /mnt/x/ and paths "
                         "are case-sensitive.")
    files = sorted(p for p in directory.rglob("*") if p.is_file())
    LOG.info("%s: %d files", directory, len(files))
    return files


def match(files: list[Path], patterns: dict) -> tuple[dict, set]:
    resolved, used = {}, set()
    for role, (pats, desc) in patterns.items():
        hit = None
        for pat in pats:
            for f in files:
                if f in used:
                    continue
                if re.search(pat, f.name.lower()):
                    hit = f
                    break
            if hit:
                break
        if hit:
            resolved[role] = hit
            used.add(hit)
            LOG.info("  %-16s -> %-46s (%s)", role, hit.name, desc)
        else:
            LOG.warning("  %-16s -> NOT FOUND  (%s)", role, desc)
    return resolved, used


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zenodo", default=None)
    ap.add_argument("--metalog", default=None)
    ap.add_argument("--out", default="config.local.yaml")
    args = ap.parse_args()

    cfg = load_config()
    zdir = Path(args.zenodo or cfg["paths"]["zenodo_dir"])
    mdir = Path(args.metalog or cfg["paths"]["metalog_dir"])

    LOG.info("Scanning Zenodo directory")
    zfiles = scan(zdir)
    zres, zused = match(zfiles, ZENODO_PATTERNS)

    LOG.info("Scanning Metalog directory")
    mfiles = scan(mdir)
    mres, mused = match(mfiles, METALOG_PATTERNS)

    _mm = mres.get("metalog_meta")
    _harmonised = _mm is not None and any(
        t in _mm.name.lower() for t in ("harmon", "extended"))
    if _mm is not None and not _harmonised:
        LOG.warning(
            "Matched the CORE metadata export (%s), not the harmonised one. "
            "Harmonisation is what standardises sex, diet, smoker and "
            "subject_disease_status across studies - your host covariate block "
            "will be much weaker without it. Re-download 'Including partially "
            "harmonized metadata' (wide, human) from the Metalog Downloads page.",
            mres["metalog_meta"].name)

    for role, p in {**zres, **mres}.items():
        cfg["paths"][role] = str(p).replace("\\", "/")
    cfg["paths"]["zenodo_dir"] = str(zdir).replace("\\", "/")
    cfg["paths"]["metalog_dir"] = str(mdir).replace("\\", "/")

    out = Path(args.out)
    out.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
                   encoding="utf-8")

    unmatched = [f for f in (zfiles + mfiles) if f not in (zused | mused)]
    print("\n" + "=" * 72)
    print(f"WROTE {out.resolve()}")
    print("=" * 72)

    missing = [r for r in {**ZENODO_PATTERNS, **METALOG_PATTERNS}
               if r not in {**zres, **mres}]
    critical = [r for r in missing
                if r in ("arg_h5", "rrna_h5", "metadata_h5", "resfinder_anno",
                         "metalog_meta", "metalog_mapping", "metalog_mpa4")]
    if critical:
        print("\nCRITICAL FILES MISSING - the pipeline cannot run without these:")
        for r in critical:
            desc = {**ZENODO_PATTERNS, **METALOG_PATTERNS}[r][1]
            print(f"  - {r:<16} {desc}")
    optional = [r for r in missing if r not in critical]
    if optional:
        print("\nOptional files missing (pipeline runs, some panels blank):")
        for r in optional:
            print(f"  - {r}")

    if unmatched:
        print(f"\nUNMATCHED FILES ({len(unmatched)}):")
        for f in unmatched:
            hints = [msg for msg, pat in BAD_HINTS.items()
                     if re.search(pat, f.name.lower())]
            note = f"   <- looks like {hints[0]}" if hints else ""
            print(f"  {f.name}{note}")
        print("\nIf one of these is actually a file the pipeline needs, set its "
              "path by hand under `paths:` in config.local.yaml.")

    if not critical:
        print("\nAll critical files located. Next:  python src/s01_inspect_inputs.py")


if __name__ == "__main__":
    main()
