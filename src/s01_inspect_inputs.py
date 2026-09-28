"""
s01_inspect_inputs.py
=====================
Run this FIRST, before anything else.

It opens every downloaded file, prints the schema, counts rows where cheap,
and writes:
    work_dir/inspection_report.txt   - human readable, send this to review
    work_dir/detected_columns.yaml   - the auto-detected column mapping

Nothing downstream will work until this stage runs clean. If a role fails to
auto-detect, copy the correct column name into config.yaml under `columns:`.

Usage:  python src/s01_inspect_inputs.py
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import yaml

from common import (LOG, detect_column, h5_keys, load_config, read_h5_head,
                    read_table, work_path)

ROLES_BY_TABLE = {
    "arg":          {"run": True, "gene": True, "count": True},
    "rrna":         {"run": True, "count": True, "taxon": False},
    "zmeta":        {"run": True, "sample": False, "project": False,
                     "date": False, "country": False, "host": False},
    "metalog_meta": {"sample": True, "study": False, "country": False,
                     "age": False, "sex": False, "disease": False, "bodysite": False},
    "mapping":      {"metalog_sample": True, "ena_run": False, "ena_sample": False},
}

FILE_FOR_TABLE = {
    "arg":          ("arg_h5", "h5"),
    "rrna":         ("rrna_h5", "h5"),
    "zmeta":        ("metadata_h5", "h5"),
    "metalog_meta": ("metalog_meta", "tsv"),
    "mapping":      ("metalog_mapping", "tsv"),
}

EXTRA_TSVS = ["metalog_mpa4", "metalog_reads", "metalog_entero",
              "metalog_load", "metalog_lineage"]


def describe(buf: io.StringIO, title: str, df: pd.DataFrame, path: Path) -> None:
    buf.write("\n" + "=" * 78 + f"\n{title}\n{path}\n" + "=" * 78 + "\n")
    buf.write(f"shape of preview: {df.shape}\n")
    buf.write(f"columns ({len(df.columns)}):\n")
    for c in df.columns:
        dt = df[c].dtype
        sample_vals = df[c].dropna().astype(str).head(3).tolist()
        buf.write(f"  - {c:<38} {str(dt):<10} e.g. {sample_vals}\n")
    buf.write("\nhead:\n")
    buf.write(df.head(5).to_string(max_cols=12, max_colwidth=28) + "\n")


def main() -> None:
    cfg = load_config()
    paths = cfg["paths"]
    buf = io.StringIO()
    detected: dict[str, dict] = {}

    buf.write("W1 RESISTOME PIPELINE - INPUT INSPECTION REPORT\n")
    buf.write(f"generated from config paths under: {paths['work_dir']}\n")

    # ---- HDF5 and primary TSVs ------------------------------------------
    for table, (path_key, kind) in FILE_FOR_TABLE.items():
        p = Path(paths[path_key])
        if not p.exists():
            buf.write(f"\n!! MISSING: {table} -> {p}\n")
            LOG.warning("Missing file for %s: %s", table, p)
            continue
        if kind == "h5":
            buf.write(f"\nHDF5 keys in {p.name}: {h5_keys(p)[:8]}"
                      f"{' ...' if len(h5_keys(p)) > 8 else ''} "
                      f"(n={len(h5_keys(p))})\n")
            head = read_h5_head(p, n=200)
        else:
            head = read_table(p, nrows=200)
        describe(buf, f"TABLE: {table}", head, p)

        got = {}
        for role, required in ROLES_BY_TABLE[table].items():
            try:
                got[role] = detect_column(head.columns, role, None, required=False)
            except KeyError:
                got[role] = None
            if got[role] is None and required:
                buf.write(f"  ** REQUIRED role '{role}' NOT DETECTED - set it in config.yaml\n")
        detected[table] = got
        buf.write(f"\nauto-detected roles: {got}\n")

    # ---- Secondary Metalog tables ---------------------------------------
    for key in EXTRA_TSVS:
        p = Path(paths[key])
        if not p.exists():
            buf.write(f"\n!! MISSING: {key} -> {p}\n")
            continue
        head = read_table(p, nrows=50)
        describe(buf, f"TABLE: {key}", head, p)
        # profile tables are usually wide (samples as columns) - flag orientation
        if head.shape[1] > 200:
            buf.write("\nNOTE: >200 columns - this file is probably samples-as-columns.\n"
                      "      s03 will transpose it. Confirm the first column holds clade names.\n")

    out_txt = work_path(cfg, "inspection_report.txt")
    out_txt.write_text(buf.getvalue(), encoding="utf-8")
    out_yaml = work_path(cfg, "detected_columns.yaml")
    out_yaml.write_text(yaml.safe_dump({"columns": detected}, sort_keys=False), encoding="utf-8")

    print(buf.getvalue())
    LOG.info("Wrote %s", out_txt)
    LOG.info("Wrote %s", out_yaml)
    LOG.info("Review the report, fix any '** REQUIRED role NOT DETECTED' lines "
             "in config.yaml, then run s02_build_sample_table.py")


if __name__ == "__main__":
    main()
