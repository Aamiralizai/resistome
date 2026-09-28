"""
s02_build_sample_table.py
=========================
THE GO / NO-GO STAGE.

Builds the master sample table: every biological sample with BOTH a resistome
profile (Zenodo) AND microbiome data + curated metadata (Metalog).

Join semantics, as confirmed against the real schemas
-----------------------------------------------------
The Metalog mapping file is LONG: one row per external identifier, with a
`kind` column taking values 'run', 'sample' or 'experiment', and the identifier
itself in `external_id`. The canonical Metalog key is `sample_alias` (not
`sample_id`) - every Metalog profile table joins on sample_alias.

Zenodo carries BOTH run_accession and sample_accession, so two independent
bridges exist and both are used:
    kind == 'run'    : external_id  ->  Zenodo run_accession    (direct)
    kind == 'sample' : external_id  ->  Zenodo sample_accession ->  its runs
Using both maximises recall, because not every Metalog sample has a 'run' row.
'experiment' rows (SRX/ERX) have no counterpart in Zenodo and are dropped.

No upload-date filter is applied. The Zenodo resource only contains runs
deposited before 2020 in the first place, so the join itself enforces the
coverage window. The `collection_date` column is the SAMPLING date, not the
deposit date, and filtering on it would wrongly discard old samples that were
deposited late.

Outputs
-------
work_dir/sample_table.parquet   one row per usable biological sample
work_dir/run_map.parquet        sample_alias <-> ena_run, usable rows only
work_dir/join_report.txt        the attrition table

Usage:  python src/s02_build_sample_table.py
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd

import common
from common import (LOG, load_config, read_table, resolve_roles, work_path)

RUN_RX = r"^[EDS]RR\d+$"
SAMPLE_RX = r"^(SAM[EDN]A?\d+|SAMN\d+)$"


def load_zenodo_metadata(cfg) -> pd.DataFrame:
    p = Path(cfg["paths"]["metadata_h5"])
    LOG.info("Loading Zenodo run metadata: %s", p)
    zmeta = pd.concat(list(common.iter_h5_chunks(p)), ignore_index=True)
    roles = resolve_roles(
        zmeta, {"run": True, "sample": False, "project": False, "date": False,
                "country": False, "host": False}, "zmeta", cfg)
    zmeta = zmeta.rename(columns={roles["run"]: "ena_run",
                                  roles["sample"]: "ena_sample"})
    zmeta["ena_run"] = zmeta["ena_run"].astype(str).str.strip()
    if "ena_sample" in zmeta.columns:
        zmeta["ena_sample"] = zmeta["ena_sample"].astype(str).str.strip()
    LOG.info("Zenodo metadata: %d runs, %d distinct samples",
             zmeta["ena_run"].nunique(),
             zmeta["ena_sample"].nunique() if "ena_sample" in zmeta else -1)
    return zmeta


def build_run_map(cfg, zmeta: pd.DataFrame, note) -> pd.DataFrame:
    """sample_alias -> ena_run, via both the run and sample bridges."""
    p = Path(cfg["paths"]["metalog_mapping"])
    LOG.info("Loading Metalog mapping file: %s", p)
    mp = read_table(p)

    need = {"sample_alias", "kind", "external_id"}
    if not need.issubset(mp.columns):
        raise KeyError(f"Mapping file lacks {need - set(mp.columns)}. Columns: {list(mp.columns)}")

    mp["sample_alias"] = mp["sample_alias"].astype(str).str.strip()
    mp["external_id"] = mp["external_id"].astype(str).str.strip()
    mp["kind"] = mp["kind"].astype(str).str.lower().str.strip()
    note(f"\n[2] Mapping rows total                      : {len(mp):>9,}")
    note("[2] Rows by kind: " + mp["kind"].value_counts().to_dict().__str__())

    z_runs = zmeta[["ena_run"]].drop_duplicates()

    # --- bridge A: direct run accessions ---------------------------------
    a = mp[(mp["kind"] == "run") & mp["external_id"].str.match(RUN_RX, na=False)]
    a = (a[["sample_alias", "external_id"]]
         .rename(columns={"external_id": "ena_run"})
         .merge(z_runs, on="ena_run", how="inner"))
    note(f"[2] Bridge A (kind=run)   -> matched runs   : {a['ena_run'].nunique():>9,} "
         f"from {a['sample_alias'].nunique():,} samples")

    # --- bridge B: sample accessions expanded to their runs --------------
    b = pd.DataFrame(columns=["sample_alias", "ena_run"])
    if "ena_sample" in zmeta.columns:
        z_s2r = zmeta[["ena_sample", "ena_run"]].drop_duplicates()
        bb = mp[(mp["kind"] == "sample") & mp["external_id"].str.match(SAMPLE_RX, na=False)]
        b = (bb[["sample_alias", "external_id"]]
             .rename(columns={"external_id": "ena_sample"})
             .merge(z_s2r, on="ena_sample", how="inner")[["sample_alias", "ena_run"]])
        note(f"[2] Bridge B (kind=sample) -> matched runs  : {b['ena_run'].nunique():>9,} "
             f"from {b['sample_alias'].nunique():,} samples")

    run_map = pd.concat([a[["sample_alias", "ena_run"]], b], ignore_index=True).drop_duplicates()
    note(f"[2] Union of both bridges -> runs           : {run_map['ena_run'].nunique():>9,}")
    note(f"[2] Union of both bridges -> Metalog samples: {run_map['sample_alias'].nunique():>9,}")

    # sanity: a run should not belong to two biological samples
    dup = run_map.groupby("ena_run")["sample_alias"].nunique()
    if (dup > 1).any():
        n = int((dup > 1).sum())
        note(f"[2] WARNING: {n:,} runs map to >1 Metalog sample; keeping first only.")
        run_map = run_map.drop_duplicates("ena_run")
    return run_map.rename(columns={"sample_alias": "sample"})


def load_metalog_metadata(cfg) -> pd.DataFrame:
    p = Path(cfg["paths"]["metalog_meta"])
    LOG.info("Loading Metalog human metadata: %s", p)
    meta = read_table(p)
    roles = resolve_roles(
        meta, {"sample": True, "study": False, "country": False, "age": False,
               "sex": False, "disease": False, "bodysite": False},
        "metalog_meta", cfg)
    keep = {v: k for k, v in roles.items() if v is not None}
    extra = [c for c in ("bmi", "diet", "smoker", "medication",
                         "medication_with_parents", "artificial", "intervention",
                         "type_of_birth", "age_category", "environmental_package",
                         "subject_id", "timepoint", "collection_date", "pmid")
             if c in meta.columns and c not in keep]
    meta = meta[list(keep) + extra].rename(columns=keep)
    meta["sample"] = meta["sample"].astype(str).str.strip()
    LOG.info("Metalog human samples: %d", len(meta))
    return meta


def derive_exposure(meta: pd.DataFrame, note) -> pd.DataFrame:
    """Antibiotic exposure from the ATC hierarchy in medication_with_parents.

    Metalog stores medications as '; '-delimited ATC codes WITH their parent
    levels included, so a simple prefix test is exact and needs no lookup:
        ATC J01   systemic antibacterials
        ATC A07A  intestinal anti-infectives
    This is the exposure variable the whole 'who is there vs what they were
    exposed to' question turns on, so it is derived here rather than left to
    ad-hoc parsing downstream.
    """
    if "medication_with_parents" not in meta.columns:
        note("[3] WARNING: no medication_with_parents column; exposure block unavailable.")
        return meta
    mwp = meta["medication_with_parents"].fillna("")
    annotated = mwp.str.len() > 0
    meta["has_medication_data"] = annotated
    # Absence of an annotation is NOT evidence of absence of exposure. Coding
    # unannotated samples as "unexposed" mixes ~85% unknowns into the negative
    # class and dilutes any real antibiotic effect to nothing. They stay NaN,
    # and the exposure analysis is restricted to the annotated subset.
    meta["abx_systemic"] = np.where(annotated, mwp.str.contains(r"ATC J01", na=False), np.nan)
    meta["abx_intestinal"] = np.where(annotated, mwp.str.contains(r"ATC A07A", na=False), np.nan)
    meta["abx_any"] = np.where(
        annotated,
        mwp.str.contains(r"ATC J01", na=False) | mwp.str.contains(r"ATC A07A", na=False),
        np.nan)
    meta["n_medications"] = mwp.apply(
        lambda s: len([x for x in str(s).split("; ") if x.strip()]) if s else 0)
    note(f"[3] Samples with any medication annotation : "
         f"{meta['has_medication_data'].sum():>9,}")
    note(f"[3] Of those, antibiotic-exposed            : "
         f"{int(np.nansum(meta['abx_any'])):>9,} "
         f"(systemic {int(np.nansum(meta['abx_systemic'])):,}, "
         f"intestinal {int(np.nansum(meta['abx_intestinal'])):,})")
    note(f"[3] Annotated AND unexposed (the controls)  : "
         f"{int(annotated.sum() - np.nansum(meta['abx_any'])):>9,}")
    return meta


def attach_side_tables(cfg, st: pd.DataFrame, note) -> pd.DataFrame:
    for key, prefix in [("metalog_reads", "qc_"), ("metalog_entero", "ent_"),
                        ("metalog_load", "load_")]:
        p = Path(cfg["paths"][key])
        if not p.exists():
            LOG.warning("Optional table missing, skipping: %s", p)
            continue
        df = read_table(p)
        df = df.rename(columns={df.columns[0]: "sample"})
        df["sample"] = df["sample"].astype(str).str.strip()
        df = df.drop_duplicates("sample")
        ren = {c: f"{prefix}{c}" for c in df.columns if c != "sample"}
        df = df.rename(columns=ren)
        st = st.merge(df, on="sample", how="left")
        first = list(ren.values())[0]
        note(f"[6] {key:<16}: {st[first].notna().sum():>9,} / {len(st):,} samples matched")
    return st


def main() -> None:
    cfg = load_config()
    f = cfg["filters"]
    buf = io.StringIO()

    def note(msg: str) -> None:
        LOG.info(msg)
        buf.write(msg + "\n")

    note("=" * 70)
    note("W1 ACCESSION JOIN - ATTRITION REPORT")
    note("=" * 70)

    zmeta = load_zenodo_metadata(cfg)
    note(f"\n[1] Zenodo runs                             : {zmeta['ena_run'].nunique():>9,}")

    run_map = build_run_map(cfg, zmeta, note)

    meta = load_metalog_metadata(cfg)
    note(f"\n[3] Metalog human samples with metadata     : {meta['sample'].nunique():>9,}")

    # Mock communities and other synthetic samples must go. Metalog flags them
    # in `artificial`; the reference usage example keeps only rows where it is
    # NA. Leaving them in would put fabricated compositions into training data.
    if "artificial" in meta.columns:
        real = meta["artificial"].isna()
        note(f"[3] Excluding artificial/mock samples       : {(~real).sum():>9,} dropped")
        meta = meta[real]

    meta = derive_exposure(meta, note)

    st = (run_map.groupby("sample").agg(n_runs=("ena_run", "nunique")).reset_index()
          .merge(meta, on="sample", how="inner"))
    note(f"[4] Joined: resistome + curated metadata    : {len(st):>9,}")

    if "bodysite" in st.columns:
        # Metalog's canonical faecal selector is the exact ENVO term; the
        # keyword match is only a fallback for other export versions.
        exact = st["bodysite"].astype(str).eq("fecal material [ENVO:00002003]")
        if exact.sum() > 0:
            gut = exact
            note(f"[5] Faecal material [ENVO:00002003]         : {gut.sum():>9,} "
                 f"(dropped {(~gut).sum():,})")
        else:
            pat = "|".join(f["bodysite_keywords"])
            gut = st["bodysite"].astype(str).str.lower().str.contains(pat, na=False)
            note(f"[5] Gut/faecal by keyword fallback          : {gut.sum():>9,} "
                 f"(dropped {(~gut).sum():,})")
        st = st[gut]
    else:
        note("[5] WARNING: no bodysite column; gut filter SKIPPED.")

    if "disease" in st.columns:
        known = st["disease"].notna()
        note(f"[5] With known disease status               : {known.sum():>9,} "
             f"(dropped {(~known).sum():,})")
        st = st[known]

    st = attach_side_tables(cfg, st, note)

    if "qc_read_count" in st.columns:
        st["reads_after_qc"] = pd.to_numeric(st["qc_read_count"], errors="coerce")
        ok = st["reads_after_qc"].isna() | (st["reads_after_qc"] >= f["min_reads_after_qc"])
        note(f"[7] Passing depth filter (>= {f['min_reads_after_qc']:,})   : {ok.sum():>9,} "
             f"(dropped {(~ok).sum():,})")
        st = st[ok]
    else:
        note("[7] WARNING: qc_read_count absent; depth filter SKIPPED.")

    note("\n" + "=" * 70)
    note(f"FINAL USABLE SAMPLES: {len(st):,}")
    note("=" * 70)

    if "study" in st.columns:
        by = st.groupby("study").size().sort_values(ascending=False)
        note(f"\nStudies represented: {by.size}")
        note(f"Studies with >= {f['min_samples_per_study']} samples: "
             f"{(by >= f['min_samples_per_study']).sum()}")
        note("\nTop 25 studies:\n" + by.head(25).to_string())
    if "country" in st.columns:
        by = st.groupby("country").size().sort_values(ascending=False)
        note(f"\nCountries: {by.size}\n" + by.head(20).to_string())
    if "disease" in st.columns:
        note("\nDisease status (top 20):\n" +
             st["disease"].astype(str).value_counts().head(20).to_string())
    if "age" in st.columns:
        a = pd.to_numeric(st["age"], errors="coerce")
        note(f"\nAge: n={a.notna().sum():,}, median={a.median():.1f}, "
             f"range={a.min():.0f}-{a.max():.0f}")

    # Per-study attrition so the cohort table describes the FILTERED cohort
    # (19,427 samples, 112 studies), not the smaller modelling subset that
    # survives alignment with taxonomic profiles.
    if "study" in st.columns:
        att = (st.groupby("study")
                 .agg(n_samples=("sample", "nunique"),
                      n_subjects=("subject_id", "nunique")
                      if "subject_id" in st.columns else ("sample", "nunique"),
                      n_countries=("country", "nunique")
                      if "country" in st.columns else ("sample", "nunique"),
                      median_reads=("reads_after_qc", "median")
                      if "reads_after_qc" in st.columns else ("n_runs", "median"))
                 .reset_index()
                 .sort_values("n_samples", ascending=False))
        if "country" in st.columns:
            att = att.merge(st.groupby("study")["country"].agg(
                lambda x: ", ".join(sorted(set(map(str, x)))[:3])).rename("countries"),
                on="study", how="left")
        if "disease" in st.columns:
            att = att.merge(st.groupby("study")["disease"].agg(
                lambda x: ", ".join(sorted(set(map(str, x)))[:3])).rename("health_status"),
                on="study", how="left")
        att.to_csv(Path(cfg["paths"]["table_dir"]) / "cohort_by_study.csv", index=False)
        note(f"\n[8] Per-study cohort table written: {len(att)} studies, "
             f"{int(att['n_samples'].sum()):,} samples")

    run_map = run_map[run_map["sample"].isin(st["sample"])]
    st.to_parquet(work_path(cfg, "sample_table.parquet"), index=False)
    run_map.to_parquet(work_path(cfg, "run_map.parquet"), index=False)
    work_path(cfg, "join_report.txt").write_text(buf.getvalue(), encoding="utf-8")

    print("\n" + buf.getvalue())
    LOG.info("Wrote sample_table.parquet (%d rows), run_map.parquet (%d rows)",
             len(st), len(run_map))


if __name__ == "__main__":
    main()
