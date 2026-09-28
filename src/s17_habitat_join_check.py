"""
s17_habitat_join_check.py
=========================
Is the cross-habitat extension viable at all?

Before building per-habitat configurations, this answers the only question
that matters: how many samples in each habitat have BOTH a resistome profile
in the Zenodo resource and a taxonomic profile in Metalog. The Zenodo resource
covers ENA deposits up to 2020, and habitats differ enormously in how much of
their sequencing predates that, so the answer cannot be guessed from the
metadata counts alone.

If a habitat yields only a few hundred joined samples, a variance partition on
it will not support any claim, and the extension should be scoped to the
habitats that do.

Usage:
    python src/s17_habitat_join_check.py \\
        --habitats /mnt/x/w1_resistome/Metalog_environmental \\
                   /mnt/x/w1_resistome/Metalog_animal \\
                   /mnt/x/w1_resistome/Metalog_ocean
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

import common
from common import LOG, load_config, read_table

RUN_RX = r"^[EDS]RR\d+$"
SAMPLE_RX = r"^(SAM[EDN]A?\d+|SAMN\d+)$"


def load_zenodo(cfg):
    p = Path(cfg["paths"]["metadata_h5"])
    LOG.info("Loading Zenodo run metadata once: %s", p)
    z = pd.concat(list(common.iter_h5_chunks(p)), ignore_index=True)
    z = z.rename(columns={"run_accession": "ena_run",
                          "sample_accession": "ena_sample"})
    for c in ("ena_run", "ena_sample"):
        if c in z.columns:
            z[c] = z[c].astype(str).str.strip()
    LOG.info("Zenodo: %d runs, %d samples", z["ena_run"].nunique(),
             z["ena_sample"].nunique() if "ena_sample" in z else -1)
    return z


def load_mapping(cfg):
    mp = read_table(cfg["paths"]["metalog_mapping"])
    mp["sample_alias"] = mp["sample_alias"].astype(str).str.strip()
    mp["external_id"] = mp["external_id"].astype(str).str.strip()
    mp["kind"] = mp["kind"].astype(str).str.lower().str.strip()
    return mp


def find(habdir: Path, *patterns):
    for pat in patterns:
        hit = sorted(glob.glob(str(habdir / f"*{pat}*")))
        if hit:
            return hit[0]
    return None


def check(habdir: Path, z: pd.DataFrame, mp: pd.DataFrame) -> dict:
    name = habdir.name.replace("Metalog_", "")
    meta_f = find(habdir, "extended_wide", "core_wide", "wide")
    prof_f = find(habdir, "metaphlan4_species", "metaphlan4")
    if not meta_f:
        LOG.warning("%s: no wide metadata file found", name)
        return {}

    try:
        meta = read_table(meta_f)
    except EOFError:
        LOG.error("%s: %s is a truncated gzip archive - the download did not "
                  "complete. Re-download it and rerun. Verify archives with: "
                  "gzip -t <file>", name, Path(meta_f).name)
        return {"habitat": name, "metadata_samples": np.nan,
                "joined_to_zenodo": np.nan, "with_taxonomic_profile": np.nan,
                "studies": np.nan, "error": "truncated metadata archive"}
    except Exception as e:  # noqa: BLE001
        LOG.error("%s: could not read %s (%s)", name, Path(meta_f).name, e)
        return {"habitat": name, "error": str(e)[:80]}
    meta["sample_alias"] = meta["sample_alias"].astype(str).str.strip()
    n_meta = meta["sample_alias"].nunique()

    if "artificial" in meta.columns:
        meta = meta[meta["artificial"].isna()]

    # bridge A: run accessions; bridge B: sample accessions expanded to runs
    a = mp[(mp["kind"] == "run") & mp["external_id"].str.match(RUN_RX, na=False)]
    a = a[a["sample_alias"].isin(meta["sample_alias"])]
    a = a.merge(z[["ena_run"]].drop_duplicates(),
                left_on="external_id", right_on="ena_run", how="inner")

    b = pd.DataFrame(columns=["sample_alias", "ena_run"])
    if "ena_sample" in z.columns:
        bb = mp[(mp["kind"] == "sample")
                & mp["external_id"].str.match(SAMPLE_RX, na=False)]
        bb = bb[bb["sample_alias"].isin(meta["sample_alias"])]
        b = bb.merge(z[["ena_sample", "ena_run"]].drop_duplicates(),
                     left_on="external_id", right_on="ena_sample", how="inner")

    joined = set(a["sample_alias"]) | set(b["sample_alias"])

    n_prof = np.nan
    both = np.nan
    if prof_f:
        try:
            prof = read_table(prof_f, usecols=lambda c: "sample" in c.lower()
                              or "alias" in c.lower())
        except EOFError:
            LOG.error("%s: %s is truncated; profile counts unavailable.",
                      name, Path(prof_f).name)
            prof = None
    if prof_f and prof is not None:
        idcol = prof.columns[0]
        prof_ids = set(prof[idcol].astype(str).str.strip())
        n_prof = len(prof_ids)
        both = len(joined & prof_ids)

    # what the habitat actually contains
    for col in ("environment_biome", "environmental_package", "environment_material"):
        if col in meta.columns:
            top = meta.loc[meta["sample_alias"].isin(joined), col] \
                      .astype(str).value_counts().head(6)
            if len(top):
                LOG.info("%s | %s of joined samples:\n%s", name, col, top.to_string())
                break

    n_studies = (meta.loc[meta["sample_alias"].isin(joined), "study_code"].nunique()
                 if "study_code" in meta.columns else np.nan)

    LOG.info("%-14s metadata=%6d | joined to Zenodo=%6d | with profiles=%6s | "
             "studies=%s", name, n_meta, len(joined),
             f"{both}" if both == both else "n/a", n_studies)
    return {"habitat": name, "metadata_samples": n_meta,
            "joined_to_zenodo": len(joined), "with_taxonomic_profile": both,
            "studies": n_studies}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--habitats", nargs="+", required=True)
    args = ap.parse_args()
    cfg = load_config()

    z = load_zenodo(cfg)
    mp = load_mapping(cfg)

    rows = [check(Path(h), z, mp) for h in args.habitats]
    rows = [r for r in rows if r]
    bad = [r for r in rows if r.get("error")]
    if bad:
        LOG.error("%d habitat(s) could not be assessed: %s", len(bad),
                  ", ".join(f"{r['habitat']} ({r['error']})" for r in bad))
    if not rows:
        raise SystemExit("No habitat could be checked.")
    df = pd.DataFrame(rows)
    out = Path(cfg["paths"]["table_dir"]) / "habitat_join_viability.csv"
    df.to_csv(out, index=False)

    print("\n" + df.to_string(index=False))
    LOG.info("Wrote %s", out)
    LOG.info("Guidance: a habitat needs roughly 1,000 joined samples across "
             "several studies before a variance partition means anything. "
             "Below a few hundred, or concentrated in one or two studies, the "
             "estimate cannot be separated from study identity and the habitat "
             "should be excluded rather than reported with a caveat.")


if __name__ == "__main__":
    main()
