"""
common.py
=========
Config loading, path handling, chunked HDF5 readers and column auto-detection
shared by every stage of the pipeline.
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

LOG_FMT = "%(asctime)s | %(levelname)-7s | %(message)s"


def setup_logging(name: str = "w1") -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format=LOG_FMT, datefmt="%H:%M:%S",
                        stream=sys.stdout)
    return logging.getLogger(name)


LOG = setup_logging()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str | Path = None) -> dict:
    """Load config.local.yaml if it exists (written by s00b_discover_files),
    otherwise config.yaml."""
    root = Path(__file__).resolve().parent.parent
    if path is None:
        for cand in (Path("config.local.yaml"), root / "config.local.yaml",
                     Path("config.yaml"), root / "config.yaml"):
            if cand.exists():
                path = cand
                break
        else:
            raise FileNotFoundError("No config.yaml or config.local.yaml found.")
    path = Path(path)
    if not path.exists():
        alt = root / path.name
        if alt.exists():
            path = alt
        else:
            raise FileNotFoundError(f"config not found at {path}")
    LOG.info("Using config: %s", path)
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    for key in ("work_dir", "fig_dir", "table_dir"):
        Path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)
    return cfg


def stratum(cfg: dict) -> str:
    return str((cfg.get("analysis", {}) or {}).get("stratum", "all") or "all")


def tag(cfg: dict) -> str:
    """Filename suffix identifying the current stratum.

    Without this every stratum overwrites the last, so adult, infant and
    combined results cannot coexist or be compared.
    """
    return f"_{stratum(cfg)}"


def work_path(cfg: dict, name: str, per_stratum: bool = False) -> Path:
    """Path in work_dir. With per_stratum=True the stratum tag is inserted
    before the extension, e.g. vp_results_adult.parquet."""
    if per_stratum:
        p = Path(name)
        name = f"{p.stem}{tag(cfg)}{p.suffix}"
    return Path(cfg["paths"]["work_dir"]) / name


def table_path(cfg: dict, name: str, per_stratum: bool = True) -> Path:
    p = Path(name)
    if per_stratum:
        name = f"{p.stem}{tag(cfg)}{p.suffix}"
    d = Path(cfg["paths"]["table_dir"])
    d.mkdir(parents=True, exist_ok=True)
    return d / name


# ---------------------------------------------------------------------------
# Column auto-detection
#
# The exact column names in the Zenodo HDF5 tables and the Metalog TSVs are
# not guaranteed stable, so nothing downstream hardcodes them. Each stage asks
# for a *role* ("run accession", "gene name") and this resolver finds the best
# matching column, or raises with the list of candidates.
# ---------------------------------------------------------------------------

ROLE_PATTERNS: dict[str, list[str]] = {
    "run":            [r"^run(_?accession)?$", r"^run_?id$", r"runaccession", r"\brun\b"],
    "sample":         [r"^sample(_?accession)?$", r"^sample_?id$", r"biosample", r"\bsample\b"],
    "project":        [r"project", r"study_?accession", r"bioproject"],
    "gene":           [r"refsequence", r"ref_?seq", r"^gene", r"template", r"arg_?name", r"^name$"],
    "count":          [r"fragmentcountaln", r"fragment_?count", r"read_?count", r"^count", r"depth", r"n_?frag"],
    "taxon":          [r"taxon", r"genus", r"phylum", r"lineage", r"clade", r"silva"],
    "date":           [r"collection_?date", r"first_?public", r"^date", r"year", r"release"],
    "country":        [r"country", r"geo_?loc", r"location", r"nation"],
    "host":           [r"host", r"scientific_?name", r"organism", r"env_?", r"habitat", r"body_?site"],
    "study":          [r"^study$", r"study_?name", r"study_?id", r"^dataset"],
    "age":            [r"^age$", r"age_?years", r"host_?age"],
    "sex":            [r"^sex$", r"gender"],
    "disease":        [r"disease", r"health_?status", r"condition", r"phenotype"],
    "bodysite":       [r"body_?site", r"material", r"environment", r"habitat", r"specimen"],
    "metalog_sample": [r"metalog", r"^sample_?id$", r"^sample$"],
    "ena_run":        [r"run", r"^srr", r"^err"],
    "ena_sample":     [r"sample", r"^srs", r"^ers", r"biosample"],
}


def detect_column(columns, role: str, override=None, required: bool = True):
    """Find the column serving `role`. Explicit overrides always win."""
    cols = list(columns)
    if override:
        if override in cols:
            return override
        raise KeyError(f"Configured column '{override}' for role '{role}' not present. Have: {cols[:40]}")
    lowered = {c.lower().strip(): c for c in cols}
    for pat in ROLE_PATTERNS.get(role, []):
        rx = re.compile(pat)
        for low, orig in lowered.items():
            if rx.search(low):
                return orig
    if required:
        raise KeyError(
            f"Could not auto-detect a column for role '{role}'.\n"
            f"Available columns: {cols}\n"
            f"Set columns.<table>.{role} in config.yaml explicitly."
        )
    return None


def resolve_roles(df: pd.DataFrame, roles: dict, table_key: str, cfg: dict) -> dict:
    """Resolve several roles at once, honouring config overrides."""
    overrides = (cfg.get("columns", {}) or {}).get(table_key, {}) or {}
    out = {}
    for role, required in roles.items():
        out[role] = detect_column(df.columns, role, overrides.get(role), required=required)
    LOG.info("Resolved columns for '%s': %s", table_key, out)
    return out


# ---------------------------------------------------------------------------
# HDF5 readers
# ---------------------------------------------------------------------------

def iter_table_chunks(path: str | Path, usecols=None, chunksize: int = 2_000_000):
    """Stream a big table in chunks, whatever the format.

    pandas 'fixed'-format HDF5 cannot be sliced, so reading a 20M+ row table
    that way means loading all of it into RAM at once. When that is detected
    and a sibling .tsv exists (the Zenodo record ships both), the TSV is
    streamed instead, with `usecols` keeping only the needed columns. This is
    both memory-safe and usually faster.
    """
    path = Path(path)
    tsv = path.with_suffix(".tsv")

    if path.suffix in (".h5", ".hdf5"):
        keys = h5_keys(path)
        batched = any(k.strip("/").startswith("table_") for k in keys)
        if not batched:
            sliceable = False
            with pd.HDFStore(str(path), mode="r") as store:
                try:
                    store.select(keys[0], start=0, stop=1)
                    sliceable = True
                except (TypeError, NotImplementedError, AttributeError):
                    sliceable = False
            if not sliceable and tsv.exists():
                LOG.info("%s is fixed-format; streaming %s instead (memory-safe).",
                         path.name, tsv.name)
                for chunk in pd.read_csv(tsv, sep="\t", chunksize=chunksize,
                                         usecols=usecols, low_memory=False):
                    yield chunk
                return
            if not sliceable:
                LOG.warning("%s is fixed-format and no sibling %s exists. Reading it "
                            "whole - this may exhaust RAM. Download the .tsv version "
                            "from the Zenodo record if this fails.",
                            path.name, tsv.name)
    yield from iter_h5_chunks(path, chunksize=chunksize)


def h5_keys(path: str | Path) -> list[str]:
    with pd.HDFStore(str(path), mode="r") as store:
        return list(store.keys())


def read_h5_head(path: str | Path, n: int = 5) -> pd.DataFrame:
    """Read the first rows of the first table in an HDF5 file."""
    keys = h5_keys(path)
    if not keys:
        raise ValueError(f"No tables in {path}")
    with pd.HDFStore(str(path), mode="r") as store:
        try:
            return store.select(keys[0], stop=n)
        except (TypeError, NotImplementedError):
            return store.get(keys[0]).head(n)


def iter_h5_chunks(path: str | Path, chunksize: int = 2_000_000):
    """Yield chunks from an HDF5 file, transparently handling the batched
    layout used by rRNA.h5 (keys table_0 ... table_4736)."""
    keys = h5_keys(path)
    batched = sorted(
        [k for k in keys if re.match(r"^/?table_\d+$", k.strip("/")) or k.strip("/").startswith("table_")],
        key=lambda k: int(re.search(r"(\d+)$", k).group(1)),
    )
    with pd.HDFStore(str(path), mode="r") as store:
        if batched:
            for k in batched:
                yield store.get(k)
        else:
            for k in keys:
                try:
                    nrows = store.get_storer(k).nrows
                    for start in range(0, nrows, chunksize):
                        yield store.select(k, start=start,
                                           stop=min(start + chunksize, nrows))
                except (AttributeError, TypeError, NotImplementedError):
                    # 'fixed' format table: cannot be read in slices
                    LOG.warning("%s/%s is fixed-format; reading it whole "
                                "(this needs enough RAM).", Path(path).name, k)
                    yield store.get(k)


# ---------------------------------------------------------------------------
# Tabular helpers
# ---------------------------------------------------------------------------

def read_table(path: str | Path, **kw) -> pd.DataFrame:
    """Read TSV/CSV, gzipped or not, sniffing the separator."""
    path = str(path)
    sep = "," if re.search(r"\.csv(\.gz)?$", path) else "\t"
    return pd.read_csv(path, sep=sep, low_memory=False, **kw)


def clr(mat: np.ndarray, pseudo: np.ndarray | float | None = None) -> np.ndarray:
    """Centred log-ratio transform, rows = samples."""
    x = np.asarray(mat, dtype=float)
    if pseudo is None:
        nz = x[x > 0]
        pseudo = (nz.min() / 2.0) if nz.size else 1e-6
    x = x + pseudo
    logx = np.log(x)
    return logx - logx.mean(axis=1, keepdims=True)


def derive_life_stage(st: pd.DataFrame) -> pd.DataFrame:
    """Collapse Metalog's age_category into a clean life-stage factor.

    Metalog allows multi-valued entries ('child, adolescent, adult') for
    study-level ranges; those are treated as unknown at the sample level.
    Infant gut communities differ structurally from adult ones, so mixing the
    two makes taxonomy look strongly predictive of the resistome largely
    because it separates babies from adults.
    """
    if "age_category" not in st.columns:
        st["life_stage"] = "unknown"
        return st
    a = st["age_category"].astype(str).str.strip().str.lower()
    st["life_stage"] = np.select(
        [a.eq("baby"), a.eq("child"), a.eq("adolescent"), a.eq("adult")],
        ["infant", "child", "adolescent", "adult"], default="unknown")
    return st


def apply_stratum(cfg: dict, st: pd.DataFrame) -> pd.DataFrame:
    """Restrict to the stratum named in config (analysis.stratum)."""
    stratum = (cfg.get("analysis", {}) or {}).get("stratum", "all")
    st = derive_life_stage(st)
    if stratum in (None, "all"):
        LOG.info("Stratum 'all': %d samples (%s)", len(st),
                 st["life_stage"].value_counts().to_dict())
        return st
    keep = st["life_stage"].eq(stratum)
    LOG.info("Stratum '%s': %d / %d samples retained", stratum, int(keep.sum()), len(st))
    if keep.sum() < 200:
        LOG.warning("Stratum '%s' has very few samples; results will be unstable.", stratum)
    return st[keep]


def namespace_subjects(st: pd.DataFrame) -> pd.DataFrame:
    """Make subject identifiers globally unique.

    Public datasets reuse identifiers such as "1", "01" or "patient1", so
    grouping on subject_id alone can silently merge unrelated participants
    from different studies into one apparent subject.
    """
    if "subject_id" not in st.columns:
        return st
    st = st.copy()
    raw = st["subject_id"].astype(str)
    study = st["study"].astype(str) if "study" in st.columns else ""
    st["subject_uid"] = study + "::" + raw
    n_raw, n_uid = raw.nunique(), st["subject_uid"].nunique()
    if n_uid > n_raw:
        LOG.warning("%d subject identifiers were shared across studies and have "
                    "been separated (%d -> %d unique subjects).",
                    n_uid - n_raw, n_raw, n_uid)
    st["subject_id"] = st["subject_uid"]
    return st


def harmonise_disease(st: pd.DataFrame) -> pd.DataFrame:
    """COHORT and CTR are administrative labels, not diagnoses.

    Metalog uses COHORT for population-cohort samples and CTR for designated
    controls. Left raw, a 'disease' block mostly models study bookkeeping.
    """
    if "disease" not in st.columns:
        return st
    d = st["disease"].astype(str).str.strip()
    healthy = d.str.upper().isin(["COHORT", "CTR", "CONTROL", "CONTROL PATIENT",
                                  "HEALTHY", "N", "NA"])
    st["disease_group"] = np.where(healthy, "non-diseased", "diseased")
    st["disease_detail"] = np.where(healthy, "non-diseased", d)
    LOG.info("Disease groups: %s", st["disease_group"].value_counts().to_dict())
    return st


def prevalence_filter(df: pd.DataFrame, min_prev: float, min_abund: float = 0.0) -> pd.DataFrame:
    """Keep columns present above `min_abund` in at least `min_prev` of rows."""
    present = (df > min_abund).mean(axis=0)
    keep = present[present >= min_prev].index
    LOG.info("Prevalence filter: kept %d / %d features (min_prev=%.2f)",
             len(keep), df.shape[1], min_prev)
    return df[keep]


def extract_year(series: pd.Series) -> pd.Series:
    """Pull a 4-digit year out of messy date strings."""
    s = series.astype(str)
    yr = s.str.extract(r"(19\d{2}|20\d{2})", expand=False)
    return pd.to_numeric(yr, errors="coerce")
