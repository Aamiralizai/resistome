"""
s00_smoke_test.py
=================
Verifies the environment before you touch real data: writes synthetic
matrices in the same shapes the pipeline produces, runs the figure stage,
and reports whether Times New Roman resolved and whether CUDA is visible.

Run this immediately after `pip install -r requirements.txt`. If it passes,
the plotting stack and the file layout are correct and any later failure is a
data problem rather than an environment problem.

It writes into work_dir/_smoke/ and fig_dir/_smoke/ so nothing real is touched.

Usage:  python src/s00_smoke_test.py
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd

import plotstyle as ps
from common import LOG, load_config

RNG = np.random.default_rng(7)


def synth(cfg) -> dict:
    n, n_sp, n_arg = 600, 120, 40
    samples = [f"S{i:05d}" for i in range(n)]
    studies = RNG.choice([f"Study_{c}" for c in "ABCDEFGH"], n)
    countries = RNG.choice(["Denmark", "China", "USA", "Fiji", "Spain"], n)

    st = pd.DataFrame({
        "sample": samples,
        "study": studies,
        "country": countries,
        "age": RNG.normal(45, 18, n).clip(0, 95),
        "sex": RNG.choice(["male", "female"], n),
        "disease": RNG.choice(["healthy", "CRC", "IBD", "T2D"], n),
        "reads_after_qc": RNG.lognormal(16, 0.6, n),
        "load_predicted": RNG.lognormal(25, 1.0, n),
        "n_runs": 1,
    })

    species = [f"Species_{i:03d}" for i in range(n_sp)]
    genera = [f"Genus_{i % 30:02d}" for i in range(n_sp)]
    families = [f"Family_{i % 12:02d}" for i in range(n_sp)]
    lineage = pd.DataFrame({"species": species, "genus": genera,
                            "family": families, "phylum": "Firmicutes",
                            "clade": species})

    relab = RNG.dirichlet(np.ones(n_sp) * 0.4, n)
    taxa_clr = pd.DataFrame(
        np.log(relab + 1e-6) - np.log(relab + 1e-6).mean(axis=1, keepdims=True),
        index=samples, columns=species)

    W = RNG.normal(0, 0.35, (n_sp, n_arg)) * (RNG.random((n_sp, n_arg)) < 0.08)
    arg = pd.DataFrame(taxa_clr.values @ W + RNG.normal(0, 1.0, (n, n_arg)) - 3.0,
                       index=samples, columns=[f"gene_{i:02d}" for i in range(n_arg)])

    return {
        "sample_table.parquet": st,
        "taxa_clr.parquet": taxa_clr,
        "taxa_relab.parquet": pd.DataFrame(relab, index=samples, columns=species),
        "taxa_lineage.parquet": lineage,
        "arg_abundance.parquet": arg,
        "arg_absolute.parquet": arg + 2.0,
        "vp_results.parquet": pd.DataFrame({
            "block": ["taxonomy", "study", "geography", "host", "technical"],
            "n_terms": [50, 7, 4, 6, 2],
            "R2": [0.21, 0.18, 0.09, 0.04, 0.02],
            "R2_adj": [0.17, 0.16, 0.08, 0.03, 0.02],
            "p_perm": [0.005, 0.005, 0.005, 0.02, 0.10],
            "unique_R2": [0.09, 0.11, 0.03, 0.01, 0.01],
            "unique_R2_adj": [0.07, 0.10, 0.02, 0.01, 0.01]}),
        "attribution.parquet": pd.DataFrame(W, index=species, columns=arg.columns),
    }


def synth_cv(arg_cols, samples) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, preds = [], []
    MODELS = {"mean": 0.0, "ridge": 0.28, "taxa_nn": 0.38,
              "mlp": 0.37, "rf": 0.40, "xgb": 0.45}
    for scheme in ("leave_one_study_out", "random"):
        for model in MODELS:
            base = MODELS[model]
            if scheme == "leave_one_study_out":
                base *= 0.6
            for fold in range(4):
                rho = np.clip(RNG.normal(base, 0.12, len(arg_cols)), -0.6, 0.95)
                r2 = rho ** 2 * RNG.uniform(0.5, 1.0, len(arg_cols))
                if scheme == "leave_one_study_out" and model == "ridge":
                    r2 = r2 - 0.35          # linear model mis-calibrates on shift
                rows.append(pd.DataFrame({
                    "gene": arg_cols, "spearman": rho, "r2": r2,
                    "r2_centered": r2 + RNG.normal(0, .01, len(arg_cols)),
                    "model": model, "scheme": scheme,
                    "fold": f"f{fold}", "n_test": 120}))
            sub = RNG.choice(samples, 120, replace=False)
            obs = RNG.normal(-3, 1.2, len(sub) * len(arg_cols))
            preds.append(pd.DataFrame({
                "sample": np.repeat(sub, len(arg_cols)),
                "gene": np.tile(arg_cols, len(sub)),
                "observed": obs,
                "predicted": obs * base + RNG.normal(0, 0.8, len(obs)) - 3 * (1 - base),
                "model": model, "scheme": scheme, "fold": "f0"}))
    return pd.concat(rows, ignore_index=True), pd.concat(preds, ignore_index=True)


def main() -> None:
    cfg = load_config()
    cfg = copy.deepcopy(cfg)
    work = Path(cfg["paths"]["work_dir"]) / "_smoke"
    figs = Path(cfg["paths"]["fig_dir"]) / "_smoke"
    tabs = Path(cfg["paths"]["table_dir"]) / "_smoke"
    for d in (work, figs, tabs):
        d.mkdir(parents=True, exist_ok=True)
    cfg["paths"]["work_dir"], cfg["paths"]["fig_dir"], cfg["paths"]["table_dir"] = \
        str(work), str(figs), str(tabs)

    data = synth(cfg)
    for name, df in data.items():
        idx = name.startswith(("taxa_clr", "taxa_relab", "arg_", "attribution"))
        df.to_parquet(work / name, index=idx)

    m, p = synth_cv(list(data["arg_abundance.parquet"].columns),
                    list(data["sample_table.parquet"]["sample"]))
    m.to_parquet(work / "cv_metrics.parquet", index=False)
    m.to_parquet(work / f"cv_metrics{__import__('common').tag(cfg)}.parquet", index=False)
    hold = m[(m["scheme"] == "leave_one_study_out")].copy()
    hold = hold.groupby(["model", "gene"], as_index=False).first()
    hold.to_parquet(work / f"holdout_metrics{__import__('common').tag(cfg)}.parquet",
                    index=False)
    p.to_parquet(work / "cv_predictions.parquet", index=False)
    p.to_parquet(work / f"cv_predictions{__import__('common').tag(cfg)}.parquet", index=False)
    (work / "join_report.txt").write_text(
        "[1] Zenodo runs with metadata : 214095\n"
        "[3] Runs in both              : 41230\n"
        "[4] With curated metadata     : 22110\n"
        "[5] Human gut samples         : 15870\n"
        "[6] Passing depth filter      : 14002\n", encoding="utf-8")
    pd.DataFrame({"category": ["confirmed", "unexplained", "missed", "absent"],
                  "n": [812, 431, 990, 44120]}).to_csv(
        tabs / "attribution_validation.csv", index=False)
    pd.DataFrame({"species": [f"Species_{i:03d}" for i in range(20)],
                  "gene": [f"gene_{i%40:02d}" for i in range(20)],
                  "attribution": np.linspace(0.9, 0.4, 20)}).to_csv(
        tabs / "unexplained_associations.csv", index=False)

    import s07_figures
    ps.apply_style()
    for name, fn in s07_figures.FIGS.items():
        fn(cfg)
        LOG.info("smoke: %s built", name)

    print("\n" + "=" * 62)
    print("SMOKE TEST COMPLETE")
    print("=" * 62)
    import matplotlib.font_manager as fm
    have_tnr = "Times New Roman" in {f.name for f in fm.fontManager.ttflist}
    print(f"Times New Roman available : {have_tnr}")
    if not have_tnr:
        print("  -> sudo apt-get install -y ttf-mscorefonts-installer fonts-liberation")
        print("  -> rm -rf ~/.cache/matplotlib")
    try:
        import torch
        print(f"PyTorch                   : {torch.__version__}")
        print(f"CUDA available            : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"Device                    : {torch.cuda.get_device_name(0)}")
            print(f"Compute capability        : {torch.cuda.get_device_capability()}"
                  "   (expect (12, 0) on RTX 5080)")
    except ImportError:
        print("PyTorch                   : NOT INSTALLED")
    print(f"\nFigures written to        : {figs}")


if __name__ == "__main__":
    main()
