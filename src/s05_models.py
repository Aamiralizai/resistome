"""
s05_models.py
=============
Predicting the resistome from community composition, evaluated under
deliberate distribution shift.

Task: multi-output regression. Input = CLR species abundances (+ optional
covariates). Output = log ARG abundance for every retained gene. This is a
structured prediction problem, not a classifier, and framing it that way is
part of the contribution - the existing literature is almost entirely
single-cohort random forests on one ARG at a time.

Models
------
mean        per-gene training mean (the floor any real model must clear)
ridge       multi-output ridge on CLR species
gbm         per-gene gradient boosting (optional, slow; set --gbm)
taxa_nn     phylogeny-aware network:
              species -> learned genus pooling -> learned family pooling
              -> concatenate all three rank representations -> MLP -> ARGs
            The pooling is a fixed sparse assignment with learned weights, so
            the model can borrow strength across related species that were
            never seen together in the same cohort. That is exactly the
            failure mode of flat models under leave-one-study-out.

Evaluation
----------
Leave-one-study-out by default. Random splits are ALSO reported, because the
gap between the two is itself a result worth a figure: it quantifies how much
published single-cohort performance is lineage and batch leakage.

Metrics: per-gene Spearman rho and R2 on held-out folds, aggregated by median.

Outputs:
    work_dir/cv_predictions.parquet   held-out predictions, all folds
    work_dir/cv_metrics.parquet       per gene x model x scheme
    work_dir/model_taxa_nn.pt         weights from a final full-data fit
    tables/model_summary.csv

Usage:
    python src/s05_models.py                 # ridge + nn, LOSO + random
    python src/s05_models.py --gbm           # add gradient boosting
    python src/s05_models.py --scheme leave_one_country_out
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from common import (LOG, apply_stratum, harmonise_disease, load_config,
                    stratum, table_path, work_path)

TAXA_TRANSFORM: dict = {}

try:
    import torch
    import torch.nn as nn
    HAVE_TORCH = True
except ImportError:  # pragma: no cover
    HAVE_TORCH = False
    LOG.warning("PyTorch not available - only ridge/mean baselines will run.")


# ---------------------------------------------------------------------------
# Splitters
# ---------------------------------------------------------------------------

def holdout_studies(groups: pd.Series, n: int, seed: int = 42) -> set:
    """Lock away whole studies as a final test set.

    Cross-validation estimates become optimistic the moment a model is CHOSEN
    on them - comparing six models and reporting the winner's CV score is a
    selection estimate, not an unbiased one. These studies are excluded from
    every fold of every scheme and scored exactly once, at the end.

    Studies are sampled across the size distribution rather than at random, so
    the held-out set is not all tiny cohorts.
    """
    if n <= 0:
        return set()
    sizes = groups.value_counts()
    eligible = sizes[sizes >= 20]
    if len(eligible) <= n:
        LOG.warning("Too few studies to hold out %d; holdout disabled.", n)
        return set()
    rng = np.random.default_rng(seed)
    strata = np.array_split(list(eligible.index), n)   # size-ordered bands
    picked = {str(rng.choice(band)) for band in strata if len(band)}
    LOG.info("Held-out studies (%d, never used for fitting or selection): %s",
             len(picked), sorted(picked))
    LOG.info("Held-out samples: %d of %d", int(groups.isin(picked).sum()), len(groups))
    return picked


def group_splits(groups: pd.Series, min_test: int = 20):
    """Leave-one-group-out, skipping groups too small to evaluate."""
    for g in groups.value_counts().index:
        test = (groups == g).values
        if test.sum() < min_test or (~test).sum() < 100:
            continue
        yield str(g), ~test, test


def random_splits(n: int, k: int = 5, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    folds = np.array_split(idx, k)
    for i, f in enumerate(folds):
        test = np.zeros(n, dtype=bool)
        test[f] = True
        yield f"fold{i}", ~test, test


def subject_splits(subjects: pd.Series, k: int = 5, seed: int = 42):
    """K-fold cross-validation splitting on SUBJECTS, not samples.

    The middle rung of the validation ladder. ``random_splits`` above puts
    repeat samples from one person on both sides of the split, so a model can
    score by recognising the person rather than by learning how composition
    maps to resistome. Whole subjects are assigned to folds here, so the test
    fold contains no one seen in training.

    The interpretation this enables is the point. Three numbers, same task:

        random            person seen, cohort seen
        subject_grouped   person unseen, cohort seen
        leave_one_study_out   person unseen, cohort unseen

    Whatever falls between the first and second is subject leakage; whatever
    falls between the second and third is genuine cohort shift. Reporting only
    the first and third attributes all of it to cohort shift.

    Folds are balanced by sample count rather than subject count, because
    subjects contribute between 1 and 81 samples; balancing on subjects would
    leave folds differing several-fold in size and make per-fold correlations
    incomparable.
    """
    subjects = pd.Series(subjects).astype(str).reset_index(drop=True)
    sizes = subjects.value_counts()
    rng = np.random.default_rng(seed)
    order = sizes.index.to_numpy()
    rng.shuffle(order)
    # Greedy largest-first assignment keeps fold sizes close despite the
    # heavily skewed number of samples per subject.
    order = sorted(order, key=lambda s: -int(sizes[s]))
    load = np.zeros(k, dtype=int)
    assign: dict[str, int] = {}
    for s in order:
        j = int(np.argmin(load))
        assign[s] = j
        load[j] += int(sizes[s])
    fold_of = subjects.map(assign).to_numpy()
    LOG.info("Subject-grouped CV: %d subjects over %d samples; fold sizes %s",
             len(sizes), len(subjects), load.tolist())
    for i in range(k):
        test = fold_of == i
        if test.sum() < 20 or (~test).sum() < 100:
            LOG.warning("Subject fold %d too small (%d test); skipped.", i, int(test.sum()))
            continue
        yield f"subjfold{i}", ~test, test


# ---------------------------------------------------------------------------
# Phylogeny-aware network
# ---------------------------------------------------------------------------

if HAVE_TORCH:

    class RankPool(nn.Module):
        """Aggregate species features into a coarser rank with learned weights.

        assign: LongTensor [n_species] giving the parent index of each species.
        Each species gets a scalar weight; pooling is a weighted sum into the
        parent slot. Initialised at 1.0 so the layer starts as a plain sum.
        """

        def __init__(self, assign: "torch.Tensor", n_parent: int):
            super().__init__()
            self.register_buffer("assign", assign)
            self.n_parent = n_parent
            self.w = nn.Parameter(torch.ones(assign.numel()))

        def forward(self, x):                      # x: [B, n_species]
            weighted = x * self.w.unsqueeze(0)
            out = x.new_zeros(x.shape[0], self.n_parent)
            out.index_add_(1, self.assign, weighted)
            return out

    class TaxaNet(nn.Module):
        def __init__(self, n_species: int, genus_assign, n_genus: int,
                     family_assign, n_family: int, n_cov: int, n_out: int,
                     hidden=(512, 256), dropout: float = 0.3):
            super().__init__()
            self.genus = RankPool(genus_assign, n_genus)
            self.family = RankPool(family_assign, n_family)
            d_in = n_species + n_genus + n_family + n_cov
            layers, d = [], d_in
            for h in hidden:
                layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(dropout)]
                d = h
            self.mlp = nn.Sequential(*layers)
            self.head = nn.Linear(d, n_out)

        def forward(self, x, cov=None):
            g = self.genus(x)
            f = self.family(x)
            parts = [x, g, f] + ([cov] if cov is not None and cov.shape[1] else [])
            return self.head(self.mlp(torch.cat(parts, dim=1)))


def _rank_quality(lin: pd.DataFrame, rank: str, n_species: int) -> float:
    """How much a rank collapses the species set, as groups per species.

    1.0 means every species sits in its own group - pooling would be a no-op.
    ZERO groups means the lineage lookup failed entirely, which must be
    rejected rather than treated as maximal collapsing: a ratio of 0.0 would
    otherwise pass any upper-bound test and the model would train with empty
    pooling branches, silently reducing the taxonomy-aware network to a plain
    one.
    """
    if rank not in lin.columns:
        return 1.0
    n_groups = lin[rank].nunique()
    if n_groups < 2:
        return 1.0            # unusable: no grouping information
    return n_groups / max(n_species, 1)


def build_assignments(lineage: pd.DataFrame, species_order, ranks=None,
                      max_ratio: float = 0.5):
    """Pick two usable pooling ranks and map species onto them.

    MetaPhlAn 4 assigns placeholder GGB genera to its unnamed SGB clades, so
    `genus` is close to one-to-one with species and pooling over it does
    nothing. Ranks are therefore screened by how much they actually collapse
    the species set, and the two coarsest usable ones are chosen. Falls back to
    a single level, or to none, rather than pretending to pool.
    """
    lin = lineage.drop_duplicates("species").set_index("species").reindex(species_order)
    n = len(species_order)
    candidates = ranks or ["genus", "family", "order", "klass", "phylum"]
    usable = []
    for r in candidates:
        q = _rank_quality(lin.reset_index(), r, n)
        n_groups = lin[r].nunique() if r in lin.columns else 0
        why = ""
        if n_groups < 2:
            why = "  <- lineage lookup failed, skipped"
        elif q > max_ratio:
            why = "  <- too little collapsing, skipped"
        LOG.info("  rank %-8s: %d groups for %d species (ratio %.2f)%s",
                 r, n_groups, n, q, why)
        if q <= max_ratio and n_groups >= 2:
            usable.append(r)

    chosen = usable[:2] if usable else []
    if not chosen:
        LOG.error("NO USABLE POOLING RANK. Every taxonomic rank either failed "
                  "to resolve or provided no grouping, so the taxonomy-aware "
                  "network would be identical to the plain network. The usual "
                  "cause is a species-name mismatch against the lineage table.")
    else:
        LOG.info("Pooling ranks selected: %s", chosen)

    out = []
    for r in chosen:
        lab = lin[r].fillna(f"unclassified_{r}").astype(str)
        levels = {v: i for i, v in enumerate(sorted(lab.unique()))}
        out.append((np.array([levels[v] for v in lab]), len(levels), r))
    while len(out) < 2:
        out.append((np.zeros(n, dtype=int), 1, None))
    return (out[0][0], out[0][1], out[1][0], out[1][1]), [c for c in chosen]


def fit_taxa_nn(Xtr, Ytr, Xva, Yva, Ctr, Cva, assign, cfg_model):
    ga, n_g, fa, n_f = assign
    dev = torch.device(cfg_model["device"] if torch.cuda.is_available() else "cpu")
    model = TaxaNet(Xtr.shape[1], torch.as_tensor(ga, dtype=torch.long), n_g,
                    torch.as_tensor(fa, dtype=torch.long), n_f,
                    Ctr.shape[1], Ytr.shape[1],
                    hidden=tuple(cfg_model["hidden"]),
                    dropout=cfg_model["dropout"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg_model["lr"],
                            weight_decay=cfg_model["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=8, factor=0.5)
    lossf = nn.SmoothL1Loss()

    tt = lambda a: torch.as_tensor(a, dtype=torch.float32, device=dev)
    Xtr_t, Ytr_t, Ctr_t = tt(Xtr), tt(Ytr), tt(Ctr)
    Xva_t, Yva_t, Cva_t = tt(Xva), tt(Yva), tt(Cva)

    n, bs = len(Xtr_t), cfg_model["batch_size"]
    best, best_state, bad = np.inf, None, 0
    for epoch in range(cfg_model["n_epochs"]):
        model.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            j = perm[i:i + bs]
            if len(j) < 2:
                continue
            opt.zero_grad()
            loss = lossf(model(Xtr_t[j], Ctr_t[j]), Ytr_t[j])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = lossf(model(Xva_t, Cva_t), Yva_t).item()
        sched.step(vl)
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg_model["patience"]:
                break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(Xva_t, Cva_t).cpu().numpy()
    return model, pred


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def fit_random_forest(Xtr, Ytr, Xte, n_trees=200, seed=0):
    """Multi-output random forest.

    One forest with vector-valued leaves predicts all genes at once, which is
    far cheaper than 99 separate forests and is the standard baseline in the
    microbiome ML literature. Its absence would be the first thing a reviewer
    asks about.
    """
    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor(n_estimators=n_trees, max_features="sqrt",
                               min_samples_leaf=5, n_jobs=-1, random_state=seed)
    rf.fit(Xtr, Ytr)
    return rf.predict(Xte)


_XGB_DEVICE = None


def _xgb_device() -> str:
    """Pick cuda if XGBoost can actually use it, else cpu.

    Probed once with a tiny fit rather than assumed: a CUDA-capable torch does
    not imply a CUDA-enabled XGBoost build, and a failed device is otherwise
    only discovered mid-fold.
    """
    global _XGB_DEVICE
    if _XGB_DEVICE is not None:
        return _XGB_DEVICE
    _XGB_DEVICE = "cpu"
    try:
        import xgboost as xgb
        rng = np.random.default_rng(0)
        xgb.XGBRegressor(n_estimators=2, device="cuda", tree_method="hist",
                         verbosity=0).fit(rng.normal(size=(32, 4)),
                                          rng.normal(size=32))
        _XGB_DEVICE = "cuda"
    except Exception as e:  # noqa: BLE001
        LOG.info("XGBoost GPU unavailable (%s); using CPU.", str(e)[:70])
    LOG.info("XGBoost device: %s", _XGB_DEVICE)
    return _XGB_DEVICE


def fit_xgboost(Xtr, Ytr, Xte, cfg_model=None, seed=0):
    """Gradient boosting on GPU where available.

    Multi-output trees predict all genes from one model, which is both faster
    and better regularised than 99 independent boosters.
    """
    try:
        import xgboost as xgb
        dev = _xgb_device()
        # multi_output_tree is one tree predicting all genes: better
        # regularised but experimental and often SLOW. one_output_per_tree is
        # the stable default. Set model.xgb_multi_output: true to switch.
        strat = ("multi_output_tree" if (cfg_model or {}).get("xgb_multi_output", False)
                 else "one_output_per_tree")
        m = xgb.XGBRegressor(n_estimators=int((cfg_model or {}).get("xgb_trees", 250)),
                             learning_rate=0.06, max_depth=5,
                             subsample=0.8, colsample_bytree=0.6,
                             tree_method="hist", device=dev,
                             multi_strategy=strat,
                             n_jobs=-1, random_state=seed, verbosity=0)
        m.fit(Xtr, Ytr)
        return m.predict(Xte)
    except Exception as e:  # noqa: BLE001
        LOG.debug("XGBoost multi-output unavailable (%s); per-gene fallback.", e)
        from sklearn.ensemble import HistGradientBoostingRegressor
        out = np.zeros((len(Xte), Ytr.shape[1]))
        for j in range(Ytr.shape[1]):
            out[:, j] = HistGradientBoostingRegressor(
                max_iter=150, learning_rate=0.08, random_state=seed
            ).fit(Xtr, Ytr[:, j]).predict(Xte)
        return out


def fit_mlp(Xtr, Ytr, Xte, cfg_model=None, seed=0):
    """Plain MLP on species abundances, with NO taxonomic pooling.

    This is the ablation that isolates the architecture's contribution: any
    difference between taxa_nn and mlp is attributable to pooling rather than
    to nonlinearity or capacity. It therefore uses the SAME hidden sizes,
    dropout, optimiser and early-stopping rule as taxa_nn - only the two
    pooling branches are removed. Runs on GPU when torch is available.
    """
    if not HAVE_TORCH:
        from sklearn.neural_network import MLPRegressor
        m = MLPRegressor(hidden_layer_sizes=(512, 256), activation="relu",
                         alpha=1e-4, batch_size=256, learning_rate_init=1e-3,
                         max_iter=300, early_stopping=True, n_iter_no_change=15,
                         validation_fraction=0.15, random_state=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(Xtr, Ytr)
        return m.predict(Xte)

    c = cfg_model or {}
    hidden = tuple(c.get("hidden", [512, 256]))
    dropout = float(c.get("dropout", 0.3))
    dev = torch.device(c.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    layers, d = [], Xtr.shape[1]
    for h in hidden:
        layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(dropout)]
        d = h
    model = nn.Sequential(*layers, nn.Linear(d, Ytr.shape[1])).to(dev)

    opt = torch.optim.AdamW(model.parameters(), lr=float(c.get("lr", 1e-3)),
                            weight_decay=float(c.get("weight_decay", 1e-5)))
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=8, factor=0.5)
    lossf = nn.SmoothL1Loss()

    idx = np.random.default_rng(seed).permutation(len(Xtr))
    cut = int(0.85 * len(idx))
    tt = lambda a: torch.as_tensor(a, dtype=torch.float32, device=dev)
    Xt, Yt = tt(Xtr[idx[:cut]]), tt(Ytr[idx[:cut]])
    Xv, Yv = tt(Xtr[idx[cut:]]), tt(Ytr[idx[cut:]])

    bs = int(c.get("batch_size", 256))
    best, best_state, bad = np.inf, None, 0
    for _ in range(int(c.get("n_epochs", 200))):
        model.train()
        perm = torch.randperm(len(Xt), device=dev)
        for i in range(0, len(Xt), bs):
            j = perm[i:i + bs]
            if len(j) < 2:
                continue
            opt.zero_grad()
            lossf(model(Xt[j]), Yt[j]).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = lossf(model(Xv), Yv).item()
        sched.step(vl)
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= int(c.get("patience", 20)):
                break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return model(tt(Xte)).cpu().numpy()


def fit_ft_transformer(Xtr, Ytr, Xte, cfg_model=None, seed=0):
    """FT-Transformer: attention over feature tokens, for tabular data.

    Why this and not a sequence transformer. The input is a fixed vector of
    species abundances with no meaningful ordering, so positional attention
    over a sequence has nothing to attend to. FT-Transformer instead embeds
    each feature as a token - value x learned weight + learned bias - and
    applies self-attention across those tokens, letting the model learn which
    species interact rather than assuming an order. That is the architecture
    designed for this data shape, and it is what a reviewer asking for "a
    transformer baseline" should be shown.

    Configured to match the other neural models where the comparison would
    otherwise be confounded: same optimiser, same early-stopping rule, same
    validation split, same loss. Differences in performance are therefore
    attributable to the architecture.

    With 449 features a full token-wise attention is O(449^2) per layer per
    sample, which is affordable but not free; the default is deliberately
    small (2 blocks, 8 heads, d=64) because the training set is a few thousand
    samples and a larger model would simply overfit.
    """
    if not HAVE_TORCH:
        LOG.warning("torch unavailable; FT-Transformer skipped.")
        return np.full((len(Xte), Ytr.shape[1]), np.nan)

    c = cfg_model or {}
    d_token = int(c.get("ft_d_token", 64))
    n_blocks = int(c.get("ft_blocks", 2))
    n_heads = int(c.get("ft_heads", 8))
    dropout = float(c.get("ft_dropout", 0.2))
    dev = torch.device(c.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    n_feat, n_out = Xtr.shape[1], Ytr.shape[1]

    class FeatureTokenizer(nn.Module):
        """One token per feature: value * w + b, plus a learned CLS token."""

        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.empty(n_feat, d_token))
            self.b = nn.Parameter(torch.zeros(n_feat, d_token))
            self.cls = nn.Parameter(torch.empty(1, 1, d_token))
            nn.init.normal_(self.w, std=d_token ** -0.5)
            nn.init.normal_(self.cls, std=d_token ** -0.5)

        def forward(self, x):
            t = x.unsqueeze(-1) * self.w + self.b          # (B, n_feat, d)
            return torch.cat([self.cls.expand(len(x), -1, -1), t], dim=1)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm1 = nn.LayerNorm(d_token)
            self.attn = nn.MultiheadAttention(d_token, n_heads, dropout=dropout,
                                              batch_first=True)
            self.norm2 = nn.LayerNorm(d_token)
            self.ff = nn.Sequential(
                nn.Linear(d_token, d_token * 2), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(d_token * 2, d_token))
            self.drop = nn.Dropout(dropout)

        def forward(self, x):
            h = self.norm1(x)
            x = x + self.drop(self.attn(h, h, h, need_weights=False)[0])
            return x + self.drop(self.ff(self.norm2(x)))

    class FTTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok = FeatureTokenizer()
            self.blocks = nn.ModuleList([Block() for _ in range(n_blocks)])
            self.head = nn.Sequential(nn.LayerNorm(d_token), nn.GELU(),
                                      nn.Linear(d_token, n_out))

        def forward(self, x):
            h = self.tok(x)
            for blk in self.blocks:
                h = blk(h)
            return self.head(h[:, 0])          # read out from the CLS token

    model = FTTransformer().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=float(c.get("ft_lr", 3e-4)),
                            weight_decay=float(c.get("weight_decay", 1e-5)))
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=8, factor=0.5)
    lossf = nn.SmoothL1Loss()

    idx = np.random.default_rng(seed).permutation(len(Xtr))
    cut = int(0.85 * len(idx))
    tt = lambda a: torch.as_tensor(a, dtype=torch.float32, device=dev)  # noqa: E731
    Xt, Yt = tt(Xtr[idx[:cut]]), tt(Ytr[idx[:cut]])
    Xv, Yv = tt(Xtr[idx[cut:]]), tt(Ytr[idx[cut:]])

    bs = int(c.get("ft_batch_size", 128))
    best, best_state, bad = np.inf, None, 0
    for _ in range(int(c.get("ft_epochs", 120))):
        model.train()
        perm = torch.randperm(len(Xt), device=dev)
        for i in range(0, len(Xt), bs):
            j = perm[i:i + bs]
            if len(j) < 2:
                continue
            opt.zero_grad()
            lossf(model(Xt[j]), Yt[j]).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(np.mean([lossf(model(Xv[i:i + 512]),
                                      Yv[i:i + 512]).item()
                                for i in range(0, len(Xv), 512)]))
        sched.step(vl)
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= int(c.get("patience", 20)):
                break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(Xte), 512):
            out.append(model(tt(Xte[i:i + 512])).cpu().numpy())
    return np.vstack(out)


def score(Y_true: np.ndarray, Y_pred: np.ndarray, genes) -> pd.DataFrame:
    """Per-gene Spearman and R2 on a held-out fold.

    Spearman is undefined when either vector is constant - the 'mean' baseline
    predicts a constant by construction, and rare ARGs are sometimes constant
    within a small fold. Those return NaN silently rather than emitting a
    warning per gene per fold.
    """
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for j, g in enumerate(genes):
            yt, yp = Y_true[:, j], Y_pred[:, j]
            if np.std(yt) < 1e-9 or np.std(yp) < 1e-9:
                rho = np.nan
            else:
                rho = spearmanr(yt, yp).statistic
            if np.std(yt) < 1e-9:
                r2 = r2c = np.nan
            else:
                ss_tot = np.sum((yt - yt.mean()) ** 2)
                r2 = 1.0 - np.sum((yt - yp) ** 2) / ss_tot
                # Calibration-free R2: remove the fold's mean offset from both
                # vectors. Every study sits at a different absolute resistome
                # level, so raw R2 is dominated by an offset no model trained
                # elsewhere can know. This isolates within-cohort structure,
                # which is what the claim is actually about.
                r2c = 1.0 - np.sum(((yt - yt.mean()) - (yp - yp.mean())) ** 2) / ss_tot
            rows.append({"gene": g, "spearman": rho, "r2": r2, "r2_centered": r2c})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="mean,ridge,taxa_nn",
                    help="comma-separated: mean,ridge,taxa_nn,rf,xgb,mlp. "
                         "'all' fits everything. rf/xgb/mlp add substantial "
                         "runtime across 60+ leave-one-study-out folds.")
    ap.add_argument("--gbm", action="store_true",
                    help="deprecated alias for --models mean,ridge,taxa_nn,xgb")
    ap.add_argument("--skip-cv", action="store_true",
                    help="reuse existing cv_metrics/cv_predictions and run only "
                         "the held-out evaluation and final fit")
    ap.add_argument("--scheme", default=None,
                    choices=["leave_one_study_out", "leave_one_country_out", "random"])
    args = ap.parse_args()

    wanted = ({"mean", "ridge", "taxa_nn", "rf", "xgb", "mlp", "ft"}
              if args.models.strip().lower() == "all"
              else {m.strip() for m in args.models.split(",") if m.strip()})
    if args.gbm:
        wanted |= {"xgb"}
    LOG.info("Models to fit: %s", sorted(wanted))

    cfg = load_config()
    mcfg = cfg["model"]
    np.random.seed(mcfg["seed"])
    if HAVE_TORCH:
        torch.manual_seed(mcfg["seed"])

    st = pd.read_parquet(work_path(cfg, "sample_table.parquet"))
    st = harmonise_disease(apply_stratum(cfg, st)).set_index("sample")
    # Prefer the unfiltered matrix: the pooled 5% filter that produced
    # arg_abundance.parquet was applied across the whole cohort, so starting
    # from it means the gene universe was chosen with the held-out studies in
    # view even though the second filter is development-only.
    _unf = work_path(cfg, "arg_abundance_unfiltered.parquet")
    if _unf.exists():
        arg = pd.read_parquet(_unf)
        LOG.info("Starting from the unfiltered ARG universe (%d genes); all "
                 "prevalence filtering will be learned on development samples "
                 "alone.", arg.shape[1])
    else:
        arg = pd.read_parquet(work_path(cfg, "arg_abundance.parquet"))
        LOG.warning("arg_abundance_unfiltered.parquet absent: the gene universe "
                    "was already filtered across the full cohort. Rerun s03 to "
                    "make feature selection genuinely development-only.")
    taxa = pd.read_parquet(work_path(cfg, "taxa_clr.parquet"))
    lineage = pd.read_parquet(work_path(cfg, "taxa_lineage.parquet"))

    shared = sorted(set(arg.index) & set(taxa.index) & set(st.index))
    arg, taxa, st = arg.loc[shared], taxa.loc[shared], st.loc[shared]
    genes = list(arg.columns)
    LOG.info("Modelling %d samples, %d species -> %d ARG genes",
             len(shared), taxa.shape[1], len(genes))

    # covariate block: log depth only. Deliberately minimal - the question is
    # what TAXONOMY explains, so host variables stay out of the predictor.
    cov = np.zeros((len(shared), 0))
    if "reads_after_qc" in st.columns:
        d = pd.to_numeric(st["reads_after_qc"], errors="coerce")
        cov = np.log1p(d.fillna(d.median())).values[:, None]

    # ---- lock away the final test studies before anything is fitted -----
    n_hold = int((cfg.get("analysis", {}) or {}).get("holdout_studies", 0))
    held = holdout_studies(st["study"], n_hold, mcfg["seed"]) \
        if ("study" in st.columns and n_hold) else set()
    is_held = st["study"].astype(str).isin(held).values if held else np.zeros(len(st), bool)
    dev = ~is_held

    X_all, Y_all = taxa.values, arg.values
    assign, chosen_ranks = build_assignments(lineage, taxa.columns)
    LOG.info("Tree pooling: %d species -> %d %s -> %d %s",
             X_all.shape[1], assign[1], chosen_ranks[0] if chosen_ranks else "(none)",
             assign[3], chosen_ranks[1] if len(chosen_ranks) > 1 else "(none)")
    if assign[1] < 2 or assign[3] < 2:
        LOG.error("POOLING IS DEGENERATE (%d and %d groups). taxa_nn would be "
                  "identical to mlp and the architecture comparison meaningless. "
                  "The usual cause is a species-name mismatch against the "
                  "lineage table - check for an s__ prefix on one side only.",
                  assign[1], assign[3])
        raise SystemExit("Refusing to run a meaningless architecture comparison.")

    # every CV scheme operates on the development set only
    dev_idx = np.where(dev)[0]
    st_dev = st.iloc[dev_idx]

    # Feature selection must be learned from the development set alone.
    # Prevalence filtering applied to the pooled data lets held-out studies
    # influence which genes and species are modelled - unsupervised leakage
    # that contradicts the claim the holdout was untouched.
    # The filter runs whether or not studies are held out. When `arg` is the
    # UNFILTERED universe, skipping it would train on every gene with no
    # prevalence threshold at all - previously safe only because `arg` arrived
    # pre-filtered. With no holdout the "development set" is every sample, so
    # the filter is simply the ordinary prevalence filter.
    _started_unfiltered = _unf.exists()
    if held or _started_unfiltered:
        if not held:
            LOG.info("No held-out studies; prevalence filtering uses all "
                     "samples, which is equivalent to the pooled filter.")
        pseudo = float(cfg["normalisation"]["arg_pseudocount"])
        thr_a = float(cfg["filters"]["arg_min_prevalence"])
        thr_t = float(cfg["filters"]["taxa_min_prevalence"])
        min_ab = float(cfg["filters"]["taxa_min_abundance"])

        # ARG prevalence from RAW COUNTS where available. Testing the log
        # matrix against its pseudocount floor is fragile; testing CLR values
        # against zero is meaningless, because CLR values are never zero -
        # a filter written that way silently retains every feature.
        cnt_p = work_path(cfg, "arg_counts_aligned.parquet")
        if cnt_p.exists():
            cnt = pd.read_parquet(cnt_p).reindex(index=shared).fillna(0.0)
            cnt = cnt.reindex(columns=arg.columns, fill_value=0.0)
            det_a = (cnt.iloc[dev_idx] > 0).mean(axis=0)
        else:
            det_a = (arg.iloc[dev_idx] > np.log(pseudo) + 1e-9).mean(axis=0)
        keep_a = det_a[det_a >= thr_a].index

        # Species prevalence from untransformed relative abundances, and the
        # CLR pseudocount re-estimated from development samples only.
        rel_p = work_path(cfg, "taxa_relab_unfiltered.parquet")
        if rel_p.exists():
            rel = pd.read_parquet(rel_p).reindex(index=shared).fillna(0.0)
            # Harmonise names with taxa_clr and the lineage table.
            if any(str(c).startswith("s__") for c in rel.columns):
                rel.columns = [str(c).replace("s__", "") for c in rel.columns]
                LOG.info("Stripped s__ prefix from the unfiltered species "
                         "matrix so names match the lineage table.")
            det_t = (rel.iloc[dev_idx] > min_ab).mean(axis=0)
            keep_t = det_t[det_t >= thr_t].index
            if len(keep_t) >= 20:
                sub_rel = rel[keep_t]
                nz = sub_rel.iloc[dev_idx].values
                nz = nz[nz > 0]
                pc = (nz.min() / 2.0) if nz.size else 1e-6
                lx = np.log(sub_rel.values + pc)
                taxa = pd.DataFrame(lx - lx.mean(axis=1, keepdims=True),
                                    index=sub_rel.index, columns=sub_rel.columns)
                # Record the transform so downstream stages can reproduce it.
                # The centred log-ratio is computed OVER A SPECIES SET, so a
                # different subset gives different values for the same species;
                # attribution must therefore rebuild the matrix rather than
                # read the filtered one.
                TAXA_TRANSFORM.update(source="taxa_relab_unfiltered.parquet",
                                      pseudocount=float(pc),
                                      species=list(sub_rel.columns))
                LOG.info("CLR re-computed with a development-only pseudocount "
                         "(%.3g); held-out studies did not influence the "
                         "transform.", pc)
        else:
            LOG.warning("taxa_relab_unfiltered.parquet absent - species "
                        "filtering falls back to the pooled matrix. Rerun s03 "
                        "to remove this leakage.")
            keep_t = taxa.columns
        LOG.info("Development-only feature selection: %d/%d ARG families, "
                 "%d/%d species retained", len(keep_a), arg.shape[1],
                 len(keep_t), taxa.shape[1])
        if len(keep_a) >= 10 and len(keep_t) >= 20:
            arg = arg[keep_a]
            taxa = taxa[[c for c in keep_t if c in taxa.columns]]
            genes = list(arg.columns)
            X_all, Y_all = taxa.values, arg.values
            assign, chosen_ranks = build_assignments(lineage, taxa.columns)
        elif _started_unfiltered:
            # Falling back to "the pooled feature set" is only meaningful when
            # one exists. Starting from the unfiltered universe there is none,
            # so retaining it would model hundreds of near-absent genes.
            LOG.error("Development-only filtering retained %d ARG families and "
                      "%d species, below the minimum. Because the run started "
                      "from the unfiltered universe there is no pooled fallback; "
                      "check the prevalence thresholds in the configuration.",
                      len(keep_a), len(keep_t))
            raise SystemExit("Refusing to model an unfiltered gene universe.")
        else:
            LOG.warning("Development-only filtering too aggressive; keeping "
                        "the pooled feature set and noting the caveat.")

    # Persist the FINAL modelled gene list. Downstream stages that describe
    # properties of the prediction targets must use the same genes; inferring
    # them from a differently filtered matrix reintroduces the mismatch that
    # development-only selection was designed to remove.
    pd.Series(genes, name="gene").to_frame().to_csv(
        table_path(cfg, "modelled_genes.csv"), index=False)
    LOG.info("Final modelled gene list (%d) -> modelled_genes.csv", len(genes))

    X_dev, Y_dev, cov_dev = X_all[dev_idx], Y_all[dev_idx], cov[dev_idx]
    shared_dev = [shared[i] for i in dev_idx]
    LOG.info("Development set: %d samples; held out: %d", len(dev_idx), int(is_held.sum()))

    schemes = {}
    scheme_arg = args.scheme or mcfg["cv_scheme"]
    if scheme_arg == "leave_one_study_out" and "study" in st.columns:
        schemes["leave_one_study_out"] = list(group_splits(st_dev["study"]))
    if scheme_arg == "leave_one_country_out" and "country" in st.columns:
        schemes["leave_one_country_out"] = list(group_splits(st_dev["country"]))
    schemes["random"] = list(random_splits(len(shared_dev), k=5, seed=mcfg["seed"]))

    # Subject-grouped CV. Without it the random-vs-LOSO gap conflates subject
    # leakage with cohort shift; 46% of adult samples are repeat measurements.
    _subj_col = "subject_uid" if "subject_uid" in st_dev.columns else (
        "subject_id" if "subject_id" in st_dev.columns else None)
    if _subj_col is None:
        LOG.warning("No subject column; subject-grouped CV skipped. Rerun s02 "
                    "to retain subject_id, or the validation ladder cannot "
                    "distinguish subject leakage from cohort shift.")
    else:
        _subj = st_dev[_subj_col].astype(str)
        _n_rep = int((_subj.value_counts() > 1).sum())
        LOG.info("Subject column '%s': %d subjects, %d with repeat samples "
                 "(%.1f%% of samples are repeats)", _subj_col, _subj.nunique(),
                 _n_rep, 100.0 * _subj.map(_subj.value_counts()).gt(1).mean())
        schemes["subject_grouped"] = list(
            subject_splits(_subj, k=5, seed=mcfg["seed"]))

    all_metrics, all_preds = [], []

    if args.skip_cv:
        mp = work_path(cfg, "cv_metrics.parquet", per_stratum=True)
        pp = work_path(cfg, "cv_predictions.parquet", per_stratum=True)
        if not mp.exists():
            raise SystemExit(f"--skip-cv needs {mp.name}; run without it first.")
        metrics = pd.read_parquet(mp)
        preds_df = pd.read_parquet(pp) if pp.exists() else pd.DataFrame()
        LOG.info("Reusing cross-validation results from %s (%d rows, models %s)",
                 mp.name, len(metrics), sorted(metrics["model"].unique()))
        schemes = {}

    for scheme, splits in schemes.items():
        LOG.info("=== scheme: %s (%d folds) ===", scheme, len(splits))
        for fold, tr, te in splits:
            xs = StandardScaler().fit(X_dev[tr])
            Xtr, Xte = xs.transform(X_dev[tr]), xs.transform(X_dev[te])
            ys = StandardScaler().fit(Y_dev[tr])
            Ytr, Yte = ys.transform(Y_dev[tr]), Y_dev[te]
            Ctr, Cte = cov_dev[tr], cov_dev[te]
            if Ctr.shape[1]:
                cs = StandardScaler().fit(Ctr)
                Ctr, Cte = cs.transform(Ctr), cs.transform(Cte)

            preds = {}
            if "mean" in wanted:
                preds["mean"] = np.tile(Y_dev[tr].mean(axis=0), (te.sum(), 1))

            if "ridge" in wanted:
                ridge = Ridge(alpha=10.0).fit(np.hstack([Xtr, Ctr]), Ytr)
                preds["ridge"] = ys.inverse_transform(ridge.predict(np.hstack([Xte, Cte])))

            if HAVE_TORCH and "taxa_nn" in wanted:
                inner = np.random.default_rng(0).permutation(tr.sum())
                cut = int(0.85 * len(inner))
                i_tr, i_va = inner[:cut], inner[cut:]
                model, _ = fit_taxa_nn(Xtr[i_tr], Ytr[i_tr], Xtr[i_va], Ytr[i_va],
                                       Ctr[i_tr], Ctr[i_va], assign, mcfg)
                dev = next(model.parameters()).device
                with torch.no_grad():
                    p = model(torch.as_tensor(Xte, dtype=torch.float32, device=dev),
                              torch.as_tensor(Cte, dtype=torch.float32, device=dev)).cpu().numpy()
                preds["taxa_nn"] = ys.inverse_transform(p)

            XtrC, XteC = np.hstack([Xtr, Ctr]), np.hstack([Xte, Cte])
            if "rf" in wanted:
                preds["rf"] = ys.inverse_transform(
                    fit_random_forest(XtrC, Ytr, XteC, seed=mcfg["seed"]))
            if "xgb" in wanted:
                preds["xgb"] = ys.inverse_transform(
                    fit_xgboost(XtrC, Ytr, XteC, cfg_model=mcfg, seed=mcfg["seed"]))
            if "ft" in wanted:
                preds["ft"] = ys.inverse_transform(
                    fit_ft_transformer(XtrC, Ytr, XteC, cfg_model=mcfg, seed=mcfg["seed"]))
            if "mlp" in wanted:
                preds["mlp"] = ys.inverse_transform(
                    fit_mlp(XtrC, Ytr, XteC, cfg_model=mcfg, seed=mcfg["seed"]))

            for name, P in preds.items():
                m = score(Yte, P, genes)
                m["model"], m["scheme"], m["fold"], m["n_test"] = name, scheme, fold, int(te.sum())
                all_metrics.append(m)
                all_preds.append(pd.DataFrame({
                    "sample": np.repeat(np.array(shared_dev)[te], len(genes)),
                    "gene": np.tile(genes, te.sum()),
                    "observed": Yte.ravel(),
                    "predicted": P.ravel(),
                    "model": name, "scheme": scheme, "fold": fold,
                }))
            fold_scores = {k: score(Yte, v, genes) for k, v in preds.items()}
            LOG.info("  fold %-28s n=%4d  rho/R2: %s", fold, te.sum(),
                     {k: f"{np.nanmedian(d['spearman']):.3f}/{np.nanmean(d['r2']):.2f}"
                      for k, d in fold_scores.items()})

    if not args.skip_cv:
        metrics = pd.concat(all_metrics, ignore_index=True)
        preds_df = pd.concat(all_preds, ignore_index=True)
        metrics.to_parquet(work_path(cfg, "cv_metrics.parquet", per_stratum=True),
                           index=False)
        preds_df.to_parquet(work_path(cfg, "cv_predictions.parquet",
                                      per_stratum=True), index=False)

    summary = (metrics.groupby(["scheme", "model"])
               .agg(median_spearman=("spearman", "median"),
                    mean_r2=("r2", "mean"),
                    median_r2=("r2", "median"),
                    mean_r2_centered=("r2_centered", "mean"),
                    p10_r2=("r2", lambda s: float(np.nanpercentile(s, 10))),
                    frac_genes_rho_gt_0_3=("spearman", lambda s: float((s > 0.3).mean())))
               .reset_index())
    out = table_path(cfg, "model_summary.csv")
    summary.to_csv(out, index=False)
    print("\n" + summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # Paired comparison per gene. A difference in medians is not evidence on
    # its own - the same 99 genes are scored by both models, so the test must
    # be paired, and the effect size matters more than the p-value.
    from scipy.stats import wilcoxon
    for sch in metrics["scheme"].unique():
        sub = metrics[metrics["scheme"] == sch]
        for metric in ("spearman", "r2", "r2_centered"):
            piv = (sub[sub["model"] != "mean"]
                   .groupby(["model", "gene"])[metric].median().unstack("model")).dropna()
            if len(piv) < 10 or piv.shape[1] < 2:
                continue
            ref = "ridge" if "ridge" in piv.columns else piv.columns[0]
            for m in piv.columns:
                if m == ref:
                    continue
                d = piv[m] - piv[ref]
                try:
                    _, pval = wilcoxon(piv[m], piv[ref])
                except ValueError:
                    continue
                LOG.info("%-20s %-11s %-8s vs %-6s: median delta %+.4f, "
                         "wins %d/%d genes, Wilcoxon p=%.2e",
                         sch, metric, m, ref, float(d.median()),
                         int((d > 0).sum()), len(d), pval)

    # ---- ONE evaluation on the locked studies ---------------------------
    if held:
        LOG.info("=" * 62)
        LOG.info("HELD-OUT EVALUATION (%d studies, scored once)", len(held))
        LOG.info("=" * 62)
        hi = np.where(is_held)[0]
        xs = StandardScaler().fit(X_dev)
        ys = StandardScaler().fit(Y_dev)
        Xtr_h, Ytr_h = xs.transform(X_dev), ys.transform(Y_dev)
        Xte_h, Yte_h = xs.transform(X_all[hi]), Y_all[hi]
        Ctr_h, Cte_h = cov_dev, cov[hi]
        if Ctr_h.shape[1]:
            cs = StandardScaler().fit(Ctr_h)
            Ctr_h, Cte_h = cs.transform(Ctr_h), cs.transform(Cte_h)

        hpreds = {}
        if "mean" in wanted:
            hpreds["mean"] = np.tile(Y_dev.mean(axis=0), (len(hi), 1))
        if "ridge" in wanted:
            r = Ridge(alpha=10.0).fit(np.hstack([Xtr_h, Ctr_h]), Ytr_h)
            hpreds["ridge"] = ys.inverse_transform(r.predict(np.hstack([Xte_h, Cte_h])))
        if HAVE_TORCH and "taxa_nn" in wanted:
            inner = np.random.default_rng(0).permutation(len(Xtr_h))
            c = int(0.85 * len(inner))
            mdl, _ = fit_taxa_nn(Xtr_h[inner[:c]], Ytr_h[inner[:c]],
                                 Xtr_h[inner[c:]], Ytr_h[inner[c:]],
                                 Ctr_h[inner[:c]], Ctr_h[inner[c:]], assign, mcfg)
            dv = next(mdl.parameters()).device
            with torch.no_grad():
                pp = mdl(torch.as_tensor(Xte_h, dtype=torch.float32, device=dv),
                         torch.as_tensor(Cte_h, dtype=torch.float32, device=dv)).cpu().numpy()
            hpreds["taxa_nn"] = ys.inverse_transform(pp)
        XtrCh, XteCh = np.hstack([Xtr_h, Ctr_h]), np.hstack([Xte_h, Cte_h])
        if "rf" in wanted:
            hpreds["rf"] = ys.inverse_transform(fit_random_forest(XtrCh, Ytr_h, XteCh))
        if "xgb" in wanted:
            hpreds["xgb"] = ys.inverse_transform(fit_xgboost(XtrCh, Ytr_h, XteCh, cfg_model=mcfg))
        if "ft" in wanted:
            hpreds["ft"] = ys.inverse_transform(
                fit_ft_transformer(XtrCh, Ytr_h, XteCh, cfg_model=mcfg))
        if "mlp" in wanted:
            hpreds["mlp"] = ys.inverse_transform(fit_mlp(XtrCh, Ytr_h, XteCh, cfg_model=mcfg))

        hrows = []
        for name, P in hpreds.items():
            d = score(Yte_h, P, genes)
            d["model"] = name
            hrows.append(d)
        hm = pd.concat(hrows, ignore_index=True)
        hm.to_parquet(work_path(cfg, "holdout_metrics.parquet", per_stratum=True),
                      index=False)

        # Per-study performance and a study-macro average: a pooled median is
        # dominated by whichever held-out study is largest.
        # `st` is already indexed by sample at this point; re-indexing it by a
        # column that no longer exists was what crashed the held-out block.
        hstudy = st.reindex(np.array(shared)[hi])["study"].astype(str).values
        prow = []
        for name, P in hpreds.items():
            for stu in pd.unique(hstudy):
                m_ = hstudy == stu
                if m_.sum() < 10:
                    continue
                d = score(Yte_h[m_], P[m_], genes)
                prow.append({"model": name, "study": stu, "n": int(m_.sum()),
                             "median_spearman": float(np.nanmedian(d["spearman"])),
                             "mean_r2_centered": float(np.nanmean(d["r2_centered"]))})
        if prow:
            ps_df = pd.DataFrame(prow)
            ps_df.to_csv(table_path(cfg, "holdout_by_study.csv"), index=False)
            macro = (ps_df.groupby("model")[["median_spearman", "mean_r2_centered"]]
                          .mean().sort_values("median_spearman", ascending=False))
            print("\nHELD-OUT, STUDY-MACRO AVERAGED (each study weighted equally)\n")
            print(macro.to_string(float_format=lambda v: f"{v:.4f}"))
        hsum = (hm.groupby("model")
                  .agg(median_spearman=("spearman", "median"),
                       mean_r2=("r2", "mean"),
                       mean_r2_centered=("r2_centered", "mean"))
                  .reset_index()
                  .sort_values("median_spearman", ascending=False))
        hsum.to_csv(table_path(cfg, "holdout_summary.csv"), index=False)
        print("\nHELD-OUT STUDIES (n=%d samples, %d studies) - single evaluation\n"
              % (len(hi), len(held)))
        print(hsum.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        LOG.info("These numbers are unbiased by model selection. The CV table "
                 "above is not, once a winner is chosen from it.")

    # final full-data fit for attribution
    if HAVE_TORCH:
        xs = StandardScaler().fit(X_all)
        ys = StandardScaler().fit(Y_all)
        idx = np.random.default_rng(0).permutation(len(shared))
        cut = int(0.9 * len(idx))
        model, _ = fit_taxa_nn(xs.transform(X_all)[idx[:cut]], ys.transform(Y_all)[idx[:cut]],
                               xs.transform(X_all)[idx[cut:]], ys.transform(Y_all)[idx[cut:]],
                               cov[idx[:cut]], cov[idx[cut:]], assign, mcfg)
        torch.save({"state_dict": model.state_dict(),
                    "species": list(taxa.columns), "genes": genes,
                    "taxa_transform": dict(TAXA_TRANSFORM),
                    "x_mean": xs.mean_, "x_scale": xs.scale_,
                    "y_mean": ys.mean_, "y_scale": ys.scale_,
                    "assign": assign, "n_cov": cov.shape[1]},
                   work_path(cfg, "model_taxa_nn.pt", per_stratum=True))
        LOG.info("Saved final model for attribution.")


if __name__ == "__main__":
    main()
