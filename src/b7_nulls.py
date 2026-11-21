"""
b7_nulls.py — Picklable parallel null-model worker for B7 (UMAP + HDBSCAN).

Ensures single-threaded BLAS/OpenMP/Numba inside each spawned worker and uses
per-task seeds from SeedSequence so 1 worker and N workers produce identical
numerical results on Windows, macOS, and Linux.
"""

import os
from concurrent.futures import ProcessPoolExecutor
import numpy as np


def _set_single_thread_env() -> None:
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[var] = "1"


def _run_one_sim(args):
    """Run a single null-model UMAP + HDBSCAN fit with a deterministic per-task seed."""
    _set_single_thread_env()
    import warnings
    warnings.filterwarnings("ignore")
    import umap
    import hdbscan
    from sklearn.metrics import silhouette_score

    Z, kind, seed_int, umap_kw, hdb_kw = args
    rng = np.random.default_rng(seed_int)
    if kind == "perm":
        Zn = np.column_stack([rng.permutation(Z[:, j]) for j in range(Z.shape[1])])
    elif kind == "gauss":
        mu = Z.mean(0)
        cov = np.cov(Z.T)
        Zn = rng.multivariate_normal(mu, cov, size=Z.shape[0], check_valid="ignore")
    else:
        raise ValueError(f"Unknown null kind: {kind}")

    emb = umap.UMAP(n_components=2, random_state=int(seed_int), **umap_kw).fit_transform(Zn)
    lab = hdbscan.HDBSCAN(**hdb_kw).fit_predict(emb)
    ok = lab >= 0
    k = int(len(set(lab[ok])))
    sil = float(silhouette_score(emb[ok], lab[ok])) if ok.sum() > 1 and k > 1 else float("nan")
    return k, sil


def format_p(count_ge: int, n_sim: int) -> str:
    """Format empirical permutation p-value using integer exceedance count."""
    p = (count_ge + 1) / (n_sim + 1)
    if count_ge == 0 and n_sim >= 1000:
        return "p <= 0.001"
    return f"{p:.4f}"


def run_null_suite(
    Z: np.ndarray,
    kind: str,
    n_sim: int,
    base_seed: int,
    umap_kw: dict,
    hdb_kw: dict,
    n_workers: int = 1,
) -> np.ndarray:
    """
    Execute n_sim null simulations with per-task SeedSequence seeds.
    Returns float array of shape (n_sim, 2) with columns [k, silhouette].
    """
    ss = np.random.SeedSequence(base_seed)
    child_seeds = [int(c.generate_state(1)[0]) for c in ss.spawn(n_sim)]
    tasks = [(Z, kind, s, umap_kw, hdb_kw) for s in child_seeds]

    if n_workers <= 1:
        res = [_run_one_sim(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=n_workers, initializer=_set_single_thread_env) as ex:
            res = list(ex.map(_run_one_sim, tasks))
    return np.array(res, dtype=float)
