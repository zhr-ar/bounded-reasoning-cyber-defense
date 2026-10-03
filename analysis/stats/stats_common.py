"""Shared statistics helpers for the paper analyses.

Uses SciPy when available; otherwise falls back to NumPy-only approximations
so HITL scripts still run in minimal environments.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

try:
    from scipy import stats as _scipy_stats

    _HAS_SCIPY = True
except ImportError:  # pragma: no cover
    _scipy_stats = None
    _HAS_SCIPY = False


def mean_ci(x: Sequence[float], alpha: float = 0.05) -> dict:
    arr = np.asarray(x, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    mean = float(np.mean(arr)) if n else float("nan")
    if n < 2:
        return {"n": n, "mean": mean, "ci_low": mean, "ci_high": mean, "std": float("nan"), "sem": float("nan")}
    std = float(np.std(arr, ddof=1))
    sem = std / np.sqrt(n)
    if _HAS_SCIPY:
        tcrit = float(_scipy_stats.t.ppf(1 - alpha / 2, df=n - 1))
    else:
        # Normal approx; fine for n≈40 reporting when SciPy missing.
        tcrit = 1.96
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "sem": float(sem),
        "ci_low": mean - tcrit * sem,
        "ci_high": mean + tcrit * sem,
    }


def bootstrap_mean_ci(x: Sequence[float], n_boot: int = 5000, alpha: float = 0.05, seed: int = 0) -> dict:
    arr = np.asarray(x, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for i in range(n_boot):
        means[i] = np.mean(rng.choice(arr, size=n, replace=True))
    low, high = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {"n": n, "mean": float(np.mean(arr)), "ci_low": float(low), "ci_high": float(high)}


def hedges_g(a: Sequence[float], b: Sequence[float]) -> float:
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return float("nan")
    mx, my = np.mean(x), np.mean(y)
    sx, sy = np.var(x, ddof=1), np.var(y, ddof=1)
    sp = np.sqrt(((nx - 1) * sx + (ny - 1) * sy) / (nx + ny - 2))
    if sp == 0:
        return 0.0
    g = (mx - my) / sp
    j = 1 - (3 / (4 * (nx + ny) - 9))
    return float(g * j)


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    gt = sum(1 for xi in x for yj in y if xi > yj)
    lt = sum(1 for xi in x for yj in y if xi < yj)
    return float((gt - lt) / (len(x) * len(y)))


def welch_ttest(a: Sequence[float], b: Sequence[float]) -> dict:
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if len(x) < 2 or len(y) < 2:
        return {"stat": float("nan"), "pvalue": float("nan")}
    if _HAS_SCIPY:
        stat, p = _scipy_stats.ttest_ind(x, y, equal_var=False)
        return {"stat": float(stat), "pvalue": float(p)}
    # Manual Welch
    nx, ny = len(x), len(y)
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    stat = (np.mean(x) - np.mean(y)) / np.sqrt(vx / nx + vy / ny)
    # Two-sided normal approx for p
    from math import erfc, sqrt

    p = float(erfc(abs(stat) / sqrt(2.0)))
    return {"stat": float(stat), "pvalue": p}


def mannwhitney(a: Sequence[float], b: Sequence[float]) -> dict:
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if len(x) < 1 or len(y) < 1:
        return {"stat": float("nan"), "pvalue": float("nan")}
    if _HAS_SCIPY:
        stat, p = _scipy_stats.mannwhitneyu(x, y, alternative="two-sided")
        return {"stat": float(stat), "pvalue": float(p)}
    # Rank-sum style fallback via brute force U; p from normal approx
    hits = sum(1 for xi in x for yj in y if xi > yj) + 0.5 * sum(1 for xi in x for yj in y if xi == yj)
    u = float(hits)
    mu = len(x) * len(y) / 2.0
    sigma = np.sqrt(len(x) * len(y) * (len(x) + len(y) + 1) / 12.0)
    z = 0.0 if sigma == 0 else (u - mu) / sigma
    from math import erfc, sqrt

    p = float(erfc(abs(z) / sqrt(2.0)))
    return {"stat": u, "pvalue": p}


def paired_wilcoxon(a: Sequence[float], b: Sequence[float]) -> dict:
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    d = x - y
    d = d[~np.isnan(d)]
    if len(d) < 1 or np.allclose(d, 0):
        return {"stat": float("nan"), "pvalue": float("nan")}
    if _HAS_SCIPY:
        stat, p = _scipy_stats.wilcoxon(d)
        return {"stat": float(stat), "pvalue": float(p)}
    # Sign test fallback
    pos = int(np.sum(d > 0))
    neg = int(np.sum(d < 0))
    n = pos + neg
    if n == 0:
        return {"stat": 0.0, "pvalue": 1.0}
    # Exact binomial two-sided
    k = min(pos, neg)
    # cumulative binomial under p=0.5
    from math import comb

    cdf = sum(comb(n, i) for i in range(0, k + 1)) / (2**n)
    p = min(1.0, 2 * cdf)
    return {"stat": float(k), "pvalue": float(p)}


def benjamini_hochberg(pvalues: Iterable[float], alpha: float = 0.05) -> list[dict]:
    ps = [float(p) for p in pvalues]
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    qvals = [0.0] * m
    prev = 1.0
    for rank in range(m, 0, -1):
        idx = order[rank - 1]
        q = min(prev, ps[idx] * m / rank)
        qvals[idx] = q
        prev = q
    return [
        {"index": i, "pvalue": ps[i], "qvalue": float(min(qvals[i], 1.0)), "reject": qvals[i] <= alpha}
        for i in range(m)
    ]


def permutation_mean_diff(
    a: Sequence[float],
    b: Sequence[float],
    n_perm: int = 5000,
    seed: int = 0,
) -> dict:
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    obs = float(np.mean(x) - np.mean(y))
    pooled = np.concatenate([x, y])
    nx = len(x)
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        diff = float(np.mean(pooled[:nx]) - np.mean(pooled[nx:]))
        if abs(diff) >= abs(obs):
            count += 1
    return {"diff": obs, "pvalue": (count + 1) / (n_perm + 1)}
