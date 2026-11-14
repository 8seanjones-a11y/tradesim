"""Overfitting / data-snooping statistics.

* Probabilistic & Deflated Sharpe Ratio (Bailey & Lopez de Prado 2012/2014)
* Probability of Backtest Overfitting via CSCV (Bailey, Borwein, Lopez de Prado, Zhu 2015)
* White's Reality Check (bootstrap) for the best of many strategy variants
"""
from __future__ import annotations

import itertools
import math

import numpy as np
from scipy import stats

EULER = 0.5772156649015329


def probabilistic_sharpe(sr: float, n_obs: int, skew: float = 0.0, kurt: float = 3.0, sr_benchmark: float = 0.0) -> float:
    """P(true SR > benchmark) given an observed per-period SR (NOT annualised)."""
    if n_obs < 2:
        return 0.5
    denom = math.sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4 * sr**2))
    return float(stats.norm.cdf((sr - sr_benchmark) * math.sqrt(n_obs - 1) / denom))


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """Expected maximum per-period SR among n_trials unskilled strategies."""
    if n_trials <= 1:
        return 0.0
    z1 = stats.norm.ppf(1 - 1 / n_trials)
    z2 = stats.norm.ppf(1 - 1 / (n_trials * math.e))
    return math.sqrt(max(sr_variance, 0.0)) * ((1 - EULER) * z1 + EULER * z2)


def deflated_sharpe(returns: np.ndarray, n_trials: int = 1, trial_sharpes: np.ndarray | None = None) -> dict:
    """DSR: probability the strategy's Sharpe is real after accounting for how many variants
    were tried. `trial_sharpes` (per-period SRs of all variants tried) gives the variance; if not
    supplied, uses the SR's own estimation variance as a conservative proxy."""
    r = np.asarray(returns, float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 3 or r.std(ddof=1) == 0:
        return dict(sr=0.0, psr=0.5, dsr=0.0, sr0=0.0, n_trials=n_trials)
    sr = r.mean() / r.std(ddof=1)
    sk = float(stats.skew(r))
    ku = float(stats.kurtosis(r, fisher=False))
    if trial_sharpes is not None and len(trial_sharpes) > 1:
        var = float(np.var(trial_sharpes, ddof=1))
        n_trials = max(n_trials, len(trial_sharpes))
    else:
        var = (1 - sk * sr + (ku - 1) / 4 * sr**2) / (n - 1)
    sr0 = expected_max_sharpe(n_trials, var)
    return dict(sr=float(sr), psr=probabilistic_sharpe(sr, n, sk, ku, 0.0),
                dsr=probabilistic_sharpe(sr, n, sk, ku, sr0), sr0=float(sr0), n_trials=int(n_trials))


def pbo_cscv(perf_matrix: np.ndarray, n_splits: int = 10, metric=None) -> dict:
    """Probability of Backtest Overfitting.

    perf_matrix: shape (T periods, N strategy variants) of per-period returns/P&L.
    Splits time into `n_splits` blocks, evaluates every combination of half the blocks as
    in-sample, picks the IS-best variant and records its OOS rank. PBO = P(OOS rank below median).
    """
    M = np.asarray(perf_matrix, float)
    T, N = M.shape
    if N < 2 or T < n_splits * 2:
        return dict(pbo=float("nan"), logits=np.array([]), n_combos=0)
    metric = metric or (lambda x: x.mean(axis=0) / (x.std(axis=0, ddof=1) + 1e-12))
    blocks = np.array_split(np.arange(T), n_splits)
    logits = []
    combos = list(itertools.combinations(range(n_splits), n_splits // 2))
    if len(combos) > 2000:
        rng = np.random.default_rng(0)
        combos = [combos[i] for i in rng.choice(len(combos), 2000, replace=False)]
    for is_blocks in combos:
        is_idx = np.concatenate([blocks[b] for b in is_blocks])
        oos_idx = np.concatenate([blocks[b] for b in range(n_splits) if b not in is_blocks])
        is_perf, oos_perf = metric(M[is_idx]), metric(M[oos_idx])
        best = int(np.argmax(is_perf))
        rank = stats.rankdata(oos_perf)[best] / (N + 1)
        logits.append(math.log(rank / (1 - rank)))
    logits = np.array(logits)
    return dict(pbo=float((logits <= 0).mean()), logits=logits, n_combos=len(combos))


def whites_reality_check(perf_matrix: np.ndarray, n_boot: int = 2000, mean_block: float = 10.0, seed: int = 0) -> dict:
    """p-value that the best of N variants beats zero (benchmark) by more than luck.
    perf_matrix: (T, N) per-period excess returns relative to the benchmark."""
    from .montecarlo import stationary_bootstrap_idx
    M = np.asarray(perf_matrix, float)
    T, N = M.shape
    rng = np.random.default_rng(seed)
    mu = M.mean(axis=0)
    stat = math.sqrt(T) * mu.max()
    centred = M - mu
    count = 0
    for _ in range(n_boot):
        idx = stationary_bootstrap_idx(T, T, mean_block, rng)
        if math.sqrt(T) * centred[idx].mean(axis=0).max() >= stat:
            count += 1
    return dict(stat=float(stat), p_value=(count + 1) / (n_boot + 1), best_variant=int(np.argmax(mu)))
