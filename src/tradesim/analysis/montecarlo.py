"""Trade-level Monte Carlo: shuffle, bootstrap, stationary-block bootstrap, execution noise.

All functions return a dict of numpy arrays (one value per simulation) plus percentile
equity "fan" bands, so reports can plot them directly.
"""
from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def _path_stats(pnl_matrix, capital, ruin_level):
    n_sims, n = pnl_matrix.shape
    final = np.empty(n_sims)
    max_dd = np.empty(n_sims)
    max_dd_abs = np.empty(n_sims)
    ruined = np.zeros(n_sims, dtype=np.bool_)
    max_consec = np.empty(n_sims, dtype=np.int64)
    for s in range(n_sims):
        eq = capital
        peak = capital
        mdd = 0.0
        mdda = 0.0
        cl = 0
        best = 0
        for i in range(n):
            x = pnl_matrix[s, i]
            eq += x
            if eq > peak:
                peak = eq
            dda = peak - eq
            if dda > mdda:
                mdda = dda
            if peak > 0 and dda / peak > mdd:
                mdd = dda / peak
            if eq <= capital * (1 - ruin_level):
                ruined[s] = True
            if x <= 0:
                cl += 1
                if cl > best:
                    best = cl
            else:
                cl = 0
        final[s] = eq
        max_dd[s] = mdd
        max_dd_abs[s] = mdda
        max_consec[s] = best
    return final, max_dd, max_dd_abs, ruined, max_consec


def _fan(pnl_matrix: np.ndarray, capital: float, qs=(5, 25, 50, 75, 95), max_points: int = 300) -> dict:
    eq = capital + np.cumsum(pnl_matrix, axis=1)
    step = max(1, eq.shape[1] // max_points)
    eq = eq[:, ::step]
    return {f"p{q}": np.percentile(eq, q, axis=0) for q in qs} | {"x": np.arange(0, pnl_matrix.shape[1], step)}


def _summarize(mat: np.ndarray, capital: float, ruin_level: float, method: str) -> dict:
    final, mdd, mdda, ruined, mcl = _path_stats(mat, float(capital), float(ruin_level))
    return dict(method=method, n_sims=mat.shape[0], n_trades=mat.shape[1], final=final, max_dd=mdd,
                max_dd_abs=mdda, ruined=ruined, max_consec_losses=mcl,
                p_loss=float((final < capital).mean()), p_ruin=float(ruined.mean()),
                fan=_fan(mat, capital))


def shuffle_mc(pnl: np.ndarray, capital: float, n_sims: int = 5000, ruin_level: float = 0.5, seed: int = 42) -> dict:
    """Reorder realised trades. Total P&L identical; answers 'how bad could the path have been?'"""
    rng = np.random.default_rng(seed)
    pnl = np.asarray(pnl, float)
    mat = np.stack([rng.permutation(pnl) for _ in range(n_sims)]) if len(pnl) else np.zeros((n_sims, 0))
    return _summarize(mat, capital, ruin_level, "shuffle")


def bootstrap_mc(pnl: np.ndarray, capital: float, n_sims: int = 5000, n_trades: int | None = None,
                 ruin_level: float = 0.5, seed: int = 42) -> dict:
    """IID resample trades with replacement. Gives confidence intervals on final P&L / DD."""
    rng = np.random.default_rng(seed)
    pnl = np.asarray(pnl, float)
    n = n_trades or len(pnl)
    mat = rng.choice(pnl, size=(n_sims, n), replace=True) if len(pnl) else np.zeros((n_sims, 0))
    return _summarize(mat, capital, ruin_level, "bootstrap")


def stationary_bootstrap_idx(n: int, n_out: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    """Politis-Romano stationary bootstrap indices (geometric block lengths, circular)."""
    idx = np.empty(n_out, dtype=np.int64)
    i = 0
    while i < n_out:
        start = rng.integers(0, n)
        L = rng.geometric(1.0 / mean_block)
        take = min(L, n_out - i)
        idx[i:i + take] = (start + np.arange(take)) % n
        i += take
    return idx


def block_bootstrap_mc(pnl: np.ndarray, capital: float, n_sims: int = 5000, mean_block: float = 5.0,
                       n_out: int | None = None, ruin_level: float = 0.5, seed: int = 42) -> dict:
    """Stationary block bootstrap: preserves streaks / serial dependence in the trade sequence."""
    rng = np.random.default_rng(seed)
    pnl = np.asarray(pnl, float)
    n = len(pnl)
    m = n_out or n
    mat = np.stack([pnl[stationary_bootstrap_idx(n, m, mean_block, rng)] for _ in range(n_sims)]) if n else np.zeros((n_sims, 0))
    return _summarize(mat, capital, ruin_level, "block_bootstrap")


def execution_noise_mc(trades, capital: float, point_value: float, tick_size: float, n_sims: int = 2000,
                       slip_ticks_max: float = 2.0, commission_jitter: float = 0.5, skip_prob: float = 0.1,
                       ruin_level: float = 0.5, seed: int = 42) -> dict:
    """Perturb execution: random extra slippage (0..max ticks per side), commission x(1±jitter),
    and randomly skip trades (missed fills / downtime). Answers 'does the edge survive real fills?'"""
    rng = np.random.default_rng(seed)
    pnl = trades["pnl"].to_numpy()
    units = trades["units"].to_numpy()
    costs = trades["costs"].to_numpy()
    n = len(pnl)
    if n == 0:
        return _summarize(np.zeros((n_sims, 0)), capital, ruin_level, "execution_noise")
    extra_slip = rng.uniform(0, slip_ticks_max, (n_sims, n)) * 2 * tick_size * units * point_value
    comm = costs * rng.uniform(-commission_jitter, commission_jitter, (n_sims, n))
    keep = rng.random((n_sims, n)) >= skip_prob
    mat = (pnl - extra_slip - comm) * keep
    return _summarize(mat, capital, ruin_level, "execution_noise")


def percentiles(x: np.ndarray, qs=(5, 25, 50, 75, 95)) -> dict:
    x = np.asarray(x, float)
    return {f"p{q}": float(np.percentile(x, q)) for q in qs} if len(x) else {}
