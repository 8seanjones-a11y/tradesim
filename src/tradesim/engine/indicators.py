"""Causal indicators on numpy arrays. Every output at index i uses only inputs [0..i].
Warm-up values are NaN; comparisons with NaN are False, so no signals fire during warm-up."""
from __future__ import annotations

import numpy as np
from numba import njit


def sma(x: np.ndarray, n: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan)
    if n <= 0 or len(x) < n:
        return out
    c = np.cumsum(np.insert(x, 0, 0.0))
    out[n - 1:] = (c[n:] - c[:-n]) / n
    return out


@njit(cache=True)
def _ema(x, alpha, n):
    out = np.full(x.shape[0], np.nan)
    if x.shape[0] < n:
        return out
    s = 0.0
    for i in range(n):
        s += x[i]
    prev = s / n
    out[n - 1] = prev
    for i in range(n, x.shape[0]):
        prev = alpha * x[i] + (1 - alpha) * prev
        out[i] = prev
    return out


def ema(x: np.ndarray, n: int) -> np.ndarray:
    return _ema(np.asarray(x, dtype=float), 2.0 / (n + 1), n)


def wilder(x: np.ndarray, n: int) -> np.ndarray:
    return _ema(np.asarray(x, dtype=float), 1.0 / n, n)


def rsi(close: np.ndarray, n: int = 14) -> np.ndarray:
    close = np.asarray(close, dtype=float)
    d = np.diff(close, prepend=np.nan)
    up = np.where(d > 0, d, 0.0)[1:]
    dn = np.where(d < 0, -d, 0.0)[1:]
    au, ad = wilder(up, n), wilder(dn, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = 100 - 100 / (1 + au / ad)
    r = np.where(ad == 0, 100.0, r)
    return np.concatenate([[np.nan], r])


def true_range(high, low, close) -> np.ndarray:
    high, low, close = (np.asarray(a, dtype=float) for a in (high, low, close))
    pc = np.roll(close, 1)
    pc[0] = np.nan
    tr = np.nanmax(np.vstack([high - low, np.abs(high - pc), np.abs(low - pc)]), axis=0)
    return tr


def atr(high, low, close, n: int = 14) -> np.ndarray:
    return wilder(true_range(high, low, close), n)


def rolling_max(x, n: int) -> np.ndarray:
    from numpy.lib.stride_tricks import sliding_window_view
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan)
    if len(x) >= n:
        out[n - 1:] = sliding_window_view(x, n).max(axis=1)
    return out


def rolling_min(x, n: int) -> np.ndarray:
    from numpy.lib.stride_tricks import sliding_window_view
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan)
    if len(x) >= n:
        out[n - 1:] = sliding_window_view(x, n).min(axis=1)
    return out


def rolling_std(x, n: int) -> np.ndarray:
    from numpy.lib.stride_tricks import sliding_window_view
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan)
    if len(x) >= n:
        out[n - 1:] = sliding_window_view(x, n).std(axis=1, ddof=1)
    return out


def shift(x, k: int = 1) -> np.ndarray:
    """Value from k bars ago (k>0). Never use negative k: that is lookahead."""
    if k < 0:
        raise ValueError("shift(k<0) would look into the future")
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan)
    out[k:] = x[: len(x) - k] if k else x
    return out


def crossover(a, b) -> np.ndarray:
    a, b = np.asarray(a, dtype=float), np.broadcast_to(np.asarray(b, dtype=float), np.shape(a))
    pa, pb = shift(a), shift(b)
    return (a > b) & (pa <= pb)


def crossunder(a, b) -> np.ndarray:
    a, b = np.asarray(a, dtype=float), np.broadcast_to(np.asarray(b, dtype=float), np.shape(a))
    pa, pb = shift(a), shift(b)
    return (a < b) & (pa >= pb)


def bollinger(close, n: int = 20, k: float = 2.0):
    m = sma(close, n)
    s = rolling_std(close, n)
    return m - k * s, m, m + k * s
