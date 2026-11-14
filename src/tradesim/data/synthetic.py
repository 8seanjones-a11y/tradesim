"""Synthetic price generators: used for tests, null-model ("is the edge just noise?") checks,
and Monte Carlo price-path simulation."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl


def _ohlc_from_close(close: np.ndarray, rng: np.random.Generator, vol: np.ndarray | float) -> dict:
    n = len(close)
    open_ = np.empty(n)
    open_[0] = close[0]
    open_[1:] = close[:-1] * np.exp(rng.normal(0, np.asarray(vol) * 0.1, n - 1) if np.ndim(vol) == 0
                                    else rng.normal(0, 1, n - 1) * np.asarray(vol)[1:] * 0.1)
    span = np.abs(rng.normal(0, 1, n)) * np.asarray(vol) * close * 0.6
    high = np.maximum(open_, close) + span * rng.uniform(0, 1, n)
    low = np.minimum(open_, close) - span * rng.uniform(0, 1, n)
    return dict(open=open_, high=high, low=low, close=close)


def _frame(ohlc: dict, start: datetime, step: timedelta) -> pl.DataFrame:
    n = len(ohlc["close"])
    ts = pl.datetime_range(start, start + step * (n - 1), step, time_zone="UTC", eager=True)
    return pl.DataFrame({"ts": ts, **ohlc, "volume": np.full(n, 1000.0)})


def gbm(n: int = 2000, s0: float = 100.0, mu: float = 0.0, sigma: float = 0.2, dt: float = 1 / 252,
        seed: int | None = 0, start: datetime | None = None, step: timedelta = timedelta(days=1)) -> pl.DataFrame:
    """Geometric Brownian motion bars (annualised mu/sigma)."""
    rng = np.random.default_rng(seed)
    r = rng.normal((mu - 0.5 * sigma**2) * dt, sigma * np.sqrt(dt), n)
    close = s0 * np.exp(np.cumsum(r))
    return _frame(_ohlc_from_close(close, rng, sigma * np.sqrt(dt)),
                  start or datetime(2015, 1, 1, tzinfo=timezone.utc), step)


def garch(n: int = 2000, s0: float = 100.0, omega: float = 2e-6, alpha: float = 0.08, beta: float = 0.9,
          mu: float = 0.0, seed: int | None = 0, start: datetime | None = None,
          step: timedelta = timedelta(days=1), t_df: float | None = 5.0) -> pl.DataFrame:
    """GARCH(1,1) returns with optional Student-t shocks (fat tails + volatility clustering)."""
    rng = np.random.default_rng(seed)
    var = omega / max(1e-12, 1 - alpha - beta)
    r = np.empty(n)
    vols = np.empty(n)
    for i in range(n):
        z = rng.standard_t(t_df) / np.sqrt(t_df / (t_df - 2)) if t_df else rng.normal()
        vols[i] = np.sqrt(var)
        r[i] = mu + vols[i] * z
        var = omega + alpha * r[i] ** 2 + beta * var
    close = s0 * np.exp(np.cumsum(r))
    return _frame(_ohlc_from_close(close, rng, vols), start or datetime(2015, 1, 1, tzinfo=timezone.utc), step)


def jump_diffusion(n: int = 2000, s0: float = 100.0, mu: float = 0.0, sigma: float = 0.15,
                   jump_rate: float = 5.0, jump_mean: float = -0.02, jump_std: float = 0.04,
                   dt: float = 1 / 252, seed: int | None = 0, start: datetime | None = None,
                   step: timedelta = timedelta(days=1)) -> pl.DataFrame:
    """Merton jump-diffusion: GBM plus Poisson jumps (gaps through stops)."""
    rng = np.random.default_rng(seed)
    diff = rng.normal((mu - 0.5 * sigma**2) * dt, sigma * np.sqrt(dt), n)
    nj = rng.poisson(jump_rate * dt, n)
    jumps = np.array([rng.normal(jump_mean, jump_std, k).sum() if k else 0.0 for k in nj])
    close = s0 * np.exp(np.cumsum(diff + jumps))
    return _frame(_ohlc_from_close(close, rng, sigma * np.sqrt(dt)),
                  start or datetime(2015, 1, 1, tzinfo=timezone.utc), step)


def regime_switching(n: int = 2000, s0: float = 100.0,
                     mus=(0.15, -0.25, 0.0), sigmas=(0.12, 0.35, 0.18),
                     transition=((0.98, 0.01, 0.01), (0.03, 0.95, 0.02), (0.02, 0.01, 0.97)),
                     dt: float = 1 / 252, seed: int | None = 0, start: datetime | None = None,
                     step: timedelta = timedelta(days=1)) -> tuple[pl.DataFrame, np.ndarray]:
    """Markov regime-switching (bull / bear / chop by default). Returns (bars, regime_ids)."""
    rng = np.random.default_rng(seed)
    P = np.asarray(transition)
    k = len(mus)
    states = np.empty(n, dtype=int)
    s = 0
    for i in range(n):
        states[i] = s
        s = rng.choice(k, p=P[s])
    mu = np.asarray(mus)[states]
    sg = np.asarray(sigmas)[states]
    r = rng.normal((mu - 0.5 * sg**2) * dt, sg * np.sqrt(dt))
    close = s0 * np.exp(np.cumsum(r))
    bars = _frame(_ohlc_from_close(close, rng, sg * np.sqrt(dt)), start or datetime(2015, 1, 1, tzinfo=timezone.utc), step)
    return bars, states


def bootstrap_bars(bars: pl.DataFrame, block: int = 20, seed: int | None = 0) -> pl.DataFrame:
    """Synthetic history by stationary-block-resampling the real bar log-returns (keeps
    vol clustering and the empirical distribution, destroys long-range timing)."""
    rng = np.random.default_rng(seed)
    c = bars["close"].to_numpy()
    o, h, l = (bars[k].to_numpy() for k in ("open", "high", "low"))
    prev = np.concatenate([[c[0]], c[:-1]])
    rel = np.vstack([np.log(o / prev), np.log(h / prev), np.log(l / prev), np.log(c / prev)]).T
    n = len(c)
    idx = np.empty(n, dtype=int)
    i = 0
    while i < n:
        start = rng.integers(1, n)
        length = rng.geometric(1 / block)
        for j in range(length):
            if i >= n:
                break
            idx[i] = 1 + (start - 1 + j) % (n - 1)
            i += 1
    rr = rel[idx]
    close = c[0] * np.exp(np.cumsum(rr[:, 3]))
    prevc = np.concatenate([[c[0]], close[:-1]])
    return bars.with_columns(open=prevc * np.exp(rr[:, 0]), high=prevc * np.exp(rr[:, 1]),
                             low=prevc * np.exp(rr[:, 2]), close=close)
