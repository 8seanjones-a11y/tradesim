"""Open-market (own-account) evaluation: horizon outcome distributions, risk of ruin,
position sizing (Kelly / optimal-f), capital requirements, benchmark comparison."""
from __future__ import annotations

import math

import numpy as np
import polars as pl

from .metrics import daily_returns, periods_per_year, sharpe
from .montecarlo import stationary_bootstrap_idx


def horizon_outcomes(result, horizons_years=(1, 3, 5), n_sims: int = 5000, mean_block: float = 10.0,
                     ruin_dd: float = 0.5, seed: int = 42) -> pl.DataFrame:
    """Block-bootstrap daily returns forward over each horizon. Per horizon: CAGR percentiles,
    P(loss), P(drawdown >= ruin_dd), median max drawdown."""
    r = daily_returns(result.equity)
    ppy = periods_per_year(result.equity)
    rng = np.random.default_rng(seed)
    rows = []
    if len(r) < 10:
        return pl.DataFrame()
    for hy in horizons_years:
        n = max(1, int(round(hy * ppy)))
        idx = np.stack([stationary_bootstrap_idx(len(r), n, mean_block, rng) for _ in range(n_sims)])
        paths = np.cumprod(1 + r[idx], axis=1)
        final = paths[:, -1]
        peak = np.maximum.accumulate(np.maximum(paths, 1.0), axis=1)
        mdd = ((peak - paths) / peak).max(axis=1)
        cagr = np.where(final > 0, final ** (1 / hy) - 1, -1.0)
        rows.append(dict(horizon_years=hy, cagr_p5=float(np.percentile(cagr, 5)), cagr_p50=float(np.median(cagr)),
                         cagr_p95=float(np.percentile(cagr, 95)), p_loss=float((final < 1).mean()),
                         p_dd_ge_ruin=float((mdd >= ruin_dd).mean()), mdd_p50=float(np.median(mdd)),
                         mdd_p95=float(np.percentile(mdd, 95))))
    return pl.DataFrame(rows)


def risk_of_ruin(pnl: np.ndarray, capital: float, ruin_frac: float = 0.5, n_trades: int | None = None,
                 n_sims: int = 5000, seed: int = 42) -> float:
    """MC probability of losing `ruin_frac` of capital within n_trades (default: 4x history)."""
    pnl = np.asarray(pnl, float)
    if len(pnl) == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    n = n_trades or 4 * len(pnl)
    eq = capital + np.cumsum(rng.choice(pnl, (n_sims, n)), axis=1)
    return float((eq.min(axis=1) <= capital * (1 - ruin_frac)).mean())


def kelly_fraction(trade_returns: np.ndarray) -> dict:
    """Kelly for a discrete trade-return distribution (returns as fraction of capital risked per trade
    at the backtest's sizing). Returns full Kelly multiplier on current size plus half-Kelly."""
    x = np.asarray(trade_returns, float)
    if len(x) < 5 or x.mean() <= 0:
        return dict(kelly_mult=0.0, half_kelly_mult=0.0, growth_at_kelly=0.0)
    fs = np.linspace(0.01, 50, 2000)
    lo = x.min()
    if lo < 0:
        fs = fs[fs < 0.999 / -lo]
    g = np.array([np.mean(np.log1p(f * x)) for f in fs])
    k = float(fs[int(np.argmax(g))])
    return dict(kelly_mult=k, half_kelly_mult=k / 2, growth_at_kelly=float(g.max()))


def optimal_f(pnl: np.ndarray) -> dict:
    """Ralph Vince optimal-f: fraction of capital to risk per 'largest loss' unit."""
    pnl = np.asarray(pnl, float)
    if len(pnl) < 5 or pnl.mean() <= 0 or pnl.min() >= 0:
        return dict(optimal_f=0.0, twr=1.0)
    worst = -pnl.min()
    fs = np.linspace(0.01, 0.99, 99)
    twr = np.array([np.prod(1 + f * pnl / worst) for f in fs])
    i = int(np.argmax(twr))
    return dict(optimal_f=float(fs[i]), twr=float(twr[i]), worst_loss=float(worst))


def capital_needed(pnl: np.ndarray, dd_tolerance: float, confidence: float = 0.95, n_sims: int = 5000,
                   seed: int = 42) -> float:
    """Account size so that the `confidence` percentile MC max drawdown (in $) stays within
    dd_tolerance of starting capital at the backtest's fixed size."""
    pnl = np.asarray(pnl, float)
    if len(pnl) == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    eq = np.cumsum(rng.choice(pnl, (n_sims, len(pnl))), axis=1)
    peak = np.maximum.accumulate(np.maximum(eq, 0), axis=1)
    dd = (peak - eq).max(axis=1)
    return float(np.percentile(dd, confidence * 100) / dd_tolerance)


def benchmark_compare(result, bench_bars: pl.DataFrame | None = None, rf: float = 0.0) -> dict:
    """Compare against buy-and-hold of `bench_bars` (defaults to the traded instrument itself)."""
    bars = bench_bars if bench_bars is not None else result.bars
    eq = result.equity.group_by("day", maintain_order=True).agg(pl.col("ts").last(), pl.col("equity").last())
    eq = eq.with_columns(pl.col("ts").dt.date().alias("d"))
    b = bars.with_columns(pl.col("ts").dt.date().alias("d")).group_by("d", maintain_order=True).agg(pl.col("close").last())
    j = eq.join(b, on="d", how="inner").sort("d")
    if j.height < 10:
        return {}
    s = j["equity"].to_numpy()
    m = j["close"].to_numpy()
    rs, rm = s[1:] / s[:-1] - 1, m[1:] / m[:-1] - 1
    ppy = periods_per_year(result.equity)
    cov = np.cov(rs, rm)
    beta = float(cov[0, 1] / cov[1, 1]) if cov[1, 1] > 0 else 0.0
    alpha = float((rs.mean() - beta * rm.mean()) * ppy)
    active = rs - rm
    ir = float(active.mean() / active.std(ddof=1) * math.sqrt(ppy)) if active.std(ddof=1) > 0 else 0.0
    years = max(1e-9, len(rs) / ppy)
    return dict(
        bench_total_return=float(m[-1] / m[0] - 1), strat_total_return=float(s[-1] / s[0] - 1),
        bench_cagr=float((m[-1] / m[0]) ** (1 / years) - 1), bench_sharpe=sharpe(rm, ppy, rf),
        alpha_ann=alpha, beta=beta, correlation=float(np.corrcoef(rs, rm)[0, 1]), information_ratio=ir,
        bench_curve=pl.DataFrame({"ts": j["ts"], "bench_equity": m / m[0] * result.initial_capital}),
    )
