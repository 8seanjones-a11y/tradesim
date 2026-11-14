"""Performance metrics from equity curves and trade lists."""
from __future__ import annotations

import math

import numpy as np
import polars as pl
from scipy import stats


def drawdown(equity: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Returns (dd series as fraction of peak, max dd fraction, max dd absolute)."""
    eq = np.asarray(equity, dtype=float)
    peak = np.maximum.accumulate(eq)
    dd_abs = peak - eq
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = np.where(peak > 0, dd_abs / peak, 0.0)
    return dd, float(dd.max(initial=0.0)), float(dd_abs.max(initial=0.0))


def longest_underwater(equity: np.ndarray) -> int:
    eq = np.asarray(equity, dtype=float)
    peak = np.maximum.accumulate(eq)
    under = eq < peak
    best = cur = 0
    for u in under:
        cur = cur + 1 if u else 0
        best = max(best, cur)
    return best


def daily_returns(eq_df: pl.DataFrame) -> np.ndarray:
    """Close-of-day equity returns using the `day` column."""
    d = eq_df.group_by("day", maintain_order=True).agg(pl.col("equity").last())["equity"].to_numpy()
    if len(d) < 2:
        return np.array([])
    return d[1:] / d[:-1] - 1


def periods_per_year(eq_df: pl.DataFrame) -> float:
    ts = eq_df["ts"]
    days = eq_df["day"].n_unique()
    span = (ts[-1] - ts[0]).total_seconds() / 86400 / 365.25
    return days / span if span > 0 else 252.0


def sharpe(r: np.ndarray, ppy: float, rf: float = 0.0) -> float:
    r = np.asarray(r, dtype=float)
    if len(r) < 2 or r.std(ddof=1) == 0:
        return 0.0
    ex = r - rf / ppy
    return float(ex.mean() / r.std(ddof=1) * math.sqrt(ppy))


def sortino(r: np.ndarray, ppy: float, rf: float = 0.0) -> float:
    r = np.asarray(r, dtype=float)
    dn = r[r < 0]
    if len(r) < 2 or len(dn) == 0:
        return 0.0
    dd = math.sqrt((dn**2).sum() / len(r))
    return float((r.mean() - rf / ppy) / dd * math.sqrt(ppy)) if dd > 0 else 0.0


def trade_stats(pnl: np.ndarray) -> dict:
    pnl = np.asarray(pnl, dtype=float)
    n = len(pnl)
    if n == 0:
        return dict(n_trades=0, win_rate=0.0, profit_factor=0.0, avg_trade=0.0, avg_win=0.0,
                    avg_loss=0.0, payoff=0.0, expectancy_r=0.0, max_consec_losses=0, t_stat=0.0)
    w, lo = pnl[pnl > 0], pnl[pnl <= 0]
    gp, gl = w.sum(), -lo.sum()
    cl = best = 0
    for x in pnl:
        cl = cl + 1 if x <= 0 else 0
        best = max(best, cl)
    sd = pnl.std(ddof=1) if n > 1 else 0.0
    return dict(
        n_trades=n, win_rate=len(w) / n,
        profit_factor=float(gp / gl) if gl > 0 else float("inf") if gp > 0 else 0.0,
        avg_trade=float(pnl.mean()), avg_win=float(w.mean()) if len(w) else 0.0,
        avg_loss=float(lo.mean()) if len(lo) else 0.0,
        payoff=float(w.mean() / -lo.mean()) if len(w) and len(lo) and lo.mean() < 0 else 0.0,
        expectancy_r=float(pnl.mean() / -lo.mean()) if len(lo) and lo.mean() < 0 else 0.0,
        max_consec_losses=int(best), t_stat=float(pnl.mean() / sd * math.sqrt(n)) if sd > 0 else 0.0,
    )


def summary(result, rf: float | None = None) -> dict:
    """Headline metrics for a BacktestResult."""
    from ..config import settings
    rf = settings().get("risk_free_rate", 0.0) if rf is None else rf
    eq = result.equity["equity"].to_numpy()
    cap = result.initial_capital
    ppy = periods_per_year(result.equity)
    r = daily_returns(result.equity)
    years = max(1e-9, (result.equity["ts"][-1] - result.equity["ts"][0]).total_seconds() / 86400 / 365.25)
    final = eq[-1]
    cagr = (final / cap) ** (1 / years) - 1 if final > 0 else -1.0
    _, mdd, mdd_abs = drawdown(eq)
    _, mdd_intra, mdd_intra_abs = drawdown(np.minimum(result.equity["equity_lo"].to_numpy(), eq))
    exposure = float((result.equity["position"].to_numpy() != 0).mean())
    out = dict(
        start=str(result.equity["ts"][0]), end=str(result.equity["ts"][-1]), years=years,
        initial_capital=cap, final_equity=float(final), net_profit=float(final - cap),
        total_return=float(final / cap - 1), cagr=float(cagr),
        sharpe=sharpe(r, ppy, rf), sortino=sortino(r, ppy, rf),
        ann_vol=float(np.std(r, ddof=1) * math.sqrt(ppy)) if len(r) > 1 else 0.0,
        max_dd=mdd, max_dd_abs=mdd_abs, max_dd_intrabar=mdd_intra, max_dd_intrabar_abs=mdd_intra_abs,
        calmar=float(cagr / mdd) if mdd > 0 else 0.0,
        longest_underwater_bars=longest_underwater(eq), exposure=exposure,
        skew=float(stats.skew(r)) if len(r) > 2 else 0.0,
        kurtosis=float(stats.kurtosis(r, fisher=False)) if len(r) > 3 else 3.0,
        n_days=len(r) + 1, periods_per_year=ppy,
        total_costs=float(result.trades["costs"].sum()) if result.trades.height else 0.0,
    )
    out.update(trade_stats(result.pnl))
    return out
