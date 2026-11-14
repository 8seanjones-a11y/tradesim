"""Strategy-level robustness tests that re-run the backtest:
parameter sensitivity, walk-forward, noise injection, permutation test, null-model (synthetic) runs,
cost sensitivity, and regime breakdown."""
from __future__ import annotations

import itertools
import math
from dataclasses import replace

import numpy as np
import polars as pl

from ..engine.backtest import ExecConfig, run_backtest
from .metrics import daily_returns, drawdown, periods_per_year, sharpe


def _score(res, objective: str = "sharpe") -> float:
    eq = res.equity
    if objective == "net_profit":
        return float(eq["equity"][-1] - res.initial_capital)
    r = daily_returns(eq)
    if objective == "sharpe":
        return sharpe(r, periods_per_year(eq))
    if objective == "calmar":
        _, mdd, _ = drawdown(eq["equity"].to_numpy())
        tot = eq["equity"][-1] / res.initial_capital - 1
        return float(tot / mdd) if mdd > 0 else 0.0
    raise ValueError(objective)


def grid(param_grid: dict) -> list[dict]:
    keys = list(param_grid)
    return [dict(zip(keys, v)) for v in itertools.product(*(param_grid[k] for k in keys))]


# ---------------------------------------------------------------- parameter sensitivity
def sensitivity(strategy, bars, inst, param_grid: dict | None = None, cfg: ExecConfig | None = None,
                objective: str = "sharpe") -> dict:
    """Evaluate every combination in the grid. Returns table + daily-return matrix (for PBO/DSR)
    + 'plateau' score: fraction of neighbours within 50% of the best score (broad plateau = robust)."""
    pg = param_grid or strategy.param_grid
    if not pg:
        return dict(table=pl.DataFrame(), returns_matrix=None, plateau=float("nan"), best=None)
    combos = grid(pg)
    rows, rets = [], []
    for c in combos:
        res = run_backtest(strategy, bars, inst, params=c, cfg=cfg)
        r = daily_returns(res.equity)
        rets.append(r)
        rows.append({**c, "score": _score(res, objective), "net_profit": float(res.equity["equity"][-1] - res.initial_capital),
                     "n_trades": res.trades.height})
    table = pl.DataFrame(rows)
    L = min(len(r) for r in rets)
    mat = np.column_stack([r[:L] for r in rets]) if L > 0 else None
    scores = table["score"].to_numpy()
    best_i = int(np.nanargmax(scores))
    best = combos[best_i]
    # neighbours = combos differing by one grid step in exactly one parameter
    keys = list(pg)
    pos = {k: {v: i for i, v in enumerate(pg[k])} for k in keys}
    neigh = []
    for j, c in enumerate(combos):
        d = [abs(pos[k][c[k]] - pos[k][best[k]]) for k in keys]
        if sum(d) == 1:
            neigh.append(scores[j])
    bs = scores[best_i]
    plateau = float(np.mean([s >= 0.5 * bs for s in neigh])) if neigh and bs > 0 else 0.0
    return dict(table=table, returns_matrix=mat, plateau=plateau, best=best, best_score=float(bs),
                objective=objective, keys=keys)


# ---------------------------------------------------------------- walk-forward
def walk_forward(strategy, bars, inst, param_grid: dict | None = None, n_folds: int = 5, train_frac: float = 0.6,
                 anchored: bool = True, cfg: ExecConfig | None = None, objective: str = "sharpe") -> dict:
    """Optimise on each in-sample window, run the chosen params on the following OOS window.
    WF efficiency = mean OOS score / mean IS score (>0.5 is generally considered healthy)."""
    pg = param_grid or strategy.param_grid
    n = bars.height
    first_train = int(n * train_frac)
    oos_len = (n - first_train) // n_folds
    if oos_len < 20 or not pg:
        return dict(folds=pl.DataFrame(), wf_efficiency=float("nan"), oos_equity=None)
    combos = grid(pg)
    folds, oos_pnl_frames = [], []
    for k in range(n_folds):
        te_start = first_train + k * oos_len
        te_end = n if k == n_folds - 1 else te_start + oos_len
        tr_start = 0 if anchored else te_start - first_train
        train = bars.slice(tr_start, te_start - tr_start)
        best, best_s = None, -np.inf
        for c in combos:
            s = _score(run_backtest(strategy, train, inst, params=c, cfg=cfg), objective)
            if s > best_s:
                best, best_s = c, s
        # include warm-up history so indicators are valid at OOS start; only score the OOS slice
        warm = min(te_start, max(200, int(0.2 * (te_start - tr_start))))
        test = bars.slice(te_start - warm, te_end - te_start + warm)
        res = run_backtest(strategy, test, inst, params=best, cfg=cfg)
        eq = res.equity.slice(warm)
        oos_ret = eq["equity"].to_numpy()
        oos_s = sharpe(daily_returns(eq), periods_per_year(eq)) if objective == "sharpe" else _score(res, objective)
        folds.append({"fold": k, "train_start": str(train["ts"][0]), "test_start": str(bars["ts"][te_start]),
                      "test_end": str(bars["ts"][te_end - 1]), "is_score": float(best_s), "oos_score": float(oos_s),
                      "oos_pnl": float(oos_ret[-1] - oos_ret[0]), **{f"p_{a}": b for a, b in best.items()}})
        oos_pnl_frames.append(eq.with_columns(pl.col("equity").diff().fill_null(0).alias("dpnl")).select("ts", "dpnl"))
    f = pl.DataFrame(folds)
    is_m, oos_m = f["is_score"].mean(), f["oos_score"].mean()
    stitched = pl.concat(oos_pnl_frames)
    stitched = stitched.with_columns(equity=pl.col("dpnl").cum_sum() + (cfg.resolved() if cfg else ExecConfig().resolved()).initial_capital)
    return dict(folds=f, wf_efficiency=float(oos_m / is_m) if is_m and is_m > 0 else 0.0,
                oos_equity=stitched, pct_profitable_folds=float((f["oos_pnl"] > 0).mean()))


# ---------------------------------------------------------------- noise injection
def noise_injection(strategy, bars, inst, params=None, cfg=None, n_runs: int = 50, atr_frac: float = 0.1,
                    seed: int = 0) -> dict:
    """Perturb each bar's OHLC by N(0, atr_frac*ATR) (kept internally consistent) and re-run.
    Strategies fitted to exact prices degrade sharply."""
    from ..engine.indicators import atr
    rng = np.random.default_rng(seed)
    a = atr(bars["high"].to_numpy(), bars["low"].to_numpy(), bars["close"].to_numpy(), 14)
    a = np.where(np.isnan(a), np.nanmedian(a), a)
    base = run_backtest(strategy, bars, inst, params=params, cfg=cfg)
    base_np = float(base.equity["equity"][-1] - base.initial_capital)
    out = []
    for _ in range(n_runs):
        eps = {k: rng.normal(0, atr_frac, bars.height) * a for k in ("open", "high", "low", "close")}
        o, h, l, c = (bars[k].to_numpy() + eps[k] for k in ("open", "high", "low", "close"))
        h2, l2 = np.maximum.reduce([o, h, l, c]), np.minimum.reduce([o, h, l, c])
        nb = bars.with_columns(open=o, high=h2, low=l2, close=c)
        r = run_backtest(strategy, nb, inst, params=params, cfg=cfg)
        out.append(float(r.equity["equity"][-1] - r.initial_capital))
    out = np.array(out)
    return dict(base_net=base_np, noisy_net=out, pct_profitable=float((out > 0).mean()),
                median_ratio=float(np.median(out) / base_np) if base_np else 0.0)


# ---------------------------------------------------------------- permutation test (Masters)
def _permute_bars(bars: pl.DataFrame, rng) -> pl.DataFrame:
    """Permute bar-to-bar log changes (close-to-open gap and open-relative intrabar shape kept
    paired) so the price path has the same return distribution but no exploitable order."""
    o, h, l, c = (bars[k].to_numpy() for k in ("open", "high", "low", "close"))
    lo_, lh, ll, lc = np.log(o), np.log(h), np.log(l), np.log(c)
    gap = lo_[1:] - lc[:-1]
    rh, rl, rc = lh[1:] - lo_[1:], ll[1:] - lo_[1:], lc[1:] - lo_[1:]
    p1, p2 = rng.permutation(len(gap)), rng.permutation(len(gap))
    gap = gap[p1]
    rh, rl, rc = rh[p2], rl[p2], rc[p2]
    nc = np.empty_like(c); no = np.empty_like(o); nh = np.empty_like(h); nl = np.empty_like(l)
    no[0], nh[0], nl[0], nc[0] = o[0], h[0], l[0], c[0]
    prev = lc[0]
    for i in range(len(gap)):
        op = prev + gap[i]
        no[i + 1] = math.exp(op); nh[i + 1] = math.exp(op + rh[i]); nl[i + 1] = math.exp(op + rl[i])
        prev = op + rc[i]
        nc[i + 1] = math.exp(prev)
    return bars.with_columns(open=no, high=nh, low=nl, close=nc)


def permutation_test(strategy, bars, inst, param_grid: dict | None = None, cfg=None, n_perm: int = 100,
                     objective: str = "sharpe", optimise: bool = True, seed: int = 0) -> dict:
    """Timothy Masters' Monte Carlo permutation test. If `optimise`, the whole parameter search
    is repeated on every permuted series, so the p-value accounts for selection bias.
    p = fraction of permuted runs scoring >= the real run."""
    rng = np.random.default_rng(seed)
    pg = (param_grid or strategy.param_grid) if optimise else None
    combos = grid(pg) if pg else [None]

    def best_score(b):
        return max(_score(run_backtest(strategy, b, inst, params=c, cfg=cfg), objective) for c in combos)

    real = best_score(bars)
    perm = np.array([best_score(_permute_bars(bars, rng)) for _ in range(n_perm)])
    p = float(((perm >= real).sum() + 1) / (n_perm + 1))
    return dict(real=real, permuted=perm, p_value=p, optimised=bool(pg), n_perm=n_perm)


# ---------------------------------------------------------------- null model
def null_model_test(strategy, bars, inst, params=None, cfg=None, n_paths: int = 30, model: str = "bootstrap",
                    seed: int = 0) -> dict:
    """Run the strategy on synthetic series with no designed edge (block-bootstrapped real returns,
    GBM or GARCH matched to the real vol). Real result should sit well above this distribution."""
    from ..data import synthetic
    c = bars["close"].to_numpy()
    lr = np.diff(np.log(c))
    ppy = 252.0
    res0 = run_backtest(strategy, bars, inst, params=params, cfg=cfg)
    real = _score(res0)
    scores = []
    for k in range(n_paths):
        if model == "bootstrap":
            sb = synthetic.bootstrap_bars(bars, block=20, seed=seed + k)
            # de-mean so drift (buy-and-hold beta) is not mistaken for edge
            cc = sb["close"].to_numpy()
            drift = np.exp(-np.arange(len(cc)) * lr.mean())
            sb = sb.with_columns(*(pl.col(x) * drift for x in ("open", "high", "low", "close")))
        elif model == "gbm":
            sb = synthetic.gbm(bars.height, c[0], 0.0, lr.std() * math.sqrt(ppy), 1 / ppy, seed + k).with_columns(ts=bars["ts"])
        elif model == "garch":
            v = lr.var()
            sb = synthetic.garch(bars.height, c[0], omega=v * 0.02, alpha=0.08, beta=0.9, seed=seed + k).with_columns(ts=bars["ts"])
        else:
            raise ValueError(model)
        scores.append(_score(run_backtest(strategy, sb, inst, params=params, cfg=cfg)))
    scores = np.array(scores)
    return dict(real=real, null_scores=scores, p_value=float(((scores >= real).sum() + 1) / (n_paths + 1)), model=model)


# ---------------------------------------------------------------- cost sensitivity
def cost_sensitivity(strategy, bars, inst, params=None, cfg=None, slip_ticks=(0, 1, 2, 4, 8)) -> dict:
    """Net profit as slippage grows; break-even slippage = ticks/side at which edge vanishes."""
    base = (cfg or ExecConfig())
    rows = []
    for s in slip_ticks:
        r = run_backtest(strategy, bars, inst, params=params, cfg=replace(base, slippage_ticks=s))
        rows.append({"slippage_ticks": s, "net_profit": float(r.equity["equity"][-1] - r.initial_capital)})
    t = pl.DataFrame(rows)
    x, y = t["slippage_ticks"].to_numpy().astype(float), t["net_profit"].to_numpy()
    be = float("nan")
    if y[0] <= 0:
        be = 0.0
    else:
        for i in range(1, len(y)):
            if y[i] <= 0:
                be = x[i - 1] + (x[i] - x[i - 1]) * y[i - 1] / (y[i - 1] - y[i])
                break
        else:
            be = float("inf")
    return dict(table=t, breakeven_slippage_ticks=be)


# ---------------------------------------------------------------- regimes
def regime_breakdown(result, window: int = 50) -> pl.DataFrame:
    """Split closed trades by market regime at entry: trend (close vs SMA), volatility tercile,
    weekday and (intraday) hour."""
    from ..engine.indicators import rolling_std, sma
    b = result.bars
    c = b["close"].to_numpy()
    ma = sma(c, window)
    vol = rolling_std(np.diff(np.log(c), prepend=np.nan), window)
    q1, q2 = np.nanpercentile(vol, [33, 66]) if np.isfinite(vol).any() else (np.nan, np.nan)
    t = result.trades
    if t.height == 0:
        return pl.DataFrame()
    ei = t["entry_i"].to_numpy()
    sig_i = np.maximum(ei - 1, 0)
    trend = np.where(c[sig_i] > ma[sig_i], "up", np.where(c[sig_i] < ma[sig_i], "down", "n/a"))
    v = vol[sig_i]
    vr = np.where(v <= q1, "low_vol", np.where(v <= q2, "mid_vol", np.where(np.isfinite(v), "high_vol", "n/a")))
    t = t.with_columns(trend=pl.Series(trend), vol_regime=pl.Series(vr),
                       weekday=pl.col("entry_ts").dt.weekday(), hour=pl.col("entry_ts").dt.hour())
    out = []
    for col in ("trend", "vol_regime", "weekday", "hour"):
        g = t.group_by(col).agg(pl.len().alias("n"), pl.col("pnl").sum().alias("pnl"),
                                (pl.col("pnl") > 0).mean().alias("win_rate"), pl.col("pnl").mean().alias("avg"))
        out.append(g.with_columns(pl.lit(col).alias("dimension"), pl.col(col).cast(pl.Utf8).alias("bucket")).drop(col))
    return pl.concat(out).select("dimension", "bucket", "n", "pnl", "win_rate", "avg").sort("dimension", "bucket")
