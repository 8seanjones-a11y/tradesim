import numpy as np
import polars as pl
import pytest

from tradesim.analysis import account, montecarlo, overfit, robustness
from tradesim.config import generic_instrument
from tradesim.data import synthetic
from tradesim.engine import indicators as ind
from tradesim.engine.backtest import ExecConfig, run_backtest
from tradesim.engine.strategy import Signals, Strategy, load_strategy

CFG = ExecConfig(slippage_ticks=0, initial_capital=100_000)


def test_shuffle_preserves_total():
    pnl = np.random.default_rng(0).normal(10, 100, 200)
    mc = montecarlo.shuffle_mc(pnl, 10_000, 500)
    assert np.allclose(mc["final"], 10_000 + pnl.sum())
    assert mc["max_dd"].min() >= 0


def test_bootstrap_ci_covers_mean():
    rng = np.random.default_rng(1)
    pnl = rng.normal(5, 50, 400)
    mc = montecarlo.bootstrap_mc(pnl, 0.0 + 1e6, 2000)
    lo, hi = np.percentile(mc["final"] - 1e6, [2.5, 97.5])
    assert lo < 5 * 400 < hi


def test_dsr_penalises_many_trials():
    r = np.random.default_rng(2).normal(0.0005, 0.01, 1000)
    one = overfit.deflated_sharpe(r, 1)["dsr"]
    many = overfit.deflated_sharpe(r, 1000)["dsr"]
    assert many < one


def test_pbo_high_for_pure_noise():
    # single-matrix PBO is noisy; on average over seeds it should be around/above 0.5
    vals = [overfit.pbo_cscv(np.random.default_rng(s).normal(0, 1, (500, 20)), 10)["pbo"] for s in range(8)]
    assert np.mean(vals) > 0.4


def test_pbo_low_for_dominant_variant():
    rng = np.random.default_rng(4)
    m = rng.normal(0, 1, (500, 20))
    m[:, 7] += 0.5
    assert overfit.pbo_cscv(m, 10)["pbo"] < 0.1


def test_indicators_are_causal():
    x = np.random.default_rng(5).normal(0, 1, 300).cumsum() + 100
    full = [ind.sma(x, 20), ind.ema(x, 20), ind.rsi(x, 14), ind.rolling_max(x, 10), ind.atr(x + 1, x - 1, x, 14)]
    for k in (50, 120, 299):
        part = [ind.sma(x[:k], 20), ind.ema(x[:k], 20), ind.rsi(x[:k], 14), ind.rolling_max(x[:k], 10),
                ind.atr(x[:k] + 1, x[:k] - 1, x[:k], 14)]
        for f, p in zip(full, part):
            np.testing.assert_allclose(f[:k], p, equal_nan=True)


@pytest.mark.parametrize("name", ["sma_cross", "rsi_reversion", "orb"])
def test_example_strategies_have_no_lookahead(name):
    """Signals at bar i must not change when future bars are removed."""
    from datetime import timedelta
    s = load_strategy(name)
    step = timedelta(minutes=5) if s.timeframe == "5m" else timedelta(days=1)
    bars = synthetic.gbm(1200, seed=7, step=step)
    full = s.signals(bars, s.params).finalize(bars.height)
    for k in (400, 900):
        part = s.signals(bars.head(k), s.params).finalize(k)
        for f in ("entry_long", "exit_long", "entry_short", "exit_short"):
            np.testing.assert_array_equal(getattr(full, f)[:k], getattr(part, f))
        np.testing.assert_allclose(full.stop_dist[:k], part.stop_dist, equal_nan=True)


class Momentum(Strategy):
    name = "mom"
    params = {"n": 1}

    def signals(self, bars, p):
        c = bars["close"].to_numpy()
        up = c > ind.shift(c, p["n"])
        return Signals(entry_long=up, entry_short=~up & np.isfinite(ind.shift(c, p["n"])), size=100)


def _trending_ar(n=3000, phi=0.3, seed=0):
    """Returns with positive autocorrelation -> momentum has a genuine edge."""
    rng = np.random.default_rng(seed)
    e = rng.normal(0, 0.01, n)
    r = np.zeros(n)
    for i in range(1, n):
        r[i] = phi * r[i - 1] + e[i]
    b = synthetic.gbm(n, seed=seed)
    c = 100 * np.exp(np.cumsum(r))
    o = np.concatenate([[100], c[:-1]])
    return b.with_columns(open=o, close=c, high=np.maximum(o, c) * 1.001, low=np.minimum(o, c) * 0.999)


def test_permutation_detects_real_edge_and_not_noise():
    inst = generic_instrument()
    edge = robustness.permutation_test(Momentum(), _trending_ar(), inst, cfg=CFG, n_perm=40, optimise=False)
    assert edge["p_value"] < 0.05
    noise = robustness.permutation_test(Momentum(), synthetic.gbm(3000, seed=11), inst, cfg=CFG, n_perm=40, optimise=False)
    assert noise["p_value"] > 0.05


def test_walk_forward_and_sensitivity_run():
    s = load_strategy("sma_cross")
    bars = synthetic.garch(2500, seed=3)
    inst = generic_instrument()
    sens = robustness.sensitivity(s, bars, inst, cfg=CFG)
    assert sens["table"].height == 16 and sens["returns_matrix"].shape[1] == 16
    wfo = robustness.walk_forward(s, bars, inst, cfg=CFG, n_folds=3)
    assert wfo["folds"].height == 3


def test_cost_sensitivity_monotone():
    inst = generic_instrument(tick_size=0.01)
    c = robustness.cost_sensitivity(Momentum(), _trending_ar(), inst, cfg=CFG)
    y = c["table"]["net_profit"].to_numpy()
    assert np.all(np.diff(y) <= 1e-9)


def test_account_outputs():
    s = load_strategy("sma_cross")
    r = run_backtest(s, _trending_ar(), generic_instrument(), cfg=CFG)
    h = account.horizon_outcomes(r, (1, 3), n_sims=300)
    assert h.height == 2 and (h["p_loss"] >= 0).all()
    k = account.kelly_fraction(np.array([0.02, -0.01] * 50))
    assert k["kelly_mult"] > 0
