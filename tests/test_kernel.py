from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl
import pytest

from tradesim.config import generic_instrument
from tradesim.engine.backtest import ExecConfig, run_backtest
from tradesim.engine.strategy import Signals, Strategy


def bars_from(rows):
    t0 = datetime(2024, 1, 1, 15, tzinfo=timezone.utc)
    o, h, l, c = zip(*rows)
    return pl.DataFrame({"ts": [t0 + timedelta(days=i) for i in range(len(rows))],
                         "open": o, "high": h, "low": l, "close": c, "volume": [1.0] * len(rows)})


class Fixed(Strategy):
    name = "fixed"

    def __init__(self, sig):
        self._sig = sig

    def signals(self, bars, p):
        return self._sig


CFG = ExecConfig(slippage_ticks=0, initial_capital=10_000)


def test_fill_next_open_and_pnl():
    b = bars_from([(10, 10, 10, 10), (11, 12, 10, 11), (12, 13, 11, 13), (14, 14, 14, 14)])
    sig = Signals(entry_long=[True, False, False, False], exit_long=[False, False, True, False], size=2)
    r = run_backtest(Fixed(sig), b, generic_instrument(point_value=1.0), cfg=CFG)
    t = r.trades.row(0, named=True)
    assert t["entry_price"] == 11 and t["entry_i"] == 1
    assert t["exit_price"] == 14 and t["exit_i"] == 3
    assert t["pnl"] == pytest.approx(2 * 3)
    assert r.equity["equity"][-1] == pytest.approx(10_006)


def test_futures_multiplier_and_commission():
    b = bars_from([(100, 100, 100, 100), (100, 101, 99, 100), (102, 102, 102, 102)])
    sig = Signals(entry_long=[True, False, False], exit_long=[False, True, False], size=1)
    inst = generic_instrument(point_value=50.0, tick_size=0.25, commission_per_unit=2.5, lot_step=1)
    r = run_backtest(Fixed(sig), b, inst, cfg=CFG)
    assert r.trades["pnl"][0] == pytest.approx(2 * 50 - 5.0)


def test_stop_before_target_same_bar_is_pessimistic():
    b = bars_from([(100, 100, 100, 100), (100, 106, 94, 100), (100, 100, 100, 100)])
    sig = Signals(entry_long=[True, False, False], stop_dist=5, target_dist=5, size=1)
    r = run_backtest(Fixed(sig), b, generic_instrument(), cfg=CFG)
    t = r.trades.row(0, named=True)
    assert t["exit_reason"] == "stop" and t["exit_price"] == 95


def test_gap_through_stop_fills_at_open():
    b = bars_from([(100, 100, 100, 100), (100, 100, 99, 99), (90, 91, 89, 90), (90, 90, 90, 90)])
    sig = Signals(entry_long=[True, False, False, False], stop_dist=5, size=1)
    r = run_backtest(Fixed(sig), b, generic_instrument(), cfg=CFG)
    assert r.trades["exit_price"][0] == 90
    assert r.equity["equity_lo"][2] == pytest.approx(10_000 - 10)


def test_short_and_reverse_and_slippage():
    b = bars_from([(100, 100, 100, 100), (100, 100, 100, 100), (90, 90, 90, 90), (95, 95, 95, 95)])
    sig = Signals(entry_long=[False, True, False, False], entry_short=[True, False, False, False], size=1)
    inst = generic_instrument(tick_size=1.0)
    r = run_backtest(Fixed(sig), b, inst, cfg=ExecConfig(slippage_ticks=1, initial_capital=10_000))
    t0, t1 = r.trades.row(0, named=True), r.trades.row(1, named=True)
    assert t0["direction"] == -1 and t0["entry_price"] == 99 and t0["exit_price"] == 91 and t0["exit_reason"] == "reverse"
    assert t1["direction"] == 1 and t1["entry_price"] == 91 and t1["exit_reason"] == "end"


def test_intrabar_equity_marks():
    b = bars_from([(100, 100, 100, 100), (100, 110, 92, 105), (105, 105, 105, 105)])
    sig = Signals(entry_long=[True, False, False], size=1)
    r = run_backtest(Fixed(sig), b, generic_instrument(), cfg=CFG)
    assert r.equity["equity_lo"][1] == pytest.approx(9_992)
    assert r.equity["equity_hi"][1] == pytest.approx(10_010)


def test_risk_sizing():
    b = bars_from([(100, 100, 100, 100), (100, 100, 100, 100), (100, 100, 100, 100)])
    sig = Signals(entry_long=[True, False, False], stop_dist=2.0)
    inst = generic_instrument(point_value=50.0, lot_step=1)
    r = run_backtest(Fixed(sig), b, inst, cfg=ExecConfig(slippage_ticks=0, initial_capital=10_000, risk_frac=0.01))
    assert r.trades["units"][0] == 1  # 100 risk / (2*50) = 1 contract
