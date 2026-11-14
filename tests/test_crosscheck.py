"""Cross-check the engine against backtesting.py on a simple SMA crossover (no costs)."""
import numpy as np
import pytest

bt = pytest.importorskip("backtesting")

from tradesim.config import generic_instrument
from tradesim.data import synthetic
from tradesim.engine import indicators as ind
from tradesim.engine.backtest import ExecConfig, run_backtest
from tradesim.engine.strategy import Signals, Strategy


class Cross(Strategy):
    name = "x"
    params = {"f": 10, "s": 30}

    def signals(self, bars, p):
        c = bars["close"].to_numpy()
        a, b = ind.sma(c, p["f"]), ind.sma(c, p["s"])
        return Signals(entry_long=ind.crossover(a, b), entry_short=ind.crossunder(a, b), size=10)


def test_matches_backtesting_py():
    from backtesting import Backtest
    from backtesting import Strategy as BS
    from backtesting.lib import crossover

    bars = synthetic.gbm(1500, sigma=0.25, seed=21)
    ours = run_backtest(Cross(), bars, generic_instrument(), cfg=ExecConfig(slippage_ticks=0, initial_capital=1e6))

    df = bars.to_pandas().set_index("ts")
    df.columns = [c.capitalize() for c in df.columns]

    class B(BS):
        def init(self):
            sma = lambda x, n: ind.sma(np.asarray(x), n)
            self.a = self.I(sma, self.data.Close, 10)
            self.b = self.I(sma, self.data.Close, 30)

        def next(self):
            if crossover(self.a, self.b):
                self.position.close(); self.buy(size=10)
            elif crossover(self.b, self.a):
                self.position.close(); self.sell(size=10)

    st = Backtest(df, B, cash=1e6, commission=0, finalize_trades=True).run()
    theirs = st._trades
    assert len(theirs) == ours.trades.height
    np.testing.assert_allclose(theirs["EntryPrice"].to_numpy(), ours.trades["entry_price"].to_numpy(), rtol=1e-9)
    # every signal-closed trade must match exactly; the final force-closed position differs by
    # convention only (backtesting.py closes at the last OPEN, we close at the last CLOSE)
    np.testing.assert_allclose(theirs["PnL"].to_numpy()[:-1], ours.trades["pnl"].to_numpy()[:-1], rtol=1e-9)
    np.testing.assert_allclose(theirs["ExitPrice"].to_numpy()[:-1], ours.trades["exit_price"].to_numpy()[:-1], rtol=1e-9)
