"""Example 1 - trend following: long when fast SMA crosses above slow SMA, exit on cross below.
Optional ATR stop. Daily bars."""
import numpy as np

from tradesim.engine import indicators as ind
from tradesim.engine.strategy import Signals, Strategy


class SmaCross(Strategy):
    name = "sma_cross"
    description = "Long-only SMA crossover with optional ATR stop"
    symbol = "SPY"
    timeframe = "1d"
    params = {"fast": 20, "slow": 100, "atr_stop": 0.0, "units": 100}
    param_grid = {"fast": [10, 20, 30, 50], "slow": [80, 100, 150, 200]}

    def signals(self, bars, p):
        c = bars["close"].to_numpy()
        f, s = ind.sma(c, p["fast"]), ind.sma(c, p["slow"])
        stop = None
        if p["atr_stop"] > 0:
            stop = ind.atr(bars["high"].to_numpy(), bars["low"].to_numpy(), c, 14) * p["atr_stop"]
        return Signals(entry_long=ind.crossover(f, s), exit_long=ind.crossunder(f, s),
                       stop_dist=stop, size=p["units"])
