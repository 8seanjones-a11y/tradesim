"""Example 3 - mean reversion: buy when RSI(n) < lower in an uptrend (close > SMA200), exit when
RSI > upper or after `max_hold` bars; mirror for shorts. Risk-based sizing via ATR stop."""
from tradesim.engine import indicators as ind
from tradesim.engine.strategy import Signals, Strategy


class RsiReversion(Strategy):
    name = "rsi_reversion"
    description = "RSI pullback in trend direction, ATR stop"
    symbol = "EURUSD"
    timeframe = "1h"
    params = {"n": 2, "lower": 10, "upper": 70, "trend": 200, "atr_mult": 2.0, "units": 100_000}
    param_grid = {"n": [2, 3, 5], "lower": [5, 10, 20], "upper": [60, 70, 80]}

    def signals(self, bars, p):
        c = bars["close"].to_numpy()
        r = ind.rsi(c, p["n"])
        t = ind.sma(c, p["trend"])
        a = ind.atr(bars["high"].to_numpy(), bars["low"].to_numpy(), c, 14)
        return Signals(
            entry_long=(c > t) & (r < p["lower"]), exit_long=r > p["upper"],
            entry_short=(c < t) & (r > 100 - p["lower"]), exit_short=r < 100 - p["upper"],
            stop_dist=a * p["atr_mult"], size=p["units"],
        )
