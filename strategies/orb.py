"""Example 2 - intraday opening-range breakout on ES (proxy data), 5m bars.

Opening range = first `range_min` minutes after 09:30 New York. Enter on a close beyond the
range, stop at the opposite side of the range, target = `rr` x risk. Flat by 15:55 NY.
One trade per day."""
import numpy as np
import polars as pl

from tradesim.engine.strategy import Signals, Strategy


class OpeningRangeBreakout(Strategy):
    name = "orb"
    description = "Opening range breakout, R-multiple target, flat by close"
    symbol = "ES"
    timeframe = "5m"
    params = {"range_min": 15, "rr": 2.0, "contracts": 1, "allow_short": True}
    param_grid = {"range_min": [5, 15, 30, 60], "rr": [1.0, 1.5, 2.0, 3.0]}

    def signals(self, bars, p):
        ny = bars.select(pl.col("ts").dt.convert_time_zone("America/New_York").alias("t"))["t"]
        mins = (ny.dt.hour().cast(pl.Int32) * 60 + ny.dt.minute().cast(pl.Int32)).to_numpy()
        day = ny.dt.date().cast(pl.Int32).to_numpy()
        h, l, c = (bars[k].to_numpy() for k in ("high", "low", "close"))
        n = len(c)
        open_m, end_m = 9 * 60 + 30, 9 * 60 + 30 + p["range_min"]
        el, es = np.zeros(n, bool), np.zeros(n, bool)
        xl = np.zeros(n, bool)
        stop = np.full(n, np.nan)
        rng_hi = rng_lo = np.nan
        cur_day, traded = None, False
        for i in range(n):
            if day[i] != cur_day:
                cur_day, rng_hi, rng_lo, traded = day[i], -np.inf, np.inf, False
            m = mins[i]
            # bar i covers [m, m+5). It belongs to the range if it ends at or before end_m.
            if open_m <= m and m + 5 <= end_m:
                rng_hi, rng_lo = max(rng_hi, h[i]), min(rng_lo, l[i])
                continue
            if m >= 15 * 60 + 50:
                xl[i] = True     # flatten before the close (fills at next bar open)
                continue
            if traded or not np.isfinite(rng_hi) or m < end_m:
                continue
            if c[i] > rng_hi:
                el[i], stop[i], traded = True, c[i] - rng_lo, True
            elif p["allow_short"] and c[i] < rng_lo:
                es[i], stop[i], traded = True, rng_hi - c[i], True
        return Signals(entry_long=el, exit_long=xl, entry_short=es, exit_short=xl,
                       stop_dist=stop, target_dist=stop * p["rr"], size=p["contracts"])
