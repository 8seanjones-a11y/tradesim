"""Dispatch `tsim data pull SYMBOL` to the right adapter based on instruments.yaml."""
from __future__ import annotations

from datetime import date, timedelta

from ..config import instrument
from . import store


def pull(symbol: str, start: date, end: date | None = None, daily: bool = False, progress=print) -> int:
    inst = instrument(symbol)
    end = end or date.today() - timedelta(days=1)
    src = inst.daily_source if daily and inst.daily_source else inst.source
    if src == "binance":
        from .adapters import binance
        df = binance.pull(inst.source_symbol, start, end, market=inst.market or "spot", progress=progress)
        tf = "1m"
    elif src == "dukascopy":
        from .adapters import dukascopy
        df = dukascopy.pull(inst.source_symbol, start, end, inst.price_divisor, progress=progress)
        tf = "1m"
        if df is not None:
            for w in dukascopy.sanity_check(df, inst.source_symbol, inst.extra.get("expected_range")):
                progress("WARNING " + w)
    elif src == "alpaca":
        from .adapters import other
        df = other.alpaca_bars(inst.source_symbol, start, end, progress=progress)
        tf = "1m"
    elif src == "yfinance":
        from .adapters import other
        df = other.yfinance_daily(inst.daily_symbol or inst.source_symbol, start, end)
        tf = "1d"
    else:
        raise ValueError(f"{symbol}: unknown source {src!r}")
    if df is None or df.is_empty():
        progress(f"{symbol}: no data returned")
        return 0
    n = store.write_bars(df, inst.asset_class, symbol, tf, source=src)
    progress(f"{symbol}: stored {n} {tf} bars from {src}")
    return n
