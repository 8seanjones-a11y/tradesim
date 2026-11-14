"""Binance public data (data.binance.vision): free, no API key. Monthly 1m kline ZIPs.

Spot archives switched open_time to MICROSECONDS on 2025-01-01; futures stayed in ms.
We detect the unit from magnitude, so both work.
"""
from __future__ import annotations

import io
import time
import zipfile
from datetime import date

import polars as pl
import requests

from ..store import raw_dir

BASE = "https://data.binance.vision/data"
COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume",
        "trades", "taker_buy_base", "taker_buy_quote", "ignore"]


def _url(symbol: str, market: str, interval: str, y: int, m: int) -> str:
    seg = "spot" if market == "spot" else f"futures/{market}"
    return f"{BASE}/{seg}/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{y}-{m:02d}.zip"


def to_datetime_any_unit(col: pl.Expr) -> pl.Expr:
    """Epoch -> UTC datetime, auto-detecting s / ms / us by magnitude."""
    return (pl.when(col > 10**14).then(col)                     # microseconds
            .when(col > 10**11).then(col * 1000)                 # milliseconds
            .otherwise(col * 1_000_000)                          # seconds
            ).cast(pl.Datetime("us")).dt.replace_time_zone("UTC")


def parse_kline_csv(data: bytes) -> pl.DataFrame:
    first = data.split(b"\n", 1)[0]
    has_header = first.startswith(b"open_time")
    df = pl.read_csv(io.BytesIO(data), has_header=has_header, new_columns=COLS if not has_header else None,
                     schema_overrides={"open_time": pl.Int64} if has_header else None)
    if not has_header:
        df.columns = COLS[: df.width]
    return df.select(to_datetime_any_unit(pl.col("open_time").cast(pl.Int64)).alias("ts"),
                     *(pl.col(c).cast(pl.Float64) for c in ("open", "high", "low", "close", "volume")))


def fetch_month(symbol: str, y: int, m: int, market: str = "spot", interval: str = "1m",
                session: requests.Session | None = None) -> pl.DataFrame | None:
    cache = raw_dir("binance") / market / symbol / f"{symbol}-{interval}-{y}-{m:02d}.zip"
    if not cache.exists():
        s = session or requests.Session()
        r = s.get(_url(symbol, market, interval, y, m), timeout=60)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(r.content)
        time.sleep(0.2)
    with zipfile.ZipFile(cache) as z:
        return parse_kline_csv(z.read(z.namelist()[0]))


def pull(symbol: str, start: date, end: date, market: str = "spot", interval: str = "1m", progress=print):
    s = requests.Session()
    y, m = start.year, start.month
    frames = []
    while (y, m) <= (end.year, end.month):
        df = fetch_month(symbol, y, m, market, interval, s)
        if df is not None:
            frames.append(df)
            progress(f"binance {symbol} {y}-{m:02d}: {df.height} bars")
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return pl.concat(frames) if frames else None
