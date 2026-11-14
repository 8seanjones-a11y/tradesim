"""Smaller adapters: yfinance (daily stocks/futures), Alpaca (1m stocks, free key),
FRED (macro, no key via CSV endpoint), CBOE VIX history."""
from __future__ import annotations

import io
import os
import time
from datetime import date, datetime, timezone

import polars as pl
import requests


# ------------------------------------------------------------------ yfinance
def yfinance_daily(ticker: str, start: date, end: date | None = None) -> pl.DataFrame | None:
    import yfinance as yf
    df = yf.download(ticker, start=str(start), end=str(end) if end else None, interval="1d",
                     auto_adjust=True, progress=False, multi_level_index=False)
    if df is None or df.empty:
        return None
    df = df.reset_index()
    out = pl.from_pandas(df).rename({"Date": "ts", "Open": "open", "High": "high", "Low": "low",
                                     "Close": "close", "Volume": "volume"})
    return out.select("ts", "open", "high", "low", "close", "volume").with_columns(
        pl.col("ts").cast(pl.Datetime("us")).dt.replace_time_zone("UTC"))


# ------------------------------------------------------------------ Alpaca (free IEX feed)
def alpaca_bars(symbol: str, start: date, end: date, timeframe: str = "1Min", progress=print) -> pl.DataFrame | None:
    """Needs ALPACA_API_KEY / ALPACA_API_SECRET env vars (free account). Free plan = IEX feed:
    volume is a small share of the consolidated tape."""
    key, sec = os.environ.get("ALPACA_API_KEY"), os.environ.get("ALPACA_API_SECRET")
    if not key or not sec:
        raise RuntimeError("Set ALPACA_API_KEY and ALPACA_API_SECRET (free at alpaca.markets)")
    url = f"https://data.alpaca.markets/v2/stocks/{symbol}/bars"
    params = {"timeframe": timeframe, "start": f"{start}T00:00:00Z", "end": f"{end}T23:59:59Z",
              "limit": 10000, "adjustment": "all", "feed": "iex"}
    h = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": sec}
    rows, tok = [], None
    while True:
        if tok:
            params["page_token"] = tok
        r = requests.get(url, params=params, headers=h, timeout=60)
        if r.status_code == 429:
            time.sleep(5)
            continue
        r.raise_for_status()
        j = r.json()
        rows += j.get("bars") or []
        tok = j.get("next_page_token")
        if rows and len(rows) % 100000 < 10000:
            progress(f"alpaca {symbol}: {len(rows)} bars, last {rows[-1]['t']}")
        if not tok:
            break
        time.sleep(0.35)
    if not rows:
        return None
    df = pl.DataFrame(rows).rename({"t": "ts", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    return df.with_columns(pl.col("ts").str.to_datetime(time_zone="UTC")).select("ts", "open", "high", "low", "close", "volume")


# ------------------------------------------------------------------ FRED (no key needed for CSV)
def fred_series(series_id: str) -> pl.DataFrame:
    r = requests.get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}", timeout=60)
    r.raise_for_status()
    df = pl.read_csv(io.BytesIO(r.content), null_values=["."], try_parse_dates=True)
    df.columns = ["date", "value"]
    return df.with_columns(pl.col("value").cast(pl.Float64)).drop_nulls()


# ------------------------------------------------------------------ CBOE
def cboe_vix() -> pl.DataFrame:
    r = requests.get("https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv", timeout=60)
    r.raise_for_status()
    df = pl.read_csv(io.BytesIO(r.content))
    df.columns = [c.lower() for c in df.columns]
    return df.with_columns(pl.col("date").str.to_date("%m/%d/%Y"))


def save_macro(name: str, df: pl.DataFrame):
    from ...config import data_dir
    p = data_dir() / "macro"
    p.mkdir(parents=True, exist_ok=True)
    df.write_parquet(p / f"{name}.parquet")
    return p / f"{name}.parquet"


def load_macro(name: str) -> pl.DataFrame:
    from ...config import data_dir
    return pl.read_parquet(data_dir() / "macro" / f"{name}.parquet")
