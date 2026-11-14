"""Dukascopy free historical feed: FX, metals, index & commodity CFDs (used as futures proxies).

Daily 1-minute candle files:
  https://datafeed.dukascopy.com/datafeed/{SYM}/{YYYY}/{MM0}/{DD}/{BID|ASK}_candles_min_1.bi5
  (MM0 is the ZERO-based month: January = 00). Files are LZMA-compressed; each record is 24 bytes
  big-endian: uint32 seconds-from-midnight-UTC, uint32 open, close, low, high (price * divisor), float32 volume.
Price divisor is per instrument (1e5 most FX, 1e3 JPY pairs / metals / indices) - set `price_divisor`
in instruments.yaml and the sanity check below will warn when it looks wrong.
Downloads BID and ASK to produce mid prices and a spread column.
"""
from __future__ import annotations

import lzma
import struct
import time
from datetime import date, datetime, timedelta, timezone

import numpy as np
import polars as pl
import requests

from ..store import raw_dir

BASE = "https://datafeed.dukascopy.com/datafeed"
REC = struct.Struct(">5If")


def decode_candles(blob: bytes, day: date, divisor: float) -> pl.DataFrame:
    if not blob:
        return pl.DataFrame()
    raw = lzma.decompress(blob)
    n = len(raw) // REC.size
    arr = np.frombuffer(raw[: n * REC.size], dtype=np.dtype([("t", ">u4"), ("o", ">u4"), ("c", ">u4"),
                                                               ("l", ">u4"), ("h", ">u4"), ("v", ">f4")]))
    t0 = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    ts = [t0 + timedelta(seconds=int(s)) for s in arr["t"]]
    return pl.DataFrame({"ts": ts, "open": arr["o"] / divisor, "high": arr["h"] / divisor,
                         "low": arr["l"] / divisor, "close": arr["c"] / divisor,
                         "volume": arr["v"].astype(np.float64)})


def _fetch(sym: str, d: date, side: str, s: requests.Session) -> bytes:
    cache = raw_dir("dukascopy") / sym / f"{d:%Y}" / f"{d:%Y%m%d}_{side}.bi5"
    if cache.exists():
        return cache.read_bytes()
    url = f"{BASE}/{sym}/{d.year}/{d.month - 1:02d}/{d.day:02d}/{side}_candles_min_1.bi5"
    for attempt in range(4):
        try:
            r = s.get(url, timeout=30)
            if r.status_code == 404:
                data = b""
                break
            r.raise_for_status()
            data = r.content
            break
        except requests.RequestException:
            time.sleep(2**attempt)
    else:
        raise RuntimeError(f"dukascopy failed: {url}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(data)
    time.sleep(0.05)
    return data


def fetch_day(sym: str, d: date, divisor: float, s: requests.Session) -> pl.DataFrame:
    bid = decode_candles(_fetch(sym, d, "BID", s), d, divisor)
    ask = decode_candles(_fetch(sym, d, "ASK", s), d, divisor)
    if bid.is_empty():
        return bid
    # Dukascopy emits flat zero-volume candles for closed-market minutes; drop them
    if ask.is_empty():
        out = bid
    else:
        j = bid.join(ask, on="ts", suffix="_a")
        out = j.select("ts", *((pl.col(c) + pl.col(f"{c}_a")) / 2 for c in ("open", "high", "low", "close")),
                       "volume", (pl.col("close_a") - pl.col("close")).alias("spread"))
    return out.filter(pl.col("volume") > 0)


def sanity_check(df: pl.DataFrame, sym: str, expected_range: tuple[float, float] | None = None) -> list[str]:
    w = []
    if df.is_empty():
        return ["no data"]
    med = float(df["close"].median())
    if expected_range and not (expected_range[0] <= med <= expected_range[1]):
        w.append(f"{sym}: median price {med:g} outside {expected_range} - check price_divisor")
    return w


def pull(sym: str, start: date, end: date, divisor: float, progress=print) -> pl.DataFrame | None:
    s = requests.Session()
    frames = []
    d = start
    while d <= end:
        if d.weekday() != 5:  # Saturdays are empty
            df = fetch_day(sym, d, divisor, s)
            if not df.is_empty():
                frames.append(df)
        if d.day == 1:
            progress(f"dukascopy {sym} {d:%Y-%m}")
        d += timedelta(days=1)
    return pl.concat(frames, how="diagonal_relaxed") if frames else None
