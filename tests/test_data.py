import io
import lzma
import struct
import zipfile
from datetime import date, datetime, timedelta, timezone

import numpy as np
import polars as pl
import pytest

from tradesim.data.adapters import binance, dukascopy


@pytest.fixture(autouse=True)
def tmp_data(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADESIM_DATA_DIR", str(tmp_path))
    from tradesim import config
    config.settings.cache_clear()
    yield
    config.settings.cache_clear()


def test_binance_ms_and_us_timestamps():
    t = datetime(2024, 12, 31, 23, 59, tzinfo=timezone.utc)
    ms = int(t.timestamp() * 1000)
    us = int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp() * 1_000_000)
    csv = f"{ms},1,2,0.5,1.5,10,0,0,0,0,0,0\n{us},1.5,2,1,1.8,11,0,0,0,0,0,0\n".encode()
    df = binance.parse_kline_csv(csv)
    assert df["ts"].to_list() == [t, datetime(2025, 1, 1, tzinfo=timezone.utc)]
    hdr = b"open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,taker_buy_quote_volume,ignore\n"
    assert binance.parse_kline_csv(hdr + csv).height == 2


def test_dukascopy_decode_roundtrip():
    recs = b"".join(struct.pack(">5If", 60 * i, 110000 + i, 110010 + i, 109990, 110020 + i, 1.5) for i in range(3))
    blob = lzma.compress(recs, format=lzma.FORMAT_ALONE)
    df = dukascopy.decode_candles(blob, date(2024, 3, 4), 100000)
    assert df.height == 3
    assert df["ts"][1] == datetime(2024, 3, 4, 0, 1, tzinfo=timezone.utc)
    assert df["open"][0] == pytest.approx(1.1) and df["high"][2] == pytest.approx(1.10022)


def test_store_write_merge_and_resample():
    from tradesim.data import store, synthetic
    b = synthetic.gbm(60 * 24 * 3, step=timedelta(minutes=1), start=datetime(2024, 1, 2, tzinfo=timezone.utc))
    store.write_bars(b.head(2000), "crypto", "BTCUSDT", "1m")
    store.write_bars(b.slice(1500), "crypto", "BTCUSDT", "1m")   # overlapping merge
    m1 = store.load_bars("BTCUSDT", "1m")
    assert m1.height == b.height and m1["ts"].is_sorted()
    h1 = store.load_bars("BTCUSDT", "1h")
    assert h1.height == 72
    first = b.head(60)
    assert h1["open"][0] == pytest.approx(first["open"][0]) and h1["close"][0] == pytest.approx(first["close"][-1])
    assert h1["high"][0] == pytest.approx(first["high"].max())
