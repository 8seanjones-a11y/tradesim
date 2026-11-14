"""Local market-data store: Parquet partitioned by asset_class/symbol/timeframe/year, queried with DuckDB.

Layout:
  <data_dir>/raw/<source>/...                               untouched vendor downloads (cache)
  <data_dir>/curated/<asset_class>/<symbol>/<tf>/year=YYYY/part.parquet
  <data_dir>/catalog.duckdb                                 ingest log
Schema: ts (UTC timestamp), open, high, low, close, volume [, spread]
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import duckdb
import polars as pl

from ..config import data_dir, instrument

TF_SECONDS = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600, "4h": 14400, "1d": 86400}
SCHEMA = {"ts": pl.Datetime("us", "UTC"), "open": pl.Float64, "high": pl.Float64, "low": pl.Float64,
          "close": pl.Float64, "volume": pl.Float64}


def curated_dir(asset_class: str, symbol: str, tf: str) -> Path:
    return data_dir() / "curated" / asset_class / symbol / tf


def raw_dir(source: str) -> Path:
    p = data_dir() / "raw" / source
    p.mkdir(parents=True, exist_ok=True)
    return p


def normalize(df: pl.DataFrame) -> pl.DataFrame:
    """Coerce to the standard schema, sort, drop duplicates and impossible bars."""
    if df["ts"].dtype.time_zone is None if hasattr(df["ts"].dtype, "time_zone") else True:
        df = df.with_columns(pl.col("ts").dt.replace_time_zone("UTC"))
    cols = [pl.col("ts").dt.convert_time_zone("UTC").cast(SCHEMA["ts"])]
    cols += [pl.col(c).cast(pl.Float64) for c in ("open", "high", "low", "close")]
    cols.append(pl.col("volume").cast(pl.Float64) if "volume" in df.columns else pl.lit(0.0).alias("volume"))
    if "spread" in df.columns:
        cols.append(pl.col("spread").cast(pl.Float64))
    df = df.select(cols).drop_nulls(["open", "high", "low", "close"])
    df = df.filter((pl.col("high") >= pl.col("low")) & (pl.col("close") > 0))
    return df.unique("ts", keep="last").sort("ts")


def write_bars(df: pl.DataFrame, asset_class: str, symbol: str, tf: str, source: str = "") -> int:
    """Merge bars into the store (idempotent; newer data overwrites same timestamps)."""
    df = normalize(df)
    if df.is_empty():
        return 0
    base = curated_dir(asset_class, symbol, tf)
    for (year,), part in df.with_columns(pl.col("ts").dt.year().alias("_y")).group_by("_y"):
        d = base / f"year={year}"
        d.mkdir(parents=True, exist_ok=True)
        f = d / "part.parquet"
        part = part.drop("_y")
        if f.exists():
            old = pl.read_parquet(f)
            part = pl.concat([old, part], how="diagonal_relaxed").unique("ts", keep="last").sort("ts")
        part.write_parquet(f, compression="zstd")
    _log(symbol, tf, source, df)
    return df.height


def _catalog() -> duckdb.DuckDBPyConnection:
    data_dir().mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(data_dir() / "catalog.duckdb"))
    con.execute("""CREATE TABLE IF NOT EXISTS ingest_log(symbol TEXT, tf TEXT, source TEXT,
                   first_ts TIMESTAMPTZ, last_ts TIMESTAMPTZ, rows BIGINT, ingested_at TIMESTAMPTZ)""")
    return con


def _log(symbol, tf, source, df):
    with _catalog() as con:
        con.execute("INSERT INTO ingest_log VALUES (?,?,?,?,?,?,?)",
                    [symbol, tf, source, df["ts"][0], df["ts"][-1], df.height, datetime.now(timezone.utc)])


def available() -> pl.DataFrame:
    """Summary of everything in the store."""
    root = data_dir() / "curated"
    rows = []
    if root.exists():
        for tf_dir in root.glob("*/*/*"):
            ac, sym, tf = tf_dir.parts[-3:]
            files = list(tf_dir.glob("year=*/part.parquet"))
            if not files:
                continue
            q = duckdb.sql(f"SELECT min(ts), max(ts), count(*) FROM read_parquet('{tf_dir}/year=*/part.parquet')").fetchall()[0]
            rows.append(dict(asset_class=ac, symbol=sym, timeframe=tf, first=q[0], last=q[1], rows=q[2]))
    return pl.DataFrame(rows) if rows else pl.DataFrame(schema={"asset_class": pl.Utf8, "symbol": pl.Utf8, "timeframe": pl.Utf8})


def _stored_tfs(asset_class: str, symbol: str) -> list[str]:
    base = data_dir() / "curated" / asset_class / symbol
    return [p.name for p in base.iterdir()] if base.exists() else []


def load_bars(symbol: str, tf: str = "1d", start: str | None = None, end: str | None = None,
              session_tz: str = "America/New_York") -> pl.DataFrame:
    """Load bars at `tf`, resampling from the finest stored timeframe if needed.
    Daily bars from intraday data use a 17:00 New York roll for futures/forex, calendar day otherwise."""
    inst = instrument(symbol)
    ac = inst.asset_class
    stored = _stored_tfs(ac, symbol)
    if not stored:
        raise FileNotFoundError(f"No data for {symbol}. Run: tsim data pull {symbol}")
    want = TF_SECONDS[tf]
    usable = sorted((t for t in stored if TF_SECONDS.get(t, 1e18) <= want and want % TF_SECONDS[t] == 0),
                    key=lambda t: -TF_SECONDS[t])
    if not usable:
        raise ValueError(f"{symbol}: stored {stored}, cannot build {tf}")
    src = usable[0]
    path = curated_dir(ac, symbol, src)
    where = []
    if start:
        where.append(f"ts >= TIMESTAMPTZ '{start}'")
    if end:
        where.append(f"ts < TIMESTAMPTZ '{end}'")
    w = ("WHERE " + " AND ".join(where)) if where else ""
    glob = f"{path}/year=*/part.parquet"
    if src == tf:
        sql = f"SELECT ts, open, high, low, close, volume FROM read_parquet('{glob}') {w} ORDER BY ts"
    elif tf == "1d" and ac in ("futures", "forex"):
        # trading day rolls at 17:00 New York
        sql = f"""SELECT min(ts) AS ts, arg_min(open, ts) AS open, max(high) AS high, min(low) AS low,
                  arg_max(close, ts) AS close, sum(volume) AS volume FROM (
                    SELECT *, CAST((timezone('{session_tz}', ts) + INTERVAL 7 HOUR) AS DATE) AS tday
                    FROM read_parquet('{glob}') {w}) GROUP BY tday ORDER BY ts"""
    else:
        sql = f"""SELECT time_bucket(INTERVAL {want} SECOND, ts) AS ts, arg_min(open, ts) AS open, max(high) AS high,
                  min(low) AS low, arg_max(close, ts) AS close, sum(volume) AS volume
                  FROM read_parquet('{glob}') {w} GROUP BY 1 ORDER BY 1"""
    con = duckdb.connect()
    con.execute("SET TimeZone='UTC'")
    df = con.sql(sql).pl()
    return df.with_columns(pl.col("ts").dt.convert_time_zone("UTC"))


def gaps(symbol: str, tf: str = "1m", max_gap_minutes: int = 60 * 72) -> pl.DataFrame:
    """Report gaps larger than expected (weekends excluded by the default threshold)."""
    b = load_bars(symbol, tf)
    d = b.with_columns(gap=(pl.col("ts").diff().dt.total_minutes()))
    return d.filter(pl.col("gap") > max_gap_minutes).select("ts", "gap")
