"""Config loading: settings.yaml, instruments.yaml, prop firm rule files."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = Path(os.environ.get("TRADESIM_CONFIG_DIR", PROJECT_ROOT / "config"))


@lru_cache
def settings() -> dict:
    with open(CONFIG_DIR / "settings.yaml") as f:
        s = yaml.safe_load(f)
    s["data_dir"] = str(Path(os.environ.get("TRADESIM_DATA_DIR", s["data_dir"])).expanduser())
    runs = Path(s.get("runs_dir", "runs"))
    s["runs_dir"] = str(runs if runs.is_absolute() else PROJECT_ROOT / runs)
    return s


def data_dir() -> Path:
    return Path(settings()["data_dir"])


@dataclass(frozen=True)
class Instrument:
    symbol: str
    asset_class: str
    point_value: float = 1.0
    tick_size: float = 0.01
    lot_step: float = 1.0
    commission_per_unit: float = 0.0
    commission_pct: float = 0.0
    half_spread: float = 0.0
    carry_per_day: float = 0.0
    source: str = ""
    source_symbol: str = ""
    price_divisor: float = 1.0
    proxy_for: str | None = None
    daily_source: str | None = None
    daily_symbol: str | None = None
    market: str | None = None
    quote_ccy: str = "USD"
    extra: dict = field(default_factory=dict)

    @property
    def is_proxy(self) -> bool:
        return self.proxy_for is not None


@lru_cache
def _instruments_raw() -> dict:
    with open(CONFIG_DIR / "instruments.yaml") as f:
        return yaml.safe_load(f)


def instrument(symbol: str) -> Instrument:
    raw = _instruments_raw()
    inst = raw["instruments"].get(symbol)
    if inst is None:
        raise KeyError(f"Unknown instrument {symbol!r}; add it to config/instruments.yaml")
    merged = {**raw["defaults"].get(inst["asset_class"], {}), **inst}
    known = set(Instrument.__dataclass_fields__) - {"symbol", "extra"}
    kw = {k: v for k, v in merged.items() if k in known}
    extra = {k: v for k, v in merged.items() if k not in known}
    return Instrument(symbol=symbol, extra=extra, **kw)


def generic_instrument(symbol: str = "SYNTH", **overrides) -> Instrument:
    """Instrument for synthetic / ad-hoc data (tests, null-model runs)."""
    base = dict(asset_class="stocks", point_value=1.0, tick_size=0.01, lot_step=0.0)
    base.update(overrides)
    return Instrument(symbol=symbol, **base)


def list_instruments() -> list[str]:
    return list(_instruments_raw()["instruments"])
