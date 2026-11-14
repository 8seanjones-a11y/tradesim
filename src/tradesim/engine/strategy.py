"""Strategy contract.

A strategy is a class with default `params` and a `signals(bars, p)` method that returns a
`Signals` object of numpy arrays aligned to `bars`. The value at index i may only use data
up to and including bar i (it is acted on at the open of bar i+1). See CLAUDE.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import polars as pl


@dataclass
class Signals:
    entry_long: np.ndarray
    exit_long: np.ndarray | None = None
    entry_short: np.ndarray | None = None
    exit_short: np.ndarray | None = None
    stop_dist: np.ndarray | float | None = None     # price distance from fill; NaN/None = no stop
    target_dist: np.ndarray | float | None = None   # price distance from fill; NaN/None = no target
    size: np.ndarray | float | None = None          # units/contracts (ignored when risk_frac>0 and stop set)

    def finalize(self, n: int) -> "Signals":
        def b(x):
            if x is None:
                return np.zeros(n, dtype=np.bool_)
            a = np.asarray(x)
            a = np.where(np.isnan(a.astype(float)), False, a).astype(np.bool_) if a.dtype != np.bool_ else a
            return np.broadcast_to(a, (n,)).copy()

        def f(x, default):
            if x is None:
                return np.full(n, default, dtype=np.float64)
            return np.broadcast_to(np.asarray(x, dtype=np.float64), (n,)).copy()

        return Signals(
            entry_long=b(self.entry_long), exit_long=b(self.exit_long),
            entry_short=b(self.entry_short), exit_short=b(self.exit_short),
            stop_dist=f(self.stop_dist, np.nan), target_dist=f(self.target_dist, np.nan),
            size=f(self.size, 1.0),
        )


class Strategy:
    """Base class. Subclass and set `name`, `params`, optionally `param_grid`, `symbols`, `timeframe`."""

    name: ClassVar[str] = "unnamed"
    description: ClassVar[str] = ""
    params: ClassVar[dict] = {}
    param_grid: ClassVar[dict] = {}          # for sensitivity / walk-forward: {"fast": [5,10,20], ...}
    symbol: ClassVar[str] = "SPY"            # default instrument
    timeframe: ClassVar[str] = "1d"          # default bar size: 1m,5m,15m,1h,4h,1d

    def signals(self, bars: pl.DataFrame, p: dict) -> Signals:  # pragma: no cover
        raise NotImplementedError

    def merged_params(self, overrides: dict | None = None) -> dict:
        p = dict(self.params)
        if overrides:
            p.update(overrides)
        return p


def load_strategy(name_or_path: str) -> Strategy:
    """Load a Strategy subclass from strategies/<name>.py or an explicit .py path."""
    import importlib.util
    from pathlib import Path

    from ..config import PROJECT_ROOT

    path = Path(name_or_path)
    if not path.suffix:
        path = PROJECT_ROOT / "strategies" / f"{name_or_path}.py"
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(f"strategies.{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    classes = [v for v in vars(mod).values()
               if isinstance(v, type) and issubclass(v, Strategy) and v is not Strategy]
    if not classes:
        raise ValueError(f"No Strategy subclass in {path}")
    return classes[-1]()
