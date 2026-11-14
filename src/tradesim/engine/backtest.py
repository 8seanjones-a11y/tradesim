"""High-level backtest runner: Strategy + bars + Instrument -> BacktestResult."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np
import polars as pl

from ..config import Instrument, settings
from .kernel import EXIT_REASONS, run_kernel
from .strategy import Signals, Strategy


def trading_day_ids(ts: pl.Series, tz: str = "America/New_York", roll_hour: int = 17) -> np.ndarray:
    """Integer trading-day id per bar. A new day starts at `roll_hour` local time
    (17:00 New York = CME futures convention). Daily bars: each bar is its own day."""
    local = ts.dt.convert_time_zone(tz) if ts.dtype.time_zone else ts.dt.replace_time_zone("UTC").dt.convert_time_zone(tz)
    shifted = local + timedelta(hours=24 - roll_hour) if roll_hour else local
    d = shifted.dt.date().cast(pl.Int32).to_numpy()
    return d.astype(np.int64)


@dataclass
class ExecConfig:
    slippage_ticks: float | None = None
    initial_capital: float | None = None
    risk_frac: float = 0.0          # >0: size = capital*risk_frac / (stop_dist*point_value)
    max_bars_held: int = 0
    commission_mult: float = 1.0    # scale instrument commissions (MC execution noise)
    extra_slip: float = 0.0         # extra price units of slippage
    day_tz: str = "America/New_York"
    day_roll_hour: int = 17

    def resolved(self) -> "ExecConfig":
        s = settings()
        return ExecConfig(
            slippage_ticks=s["slippage_ticks"] if self.slippage_ticks is None else self.slippage_ticks,
            initial_capital=s["initial_capital"] if self.initial_capital is None else self.initial_capital,
            risk_frac=self.risk_frac, max_bars_held=self.max_bars_held,
            commission_mult=self.commission_mult, extra_slip=self.extra_slip,
            day_tz=self.day_tz, day_roll_hour=self.day_roll_hour,
        )


@dataclass
class BacktestResult:
    strategy: str
    symbol: str
    params: dict
    bars: pl.DataFrame
    equity: pl.DataFrame          # ts, equity, equity_lo, equity_hi, position, day
    trades: pl.DataFrame          # one row per closed trade
    instrument: Instrument
    exec_cfg: ExecConfig
    notes: list[str] = field(default_factory=list)

    @property
    def initial_capital(self) -> float:
        return float(self.exec_cfg.initial_capital)

    @property
    def pnl(self) -> np.ndarray:
        return self.trades["pnl"].to_numpy()

    def daily_pnl(self) -> pl.DataFrame:
        """P&L per trading day from the equity curve (includes open-trade MTM)."""
        eq = self.equity.group_by("day", maintain_order=True).agg(
            pl.col("ts").last(), pl.col("equity").last(), pl.col("equity_lo").min(), pl.col("equity_hi").max())
        prev = eq["equity"].shift(1).fill_null(self.initial_capital)
        return eq.with_columns(pnl=pl.col("equity") - prev, low_vs_open=pl.col("equity_lo") - prev,
                               high_vs_open=pl.col("equity_hi") - prev)


def run_backtest(strategy: Strategy, bars: pl.DataFrame, inst: Instrument,
                 params: dict | None = None, cfg: ExecConfig | None = None,
                 signals: Signals | None = None) -> BacktestResult:
    cfg = (cfg or ExecConfig()).resolved()
    p = strategy.merged_params(params)
    n = bars.height
    if n < 2:
        raise ValueError("need at least 2 bars")
    sig = (signals or strategy.signals(bars, p)).finalize(n)
    day = trading_day_ids(bars["ts"], cfg.day_tz, cfg.day_roll_hour)

    slip = cfg.slippage_ticks * inst.tick_size + inst.half_spread + cfg.extra_slip
    out = run_kernel(
        bars["open"].to_numpy().astype(np.float64), bars["high"].to_numpy().astype(np.float64),
        bars["low"].to_numpy().astype(np.float64), bars["close"].to_numpy().astype(np.float64),
        sig.entry_long, sig.exit_long, sig.entry_short, sig.exit_short,
        sig.stop_dist, sig.target_dist, sig.size, day,
        float(inst.point_value), float(slip),
        float(inst.commission_per_unit * cfg.commission_mult), float(inst.commission_pct * cfg.commission_mult),
        float(inst.carry_per_day), float(cfg.initial_capital), float(cfg.risk_frac),
        float(inst.lot_step), int(cfg.max_bars_held),
    )
    (eq, eq_lo, eq_hi, pos, ei, xi, d, ep, xp, u, pnl, cost, mae, mfe, reason) = out
    ts = bars["ts"]
    equity = pl.DataFrame({"ts": ts, "equity": eq, "equity_lo": eq_lo, "equity_hi": eq_hi,
                           "position": pos, "day": day})
    trades = pl.DataFrame({
        "entry_ts": ts.gather(ei) if len(ei) else pl.Series("entry_ts", [], dtype=ts.dtype),
        "exit_ts": ts.gather(xi) if len(xi) else pl.Series("exit_ts", [], dtype=ts.dtype),
        "entry_i": ei, "exit_i": xi, "direction": d, "entry_price": ep, "exit_price": xp,
        "units": u, "pnl": pnl, "costs": cost,
        "mae": mae * u * inst.point_value, "mfe": mfe * u * inst.point_value,
        "mae_price": mae, "mfe_price": mfe,
        "exit_reason": [EXIT_REASONS[int(r)] for r in reason],
        "bars_held": xi - ei,
        "exit_day": day[xi] if len(xi) else np.array([], dtype=np.int64),
    })
    notes = []
    if inst.is_proxy:
        notes.append(f"Proxy data: {inst.symbol} priced from {inst.source_symbol} as a stand-in for {inst.proxy_for}.")
    if inst.quote_ccy != "USD":
        notes.append(f"P&L is in {inst.quote_ccy} (no FX conversion applied).")
    return BacktestResult(strategy.name, inst.symbol, p, bars, equity, trades, inst, cfg, notes)
