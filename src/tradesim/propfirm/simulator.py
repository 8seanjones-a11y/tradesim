"""Prop firm rule state machine operating on daily profiles.

A daily profile is, per trading day (in the firm's day convention):
    pnl   - close-to-close change in equity
    low   - worst intraday equity relative to the prior close (<= 0 usually)
    high  - best intraday equity relative to the prior close
    traded- whether a position was held / a trade closed that day

Intraday ordering is unknown at daily resolution, so the simulator is PESSIMISTIC: on a day,
the high is assumed to occur first (raising an intraday trailing floor), then the low.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from .rules import FirmRuleSet, Phase


def daily_profile(result, tz: str, roll_hour: int) -> pl.DataFrame:
    from ..engine.backtest import trading_day_ids
    eq = result.equity.with_columns(fday=pl.Series(trading_day_ids(result.equity["ts"], tz, roll_hour)))
    g = eq.group_by("fday", maintain_order=True).agg(
        pl.col("equity").last().alias("close"), pl.col("equity_lo").min().alias("lo"),
        pl.col("equity_hi").max().alias("hi"), (pl.col("position") != 0).any().alias("held"),
        pl.col("position").abs().max().alias("max_pos"))
    prev = g["close"].shift(1).fill_null(result.initial_capital)
    tdays = set(result.trades["exit_ts"].to_list()) if result.trades.height else set()
    exit_days = set(trading_day_ids(pl.Series(list(tdays), dtype=result.equity["ts"].dtype), tz, roll_hour).tolist()) if tdays else set()
    traded = g["held"].to_numpy() | np.isin(g["fday"].to_numpy(), list(exit_days))
    return pl.DataFrame({"day": g["fday"], "pnl": g["close"] - prev, "low": np.minimum(g["lo"] - prev, 0),
                         "high": np.maximum(g["hi"] - prev, 0), "traded": traded, "max_pos": g["max_pos"]})


@dataclass
class PhaseOutcome:
    status: str                 # passed | failed | timeout | survived (funded horizon reached)
    reason: str = ""
    days: int = 0
    trading_days: int = 0
    end_balance: float = 0.0
    payouts: list = field(default_factory=list)


def _amt(x, unit, size):
    return x * size if unit == "pct" else x


def run_phase(ph: Phase, size: float, pnl, low, high, traded, start_i: int = 0,
              horizon: int | None = None) -> tuple[PhaseOutcome, int]:
    """Simulate one phase starting at day index start_i. Returns (outcome, next_day_index)."""
    start = size
    bal = size
    ml = ph.max_loss
    ml_amt = _amt(ml.amount, ml.unit, size)
    floor = start - ml_amt
    peak = start
    lock_level = {"none": np.inf, "start_balance": start, "start_plus_x": start + ml.lock_offset}[ml.lock_at]
    target = _amt(ph.profit_target, ph.target_unit, size) if ph.profit_target is not None else None
    dl = ph.daily_loss
    best_day = 0.0
    tdays = 0
    qual_days = 0
    payouts = []
    cycle_qual = 0
    total_withdrawn = 0.0
    n = len(pnl)
    i = start_i
    d = 0
    limit = horizon if horizon is not None else (ph.max_calendar_days or 10_000)
    while i < n and d < limit:
        p, lo, hi, tr = pnl[i], low[i], high[i], traded[i]
        day_open = bal
        i += 1
        d += 1
        if tr:
            tdays += 1
        # --- intraday trailing floor (pessimistic: high first) ---
        if ml.type == "trailing" and ml.trail_mode == "intraday_equity":
            pk = day_open + hi
            if pk > peak:
                peak = pk
                floor = min(max(floor, peak - ml_amt), lock_level) if lock_level != np.inf else max(floor, peak - ml_amt)
        # --- daily loss ---
        if dl is not None:
            dl_amt = _amt(dl.amount, dl.unit, size)          # pct is of the INITIAL balance
            worst = day_open + (lo if dl.basis == "equity" else min(p, 0.0))
            ref = day_open if dl.anchor == "prior_day_close" else start
            if worst <= ref - dl_amt:
                if dl.breach_action == "fail":
                    return PhaseOutcome("failed", "daily_loss", d, tdays, worst, payouts), i
                # pause_day: trading halts for the day at the limit
                cap = ref - dl_amt - day_open
                p = max(p, cap)
                lo = max(lo, cap)
        # --- max loss (intraday equity vs current floor) ---
        if day_open + lo <= floor:
            return PhaseOutcome("failed", "max_loss" if ml.type == "static" else "trailing_drawdown", d, tdays,
                                floor, payouts), i
        bal = day_open + p
        # --- EOD trailing update ---
        if ml.type == "trailing" and ml.trail_mode == "eod_balance" and bal > peak:
            peak = bal
            nf = peak - ml_amt
            floor = max(floor, min(nf, lock_level))
        if p > best_day:
            best_day = p
        if ph.min_day_profit_to_count is None or p >= ph.min_day_profit_to_count:
            qual_days += 1 if tr else 0
        profit = bal - start

        # --- evaluation pass check ---
        if ph.kind == "evaluation" and target is not None and profit >= target:
            ok_days = (qual_days if ph.min_day_profit_to_count is not None else tdays) >= ph.min_trading_days
            ok_cons = True
            for c in ph.consistency:
                denom = profit if c.metric == "best_day_over_profit" else target
                if denom > 0 and best_day / denom > c.max_ratio:
                    if c.effect == "fail":
                        return PhaseOutcome("failed", "consistency", d, tdays, bal, payouts), i
                    ok_cons = False
            if ok_days and ok_cons:
                return PhaseOutcome("passed", "", d, tdays, bal, payouts), i

        # --- funded payouts ---
        if ph.kind == "funded" and ph.payout is not None:
            po = ph.payout
            if tr and p >= po.min_day_profit:
                cycle_qual += 1
            above = bal - start
            net_req = po.safety_net or 0.0
            if cycle_qual >= po.min_days and above > net_req and (po.max_payouts is None or len(payouts) < po.max_payouts):
                ok = True
                if po.consistency is not None and above > 0 and best_day / above > po.consistency.max_ratio:
                    ok = False
                if ok:
                    cap = po.caps[min(len(payouts), len(po.caps) - 1)] if po.caps else np.inf
                    amt = min(cap, po.max_pct_of_profit * above, above - net_req)
                    if amt >= max(po.min_amount, 1e-9):
                        payouts.append(amt * po.split)
                        total_withdrawn += amt
                        bal -= amt
                        cycle_qual = 0
                        best_day = 0.0
    status = "survived" if ph.kind == "funded" else "timeout"
    if i >= n and ph.kind == "evaluation" and status == "timeout" and (ph.max_calendar_days is None or d < ph.max_calendar_days):
        status = "incomplete"
    return PhaseOutcome(status, "out_of_days" if status != "survived" else "", d, tdays, bal, payouts), i


def run_account(rules: FirmRuleSet, pnl, low, high, traded, funded_horizon: int = 120) -> dict:
    """Run all phases in sequence on one day-path. Funded phase runs for `funded_horizon` days."""
    i = 0
    phases = []
    for ph in rules.phases:
        hz = funded_horizon if ph.kind == "funded" else None
        out, i = run_phase(ph, rules.account_size, pnl, low, high, traded, i, hz)
        phases.append((ph.name, out))
        if ph.kind == "evaluation" and out.status != "passed":
            break
    return dict(phases=phases)
