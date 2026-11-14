"""Numba execution kernel.

Conventions (these are invariants the rest of the system relies on):
  * Signals are evaluated on the CLOSE of bar i and filled at the OPEN of bar i+1.
  * Stops/targets are attached at fill, measured as a price distance from the fill price,
    and checked intrabar from the fill bar onward using high/low.
  * Gaps through a stop/target fill at the bar open.
  * If stop and target are both touched in the same bar, the STOP is assumed first (pessimistic).
  * Market and stop fills pay `slip` (price units) adversely; target (limit) fills do not.
  * One position at a time per instrument: flat, long or short. An opposite entry signal reverses.
  * equity[i]     = mark-to-market at close of bar i
    equity_lo[i]  = worst intrabar equity in bar i (long marked at low, short at high)
    equity_hi[i]  = best intrabar equity in bar i  (for intraday trailing-drawdown rules)
"""
from __future__ import annotations

import numpy as np
from numba import njit

EXIT_SIGNAL, EXIT_STOP, EXIT_TARGET, EXIT_END, EXIT_REVERSE, EXIT_TIME = 0, 1, 2, 3, 4, 5
EXIT_REASONS = {0: "signal", 1: "stop", 2: "target", 3: "end", 4: "reverse", 5: "time"}


@njit(cache=True)
def _round_lot(units, lot_step):
    if lot_step > 0:
        return np.floor(units / lot_step + 1e-9) * lot_step
    return units


@njit(cache=True)
def run_kernel(
    open_, high, low, close,
    entry_long, exit_long, entry_short, exit_short,
    stop_dist, target_dist, size,
    day_index,              # int array: trading-day id per bar (for carry costs)
    point_value, slip, commission_per_unit, commission_pct, carry_per_day,
    initial_capital, risk_frac, lot_step, max_bars_held,
):
    n = open_.shape[0]
    equity = np.empty(n)
    equity_lo = np.empty(n)
    equity_hi = np.empty(n)
    position = np.zeros(n)

    # trade records
    t_entry_i = np.empty(n, np.int64)
    t_exit_i = np.empty(n, np.int64)
    t_dir = np.empty(n, np.int64)
    t_entry_p = np.empty(n)
    t_exit_p = np.empty(n)
    t_units = np.empty(n)
    t_pnl = np.empty(n)
    t_cost = np.empty(n)
    t_mae = np.empty(n)
    t_mfe = np.empty(n)
    t_reason = np.empty(n, np.int64)
    nt = 0

    cash = initial_capital      # realized equity
    pos_dir = 0
    units = 0.0
    entry_p = 0.0
    entry_i = 0
    stop_p = np.nan
    target_p = np.nan
    trade_cost = 0.0
    mae = 0.0
    mfe = 0.0

    pending = 0  # 0 none, 1 open long, -1 open short, 2 close only
    pend_stop = np.nan
    pend_target = np.nan
    pend_size = 0.0
    pend_time = False

    for i in range(n):
        o = open_[i]
        h = high[i]
        l = low[i]
        c = close[i]

        # ---------- 1) execute pending market orders at this bar's open ----------
        if pending != 0 and i > 0:
            want = pending if pending != 2 else 0
            if pos_dir != 0 and want != pos_dir:
                px = o - slip * pos_dir
                cost = commission_per_unit * units + commission_pct * abs(px * units * point_value)
                pnl = (px - entry_p) * pos_dir * units * point_value
                cash += pnl - cost
                trade_cost += cost
                t_entry_i[nt] = entry_i; t_exit_i[nt] = i; t_dir[nt] = pos_dir
                t_entry_p[nt] = entry_p; t_exit_p[nt] = px; t_units[nt] = units
                t_pnl[nt] = pnl - trade_cost; t_cost[nt] = trade_cost
                t_mae[nt] = mae; t_mfe[nt] = mfe
                t_reason[nt] = EXIT_REVERSE if want != 0 else (EXIT_TIME if pend_time else EXIT_SIGNAL)
                nt += 1
                pos_dir = 0
                units = 0.0
            if want != 0 and pos_dir == 0:
                px = o + slip * want
                u = pend_size
                if risk_frac > 0 and pend_stop == pend_stop and pend_stop > 0:
                    u = (cash * risk_frac) / (pend_stop * point_value)
                u = _round_lot(u, lot_step)
                if u > 0:
                    pos_dir = want
                    units = u
                    entry_p = px
                    entry_i = i
                    cost = commission_per_unit * units + commission_pct * abs(px * units * point_value)
                    cash -= cost
                    trade_cost = cost
                    mae = 0.0
                    mfe = 0.0
                    stop_p = px - pos_dir * pend_stop if pend_stop == pend_stop else np.nan
                    target_p = px + pos_dir * pend_target if pend_target == pend_target else np.nan
        pending = 0
        pend_time = False

        # ---------- 2) intrabar stop / target / time exits ----------
        exited_intrabar = False
        if pos_dir != 0:
            hit_px = np.nan
            reason = -1
            if pos_dir == 1:
                if stop_p == stop_p and l <= stop_p:
                    hit_px = min(o, stop_p) - slip
                    reason = EXIT_STOP
                elif target_p == target_p and h >= target_p:
                    hit_px = max(o, target_p)
                    reason = EXIT_TARGET
            else:
                if stop_p == stop_p and h >= stop_p:
                    hit_px = max(o, stop_p) + slip
                    reason = EXIT_STOP
                elif target_p == target_p and l <= target_p:
                    hit_px = min(o, target_p)
                    reason = EXIT_TARGET

            # excursions use the extreme reached before exit (approximation within the bar)
            adverse = (entry_p - l) if pos_dir == 1 else (h - entry_p)
            favor = (h - entry_p) if pos_dir == 1 else (entry_p - l)
            if reason == EXIT_STOP:
                adverse = abs(entry_p - hit_px)
            if reason == EXIT_TARGET:
                favor = abs(hit_px - entry_p)
            if adverse > mae:
                mae = adverse
            if favor > mfe:
                mfe = favor

            if reason >= 0:
                # worst/best intrabar equity before exit
                lo_mark = l if pos_dir == 1 else h
                hi_mark = h if pos_dir == 1 else l
                if reason == EXIT_STOP:
                    lo_mark = hit_px
                if reason == EXIT_TARGET:
                    hi_mark = hit_px
                equity_lo[i] = cash + (lo_mark - entry_p) * pos_dir * units * point_value
                equity_hi[i] = cash + (hi_mark - entry_p) * pos_dir * units * point_value
                cost = commission_per_unit * units + commission_pct * abs(hit_px * units * point_value)
                pnl = (hit_px - entry_p) * pos_dir * units * point_value
                cash += pnl - cost
                trade_cost += cost
                t_entry_i[nt] = entry_i; t_exit_i[nt] = i; t_dir[nt] = pos_dir
                t_entry_p[nt] = entry_p; t_exit_p[nt] = hit_px; t_units[nt] = units
                t_pnl[nt] = pnl - trade_cost; t_cost[nt] = trade_cost
                t_mae[nt] = mae; t_mfe[nt] = mfe; t_reason[nt] = reason
                nt += 1
                pos_dir = 0
                units = 0.0
                exited_intrabar = True

        # ---------- 3) carry cost (swap / funding / borrow) at day change ----------
        if pos_dir != 0 and carry_per_day > 0 and i > 0 and day_index[i] != day_index[i - 1]:
            cc = carry_per_day * abs(c * units * point_value)
            cash -= cc
            trade_cost += cc

        # ---------- 4) mark to market ----------
        if pos_dir != 0:
            equity[i] = cash + (c - entry_p) * pos_dir * units * point_value
            lo_mark = l if pos_dir == 1 else h
            hi_mark = h if pos_dir == 1 else l
            equity_lo[i] = cash + (lo_mark - entry_p) * pos_dir * units * point_value
            equity_hi[i] = cash + (hi_mark - entry_p) * pos_dir * units * point_value
        else:
            equity[i] = cash
            if not exited_intrabar:
                equity_lo[i] = cash
                equity_hi[i] = cash
            else:
                if equity_lo[i] > cash:
                    equity_lo[i] = cash
                if equity_hi[i] < cash:
                    equity_hi[i] = cash
        position[i] = pos_dir * units

        # ---------- 5) read signals at close -> schedule orders for next open ----------
        if i < n - 1:
            if pos_dir != 0 and max_bars_held > 0 and (i - entry_i + 1) >= max_bars_held:
                pending = 2
                pend_time = True
            if entry_long[i] and pos_dir != 1:
                pending = 1
            elif entry_short[i] and pos_dir != -1:
                pending = -1
            elif pos_dir == 1 and exit_long[i]:
                pending = 2
            elif pos_dir == -1 and exit_short[i]:
                pending = 2
            if pending == 1 or pending == -1:
                pend_stop = stop_dist[i]
                pend_target = target_dist[i]
                pend_size = size[i]

    # ---------- close any open position at the final close ----------
    if pos_dir != 0:
        px = close[n - 1] - slip * pos_dir
        cost = commission_per_unit * units + commission_pct * abs(px * units * point_value)
        pnl = (px - entry_p) * pos_dir * units * point_value
        cash += pnl - cost
        trade_cost += cost
        t_entry_i[nt] = entry_i; t_exit_i[nt] = n - 1; t_dir[nt] = pos_dir
        t_entry_p[nt] = entry_p; t_exit_p[nt] = px; t_units[nt] = units
        t_pnl[nt] = pnl - trade_cost; t_cost[nt] = trade_cost
        t_mae[nt] = mae; t_mfe[nt] = mfe; t_reason[nt] = EXIT_END
        nt += 1
        equity[n - 1] = cash
        if equity_lo[n - 1] > cash:
            equity_lo[n - 1] = cash
        position[n - 1] = 0.0

    return (equity, equity_lo, equity_hi, position,
            t_entry_i[:nt], t_exit_i[:nt], t_dir[:nt], t_entry_p[:nt], t_exit_p[:nt],
            t_units[:nt], t_pnl[:nt], t_cost[:nt], t_mae[:nt], t_mfe[:nt], t_reason[:nt])
