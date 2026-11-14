"""Monte Carlo prop firm evaluation: pass probability, time to pass, failure modes, payouts, EV."""
from __future__ import annotations

from collections import Counter

import numpy as np
import polars as pl

from ..analysis.montecarlo import stationary_bootstrap_idx
from .rules import FirmRuleSet
from .simulator import daily_profile, run_account


def evaluate_firm(result, rules: FirmRuleSet, n_sims: int = 2000, scale: float = 1.0, mean_block: float = 3.0,
                  funded_horizon: int = 120, path_days: int = 500, seed: int = 42, profile: pl.DataFrame | None = None) -> dict:
    prof = profile if profile is not None else daily_profile(result, rules.day_tz, rules.day_roll_hour)
    pnl = prof["pnl"].to_numpy() * scale
    low = prof["low"].to_numpy() * scale
    high = prof["high"].to_numpy() * scale
    traded = prof["traded"].to_numpy()
    max_pos = float(prof["max_pos"].max() or 0) * scale
    n = len(pnl)
    eval_phases = [p for p in rules.phases if p.kind == "evaluation"]
    cap = min((p.max_contracts for p in rules.phases if p.max_contracts), default=None)
    if cap is not None and max_pos > cap + 1e-9:
        return dict(firm=rules.key, scale=scale, feasible=False, reason=f"max position {max_pos:g} > limit {cap:g}")
    if n < 5:
        return dict(firm=rules.key, scale=scale, feasible=False, reason="too few trading days")
    rng = np.random.default_rng(seed)
    passed = 0
    days_to_pass, fail_reasons, payouts_total, n_payouts, costs = [], Counter(), [], [], []
    first_payout = 0
    funded_blown = 0
    for _ in range(n_sims):
        idx = stationary_bootstrap_idx(n, path_days, mean_block, rng)
        out = run_account(rules, pnl[idx], low[idx], high[idx], traded[idx], funded_horizon)
        ph = out["phases"]
        eval_ok = all(o.status == "passed" for name, o in ph if any(e.name == name for e in eval_phases)) and \
            len([1 for name, o in ph if any(e.name == name for e in eval_phases)]) == len(eval_phases)
        eval_days = sum(o.days for name, o in ph if any(e.name == name for e in eval_phases))
        months = max(1, int(np.ceil(eval_days / 21)))
        cost = rules.fee * (months if rules.fee_type == "monthly" else 1)
        if eval_ok:
            passed += 1
            days_to_pass.append(eval_days)
            cost += rules.activation_fee
            funded = [o for name, o in ph if name not in {e.name for e in eval_phases}]
            pays = sum(sum(o.payouts) for o in funded)
            npay = sum(len(o.payouts) for o in funded)
            payouts_total.append(pays)
            n_payouts.append(npay)
            if npay:
                first_payout += 1
            if any(o.status == "failed" for o in funded):
                funded_blown += 1
                fail_reasons["funded:" + next(o.reason for o in funded if o.status == "failed")] += 1
        else:
            last = ph[-1][1]
            fail_reasons[last.reason or last.status] += 1
            payouts_total.append(0.0)
            n_payouts.append(0)
        costs.append(cost)
    p_pass = passed / n_sims
    mean_pay = float(np.mean(payouts_total))
    mean_cost = float(np.mean(costs))
    return dict(
        firm=rules.key, scale=scale, feasible=True, verified=rules.verified, n_sims=n_sims,
        p_pass=p_pass, median_days_to_pass=float(np.median(days_to_pass)) if days_to_pass else float("nan"),
        p_first_payout=first_payout / n_sims, p_funded_blown_given_pass=funded_blown / passed if passed else float("nan"),
        mean_payouts=mean_pay, mean_n_payouts=float(np.mean(n_payouts)), mean_cost=mean_cost,
        ev_per_attempt=mean_pay - mean_cost, expected_attempts=1 / p_pass if p_pass > 0 else float("inf"),
        fail_reasons={k: v / n_sims for k, v in fail_reasons.most_common()}, account_size=rules.account_size,
        fee=rules.fee, fee_type=rules.fee_type, funded_horizon_days=funded_horizon,
    )


def best_scale(result, rules: FirmRuleSet, scales=(0.25, 0.5, 1, 2, 3, 4, 6, 8, 10), n_sims: int = 800, **kw) -> dict:
    """Search the position-size multiplier maximising EV per attempt (then re-run at full n_sims)."""
    prof = daily_profile(result, rules.day_tz, rules.day_roll_hour)
    lot = result.instrument.lot_step or 0
    base_units = float(result.trades["units"].median()) if result.trades.height else 1.0
    best = None
    for s in scales:
        if lot >= 1 and abs(round(base_units * s) - base_units * s) > 1e-9:
            continue  # whole contracts only
        r = evaluate_firm(result, rules, n_sims=n_sims, scale=s, profile=prof, **kw)
        if r["feasible"] and (best is None or r["ev_per_attempt"] > best["ev_per_attempt"]):
            best = r
    return best or dict(firm=rules.key, feasible=False, reason="no feasible size")


def compare_firms(result, rules_list: list[FirmRuleSet], n_sims: int = 2000, optimise_size: bool = True, **kw) -> pl.DataFrame:
    rows = []
    for rules in rules_list:
        if optimise_size:
            b = best_scale(result, rules, **kw)
            r = evaluate_firm(result, rules, n_sims=n_sims, scale=b.get("scale", 1.0), **kw) if b.get("feasible") else b
        else:
            r = evaluate_firm(result, rules, n_sims=n_sims, **kw)
        rows.append(r)
    keep = ["firm", "feasible", "verified", "scale", "p_pass", "median_days_to_pass", "p_first_payout",
            "mean_payouts", "mean_cost", "ev_per_attempt", "expected_attempts", "account_size", "fee", "fee_type"]
    flat = []
    for r in rows:
        d = {k: r.get(k) for k in keep}
        d["top_fail_reason"] = next(iter(r.get("fail_reasons", {}) or {}), r.get("reason", ""))
        flat.append(d)
    return pl.DataFrame(flat, infer_schema_length=None).sort("ev_per_attempt", descending=True, nulls_last=True), rows
