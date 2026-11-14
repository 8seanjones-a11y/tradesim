"""Robustness scorecard: combine many independent checks into a single verdict.

Each check yields PASS / WARN / FAIL with a short reason. Verdict:
  STRONG        no FAILs and at most one WARN
  PROMISING     no FAILs
  FRAGILE       one or two FAILs
  LIKELY OVERFIT / NO EDGE   three or more FAILs, or the core significance checks fail
"""
from __future__ import annotations

import math

PASS, WARN, FAIL, SKIP = "PASS", "WARN", "FAIL", "SKIP"


def _chk(name, status, value, detail):
    return dict(check=name, status=status, value=value, detail=detail)


def build_scorecard(summary: dict, dsr: dict | None = None, wfo: dict | None = None, sens: dict | None = None,
                    perm: dict | None = None, null: dict | None = None, noise: dict | None = None,
                    cost: dict | None = None, mc_noise: dict | None = None, pbo: dict | None = None,
                    regimes=None) -> dict:
    c = []
    n = summary.get("n_trades", 0)
    c.append(_chk("Sample size", PASS if n >= 100 else WARN if n >= 30 else FAIL, n,
                  "≥100 trades for reliable statistics; <30 is anecdotal"))
    c.append(_chk("Net profit after costs", PASS if summary["net_profit"] > 0 else FAIL,
                  round(summary["net_profit"], 2), "must be positive after commissions/slippage"))
    t = summary.get("t_stat", 0)
    c.append(_chk("Trade t-statistic", PASS if t >= 2 else WARN if t >= 1 else FAIL, round(t, 2),
                  "mean trade / std error; ≥2 ≈ 95% significance"))
    if dsr:
        d = dsr["dsr"]
        c.append(_chk("Deflated Sharpe", PASS if d >= 0.95 else WARN if d >= 0.8 else FAIL, round(d, 3),
                      f"P(real Sharpe) after {dsr['n_trials']} trial(s)"))
    if perm:
        p = perm["p_value"]
        st = FAIL if perm["real"] <= 0 else PASS if p <= 0.05 else WARN if p <= 0.15 else FAIL
        c.append(_chk("Permutation test", st, round(p, 3),
                      "p-value vs. permuted (order-destroyed) prices" +
                      (" for the best grid params, incl. re-optimisation" if perm.get("optimised") else "") +
                      ("; real score ≤ 0 so nothing to defend" if perm["real"] <= 0 else "")))
    if null:
        p = null["p_value"]
        c.append(_chk(f"Null model ({null['model']})", PASS if p <= 0.05 else WARN if p <= 0.2 else FAIL, round(p, 3),
                      "share of edge-less synthetic markets that score as well"))
    if wfo and not math.isnan(wfo.get("wf_efficiency", float("nan"))):
        e = wfo["wf_efficiency"]
        c.append(_chk("Walk-forward efficiency", PASS if e >= 0.5 else WARN if e >= 0.2 else FAIL, round(e, 2),
                      f"OOS/IS score with params re-chosen each fold; {wfo.get('pct_profitable_folds', 0):.0%} of OOS folds profitable"))
    if sens and sens.get("best") is not None:
        p = sens["plateau"]
        c.append(_chk("Parameter plateau", PASS if p >= 0.6 else WARN if p >= 0.3 else FAIL, round(p, 2),
                      "share of neighbouring params within 50% of best (broad = robust)"))
    if pbo and not math.isnan(pbo.get("pbo", float("nan"))):
        p = pbo["pbo"]
        c.append(_chk("Prob. of backtest overfitting", PASS if p <= 0.2 else WARN if p <= 0.5 else FAIL, round(p, 2),
                      "CSCV: chance the in-sample best is below median out of sample"))
    if noise:
        p = noise["pct_profitable"]
        c.append(_chk("Noise injection", PASS if p >= 0.8 else WARN if p >= 0.5 else FAIL, round(p, 2),
                      "share of runs still profitable with jittered prices"))
    if cost:
        be = cost["breakeven_slippage_ticks"]
        c.append(_chk("Cost headroom", PASS if be >= 3 else WARN if be >= 1 else FAIL,
                      "∞" if be == float("inf") else round(be, 2), "extra slippage ticks/side before edge disappears"))
    if mc_noise:
        p = 1 - mc_noise["p_loss"]
        c.append(_chk("Execution-noise MC", PASS if p >= 0.9 else WARN if p >= 0.7 else FAIL, round(p, 2),
                      "P(profitable) with random slippage, fees, missed trades"))
    if regimes is not None and getattr(regimes, "height", 0):
        tr = regimes.filter(regimes["dimension"] == "vol_regime")
        pos = int((tr["pnl"] > 0).sum()) if tr.height else 0
        c.append(_chk("Regime consistency", PASS if tr.height and pos == tr.height else WARN if pos >= 2 else FAIL,
                      f"{pos}/{tr.height}", "volatility regimes with positive P&L"))

    fails = sum(x["status"] == FAIL for x in c)
    warns = sum(x["status"] == WARN for x in c)
    core_fail = any(x["status"] == FAIL for x in c if x["check"] in ("Net profit after costs", "Permutation test", "Deflated Sharpe"))
    if fails >= 3 or (core_fail and fails >= 2):
        verdict = "LIKELY OVERFIT / NO EDGE"
    elif fails >= 1:
        verdict = "FRAGILE"
    elif warns <= 1:
        verdict = "STRONG"
    else:
        verdict = "PROMISING"
    return dict(checks=c, verdict=verdict, n_fail=fails, n_warn=warns)
