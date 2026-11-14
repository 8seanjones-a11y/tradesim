"""One-call evaluation: backtest -> Monte Carlo -> robustness -> scorecard -> (prop firms) -> report.
This is what `tsim evaluate` runs."""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import polars as pl

from .analysis import account, montecarlo, overfit, robustness
from .analysis.metrics import daily_returns, summary
from .analysis.scorecard import build_scorecard
from .config import Instrument, settings
from .engine.backtest import BacktestResult, ExecConfig, run_backtest


def evaluate(strategy, bars: pl.DataFrame, inst: Instrument, params: dict | None = None,
             cfg: ExecConfig | None = None, level: str = "standard", prop: list[str] | str | None = None,
             benchmark_bars: pl.DataFrame | None = None, n_sims: int | None = None, progress=print) -> dict:
    """level: quick (backtest+MC), standard (+WFO, sensitivity, noise, costs, null), full (+permutation w/ optimisation, PBO)."""
    s = settings()
    cfg = cfg or ExecConfig()
    n_sims = n_sims or s["monte_carlo"]["n_sims"]
    seed = s["monte_carlo"]["seed"]
    acct = s["account"]
    t0 = time.time()
    progress("backtest...")
    res = run_backtest(strategy, bars, inst, params=params, cfg=cfg)
    summ = summary(res)
    pnl = res.pnl
    cap = res.initial_capital
    out: dict = dict(result=res, summary=summ, level=level)

    progress("monte carlo...")
    out["mc_shuffle"] = montecarlo.shuffle_mc(pnl, cap, n_sims, acct["ruin_drawdown"], seed)
    out["mc_bootstrap"] = montecarlo.bootstrap_mc(pnl, cap, n_sims, ruin_level=acct["ruin_drawdown"], seed=seed)
    out["mc_block"] = montecarlo.block_bootstrap_mc(pnl, cap, n_sims, ruin_level=acct["ruin_drawdown"], seed=seed)
    out["mc_exec"] = montecarlo.execution_noise_mc(res.trades, cap, inst.point_value, inst.tick_size,
                                                   min(n_sims, 2000), seed=seed)

    progress("account analysis...")
    out["horizons"] = account.horizon_outcomes(res, acct["horizons_years"], min(n_sims, 3000), ruin_dd=acct["ruin_drawdown"], seed=seed)
    out["risk_of_ruin"] = account.risk_of_ruin(pnl, cap, acct["ruin_drawdown"], seed=seed)
    out["capital_needed"] = account.capital_needed(pnl, acct["dd_tolerance"], seed=seed)
    tr = pnl / cap
    out["kelly"] = account.kelly_fraction(tr)
    out["optimal_f"] = account.optimal_f(pnl)
    out["benchmark"] = account.benchmark_compare(res, benchmark_bars)
    out["regimes"] = robustness.regime_breakdown(res)

    sens = wfo = noise = cost = null = perm = pbo = None
    trial_sr = None
    if level in ("standard", "full"):
        if strategy.param_grid:
            progress("parameter sensitivity...")
            sens = robustness.sensitivity(strategy, bars, inst, cfg=cfg)
            if sens["returns_matrix"] is not None:
                m = sens["returns_matrix"]
                sd = m.std(axis=0, ddof=1)
                trial_sr = np.where(sd > 0, m.mean(axis=0) / np.where(sd > 0, sd, 1), 0)
            progress("walk-forward...")
            wfo = robustness.walk_forward(strategy, bars, inst, cfg=cfg)
        progress("noise injection / costs / null model...")
        noise = robustness.noise_injection(strategy, bars, inst, params, cfg, n_runs=30 if level == "standard" else 100)
        cost = robustness.cost_sensitivity(strategy, bars, inst, params, cfg)
        null = robustness.null_model_test(strategy, bars, inst, params, cfg, n_paths=30 if level == "standard" else 100)
    if level == "full":
        progress("permutation test (re-optimising each permutation)...")
        perm = robustness.permutation_test(strategy, bars, inst, cfg=cfg, n_perm=100, optimise=bool(strategy.param_grid))
        if sens and sens["returns_matrix"] is not None and sens["returns_matrix"].shape[1] >= 4:
            pbo = overfit.pbo_cscv(sens["returns_matrix"], n_splits=10)
    n_trials = len(sens["table"]) if sens and sens.get("best") else 1
    out["dsr"] = overfit.deflated_sharpe(daily_returns(res.equity), n_trials, trial_sr)
    out.update(sensitivity=sens, wfo=wfo, noise=noise, cost=cost, null=null, permutation=perm, pbo=pbo)
    out["scorecard"] = build_scorecard(summ, out["dsr"], wfo, sens, perm, null, noise, cost, out["mc_exec"], pbo, out["regimes"])

    if prop:
        from .propfirm.evaluate import compare_firms
        from .propfirm.rules import all_rules, find_rules
        rules = all_rules(inst.asset_class) if prop in ("all", ["all"]) else find_rules(prop)
        progress(f"prop firms ({len(rules)})...")
        table, details = compare_firms(res, rules, n_sims=min(n_sims, 2000))
        out["prop"] = dict(table=table, details=details)
    out["elapsed_s"] = time.time() - t0
    return out


def save_run(ev: dict, name: str | None = None) -> Path:
    from .reports.html import render_report
    res: BacktestResult = ev["result"]
    run_id = name or f"{time.strftime('%Y%m%d-%H%M%S')}_{res.strategy}_{res.symbol}"
    d = Path(settings()["runs_dir"]) / run_id
    d.mkdir(parents=True, exist_ok=True)
    res.trades.write_parquet(d / "trades.parquet")
    res.equity.write_parquet(d / "equity.parquet")
    meta = dict(strategy=res.strategy, symbol=res.symbol, params=res.params, exec=asdict(res.exec_cfg),
                summary=ev["summary"], verdict=ev["scorecard"]["verdict"], checks=ev["scorecard"]["checks"],
                notes=res.notes, level=ev["level"])
    if "prop" in ev:
        meta["prop"] = ev["prop"]["table"].to_dicts()
    (d / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    (d / "report.html").write_text(render_report(ev))
    return d
