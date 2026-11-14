"""`tsim` command line."""
from __future__ import annotations

import json
from datetime import date, timedelta
from typing import Optional

import typer

app = typer.Typer(help="tradesim - backtest, stress-test and prop-firm-test trading strategies", no_args_is_help=True)
data_app = typer.Typer(help="Market data: pull, list, check", no_args_is_help=True)
app.add_typer(data_app, name="data")


def _parse_params(items: list[str] | None) -> dict:
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def _bars(strategy, symbol, tf, start, end, synthetic):
    from .config import generic_instrument, instrument
    if synthetic:
        from .data import synthetic as syn
        n = 3000
        b = {"gbm": syn.gbm, "garch": syn.garch, "jump": syn.jump_diffusion}[synthetic](n=n, seed=0)
        return b, generic_instrument(symbol or "SYNTH")
    from .data.store import load_bars
    sym = symbol or strategy.symbol
    return load_bars(sym, tf or strategy.timeframe, start, end), instrument(sym)


@app.command()
def evaluate(strategy: str = typer.Argument(..., help="strategy name in strategies/ or path"),
             symbol: Optional[str] = typer.Option(None, "--symbol", "-s"),
             tf: Optional[str] = typer.Option(None, "--tf", help="1m,5m,15m,1h,4h,1d"),
             start: Optional[str] = None, end: Optional[str] = None,
             param: Optional[list[str]] = typer.Option(None, "--param", "-p", help="key=value (repeatable)"),
             level: str = typer.Option("standard", help="quick | standard | full"),
             prop: Optional[str] = typer.Option(None, help="'all' or comma list of firm files, e.g. topstep,apex"),
             capital: Optional[float] = None, slippage: Optional[float] = None,
             risk: float = typer.Option(0.0, help="risk fraction per trade (needs a stop); 0 = use strategy size"),
             synthetic: Optional[str] = typer.Option(None, help="run on synthetic data: gbm|garch|jump"),
             benchmark: Optional[str] = typer.Option(None, help="benchmark symbol (default: buy & hold of traded symbol)"),
             open_report: bool = typer.Option(False, "--open")):
    """Full evaluation: backtest + Monte Carlo + robustness scorecard (+ prop firms) -> HTML report."""
    from .engine.backtest import ExecConfig
    from .engine.strategy import load_strategy
    from .pipeline import evaluate as run_eval, save_run
    strat = load_strategy(strategy)
    bars, inst = _bars(strat, symbol, tf, start, end, synthetic)
    bench = None
    if benchmark:
        from .data.store import load_bars
        bench = load_bars(benchmark, "1d", start, end)
    cfg = ExecConfig(slippage_ticks=slippage, initial_capital=capital, risk_frac=risk)
    firms = None if not prop else ("all" if prop == "all" else prop.split(","))
    ev = run_eval(strat, bars, inst, _parse_params(param), cfg, level, firms, bench, progress=lambda m: typer.echo(f"  · {m}"))
    d = save_run(ev)
    s, sc = ev["summary"], ev["scorecard"]
    typer.echo(f"\n{strat.name} on {inst.symbol}: net {s['net_profit']:,.0f}  CAGR {s['cagr']:.1%}  Sharpe {s['sharpe']:.2f}  "
               f"maxDD {s['max_dd']:.1%}  trades {s['n_trades']}")
    typer.echo(f"VERDICT: {sc['verdict']}  ({sc['n_fail']} fail, {sc['n_warn']} warn)")
    for c in sc["checks"]:
        typer.echo(f"  [{c['status']:4}] {c['check']}: {c['value']}")
    if "prop" in ev:
        typer.echo("\nProp firms (best size):")
        for r in ev["prop"]["table"].iter_rows(named=True):
            if r["feasible"]:
                typer.echo(f"  {r['firm']:40} pass {r['p_pass']:.0%}  EV/attempt ${r['ev_per_attempt']:,.0f}  x{r['scale']}")
            else:
                typer.echo(f"  {r['firm']:40} not feasible ({r['top_fail_reason']})")
    typer.echo(f"\nReport: {d / 'report.html'}")
    if open_report:
        import webbrowser
        webbrowser.open((d / "report.html").as_uri())


@app.command()
def backtest(strategy: str, symbol: Optional[str] = typer.Option(None, "--symbol", "-s"), tf: Optional[str] = None,
             start: Optional[str] = None, end: Optional[str] = None,
             param: Optional[list[str]] = typer.Option(None, "--param", "-p"), synthetic: Optional[str] = None):
    """Quick backtest only (no report). Prints summary metrics."""
    from .analysis.metrics import summary
    from .engine.backtest import run_backtest
    from .engine.strategy import load_strategy
    strat = load_strategy(strategy)
    bars, inst = _bars(strat, symbol, tf, start, end, synthetic)
    r = run_backtest(strat, bars, inst, _parse_params(param))
    for k, v in summary(r).items():
        typer.echo(f"{k:28} {v:,.4f}" if isinstance(v, float) else f"{k:28} {v}")


@app.command()
def prop(strategy: str, firms: str = typer.Option("all"), symbol: Optional[str] = typer.Option(None, "--symbol", "-s"),
         tf: Optional[str] = None, start: Optional[str] = None, end: Optional[str] = None,
         param: Optional[list[str]] = typer.Option(None, "--param", "-p"), n_sims: int = 2000):
    """Prop firm pass-rate / payout Monte Carlo only."""
    from .engine.backtest import run_backtest
    from .engine.strategy import load_strategy
    from .propfirm.evaluate import compare_firms
    from .propfirm.rules import all_rules, find_rules
    strat = load_strategy(strategy)
    bars, inst = _bars(strat, symbol, tf, start, end, None)
    r = run_backtest(strat, bars, inst, _parse_params(param))
    rules = all_rules(inst.asset_class) if firms == "all" else find_rules(firms.split(","))
    table, _ = compare_firms(r, rules, n_sims=n_sims)
    import polars as pl
    with pl.Config(tbl_cols=-1, tbl_width_chars=200):
        typer.echo(table)


@app.command("firms")
def list_firms():
    """List configured prop firm rule files."""
    from .propfirm.rules import PROPFIRM_DIR, load_rules
    for p in sorted(PROPFIRM_DIR.glob("*.yaml")):
        r = load_rules(p)
        typer.echo(f"{p.stem:32} {r.key:40} {r.asset_class:8} ${r.account_size:>9,.0f}  "
                   f"{'verified ' + str(r.verified_at) if r.verified else 'UNVERIFIED'}")


@data_app.command("pull")
def data_pull(symbols: list[str], start: str = typer.Option("2020-01-01"), end: Optional[str] = None,
              daily: bool = typer.Option(False, help="use the instrument's daily source (e.g. yfinance)")):
    """Download and store data, e.g. `tsim data pull BTCUSDT EURUSD ES --start 2018-01-01`."""
    from .data.pull import pull
    for s in symbols:
        pull(s, date.fromisoformat(start), date.fromisoformat(end) if end else None, daily=daily, progress=typer.echo)


@data_app.command("list")
def data_list():
    """Show what's in the local store."""
    import polars as pl
    from .data.store import available
    with pl.Config(tbl_rows=200, tbl_width_chars=160):
        typer.echo(available())


@data_app.command("instruments")
def data_instruments():
    """List configured instruments."""
    from .config import instrument, list_instruments
    for s in list_instruments():
        i = instrument(s)
        typer.echo(f"{s:14} {i.asset_class:8} pv={i.point_value:<6g} tick={i.tick_size:<8g} src={i.source}:{i.source_symbol}"
                   + (f"  (proxy for {i.proxy_for})" if i.is_proxy else ""))


@data_app.command("macro")
def data_macro(series: list[str] = typer.Argument(None, help="FRED ids, default VIXCLS DGS10 DFF")):
    """Pull FRED macro series (no key needed) into <data_dir>/macro."""
    from .data.adapters.other import fred_series, save_macro
    for sid in series or ["VIXCLS", "DGS10", "DFF"]:
        p = save_macro(sid, fred_series(sid))
        typer.echo(f"{sid} -> {p}")


if __name__ == "__main__":
    app()
