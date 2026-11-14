# tradesim – guide for Claude

Personal trading simulator. The user describes strategies in plain English; you turn them into a
`Strategy` file in `strategies/`, run `tsim evaluate`, and explain the results. **The primary question
is always: does this have a real edge in open markets on the user's own account?** The prop firm test
(`--prop`) is an optional extra layered on top.

## Commands (run with `uv run`)
- `tsim evaluate <strategy> [-s SYMBOL] [--tf 5m] [--start 2020-01-01] [-p key=value ...] [--level quick|standard|full] [--prop all|topstep,apex] [--risk 0.01]`
  Full pipeline -> `runs/<id>/report.html` + `meta.json`. `standard` is the default; use `full` (adds the
  Masters permutation test with re-optimisation and PBO) before trusting a strategy.
- `tsim backtest <strategy> ...` quick metrics only. `--synthetic gbm|garch|jump` runs on generated data.
- `tsim prop <strategy> --firms all` prop firm table only. `tsim firms` lists rule files.
- `tsim data pull SYMBOL... --start YYYY-MM-DD` / `tsim data list` / `tsim data instruments` / `tsim data macro`
- `uv run pytest -q` – must stay green.

## Strategy contract (strategies/<name>.py)
```python
import numpy as np
from tradesim.engine import indicators as ind
from tradesim.engine.strategy import Signals, Strategy

class MyStrat(Strategy):
    name = "my_strat"                   # == file stem
    description = "one line"
    symbol = "ES"                       # default instrument (config/instruments.yaml)
    timeframe = "5m"                    # 1m 5m 15m 30m 1h 4h 1d
    params = {"n": 20, "rr": 2.0, "contracts": 1}
    param_grid = {"n": [10, 20, 40], "rr": [1.5, 2, 3]}   # drives sensitivity, WFO, PBO, DSR trial count

    def signals(self, bars, p) -> Signals:
        c = bars["close"].to_numpy()    # bars: polars DataFrame ts(UTC), open, high, low, close, volume
        ...
        return Signals(entry_long=..., exit_long=..., entry_short=..., exit_short=...,
                       stop_dist=..., target_dist=..., size=p["contracts"])
```
- Arrays are length `len(bars)`; scalars broadcast. `stop_dist` / `target_dist` are **price distances from the fill**
  (NaN = none). `size` is units (contracts, shares, coins, FX base units). With `--risk f` and a stop, size is
  computed as capital·f / (stop_dist·point_value) instead.
- Use `tradesim.engine.indicators` (sma, ema, rsi, atr, bollinger, rolling_max/min/std, crossover/crossunder, shift).
  All are causal. For session logic, convert time: `bars["ts"].dt.convert_time_zone("America/New_York")`.
- Write simple per-bar Python loops when logic is stateful (see strategies/orb.py); it's fast enough.

## Invariants – never break these
1. **No lookahead.** The signal at index i may use only bars 0..i. It fills at the OPEN of bar i+1. Never use
   `shift(-k)`, centred windows, full-sample normalisation (z-score with global mean), or future session highs.
   `tests/test_analysis.py::test_example_strategies_have_no_lookahead` shows the check. Add each new strategy name to
   that parametrize list.
2. Stops and targets are checked intrabar. When both are hit in one bar, the stop wins (pessimistic). Gaps fill at the open.
3. One position per instrument. An opposite entry reverses the position.
4. Costs come from `config/instruments.yaml` (commission, half-spread, carry) plus `slippage_ticks` (settings.yaml).
   Don't silently set costs to zero. If the user wants that, say it is optimistic.
5. Futures intraday data is **proxy data** (Dukascopy CFDs). Reports flag it. Mention it when you discuss results.

## Interpreting results for the user
- Lead with the **scorecard verdict** and the 2–3 checks that drove it, then net profit / CAGR / max DD vs buy & hold.
- "STRONG" still isn't a guarantee. Watch especially for: under 100 trades, a deflated Sharpe below 0.95, walk-forward
  efficiency below 0.5, a narrow parameter plateau, and break-even slippage under 2 ticks.
- Prop results: quote p_pass, median days, top failure reason and EV per attempt. If the open-market verdict is weak
  but prop EV is positive, call it the lottery-ticket effect, not an edge.
- Prop rule files in `config/propfirms/` are **UNVERIFIED** unless `verified_at` is set. Say so.

## Layout
`src/tradesim/` – `engine/` (kernel.py numba loop, backtest.py runner, strategy.py, indicators.py),
`analysis/` (metrics, montecarlo, robustness, overfit, account, scorecard), `propfirm/` (rules schema,
simulator state machine, evaluate MC), `data/` (store.py Parquet+DuckDB, adapters/, synthetic.py, pull.py),
`reports/html.py`, `pipeline.py` (evaluate + save_run), `cli.py`.
Data lives in `~/tradesim-data` (override with `TRADESIM_DATA_DIR`).
