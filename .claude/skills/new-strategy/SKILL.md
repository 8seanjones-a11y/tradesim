---
name: new-strategy
description: Turn a plain-English trading strategy description into a tradesim Strategy file, verify it has no lookahead, run the full evaluation (open-market scorecard, optional prop firms) and explain the results. Use when the user describes a trading idea, entry/exit rules, or asks to test/backtest a strategy.
---

# New strategy workflow

1. **Restate the spec** in a short bullet list: instrument(s), timeframe, entry long/short, exit, stop, target,
   sizing, session/time filters, parameters worth varying. List every ambiguity, such as "close above" vs
   "trade above", or which timezone a session is in. Pick sensible defaults and say what you chose. Ask the user
   only if a choice would change the strategy's logic.
2. **Check data**: run `uv run tsim data list`. If the symbol or timeframe is missing, tell the user the
   `tsim data pull` command to run. If they can't pull data right now, offer `--synthetic garch` as a smoke test.
3. **Write `strategies/<name>.py`** following the contract in CLAUDE.md. Include a `param_grid` of 2–3
   parameters with 3–5 values each, kept small, because every grid point is a full backtest during
   sensitivity, walk-forward and permutation runs. Prefer vectorised indicators. Use a loop when state
   (one trade per day, range building) requires it.
4. **Lookahead check**: add the name to the parametrize list in
   `tests/test_analysis.py::test_example_strategies_have_no_lookahead`, then run
   `uv run pytest -q -k lookahead`. Fix any failure before going on.
5. **Quick run**: `uv run tsim backtest <name>`. Sanity-check the trade count and exposure. Zero or a
   handful of trades usually means a bug.
6. **Evaluate**: `uv run tsim evaluate <name> --level standard`. Add `--prop all` if the user cares about prop
   firms. Use `--level full` once the user is serious about a strategy.
7. **Explain**: give the verdict, the decisive scorecard checks, how it compares with buy & hold, the MC
   drawdown 95th percentile and risk of ruin. If prop was run, add the best firm's p_pass, days and EV.
   Point to the report path. Suggest at most 2 concrete next experiments, such as a different filter or
   exit, checked on held-out data.

Never tune parameters on the full history and then present the in-sample result as the expected outcome.
Use walk-forward / OOS numbers when you talk about what to expect.
