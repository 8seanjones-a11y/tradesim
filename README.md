# tradesim

A personal trading simulator driven by Claude Code. Describe a strategy in plain English, and Claude
writes it, backtests it on free historical data, and stress-tests it with Monte Carlo and about 15 other
robustness methods. It then gives a verdict on whether there's a real edge in open markets. An optional
step tests whether the strategy would pass prop firm evaluations: Topstep, Apex, TPT, MFF, Tradeify,
Lucid, FTMO, FundedNext, The5ers and E8.

## Setup (on your own machine)
```bash
git clone <your repo> tradesim && cd tradesim
uv sync                       # Python 3.12+, installs everything
uv run pytest -q              # should be all green
```

## Get data (free)
| Market | Command | Source |
|---|---|---|
| Crypto 1m | `uv run tsim data pull BTCUSDT ETHUSDT --start 2019-01-01` | Binance public archives |
| Forex / metals 1m | `uv run tsim data pull EURUSD GBPUSD XAUUSD --start 2015-01-01` | Dukascopy (bid+ask → mid + spread) |
| Futures 1m (proxy) | `uv run tsim data pull ES NQ CL GC --start 2015-01-01` | Dukascopy index/commodity CFDs |
| Futures daily | `uv run tsim data pull ES --daily --start 2005-01-01` | yfinance continuous (`ES=F`) |
| Stocks 1m | `export ALPACA_API_KEY=... ALPACA_API_SECRET=...` then `uv run tsim data pull SPY QQQ --start 2016-01-01` | Alpaca free (IEX feed) |
| Stocks daily | `uv run tsim data pull SPY --daily --start 2000-01-01` | yfinance |
| Macro | `uv run tsim data macro VIXCLS DGS10` | FRED |

Data is stored as Parquet under `~/tradesim-data` (set `TRADESIM_DATA_DIR` to move it). Downloads are
cached, so re-running a pull is cheap. **Pull a short range first** (`--start 2024-01-01 --end 2024-01-31`)
and check `tsim data list`. For Dukascopy symbols, confirm the prices look right; `price_divisor` in
`config/instruments.yaml` may need adjusting for some CFDs.

## Use it
In Claude Code inside this folder, just describe the idea:
> "Test a 15-minute opening range breakout on ES, 2R target, stop at the other side of the range, flat by the close. Also check prop firms."

Claude follows `.claude/skills/new-strategy`. You can also run the CLI directly:
```bash
uv run tsim evaluate orb --level standard --prop all
uv run tsim evaluate sma_cross -s SPY --tf 1d -p fast=20 -p slow=150 --level full
uv run tsim evaluate rsi_reversion --synthetic garch      # smoke test without data
```
Each run writes `runs/<id>/report.html` with these sections: scorecard verdict, equity vs buy & hold,
own-account outlook (CAGR and drawdown over 1/3/5 years, risk of ruin, Kelly), Monte Carlo fans, robustness
charts, and an optional prop firm table.

## What's tested
| Area | Methods |
|---|---|
| Monte Carlo | trade shuffle, bootstrap, stationary block bootstrap, execution noise (slippage/fees/missed trades) |
| Overfitting | Deflated Sharpe, PBO (CSCV), White's Reality Check, Masters permutation test (with re-optimisation), parameter-plateau heatmaps, walk-forward efficiency |
| Robustness | noise injection on OHLC, null-model runs on synthetic markets (block bootstrap / GBM / GARCH), cost break-even, regime breakdown |
| Own account | horizon outcome distributions, risk of ruin, capital needed for a drawdown tolerance, Kelly / optimal-f, alpha/beta vs benchmark |
| Prop firms | per-firm state machine (static / EOD / intraday trailing, locks, daily loss on equity or balance, consistency, min days, payouts), daily-block MC, size optimisation, EV per attempt |

## Important caveats
- **Prop firm rules in `config/propfirms/` are unverified.** They come from public summaries (Nov 2026).
  Check each against the firm's help centre and set `verified_at`.
- Futures intraday data is proxied by CFDs. P&L uses real contract specs, but the CFD's price path, hours and gaps
  differ slightly from the exchange contract.
- The free Alpaca feed is IEX-only, so its volume is a fraction of the real tape.
- Backtests are hypothetical. Costs and fills are modelled pessimistically, but they're still approximations.

## Roadmap
Streamlit dashboard over `runs/`, CPCV, HMM regime generator, Databento adapter for real futures data,
economic-calendar news blackout, multi-instrument portfolios.
