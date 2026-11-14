"""Standalone HTML report (Plotly charts + tables)."""
from __future__ import annotations

import html
import math

import numpy as np
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots

C = dict(strat="#2563eb", bench="#9ca3af", dd="#dc2626", band="rgba(37,99,235,0.15)", band2="rgba(37,99,235,0.30)",
         good="#16a34a", warn="#d97706", bad="#dc2626", null="#9ca3af")
_first = [True]


def _fig_html(fig: go.Figure, height: int = 380) -> str:
    fig.update_layout(height=height, margin=dict(l=50, r=20, t=40, b=40), template="plotly_white",
                      legend=dict(orientation="h", y=1.08, x=0), font=dict(family="Inter, system-ui, sans-serif", size=12))
    from ..config import settings
    inc = (True if settings().get("report_plotly", "cdn") == "inline" else "cdn") if _first[0] else False
    _first[0] = False
    return fig.to_html(full_html=False, include_plotlyjs=inc, config={"displaylogo": False, "responsive": True})


def _fmt(v, kind=""):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    if isinstance(v, float) and math.isinf(v):
        return "∞"
    if kind == "pct":
        return f"{v * 100:.1f}%"
    if kind == "$":
        return f"${v:,.0f}"
    if isinstance(v, float):
        return f"{v:,.3f}" if abs(v) < 10 else f"{v:,.1f}"
    return html.escape(str(v))


def _table(rows: list[dict], fmts: dict | None = None, cls: str = "") -> str:
    if not rows:
        return "<p class='muted'>No data.</p>"
    fmts = fmts or {}
    cols = list(rows[0].keys())
    h = "".join(f"<th>{html.escape(str(c)).replace('_', ' ')}</th>" for c in cols)
    body = ""
    for r in rows:
        tds = ""
        for c in cols:
            v = r[c]
            if c == "status":
                tds += f"<td><span class='pill {str(v).lower()}'>{v}</span></td>"
            else:
                tds += f"<td>{_fmt(v, fmts.get(c, ''))}</td>"
        body += f"<tr>{tds}</tr>"
    return f"<div class='tw'><table class='{cls}'><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table></div>"


def _tiles(items: list[tuple[str, str]]) -> str:
    return "<div class='tiles'>" + "".join(f"<div class='tile'><div class='tl'>{k}</div><div class='tv'>{v}</div></div>" for k, v in items) + "</div>"


def _equity_chart(ev) -> str:
    res = ev["result"]
    eq = res.equity
    if eq.height > 4000:  # keep reports small: one point per trading day (close), max ~4000 points
        eq = eq.group_by("day", maintain_order=True).agg(pl.all().last())
        if eq.height > 4000:
            eq = eq.gather_every(eq.height // 4000 + 1)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.04)
    ts = eq["ts"].to_list()
    e = eq["equity"].to_numpy()
    fig.add_trace(go.Scatter(x=ts, y=e, name="Strategy", line=dict(color=C["strat"], width=1.6)), 1, 1)
    b = ev.get("benchmark") or {}
    if b.get("bench_curve") is not None:
        bc = b["bench_curve"]
        fig.add_trace(go.Scatter(x=bc["ts"].to_list(), y=bc["bench_equity"].to_numpy(), name=f"Buy & hold {res.symbol}",
                                 line=dict(color=C["bench"], width=1.2, dash="dot")), 1, 1)
    peak = np.maximum.accumulate(e)
    fig.add_trace(go.Scatter(x=ts, y=-(peak - e) / peak * 100, name="Drawdown %", fill="tozeroy",
                             line=dict(color=C["dd"], width=1), fillcolor="rgba(220,38,38,0.2)"), 2, 1)
    fig.update_yaxes(title_text="Equity", row=1, col=1)
    fig.update_yaxes(title_text="DD %", row=2, col=1)
    return _fig_html(fig, 460)


def _fan_chart(mc, cap, title) -> str:
    f = mc["fan"]
    x = f["x"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=f["p95"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=f["p5"], fill="tonexty", fillcolor=C["band"], line=dict(width=0), name="5–95%"))
    fig.add_trace(go.Scatter(x=x, y=f["p75"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=f["p25"], fill="tonexty", fillcolor=C["band2"], line=dict(width=0), name="25–75%"))
    fig.add_trace(go.Scatter(x=x, y=f["p50"], line=dict(color=C["strat"], width=2), name="Median"))
    fig.add_hline(y=cap, line=dict(color=C["bench"], dash="dot"))
    fig.update_layout(title=title, xaxis_title="Trade #", yaxis_title="Equity")
    return _fig_html(fig, 340)


def _hist(values, title, xlabel, marker=None, marker_label="", color=None, pct=False) -> str:
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    fig = go.Figure(go.Histogram(x=v * (100 if pct else 1), nbinsx=50, marker_color=color or C["strat"], opacity=0.8, name=xlabel))
    if marker is not None and np.isfinite(marker):
        fig.add_vline(x=marker * (100 if pct else 1), line=dict(color=C["dd"], width=2),
                      annotation_text=marker_label, annotation_position="top")
    fig.update_layout(title=title, xaxis_title=xlabel + (" (%)" if pct else ""), yaxis_title="Count", showlegend=False, bargap=0.05)
    return _fig_html(fig, 300)


def _heatmap(sens) -> str:
    t = sens["table"]
    keys = sens["keys"]
    if len(keys) < 2:
        fig = go.Figure(go.Scatter(x=t[keys[0]].to_list(), y=t["score"].to_list(), mode="lines+markers", line=dict(color=C["strat"])))
        fig.update_layout(title=f"Sensitivity: {sens['objective']} vs {keys[0]}", xaxis_title=keys[0], yaxis_title=sens["objective"])
        return _fig_html(fig, 320)
    a, b = keys[0], keys[1]
    best = sens["best"]
    sub = t
    for k in keys[2:]:
        sub = sub.filter(sub[k] == best[k])
    xs, ys = sorted(set(sub[a].to_list())), sorted(set(sub[b].to_list()))
    z = np.full((len(ys), len(xs)), np.nan)
    for r in sub.iter_rows(named=True):
        z[ys.index(r[b]), xs.index(r[a])] = r["score"]
    fig = go.Figure(go.Heatmap(x=[str(x) for x in xs], y=[str(y) for y in ys], z=z, colorscale="RdBu", zmid=0,
                               colorbar=dict(title=sens["objective"]), text=np.round(z, 2), texttemplate="%{text}"))
    fig.update_layout(title=f"Parameter sensitivity ({sens['objective']})" + (f" – other params at best" if len(keys) > 2 else ""),
                      xaxis_title=a, yaxis_title=b)
    return _fig_html(fig, 380)


def _mae_mfe(res) -> str:
    t = res.trades
    if t.height == 0:
        return ""
    win = t["pnl"].to_numpy() > 0
    fig = go.Figure()
    for m, name, col in ((win, "Winners", C["good"]), (~win, "Losers", C["bad"])):
        fig.add_trace(go.Scatter(x=t["mae"].to_numpy()[m], y=t["mfe"].to_numpy()[m], mode="markers", name=name,
                                 marker=dict(color=col, size=5, opacity=0.6)))
    fig.update_layout(title="MAE vs MFE per trade ($)", xaxis_title="Max adverse excursion", yaxis_title="Max favourable excursion")
    return _fig_html(fig, 340)


def _wfo_chart(wfo, cap) -> str:
    e = wfo["oos_equity"]
    fig = go.Figure(go.Scatter(x=e["ts"].to_list(), y=e["equity"].to_numpy(), line=dict(color=C["strat"]), name="Stitched OOS"))
    fig.add_hline(y=cap, line=dict(color=C["bench"], dash="dot"))
    fig.update_layout(title="Walk-forward: stitched out-of-sample equity", yaxis_title="Equity")
    return _fig_html(fig, 300)


CSS = """
:root{--bg:#f8fafc;--card:#fff;--fg:#0f172a;--muted:#64748b;--line:#e2e8f0;--accent:#2563eb}
@media (prefers-color-scheme:dark){:root{--bg:#0b1220;--card:#111a2e;--fg:#e2e8f0;--muted:#94a3b8;--line:#1e293b}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 Inter,system-ui,sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:24px 16px 60px}h1{font-size:24px;margin:0 0 4px}h2{font-size:17px;margin:0 0 12px}
.sub{color:var(--muted)}.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px;margin:16px 0}
.verdict{display:inline-block;padding:6px 14px;border-radius:999px;font-weight:700;margin-top:10px}
.v-strong{background:#dcfce7;color:#166534}.v-promising{background:#dbeafe;color:#1e40af}.v-fragile{background:#fef3c7;color:#92400e}.v-overfit{background:#fee2e2;color:#991b1b}
.tiles{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px}.tile{border:1px solid var(--line);border-radius:10px;padding:10px 12px}
.tl{color:var(--muted);font-size:12px}.tv{font-size:19px;font-weight:650}.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:800px){.grid2{grid-template-columns:1fr}}.tw{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:6px 10px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}th{color:var(--muted);font-weight:600;text-transform:capitalize}
.pill{padding:2px 8px;border-radius:6px;font-weight:600;font-size:12px}.pass{background:#dcfce7;color:#166534}.warn{background:#fef3c7;color:#92400e}
.fail{background:#fee2e2;color:#991b1b}.skip{background:#e2e8f0;color:#475569}.note{background:#fef3c7;color:#78350f;border-radius:8px;padding:8px 12px;margin:6px 0}
.muted{color:var(--muted)}nav a{margin-right:14px;color:var(--accent);text-decoration:none;font-size:13px}
"""


def render_report(ev: dict) -> str:
    _first[0] = True
    res = ev["result"]
    s = ev["summary"]
    sc = ev["scorecard"]
    cap = res.initial_capital
    vcls = {"STRONG": "v-strong", "PROMISING": "v-promising", "FRAGILE": "v-fragile"}.get(sc["verdict"], "v-overfit")
    parts = []
    notes = "".join(f"<div class='note'>{html.escape(n)}</div>" for n in res.notes)
    parts.append(f"""<div class='card'><h1>{html.escape(res.strategy)} · {html.escape(res.symbol)}</h1>
      <div class='sub'>{s['start'][:10]} → {s['end'][:10]} · {s['years']:.1f} years · {res.bars.height:,} bars ·
      params {html.escape(str(res.params))} · evaluation level: {ev['level']}</div>
      <div class='verdict {vcls}'>Verdict: {sc['verdict']}</div>
      <span class='sub'> &nbsp;{sc['n_fail']} fail · {sc['n_warn']} warn</span>{notes}
      <nav style='margin-top:12px'><a href='#open'>Open market</a><a href='#mc'>Monte Carlo</a><a href='#robust'>Robustness</a>
      {"<a href='#prop'>Prop firms</a>" if 'prop' in ev else ''}<a href='#trades'>Trades</a></nav></div>""")

    parts.append("<div class='card'>" + _tiles([
        ("Net profit", _fmt(s["net_profit"], "$")), ("CAGR", _fmt(s["cagr"], "pct")), ("Sharpe", f"{s['sharpe']:.2f}"),
        ("Sortino", f"{s['sortino']:.2f}"), ("Max DD (close)", _fmt(s["max_dd"], "pct")),
        ("Max DD (intrabar)", _fmt(s["max_dd_intrabar"], "pct")), ("Calmar", f"{s['calmar']:.2f}"),
        ("Trades", f"{s['n_trades']:,}"), ("Win rate", _fmt(s["win_rate"], "pct")),
        ("Profit factor", _fmt(s["profit_factor"])), ("Avg trade", _fmt(s["avg_trade"], "$")),
        ("Exposure", _fmt(s["exposure"], "pct")), ("Total costs", _fmt(s["total_costs"], "$")),
        ("Deflated Sharpe", f"{ev['dsr']['dsr']:.2f}"),
    ]) + "</div>")

    parts.append("<div class='card'><h2>Robustness scorecard</h2>" + _table(sc["checks"]) + "</div>")
    parts.append("<div class='card' id='open'><h2>Equity vs buy &amp; hold</h2>" + _equity_chart(ev) + "</div>")

    # ---- open-market account section
    b = ev.get("benchmark") or {}
    acct_tiles = [("Risk of ruin", _fmt(ev["risk_of_ruin"], "pct")), ("Capital for DD tolerance", _fmt(ev["capital_needed"], "$")),
                  ("Kelly × current size", f"{ev['kelly']['kelly_mult']:.2f}"), ("Half-Kelly ×", f"{ev['kelly']['half_kelly_mult']:.2f}"),
                  ("Optimal-f", f"{ev['optimal_f']['optimal_f']:.2f}")]
    if b:
        acct_tiles += [("Benchmark CAGR", _fmt(b["bench_cagr"], "pct")), ("Alpha (ann.)", _fmt(b["alpha_ann"], "pct")),
                       ("Beta", f"{b['beta']:.2f}"), ("Correlation", f"{b['correlation']:.2f}"), ("Info ratio", f"{b['information_ratio']:.2f}")]
    hz = ev["horizons"]
    parts.append("<div class='card'><h2>Own-account outlook (block-bootstrap of daily returns)</h2>" + _tiles(acct_tiles) +
                 "<p class='muted'>Kelly multiplier is relative to the backtest's position size; use ≤ half-Kelly in practice. "
                 "Capital needed = 95th-percentile MC drawdown ÷ your DD tolerance (settings.yaml).</p>" +
                 _table(hz.to_dicts() if hz.height else [], {k: "pct" for k in hz.columns if k != "horizon_years"}) + "</div>")

    # ---- Monte Carlo
    parts.append(f"""<div class='card' id='mc'><h2>Monte Carlo</h2><div class='grid2'>
      <div>{_fan_chart(ev['mc_shuffle'], cap, 'Trade-order shuffle (same total, different paths)')}</div>
      <div>{_fan_chart(ev['mc_bootstrap'], cap, 'Bootstrap resample (with replacement)')}</div>
      <div>{_hist(ev['mc_shuffle']['max_dd'], 'Max drawdown distribution (shuffle)', 'Max DD', s['max_dd'], 'actual', C['dd'], pct=True)}</div>
      <div>{_hist(ev['mc_exec']['final'] - cap, 'Net profit with execution noise', 'Net profit $', s['net_profit'], 'actual')}</div>
      </div>""" + _table([
        {"method": m["method"], "p_loss": m["p_loss"], "p_ruin": m["p_ruin"],
         "final_p5": float(np.percentile(m["final"], 5)) - cap if len(m["final"]) else None,
         "final_p50": float(np.median(m["final"])) - cap if len(m["final"]) else None,
         "max_dd_p50": float(np.median(m["max_dd"])), "max_dd_p95": float(np.percentile(m["max_dd"], 95)),
         "losing_streak_p95": int(np.percentile(m["max_consec_losses"], 95))}
        for m in (ev["mc_shuffle"], ev["mc_bootstrap"], ev["mc_block"], ev["mc_exec"])],
        {"p_loss": "pct", "p_ruin": "pct", "final_p5": "$", "final_p50": "$", "max_dd_p50": "pct", "max_dd_p95": "pct"}) + "</div>")

    # ---- robustness
    rob = []
    if ev.get("sensitivity") and ev["sensitivity"].get("best") is not None:
        rob.append(f"<div>{_heatmap(ev['sensitivity'])}</div>")
    if ev.get("wfo") and ev["wfo"].get("oos_equity") is not None:
        rob.append(f"<div>{_wfo_chart(ev['wfo'], cap)}</div>")
    if ev.get("null"):
        n = ev["null"]
        title = f"Null model ({n['model']}): Sharpe on edge-less markets"
        rob.append(f"<div>{_hist(n['null_scores'], title, 'Sharpe', n['real'], 'real', C['null'])}</div>")
    if ev.get("permutation"):
        p = ev["permutation"]
        rob.append(f"<div>{_hist(p['permuted'], 'Permutation test: best score on permuted prices', 'Score', p['real'], 'real', C['null'])}</div>")
    if ev.get("noise"):
        nz = ev["noise"]
        rob.append(f"<div>{_hist(nz['noisy_net'], 'Noise injection: net profit on jittered prices', 'Net $', nz['base_net'], 'original')}</div>")
    if ev.get("cost"):
        ct = ev["cost"]["table"]
        fig = go.Figure(go.Scatter(x=ct["slippage_ticks"].to_list(), y=ct["net_profit"].to_list(), mode="lines+markers", line=dict(color=C["strat"])))
        fig.add_hline(y=0, line=dict(color=C["dd"], dash="dot"))
        fig.update_layout(title="Cost sensitivity", xaxis_title="Slippage ticks / side", yaxis_title="Net profit")
        rob.append(f"<div>{_fig_html(fig, 300)}</div>")
    rob_html = "<div class='grid2'>" + "".join(rob) + "</div>" if rob else "<p class='muted'>Run with --level standard or full for robustness tests.</p>"
    if ev.get("wfo") and ev["wfo"]["folds"].height:
        rob_html += "<h2 style='margin-top:16px'>Walk-forward folds</h2>" + _table(ev["wfo"]["folds"].to_dicts())
    if ev.get("regimes") is not None and ev["regimes"].height:
        rob_html += "<h2 style='margin-top:16px'>Regime breakdown</h2>" + _table(ev["regimes"].to_dicts(), {"win_rate": "pct", "pnl": "$", "avg": "$"})
    parts.append("<div class='card' id='robust'><h2>Robustness tests</h2>" + rob_html + "</div>")

    if "prop" in ev:
        t = ev["prop"]["table"]
        unver = not all(t["verified"].fill_null(False).to_list()) if t.height else False
        warn = "<div class='note'>Some firm rule files are UNVERIFIED (values from public summaries). Check config/propfirms/*.yaml against each firm's help centre.</div>" if unver else ""
        pos_ev = t.height and (t["ev_per_attempt"].fill_null(-1) > 0).any()
        if pos_ev and (s["net_profit"] <= 0 or sc["verdict"] == "LIKELY OVERFIT / NO EDGE"):
            warn += ("<div class='note'><b>Lottery-ticket effect:</b> this strategy has no demonstrated edge in open markets, yet some firms "
                     "show positive EV. That comes from capped downside (you can only lose the fee) combined with oversizing - "
                     "it is gambling on variance, not a durable edge, and firm rules/fees make it fragile.</div>")
        parts.append("<div class='card' id='prop'><h2>Prop firm evaluation (optional)</h2>" + warn +
                     "<p class='muted'>Daily-block bootstrap of the strategy's trading days, run through each firm's rules (pessimistic intraday ordering). "
                     "Size = multiplier on backtest size chosen to maximise EV per attempt. EV = expected payouts − fees per attempt over the funded horizon.</p>" +
                     _table(t.to_dicts(), {"p_pass": "pct", "p_first_payout": "pct", "mean_payouts": "$", "mean_cost": "$",
                                           "ev_per_attempt": "$", "account_size": "$", "fee": "$"}) + "</div>")

    parts.append("<div class='card' id='trades'><h2>MAE / MFE</h2>" + _mae_mfe(res) +
                 "<h2>Last 50 trades</h2>" + _table(res.trades.tail(50).select(
                     "entry_ts", "exit_ts", "direction", "entry_price", "exit_price", "units", "pnl", "exit_reason", "bars_held"
                 ).with_columns(*(pl.col(c).cast(pl.Utf8).str.slice(0, 16) for c in ("entry_ts", "exit_ts"))).to_dicts(),
                     {"pnl": "$"}) + "</div>")
    parts.append(f"<p class='muted'>Generated by tradesim in {ev.get('elapsed_s', 0):.1f}s. Backtests are hypothetical; costs, fills and proxy data are approximations.</p>")
    return f"""<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>{html.escape(res.strategy)} {html.escape(res.symbol)} report</title><style>{CSS}</style></head><body><div class='wrap'>{''.join(parts)}</div></body></html>"""
