import numpy as np

from tradesim.propfirm.rules import DailyLoss, FirmRuleSet, MaxLoss, Payout, Phase, all_rules
from tradesim.propfirm.simulator import run_account, run_phase


def phase(**kw):
    base = dict(name="eval", profit_target=3000, max_loss=MaxLoss(amount=2000, type="trailing", trail_mode="eod_balance"))
    base.update(kw)
    return Phase(**base)


def arr(*days):
    """days: (pnl, low, high)"""
    p = np.array([d[0] for d in days], float)
    lo = np.array([d[1] for d in days], float)
    hi = np.array([d[2] for d in days], float)
    return p, lo, hi, np.ones(len(days), bool)


def test_pass_target():
    out, _ = run_phase(phase(), 50_000, *arr((1500, 0, 1500), (1600, 0, 1600)))
    assert out.status == "passed" and out.days == 2


def test_eod_trailing_floor_rises_then_breach():
    # +1500 close -> floor 49_500; day2 low 51_500-1600 = 49_900 OK, closes 50_500; day3 low 50_500-1100 = 49_400 breach
    out, _ = run_phase(phase(), 50_000, *arr((1500, 0, 1500), (-1000, -1600, 0), (0, -1100, 0)))
    assert out.status == "failed" and out.reason == "trailing_drawdown" and out.days == 3


def test_intraday_trailing_uses_highs():
    ml = MaxLoss(amount=2000, type="trailing", trail_mode="intraday_equity")
    # spikes +1900 intraday then closes flat: floor -> 49_900; same day low -200 -> 49_800 breach
    out, _ = run_phase(phase(max_loss=ml), 50_000, *arr((0, -200, 1900)))
    assert out.status == "failed"
    eod = MaxLoss(amount=2000, type="trailing", trail_mode="eod_balance")
    out2, _ = run_phase(phase(max_loss=eod), 50_000, *arr((0, -200, 1900)))
    assert out2.status != "failed"


def test_lock_at_start_balance():
    ml = MaxLoss(amount=2000, type="trailing", trail_mode="eod_balance", lock_at="start_balance")
    p = phase(max_loss=ml, profit_target=None, kind="funded")
    # run to +5000; floor would be 53_000 without lock, locked at 50_000. Drop of 4000 survives.
    out, _ = run_phase(p, 50_000, *arr((5000, 0, 5000), (-4000, -4000, 0)), horizon=2)
    assert out.status == "survived"


def test_daily_loss_equity_vs_balance():
    dl_eq = DailyLoss(amount=1000, basis="equity")
    out, _ = run_phase(phase(daily_loss=dl_eq), 50_000, *arr((200, -1100, 300)))
    assert out.status == "failed" and out.reason == "daily_loss"
    dl_bal = DailyLoss(amount=1000, basis="balance")
    out, _ = run_phase(phase(daily_loss=dl_bal), 50_000, *arr((200, -1100, 300)))
    assert out.status != "failed"


def test_daily_loss_pause_day_does_not_fail():
    dl = DailyLoss(amount=1000, basis="equity", breach_action="pause_day")
    out, _ = run_phase(phase(daily_loss=dl), 50_000, *arr((-1500, -1500, 0), (3500, 0, 3500), (600, 0, 600)))
    assert out.status == "passed"


def test_consistency_blocks_pass_until_spread_out():
    from tradesim.propfirm.rules import Consistency
    ph = phase(consistency=[Consistency(max_ratio=0.5)])
    out, _ = run_phase(ph, 50_000, *arr((3000, 0, 3000), (500, 0, 500), (2600, 0, 2600)))
    # day1: best 3000/3000 = 100% blocked; day2: 3000/3500 blocked; day3: 3000/6100 = 49% -> pass
    assert out.status == "passed" and out.days == 3


def test_pct_static_ftmo_style():
    ph = Phase(name="c", profit_target=0.1, target_unit="pct", max_loss=MaxLoss(amount=0.1, unit="pct"),
               daily_loss=DailyLoss(amount=0.05, unit="pct"))
    out, _ = run_phase(ph, 100_000, *arr((-4000, -4900, 0), (-4000, -4900, 0), (-1000, -2100, 0)))
    # 92_000 - 2_100 = 89_900 < static floor 90_000 (daily limit 5_000 not hit)
    assert out.status == "failed" and out.reason == "max_loss"


def test_funded_payout_flow():
    pay = Payout(min_days=2, min_day_profit=100, max_pct_of_profit=0.5, caps=[1000], split=0.9)
    rules = FirmRuleSet(firm="T", program="p", account_size=50_000, phases=[
        phase(profit_target=1000, min_trading_days=1),
        Phase(name="funded", kind="funded", max_loss=MaxLoss(amount=2000, type="trailing", lock_at="start_balance"), payout=pay)])
    days = [(1000, 0, 1000)] + [(1500, 0, 1500)] * 6
    out = run_account(rules, *arr(*days), funded_horizon=6)
    funded = out["phases"][1][1]
    assert funded.payouts and funded.payouts[0] == 900  # capped at 1000 * 90% split


def test_all_shipped_rule_files_parse():
    rules = all_rules()
    assert len(rules) >= 10
    assert all(r.phases and r.account_size > 0 for r in rules)
