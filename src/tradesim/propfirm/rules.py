"""Prop firm rule schema. One YAML file per firm/program in config/propfirms/.

Rule values change often - every file carries `verified_at` / `source_url`. Unverified values are
starting points from public summaries; check the firm's help centre before relying on them.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from ..config import CONFIG_DIR


class MaxLoss(BaseModel):
    amount: float                                   # $ (or fraction if unit=pct)
    unit: Literal["abs", "pct"] = "abs"
    type: Literal["static", "trailing"] = "static"
    trail_mode: Literal["intraday_equity", "eod_balance"] = "eod_balance"
    lock_at: Literal["none", "start_balance", "start_plus_x"] = "none"
    lock_offset: float = 0.0                        # $ above start where the floor stops (start_plus_x)


class DailyLoss(BaseModel):
    amount: float
    unit: Literal["abs", "pct"] = "abs"
    basis: Literal["equity", "balance"] = "equity"  # equity = includes open-trade drawdown
    anchor: Literal["initial", "prior_day_close"] = "prior_day_close"
    breach_action: Literal["fail", "pause_day"] = "fail"


class Consistency(BaseModel):
    metric: Literal["best_day_over_profit", "best_day_over_target"] = "best_day_over_profit"
    max_ratio: float
    effect: Literal["block_pass", "fail", "block_payout"] = "block_pass"


class Payout(BaseModel):
    safety_net: float | None = None          # balance (above start) required before first payout
    min_days: int = 5                        # qualifying days per payout cycle
    min_day_profit: float = 0.0              # a day qualifies if its P&L >= this
    max_pct_of_profit: float = 0.5           # withdraw at most this share of profit above start
    caps: list[float] = Field(default_factory=list)   # per-payout cap; last value repeats
    min_amount: float = 0.0
    split: float = 0.9
    max_payouts: int | None = None
    consistency: Consistency | None = None


class Phase(BaseModel):
    name: str
    kind: Literal["evaluation", "funded"] = "evaluation"
    profit_target: float | None = None
    target_unit: Literal["abs", "pct"] = "abs"
    max_loss: MaxLoss
    daily_loss: DailyLoss | None = None
    consistency: list[Consistency] = Field(default_factory=list)
    min_trading_days: int = 0
    min_day_profit_to_count: float | None = None
    max_calendar_days: int | None = None
    max_contracts: float | None = None       # in units of the instrument (contracts / lots*units)
    flat_by_eod: bool = False
    payout: Payout | None = None


class FirmRuleSet(BaseModel):
    firm: str
    program: str
    asset_class: Literal["futures", "forex", "any"] = "futures"
    account_size: float
    fee: float = 0.0
    fee_type: Literal["one_time", "monthly"] = "one_time"
    activation_fee: float = 0.0
    day_tz: str = "America/New_York"
    day_roll_hour: int = 17
    phases: list[Phase]
    version: str = "1"
    effective_date: date | None = None
    source_url: str | None = None
    verified_at: date | None = None
    notes: str = ""

    @property
    def key(self) -> str:
        return f"{self.firm}/{self.program}"

    @property
    def verified(self) -> bool:
        return self.verified_at is not None


PROPFIRM_DIR = CONFIG_DIR / "propfirms"


def load_rules(path: str | Path) -> FirmRuleSet:
    with open(path) as f:
        return FirmRuleSet.model_validate(yaml.safe_load(f))


def all_rules(asset_class: str | None = None) -> list[FirmRuleSet]:
    out = [load_rules(p) for p in sorted(PROPFIRM_DIR.glob("*.yaml"))]
    if asset_class:
        out = [r for r in out if r.asset_class in (asset_class, "any")]
    return out


def find_rules(names: list[str] | str) -> list[FirmRuleSet]:
    """Match by file stem, firm name or 'all'."""
    if names == "all" or names == ["all"]:
        return all_rules()
    names = [names] if isinstance(names, str) else names
    out = []
    for p in sorted(PROPFIRM_DIR.glob("*.yaml")):
        r = load_rules(p)
        if any(n.lower() in (p.stem.lower(), r.firm.lower()) or p.stem.lower().startswith(n.lower()) for n in names):
            out.append(r)
    return out
