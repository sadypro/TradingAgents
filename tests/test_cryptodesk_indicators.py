"""Indicator arithmetic and the triggers that decide when to spend on an LLM."""

import pytest

from cryptodesk.config import LLMBudget
from cryptodesk.engine.triggers import TriggerContext, evaluate
from cryptodesk.feeds.base import Candle
from cryptodesk.indicators import Snapshot, atr, ema, realized_vol, rsi, sma, snapshot


def bars(closes):
    return [Candle(ts=i * 300, open=c, high=c * 1.01, low=c * 0.99, close=c, volume=1)
            for i, c in enumerate(closes)]


# ---------------------------------------------------------------- maths
@pytest.mark.parametrize("fn, args", [
    (sma, ([1, 2], 5)), (ema, ([1, 2], 5)), (rsi, ([1, 2], 14)),
    (realized_vol, ([1, 2], 48)),
])
def test_indicators_return_none_rather_than_guessing_on_short_history(fn, args):
    assert fn(*args) is None


def test_atr_needs_more_bars_than_its_period():
    assert atr(bars([1] * 5), 14) is None
    assert atr(bars(range(1, 40)), 14) is not None


def test_sma_and_ema_agree_on_a_flat_series():
    flat = [100.0] * 50
    assert sma(flat, 10) == pytest.approx(100.0)
    assert ema(flat, 10) == pytest.approx(100.0)


def test_ema_reacts_to_a_recent_shock_faster_than_sma():
    """The property that makes the fast/slow EMA pair a usable trend signal."""
    flat = [100.0] * 60
    shocked = flat[:-1] + [150.0]
    ema_move = ema(shocked, 10) - ema(flat, 10)
    sma_move = sma(shocked, 10) - sma(flat, 10)
    assert ema_move > sma_move > 0


@pytest.mark.parametrize("closes, expected", [
    (list(range(1, 30)), 100.0),      # only gains
    ([100.0] * 30, 50.0),             # no movement
])
def test_rsi_handles_the_degenerate_cases_without_dividing_by_zero(closes, expected):
    assert rsi(closes) == pytest.approx(expected)


def test_rsi_stays_inside_its_bounds():
    import random
    rng = random.Random(1)
    closes = [100.0]
    for _ in range(500):
        closes.append(max(closes[-1] * (1 + rng.gauss(0, 0.01)), 0.01))
    assert 0.0 <= rsi(closes) <= 100.0


def test_atr_is_positive_and_scales_with_range():
    calm = [Candle(ts=i, open=100, high=101, low=99, close=100, volume=1) for i in range(40)]
    wild = [Candle(ts=i, open=100, high=110, low=90, close=100, volume=1) for i in range(40)]
    assert 0 < atr(calm) < atr(wild)


# ---------------------------------------------------------------- snapshot
def test_snapshot_labels_the_trend_from_the_moving_averages():
    up = snapshot("BTC-USD", bars([100 + i for i in range(200)]))
    down = snapshot("BTC-USD", bars([300 - i for i in range(200)]))
    assert up.trend == "up"
    assert down.trend == "down"
    assert snapshot("BTC-USD", bars([100.0] * 200)).trend == "flat"


def test_snapshot_without_enough_history_reports_unknown_not_a_guess():
    assert snapshot("BTC-USD", bars([100, 101, 102])).trend == "unknown"


def test_snapshot_requires_at_least_one_candle():
    with pytest.raises(ValueError):
        snapshot("BTC-USD", [])


def test_atr_pct_normalises_across_price_levels():
    snap = Snapshot(symbol="X", ts=0, price=60_000, atr14=600)
    assert snap.atr_pct == pytest.approx(0.01)
    assert Snapshot(symbol="X", ts=0, price=60_000).atr_pct is None


# ---------------------------------------------------------------- triggers
def make_snapshot(**overrides):
    base = {"symbol": "BTC-USD", "ts": 0, "price": 60_000.0, "atr14": 600.0,
                "rsi14": 50.0, "ema_fast": 61_000.0, "ema_slow": 60_000.0,
                "vol_short": 0.01, "vol_long": 0.01, "bars": 200}
    base.update(overrides)
    return Snapshot(**base)


@pytest.fixture
def budget():
    return LLMBudget(daily_usd_cap=5.0, min_minutes_between_calls=45,
                     scheduled_interval_minutes=240, estimated_cost_per_run_usd=0.35)


def test_the_first_look_at_a_symbol_always_fires(budget):
    assert evaluate(make_snapshot(), TriggerContext(), budget, now=0).reason == "cold_start"


@pytest.mark.parametrize("spend, reason", [(5.0, "budget_exhausted"), (4.9, "budget_reserved")])
def test_the_daily_budget_gates_everything(budget, spend, reason):
    trigger = evaluate(make_snapshot(), TriggerContext(spend_today=spend), budget, now=0)
    assert not trigger.fired and trigger.reason == reason


def test_no_analysis_is_bought_before_there_is_history(budget):
    trigger = evaluate(make_snapshot(atr14=None, bars=10), TriggerContext(), budget, now=0)
    assert not trigger.fired and trigger.reason == "warming_up"


def test_the_per_symbol_cooldown_blocks_rapid_re_runs(budget):
    ctx = TriggerContext(last_decision_ts=0, last_decision_price=60_000)
    assert evaluate(make_snapshot(), ctx, budget, now=60).reason == "cooldown"


@pytest.mark.parametrize("snap_kwargs, ctx_kwargs, expected", [
    ({"price": 61_000}, {}, "atr_move"),
    ({}, {"last_trend": "down"}, "trend_flip"),
    ({"vol_short": 0.02}, {"last_trend": "up"}, "vol_expansion"),
    ({"rsi14": 80}, {"last_trend": "up", "has_position": True}, "rsi_overbought"),
    ({"rsi14": 20}, {"last_trend": "up"}, "rsi_oversold"),
])
def test_material_changes_wake_the_committee(budget, snap_kwargs, ctx_kwargs, expected):
    ctx = TriggerContext(last_decision_ts=-99_999, last_decision_price=60_000, **ctx_kwargs)
    trigger = evaluate(make_snapshot(**snap_kwargs), ctx, budget, now=0)
    assert trigger.fired and trigger.reason == expected


def test_a_quiet_market_still_gets_reviewed_on_the_floor_cadence(budget):
    ctx = TriggerContext(last_decision_ts=-20_000, last_decision_price=60_000, last_trend="up")
    assert evaluate(make_snapshot(), ctx, budget, now=0).reason == "scheduled"


def test_nothing_material_means_no_spend(budget):
    ctx = TriggerContext(last_decision_ts=-3_000, last_decision_price=60_000, last_trend="up")
    trigger = evaluate(make_snapshot(), ctx, budget, now=0)
    assert not trigger.fired and trigger.reason == "no_trigger"
