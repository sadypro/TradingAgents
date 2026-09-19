"""The risk engine: volatility-targeted sizing and the four independent brakes.

Each brake is tested in isolation, because in production they must each be able
to stop trading on their own — a desk that only halts when every limit agrees
does not have limits.
"""

import pytest

from cryptodesk.broker import PaperBroker
from cryptodesk.config import RiskLimits
from cryptodesk.engine.risk import (
    RiskState,
    plan_entry,
    size_position,
    stop_hit,
    update_trailing_stop,
)
from cryptodesk.indicators import Snapshot


def snapshot(**overrides):
    base = {"symbol": "BTC-USD", "ts": 0, "price": 60_000.0, "atr14": 600.0,
                "rsi14": 50.0, "ema_fast": 61_000.0, "ema_slow": 60_000.0,
                "vol_short": 0.01, "vol_long": 0.01, "bars": 200}
    base.update(overrides)
    return Snapshot(**base)


def fresh_state(equity=10_000.0):
    return RiskState(peak_equity=equity, day_start_equity=equity, day_key="2026-01-01")


@pytest.fixture
def limits():
    return RiskLimits()


# ---------------------------------------------------------------- sizing
def test_dollar_risk_is_constant_as_volatility_changes(limits):
    """The point of ATR sizing: 2x the volatility, half the size, same risk."""
    calm = size_position(10_000, 60_000, 2_400, limits)
    wild = size_position(10_000, 60_000, 4_800, limits)

    assert wild == pytest.approx(calm / 2)
    risk_calm = calm * 2_400 * limits.stop_atr_mult
    risk_wild = wild * 4_800 * limits.stop_atr_mult
    assert risk_calm == pytest.approx(risk_wild)
    assert risk_calm == pytest.approx(10_000 * limits.risk_per_trade)


def test_weight_cap_binds_when_volatility_is_low(limits):
    """A tiny ATR would imply an enormous position; the weight cap stops it."""
    qty = size_position(10_000, 60_000, 1, limits)
    assert qty * 60_000 == pytest.approx(10_000 * limits.max_symbol_weight)


@pytest.mark.parametrize("conviction, factor", [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)])
def test_conviction_scales_size(limits, conviction, factor):
    full = size_position(10_000, 60_000, 2_400, limits, 1.0)
    assert size_position(10_000, 60_000, 2_400, limits, conviction) == pytest.approx(full * factor)


def test_conviction_above_one_cannot_increase_size(limits):
    """An over-confident committee must not be able to widen its own limit."""
    full = size_position(10_000, 60_000, 2_400, limits, 1.0)
    assert size_position(10_000, 60_000, 2_400, limits, 99.0) == pytest.approx(full)


@pytest.mark.parametrize("equity, price, atr", [(0, 60_000, 600), (10_000, 0, 600), (10_000, 60_000, 0)])
def test_degenerate_inputs_size_to_zero(limits, equity, price, atr):
    assert size_position(equity, price, atr, limits) == 0.0


# ---------------------------------------------------------------- brakes
def test_a_clean_entry_is_allowed_and_carries_a_stop(limits):
    plan = plan_entry("BTC-USD", snapshot(), 10_000, 10_000, 0.0, 0.0,
                      fresh_state(), limits, 1.0, now=0)
    assert plan.allowed
    assert plan.qty > 0
    assert plan.stop_price == pytest.approx(60_000 - 600 * limits.stop_atr_mult)


def test_halted_desk_refuses_entries(limits):
    state = fresh_state()
    state.halted, state.halt_reason = True, "kill-switch"
    plan = plan_entry("BTC-USD", snapshot(), 10_000, 10_000, 0.0, 0.0, state, limits, 1.0, now=0)
    assert not plan.allowed and "halted" in plan.reason


def test_drawdown_past_the_limit_refuses_entries(limits):
    state = fresh_state()
    state.peak_equity = 20_000       # equity 10k => 50% drawdown
    plan = plan_entry("BTC-USD", snapshot(), 10_000, 10_000, 0.0, 0.0, state, limits, 1.0, now=0)
    assert not plan.allowed and "drawdown" in plan.reason


def test_daily_loss_limit_refuses_entries(limits):
    state = fresh_state()
    state.day_start_equity = 11_000  # down ~9% on the day
    plan = plan_entry("BTC-USD", snapshot(), 10_000, 10_000, 0.0, 0.0, state, limits, 1.0, now=0)
    assert not plan.allowed and "daily loss" in plan.reason


def test_cooldown_refuses_re_entry_after_a_stop_out(limits):
    state = fresh_state()
    state.start_cooldown("BTC-USD", limits.cooldown_minutes, now=0)
    plan = plan_entry("BTC-USD", snapshot(), 10_000, 10_000, 0.0, 0.0, state, limits, 1.0, now=0)
    assert not plan.allowed and "cooldown" in plan.reason


def test_cooldown_expires(limits):
    state = fresh_state()
    state.start_cooldown("BTC-USD", 60, now=0)
    assert state.cooling_down("BTC-USD", now=0) is True
    assert state.cooling_down("BTC-USD", now=3_601) is False


def test_gross_exposure_cap_refuses_further_entries(limits):
    plan = plan_entry("BTC-USD", snapshot(), 10_000, 10_000, 0.0,
                      limits.max_gross_exposure, fresh_state(), limits, 1.0, now=0)
    assert not plan.allowed and "gross exposure" in plan.reason


def test_missing_atr_refuses_to_size_blind(limits):
    plan = plan_entry("BTC-USD", snapshot(atr14=None, bars=3), 10_000, 10_000, 0.0, 0.0,
                      fresh_state(), limits, 1.0, now=0)
    assert not plan.allowed and "ATR" in plan.reason


def test_dust_sized_orders_are_refused(limits):
    plan = plan_entry("BTC-USD", snapshot(), 10_000, cash=10.0, current_qty=0.0,
                      gross_exposure=0.0, state=fresh_state(), limits=limits,
                      conviction=1.0, now=0)
    assert not plan.allowed and "below minimum" in plan.reason


def test_an_existing_position_counts_toward_the_cap(limits):
    plan = plan_entry("BTC-USD", snapshot(), 10_000, 10_000, current_qty=99.0,
                      gross_exposure=0.0, state=fresh_state(), limits=limits,
                      conviction=1.0, now=0)
    assert not plan.allowed and "already at or above target" in plan.reason


def test_entry_is_capped_by_remaining_room_under_the_gross_limit(limits):
    plan = plan_entry("BTC-USD", snapshot(atr14=60.0), 10_000, 10_000, 0.0,
                      gross_exposure=0.55, state=fresh_state(), limits=limits,
                      conviction=1.0, now=0)
    assert plan.allowed
    # 5% of equity is all that is left under a 60% cap.
    assert plan.notional <= 10_000 * 0.05 + 1e-6


@pytest.mark.parametrize("fee_bps, slippage_bps", [(10, 5), (0, 0), (100, 100)])
def test_a_cash_bound_entry_actually_fills_at_the_broker(fee_bps, slippage_bps):
    """The clamp must leave room for the costs the broker will really charge.

    A flat 0.1% buffer was smaller than the default 0.15% of fee plus slippage,
    so every cash-bound entry was planned and then refused by the broker.
    """
    limits = RiskLimits(max_gross_exposure=2.0, max_symbol_weight=1.0, risk_per_trade=0.5)
    broker = PaperBroker(1_000, fee_bps=fee_bps, slippage_bps=slippage_bps)
    cost_rate = (fee_bps + slippage_bps) / 10_000
    plan = plan_entry("BTC-USD", snapshot(), equity=1_000, cash=broker.cash(),
                      current_qty=0.0, gross_exposure=0.0, state=fresh_state(1_000),
                      limits=limits, conviction=1.0, now=0, cost_rate=cost_rate)
    assert plan.allowed
    assert plan.notional <= 1_000

    fill = broker.market_order("BTC-USD", "buy", plan.qty, 60_000, ts=1)
    assert fill.qty == pytest.approx(plan.qty)
    # Nearly all the cash is deployed: the buffer covers the compounding of
    # slippage and fee (at most cost_rate**2) plus a rounding margin, no more.
    assert broker.cash() <= 1_000 * (cost_rate ** 2 + 2e-6)


# ---------------------------------------------------------------- stops
def test_trailing_stop_only_ever_ratchets_up(limits):
    stop, extreme = None, 60_000.0
    seen = []
    for price in (60_000, 61_000, 62_000, 61_000, 59_000):
        stop, extreme = update_trailing_stop(60_000, stop, extreme, price, 600, limits)
        seen.append(stop)
    assert seen == sorted(seen), "a stop that loosens is not a stop"
    assert extreme == 62_000


def test_stop_fires_when_price_trades_through_it(limits):
    stop, extreme = update_trailing_stop(60_000, None, 60_000, 62_000, 600, limits)
    assert stop_hit(stop, 62_000) is False
    assert stop_hit(stop, stop - 1) is True
    assert stop_hit(None, 1) is False


# ---------------------------------------------------------------- state
def test_risk_state_survives_serialisation():
    state = fresh_state()
    state.start_cooldown("BTC-USD", 30, now=0)
    state.halted, state.halt_reason = True, "test"
    restored = RiskState.from_dict(state.to_dict())
    assert restored == state


def test_day_rolls_at_utc_midnight_and_rebases_the_daily_limit():
    state = RiskState()
    state.observe(10_000, now=0)                  # 1970-01-01
    assert state.day_key == "1970-01-01"
    state.observe(9_000, now=86_400 + 60)         # next UTC day
    assert state.day_key == "1970-01-02"
    assert state.day_start_equity == 9_000
    assert state.day_pnl_pct(9_000) == pytest.approx(0.0)


def test_peak_equity_is_a_high_water_mark():
    state = RiskState()
    for equity in (10_000, 12_000, 9_000):
        state.observe(equity, now=0)
    assert state.peak_equity == 12_000
    assert state.drawdown(9_000) == pytest.approx(0.25)


def test_max_drawdown_is_lifetime_and_survives_recovery():
    """The trough must still be reported after equity makes a new high."""
    state = RiskState()
    for equity in (10_000, 12_000, 9_000, 13_000, 12_500):
        state.observe(equity, now=0)
    assert state.max_drawdown == pytest.approx(0.25)
    assert state.drawdown(12_500) < state.max_drawdown

    restored = RiskState.from_dict(state.to_dict())
    assert restored.max_drawdown == pytest.approx(0.25)


def test_first_benchmark_price_round_trips_and_defaults_to_none():
    assert RiskState().first_benchmark_price is None
    state = fresh_state()
    state.first_benchmark_price = 60_000.0
    assert RiskState.from_dict(state.to_dict()).first_benchmark_price == 60_000.0
    # A state persisted before the field existed still loads.
    assert RiskState.from_dict({"peak_equity": 1.0}).first_benchmark_price is None
