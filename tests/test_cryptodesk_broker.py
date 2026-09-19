"""Paper-broker accounting: cash, fees, realised P&L, and refusals.

These are the invariants everything else rests on — if the ledger's arithmetic
is wrong, every performance figure the desk reports is fiction.
"""

import pytest

from cryptodesk.broker import OrderRejected, PaperBroker


@pytest.fixture
def broker():
    return PaperBroker(starting_equity=10_000, fee_bps=10, slippage_bps=5)


def test_round_trip_at_a_flat_price_loses_exactly_fees_and_slippage(broker):
    entry = broker.market_order("BTC-USD", "buy", 0.1, 60_000, ts=1)
    exit_ = broker.market_order("BTC-USD", "sell", 0.1, 60_000, ts=2)

    buy_price, sell_price = 60_000 * 1.0005, 60_000 * 0.9995
    expected_fees = 0.1 * buy_price * 0.001 + 0.1 * sell_price * 0.001
    slippage_cost = 0.1 * (buy_price - sell_price)
    expected = 10_000 - slippage_cost - expected_fees

    assert broker.equity({}) == pytest.approx(expected)
    assert broker.positions() == {}
    # Realised P&L must carry the entry-side fee too, or every per-trade
    # statistic is gross of half the costs while equity is net of all of them.
    assert exit_.realized_pnl == pytest.approx(-(entry.fee + exit_.fee) - slippage_cost)
    assert exit_.realized_pnl == pytest.approx(broker.equity({}) - 10_000)
    assert broker.total_realized_pnl == pytest.approx(broker.equity({}) - 10_000)


def test_a_marginal_winner_after_the_exit_fee_is_a_loser_after_both_fees():
    """The case the entry-fee omission mis-scored as a win."""
    broker = PaperBroker(10_000, fee_bps=10, slippage_bps=5)
    broker.market_order("BTC-USD", "buy", 0.1, 60_000, ts=1)
    fill = broker.market_order("BTC-USD", "sell", 0.1, 60_000 * 1.0025, ts=2)
    assert fill.realized_pnl < 0
    assert fill.realized_pnl == pytest.approx(broker.equity({}) - 10_000)


def test_entry_fees_are_released_pro_rata_over_partial_exits():
    broker = PaperBroker(100_000, fee_bps=10, slippage_bps=0)
    broker.market_order("BTC-USD", "buy", 2.0, 100, ts=1)     # entry fee 0.20
    pos = broker.position("BTC-USD")
    assert pos.entry_fees == pytest.approx(0.20)
    assert pos.avg_price == pytest.approx(100), "avg_price stays gross of fees"

    first = broker.market_order("BTC-USD", "sell", 0.5, 100, ts=2)
    # A quarter of the position closes: a quarter of the entry fee plus the
    # exit fee on 50 notional.
    assert first.realized_pnl == pytest.approx(-(0.05 + 0.05))
    assert broker.position("BTC-USD").entry_fees == pytest.approx(0.15)

    second = broker.market_order("BTC-USD", "sell", 1.5, 100, ts=3)
    assert second.realized_pnl == pytest.approx(-(0.15 + 0.15))
    assert broker.positions() == {}
    assert broker.total_realized_pnl == pytest.approx(broker.equity({}) - 100_000)


def test_adding_to_a_position_accumulates_entry_fees_under_one_trade_id():
    broker = PaperBroker(100_000, fee_bps=10, slippage_bps=0)
    first = broker.market_order("BTC-USD", "buy", 1.0, 100, ts=1)
    second = broker.market_order("BTC-USD", "buy", 1.0, 100, ts=2)
    pos = broker.position("BTC-USD")
    assert pos.entry_fees == pytest.approx(0.20)
    assert first.trade_id == second.trade_id == pos.trade_id
    assert pos.trade_id == "BTC-USD-1-1"

    closing = broker.market_order("BTC-USD", "sell", 2.0, 100, ts=3)
    assert closing.trade_id == pos.trade_id
    # A fresh lifecycle gets a fresh id, even at the same timestamp.
    reopened = broker.market_order("BTC-USD", "buy", 1.0, 100, ts=3)
    assert reopened.trade_id == "BTC-USD-3-2"
    assert reopened.trade_id != closing.trade_id


def test_costless_broker_captures_the_whole_move():
    broker = PaperBroker(10_000, fee_bps=0, slippage_bps=0)
    broker.market_order("ETH-USD", "buy", 1.0, 3_000, ts=1)
    broker.market_order("ETH-USD", "sell", 1.0, 3_300, ts=2)
    assert broker.equity({}) == pytest.approx(10_300)


def test_adding_to_a_position_averages_the_entry():
    broker = PaperBroker(100_000, fee_bps=0, slippage_bps=0)
    broker.market_order("BTC-USD", "buy", 1.0, 100, ts=1)
    broker.market_order("BTC-USD", "buy", 1.0, 200, ts=2)
    position = broker.position("BTC-USD")
    assert position.avg_price == pytest.approx(150.0)
    assert position.qty == pytest.approx(2.0)


def test_partial_exit_books_pnl_against_the_average_entry():
    broker = PaperBroker(100_000, fee_bps=0, slippage_bps=0)
    broker.market_order("BTC-USD", "buy", 2.0, 150, ts=1)
    fill = broker.market_order("BTC-USD", "sell", 0.5, 300, ts=2)
    assert fill.realized_pnl == pytest.approx(0.5 * (300 - 150))
    assert broker.position("BTC-USD").qty == pytest.approx(1.5)


def test_closing_the_last_unit_removes_the_position(broker):
    broker.market_order("SOL-USD", "buy", 3.0, 100, ts=1)
    broker.close("SOL-USD", 110, ts=2)
    assert "SOL-USD" not in broker.positions()


@pytest.mark.parametrize(
    "order, message",
    [
        (("BTC-USD", "buy", 1.0, 60_000_000), "Insufficient cash"),
        (("BTC-USD", "buy", 0.0, 60_000), "must be positive"),
        (("BTC-USD", "buy", 1.0, -5), "non-positive price"),
        (("BTC-USD", "sell", 1.0, 60_000), "Short selling is disabled"),
    ],
)
def test_invalid_orders_are_refused(broker, order, message):
    with pytest.raises(OrderRejected, match=message):
        broker.market_order(*order)


def test_overselling_a_position_is_refused_when_shorts_are_off(broker):
    broker.market_order("BTC-USD", "buy", 0.01, 60_000, ts=1)
    with pytest.raises(OrderRejected, match="only"):
        broker.market_order("BTC-USD", "sell", 5.0, 60_000, ts=2)


def test_overselling_is_refused_rather_than_truncated_even_with_shorts_on():
    """A reversal must not be silently filled as a plain close."""
    broker = PaperBroker(10_000, fee_bps=10, slippage_bps=5, allow_shorts=True)
    broker.market_order("BTC-USD", "buy", 0.1, 60_000, ts=1)
    with pytest.raises(OrderRejected, match="reversing direction"):
        broker.market_order("BTC-USD", "sell", 0.3, 60_000, ts=2)
    assert broker.position("BTC-USD").qty == pytest.approx(0.1)
    assert broker.total_fees == pytest.approx(sum(f.fee for f in broker.fills))


def test_total_fees_reconciles_with_the_fills_and_with_cash():
    broker = PaperBroker(10_000, fee_bps=10, slippage_bps=5)
    broker.market_order("BTC-USD", "buy", 0.1, 60_000, ts=1)
    broker.market_order("BTC-USD", "sell", 0.04, 60_000, ts=2)
    broker.close("BTC-USD", 60_000, ts=3)
    assert broker.total_fees == pytest.approx(sum(f.fee for f in broker.fills))
    slippage_cost = sum(abs(f.price - f.reference_price) * f.qty for f in broker.fills)
    assert 10_000 - broker.cash() == pytest.approx(broker.total_fees + slippage_cost)


def test_missing_price_marks_at_entry_rather_than_dropping_the_position():
    """A feed hiccup must not make equity jump — that would trip the kill-switch."""
    broker = PaperBroker(10_000, fee_bps=0, slippage_bps=0)
    broker.market_order("SOL-USD", "buy", 10, 100, ts=1)
    assert broker.equity({}) == pytest.approx(10_000)


def test_gross_exposure_is_notional_over_equity():
    broker = PaperBroker(10_000, fee_bps=0, slippage_bps=0)
    broker.market_order("SOL-USD", "buy", 10, 100, ts=1)
    assert broker.gross_exposure({"SOL-USD": 100}) == pytest.approx(0.1)


def test_state_round_trips_through_serialisation():
    broker = PaperBroker(10_000, fee_bps=10, slippage_bps=5)
    broker.market_order("BTC-USD", "buy", 0.05, 60_000, ts=1)
    broker.position("BTC-USD").stop_price = 58_000
    broker.market_order("ETH-USD", "buy", 1.0, 3_000, ts=2)

    restored = PaperBroker(10_000, fee_bps=10, slippage_bps=5)
    assert restored.load_state(broker.state()) is True

    assert restored.cash() == pytest.approx(broker.cash())
    assert restored.total_fees == pytest.approx(broker.total_fees)
    assert set(restored.positions()) == set(broker.positions())
    assert restored.position("BTC-USD").stop_price == pytest.approx(58_000)
    assert restored.position("BTC-USD").qty == pytest.approx(broker.position("BTC-USD").qty)
    assert restored.position("BTC-USD").trade_id == broker.position("BTC-USD").trade_id
    assert restored.position("BTC-USD").entry_fees == pytest.approx(
        broker.position("BTC-USD").entry_fees)

    # The restored book books the same realised P&L and never reuses an id.
    original_close = broker.close("ETH-USD", 3_000, ts=3)
    restored_close = restored.close("ETH-USD", 3_000, ts=3)
    assert restored_close.realized_pnl == pytest.approx(original_close.realized_pnl)
    assert restored.market_order("SOL-USD", "buy", 1.0, 100, ts=3).trade_id == "SOL-USD-3-3"


def test_state_written_before_trade_ids_still_loads():
    """Old ledgers have no trade_id/entry_fees; they must restore, not be refused."""
    broker = PaperBroker(10_000)
    assert broker.load_state({
        "cash": 4_000, "starting_equity": 10_000,
        "positions": [{"symbol": "BTC-USD", "qty": 0.1, "avg_price": 60_000,
                       "opened_ts": 1_700_000_000}],
    }) is True
    pos = broker.position("BTC-USD")
    assert pos.trade_id == "BTC-USD-1700000000"
    assert pos.entry_fees == 0.0
    assert broker.close("BTC-USD", 60_000, ts=1_700_000_100).trade_id == pos.trade_id


@pytest.mark.parametrize("bad", [None, {}, {"cash": 5_000, "positions": [{"symbol": "X"}]}])
def test_malformed_state_is_refused_without_partial_application(bad):
    broker = PaperBroker(10_000)
    assert broker.load_state(bad) is False
    assert broker.cash() == pytest.approx(10_000)
    assert broker.positions() == {}


def test_restore_keeps_the_original_starting_capital():
    """Returns must be measured from the capital the run actually began with."""
    broker = PaperBroker(50_000)
    broker.load_state({"cash": 9_000, "starting_equity": 10_000, "positions": []})
    assert broker.starting_equity == pytest.approx(10_000)
