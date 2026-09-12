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
    broker.market_order("BTC-USD", "buy", 0.1, 60_000, ts=1)
    broker.market_order("BTC-USD", "sell", 0.1, 60_000, ts=2)

    buy_price, sell_price = 60_000 * 1.0005, 60_000 * 0.9995
    expected_fees = 0.1 * buy_price * 0.001 + 0.1 * sell_price * 0.001
    expected = 10_000 - 0.1 * (buy_price - sell_price) - expected_fees

    assert broker.equity({}) == pytest.approx(expected)
    assert broker.positions() == {}


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
