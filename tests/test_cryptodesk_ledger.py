"""Ledger durability and accounting: atomic writes, migration, per-trade stats.

The ledger is what a restarted desk trusts. These tests pin the properties
that make it trustworthy: a fill and the book it produced land together or not
at all, old databases open after schema additions, and per-trade statistics
score a position's lifecycle as one trade however many fills closed it.
"""

import sqlite3

import pytest

from cryptodesk.broker import PaperBroker
from cryptodesk.engine.ledger import Ledger


@pytest.fixture
def ledger(tmp_path):
    ledger = Ledger(tmp_path / "desk.sqlite")
    yield ledger
    ledger.close()


def _fill(ts, side, qty, realized=0.0, trade_id="BTC-USD-1-1", symbol="BTC-USD"):
    return {"ts": ts, "symbol": symbol, "side": side, "qty": qty, "price": 100.0,
            "reference_price": 100.0, "fee": 0.1, "realized_pnl": realized,
            "reason": "test", "trade_id": trade_id}


# ------------------------------------------------------------ transactions
def test_transaction_commits_every_write_together(ledger):
    with ledger.transaction() as tx:
        tx.record_fill(_fill(1, "buy", 1.0))
        tx.set_state("broker_state", {"cash": 1.0, "positions": []})
        tx.record_event("info", "test", "hello", detail={"k": 1}, ts=1)
        tx.record_decision(ts=1, symbol="BTC-USD", rating="Buy", detail={"trend": "up"})
        tx.record_equity(ts=1, equity=1.0, cash=1.0, gross_exposure=0.0, drawdown=0.0)

    assert len(ledger.recent_fills()) == 1
    assert ledger.get_state("broker_state") == {"cash": 1.0, "positions": []}
    assert ledger.recent_events()[0]["message"] == "hello"
    assert ledger.recent_decisions()[0]["detail"] == {"trend": "up"}
    assert len(ledger.equity_curve()) == 1


def test_transaction_rolls_back_everything_on_exception(ledger):
    ledger.set_state("broker_state", {"cash": 10.0, "positions": []})
    with pytest.raises(RuntimeError, match="boom"), ledger.transaction() as tx:
        tx.record_fill(_fill(1, "buy", 1.0))
        tx.set_state("broker_state", {"cash": 1.0, "positions": []})
        raise RuntimeError("boom")

    # Neither half landed: the blotter and the book still agree.
    assert ledger.recent_fills() == []
    assert ledger.get_state("broker_state") == {"cash": 10.0, "positions": []}
    # And the ledger is still usable afterwards.
    ledger.record_event("info", "test", "after", ts=2)
    assert ledger.recent_events()[0]["message"] == "after"


def test_transaction_is_invisible_to_a_reader_until_commit(tmp_path):
    """A second connection (the API thread's view) sees nothing mid-transaction."""
    path = tmp_path / "desk.sqlite"
    ledger = Ledger(path)
    reader = sqlite3.connect(str(path))
    try:
        with ledger.transaction() as tx:
            tx.record_fill(_fill(1, "buy", 1.0))
            assert reader.execute("SELECT COUNT(*) FROM fills").fetchone()[0] == 0
        assert reader.execute("SELECT COUNT(*) FROM fills").fetchone()[0] == 1
    finally:
        reader.close()
        ledger.close()


def test_non_transactional_writes_still_commit_on_their_own(tmp_path):
    path = tmp_path / "desk.sqlite"
    ledger = Ledger(path)
    ledger.record_fill(_fill(1, "buy", 1.0))
    ledger.set_state("k", [1, 2])
    ledger.close()
    reopened = Ledger(path)
    try:
        assert len(reopened.recent_fills()) == 1
        assert reopened.recent_fills()[0]["trade_id"] == "BTC-USD-1-1"
        assert reopened.get_state("k") == [1, 2]
    finally:
        reopened.close()


# -------------------------------------------------------------- migration
def test_a_ledger_created_before_trade_ids_is_migrated_on_open(tmp_path):
    path = tmp_path / "old.sqlite"
    conn = sqlite3.connect(str(path))
    conn.executescript("""
        CREATE TABLE fills (
            id INTEGER PRIMARY KEY AUTOINCREMENT, ts INTEGER NOT NULL,
            symbol TEXT NOT NULL, side TEXT NOT NULL, qty REAL NOT NULL,
            price REAL NOT NULL, reference_price REAL NOT NULL, fee REAL NOT NULL,
            realized_pnl REAL NOT NULL, reason TEXT);
        INSERT INTO fills(ts, symbol, side, qty, price, reference_price, fee, realized_pnl, reason)
        VALUES (1, 'BTC-USD', 'buy', 1, 100, 100, 0.1, 0, 'old'),
               (2, 'BTC-USD', 'sell', 1, 110, 110, 0.1, 9.8, 'old');
    """)
    conn.commit()
    conn.close()

    ledger = Ledger(path)
    try:
        columns = {row["name"] for row in ledger._read("PRAGMA table_info(fills)")}
        assert "trade_id" in columns
        # New fills carry the id; old rows read back as NULL and still count.
        ledger.record_fill(_fill(3, "buy", 1.0))
        fills = ledger.recent_fills()
        assert fills[0]["trade_id"] == "BTC-USD-1-1"
        assert fills[-1]["trade_id"] is None
        assert ledger.trade_stats()["closed_trades"] == 1
    finally:
        ledger.close()

    # Opening again must not try to add the column twice.
    Ledger(path).close()


# ------------------------------------------------------------ trade_stats
def _persist_book(ledger, broker):
    ledger.set_state("broker_state", broker.state())


def test_a_trade_exited_in_two_partial_fills_is_one_closed_trade(ledger):
    """The numeric example: flat-price round trip, two exits, one losing trade."""
    broker = PaperBroker(10_000, fee_bps=10, slippage_bps=5)
    entry = broker.market_order("BTC-USD", "buy", 0.1, 60_000, ts=1)
    first = broker.market_order("BTC-USD", "sell", 0.04, 60_000, ts=2)
    second = broker.market_order("BTC-USD", "sell", 0.06, 60_000, ts=3)
    for fill in (entry, first, second):
        ledger.record_fill(fill.to_dict())
    _persist_book(ledger, broker)

    stats = ledger.trade_stats()
    assert stats["closed_trades"] == 1
    assert stats["partial_fills"] == 0
    assert stats["wins"] == 0 and stats["losses"] == 1
    assert stats["win_rate"] == 0.0

    slippage_cost = 0.1 * (60_000 * 1.0005 - 60_000 * 0.9995)
    expected = -(entry.fee + first.fee + second.fee) - slippage_cost
    assert stats["net_realized"] == pytest.approx(expected)
    assert stats["net_realized"] == pytest.approx(broker.equity({}) - 10_000)
    assert stats["avg_loss"] == pytest.approx(-expected)
    assert stats["profit_factor"] == pytest.approx(0.0)


def test_a_trimmed_winner_that_stops_out_is_one_losing_trade(ledger):
    """Buy 1 @100, trim 0.5 @110 (+5), stop 0.5 @80 (-10): one trade, net -5."""
    broker = PaperBroker(10_000, fee_bps=0, slippage_bps=0)
    for fill in (broker.market_order("BTC-USD", "buy", 1.0, 100, ts=1),
                 broker.market_order("BTC-USD", "sell", 0.5, 110, ts=2),
                 broker.market_order("BTC-USD", "sell", 0.5, 80, ts=3)):
        ledger.record_fill(fill.to_dict())
    _persist_book(ledger, broker)

    stats = ledger.trade_stats()
    assert (stats["closed_trades"], stats["wins"], stats["losses"]) == (1, 0, 1)
    assert stats["net_realized"] == pytest.approx(-5.0)
    assert stats["gross_profit"] == 0.0 and stats["gross_loss"] == pytest.approx(5.0)


def test_a_partial_exit_of_an_open_trade_is_not_a_closed_trade(ledger):
    broker = PaperBroker(10_000, fee_bps=0, slippage_bps=0)
    ledger.record_fill(broker.market_order("BTC-USD", "buy", 1.0, 100, ts=1).to_dict())
    ledger.record_fill(broker.market_order("BTC-USD", "sell", 0.5, 110, ts=2).to_dict())
    _persist_book(ledger, broker)

    stats = ledger.trade_stats()
    assert stats["closed_trades"] == 0
    assert stats["partial_fills"] == 1
    assert stats["wins"] == 0
    assert stats["net_realized"] == 0.0

    # Once the remainder closes, the whole lifecycle scores once.
    ledger.record_fill(broker.market_order("BTC-USD", "sell", 0.5, 120, ts=3).to_dict())
    _persist_book(ledger, broker)
    stats = ledger.trade_stats()
    assert (stats["closed_trades"], stats["partial_fills"], stats["wins"]) == (1, 0, 1)
    assert stats["net_realized"] == pytest.approx(5.0 + 10.0)


def test_a_flat_close_in_a_costless_broker_still_counts_as_a_closed_trade(ledger):
    """Zero realised P&L is a closed trade, not an absent one."""
    broker = PaperBroker(10_000, fee_bps=0, slippage_bps=0)
    ledger.record_fill(broker.market_order("BTC-USD", "buy", 1.0, 100, ts=1).to_dict())
    ledger.record_fill(broker.market_order("BTC-USD", "sell", 1.0, 100, ts=2).to_dict())
    _persist_book(ledger, broker)
    assert ledger.trade_stats()["closed_trades"] == 1


def test_trade_stats_separate_lifecycles_of_the_same_symbol(ledger):
    broker = PaperBroker(10_000, fee_bps=0, slippage_bps=0)
    for args in (("buy", 1.0, 100, 1), ("sell", 1.0, 110, 2),
                 ("buy", 1.0, 100, 3), ("sell", 1.0, 90, 4)):
        side, qty, price, ts = args
        ledger.record_fill(broker.market_order("BTC-USD", side, qty, price, ts=ts).to_dict())
    _persist_book(ledger, broker)
    stats = ledger.trade_stats()
    assert (stats["closed_trades"], stats["wins"], stats["losses"]) == (2, 1, 1)
    assert stats["profit_factor"] == pytest.approx(1.0)


# ------------------------------------------------------ sampled curve
def test_equity_curve_sampled_spans_the_full_history_evenly(ledger):
    for i in range(1_000):
        ledger.record_equity(ts=1_000 + i * 60, equity=10_000 + i, cash=0.0,
                             gross_exposure=0.0, drawdown=0.0)
    full = ledger.equity_curve_sampled(max_points=100_000)
    assert len(full) == 1_000
    assert [r["ts"] for r in full] == sorted(r["ts"] for r in full)

    thin = ledger.equity_curve_sampled(max_points=100)
    assert 100 <= len(thin) <= 101
    assert thin[0]["ts"] == 1_000, "the first mark anchors lifetime return"
    assert thin[-1]["ts"] == 1_000 + 999 * 60, "the last mark is the current equity"
    gaps = {b["ts"] - a["ts"] for a, b in zip(thin, thin[1:], strict=False)}
    assert max(gaps) <= 600, "points are spread over time, not bunched at one end"


def test_equity_curve_sampled_on_an_empty_ledger(ledger):
    assert ledger.equity_curve_sampled() == []


# --------------------------------------------------------- last_decision
def test_last_decision_returns_the_newest_row_with_detail_parsed(ledger):
    ledger.record_decision(ts=1, symbol="BTC-USD", price=100.0, detail={"trend": "up"})
    ledger.record_decision(ts=2, symbol="ETH-USD", price=5.0, detail={"trend": "down"})
    ledger.record_decision(ts=3, symbol="BTC-USD", price=120.0, detail={"trend": "flat"})

    row = ledger.last_decision("BTC-USD")
    assert row["price"] == 120.0
    assert row["detail"] == {"trend": "flat"}
    assert ledger.last_decision("ETH-USD")["detail"] == {"trend": "down"}
    assert ledger.last_decision("SOL-USD") is None
