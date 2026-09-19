"""End-to-end desk behaviour over compressed time.

Drives the real engine — real broker, real ledger, real risk logic — against a
deterministic synthetic market, so the tests exercise the code that actually
runs unattended rather than a mock of it.
"""

import sqlite3
import threading
import time

import pytest

from cryptodesk.config import DeskConfig
from cryptodesk.engine import risk, triggers
from cryptodesk.engine.committee import Proposal
from cryptodesk.engine.ledger import Ledger
from cryptodesk.engine.loop import Desk
from cryptodesk.feeds import ChainFeed, FeedError, SyntheticFeed
from cryptodesk.indicators import snapshot as compute_snapshot


class Clock:
    """A manually advanced clock shared by the desk and its feed."""

    def __init__(self, start=1_760_000_000):
        self.t = float(start)

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


class FixedCommittee:
    """A committee that always returns the same rating, for forcing a path."""

    source = "heuristic"

    def __init__(self, rating="Buy", conviction=1.0, cost=0.0, name="fixed"):
        self.rating, self.conviction, self.cost, self.name = rating, conviction, cost, name
        self.calls = 0

    def decide(self, symbol, snapshot, trigger_reason="", now=None):
        self.calls += 1
        return Proposal(symbol=symbol, rating=self.rating, conviction=self.conviction,
                        summary=f"always {self.rating}", source=self.source,
                        cost_usd=self.cost, cost_is_estimate=bool(self.cost))


class ToggleFeed(SyntheticFeed):
    """A synthetic feed whose per-symbol outage a test can switch on and off."""

    name = "toggle"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dark: set[str] = set()

    def _check(self, symbol):
        if symbol in self.dark:
            raise FeedError(f"{symbol} is dark")

    def candles(self, symbol, interval="5m", limit=200):
        self._check(symbol)
        return super().candles(symbol, interval, limit)

    def price(self, symbol):
        self._check(symbol)
        return super().price(symbol)

    def market(self, symbol, interval="5m", limit=200):
        self._check(symbol)
        return super().market(symbol, interval, limit)


def make_desk(tmp_path, committee=None, clock=None, **overrides):
    clock = clock or Clock()
    config = {
        "symbols": ["BTC-USD", "ETH-USD"], "starting_equity": 10_000,
        "home": str(tmp_path), "committee": "heuristic",
        "llm": {"min_minutes_between_calls": 20, "scheduled_interval_minutes": 60},
    }
    config.update(overrides)
    cfg = DeskConfig.from_dict(config)
    desk = Desk(cfg, feed=SyntheticFeed(seed=9, clock=clock),
                committee=committee or FixedCommittee(), clock=clock)
    return desk, cfg, clock


def run(desk, clock, ticks, step=300):
    for _ in range(ticks):
        desk.tick()
        clock.advance(step)


# ---------------------------------------------------------------- basics
def test_a_tick_marks_the_book_and_records_an_equity_point(tmp_path):
    desk, _, clock = make_desk(tmp_path)
    result = desk.tick()
    assert result["equity"] == pytest.approx(10_000)
    assert len(desk.ledger.equity_curve()) == 1


def test_the_desk_trades_and_respects_every_cap(tmp_path):
    desk, cfg, clock = make_desk(tmp_path)
    run(desk, clock, 600)

    state = desk.state()
    account = state["account"]
    assert account["fill_count"] > 0, "the desk never traded"
    assert account["gross_exposure"] <= cfg.risk.max_gross_exposure + 1e-9
    assert account["cash"] >= -1e-6, "cash went negative — leverage is not modelled"
    ceiling = cfg.risk.max_symbol_weight * (1 + cfg.risk.max_weight_drift)
    for position in state["positions"]:
        weight = abs(position["market_value"]) / account["equity"]
        assert weight <= ceiling + 1e-6, "weight drifted past the trim band"


def test_every_open_position_carries_a_stop(tmp_path):
    """An unstopped position is the one that turns a bad tick into a bad month."""
    desk, _, clock = make_desk(tmp_path)
    run(desk, clock, 400)
    positions = desk.state()["positions"]
    assert positions, "no positions were opened, so the assertion is vacuous"
    assert all(p["stop_price"] is not None for p in positions)


def test_a_sell_rating_flattens_an_existing_position(tmp_path):
    desk, _, clock = make_desk(tmp_path)
    run(desk, clock, 200)
    assert desk.broker.positions(), "expected an open position to sell"

    desk.committee = FixedCommittee(rating="Sell", conviction=0.0)
    run(desk, clock, 60)
    assert desk.broker.positions() == {}


def test_hold_changes_nothing(tmp_path):
    desk, _, clock = make_desk(tmp_path, committee=FixedCommittee(rating="Hold", conviction=0.0))
    run(desk, clock, 200)
    assert desk.broker.positions() == {}
    assert desk.broker.snapshot({})["fill_count"] == 0


# ---------------------------------------------------------------- kill-switch
def test_kill_switch_halts_flattens_and_stops_all_activity(tmp_path):
    desk, cfg, clock = make_desk(
        tmp_path,
        # Large exposure with a stop far wider than the 2% halt, so the drawdown
        # brake is what fires. risk_per_trade sits at the validator's ceiling;
        # with a 5-ATR stop the risk term implies >100% weight and the 90%
        # symbol cap binds, which is the oversized book the test wants.
        risk={"max_drawdown_halt": 0.02, "stop_atr_mult": 5, "trail_atr_mult": 5,
              "max_symbol_weight": 0.9, "max_gross_exposure": 0.9, "risk_per_trade": 0.1},
        llm={"min_minutes_between_calls": 5, "scheduled_interval_minutes": 10},
    )
    desk.feed = SyntheticFeed(seed=5, drift_per_year=-8.0, annual_vol=1.2, clock=clock)
    run(desk, clock, 1_500)

    assert desk.risk_state.halted, "a 2% limit should have tripped on a falling market"
    assert desk.broker.positions() == {}, "a halt must flatten the book"

    halt_ts = desk.ledger._read("SELECT MIN(ts) AS ts FROM events WHERE kind='halt'")[0]["ts"]
    assert not desk.ledger._read("SELECT 1 FROM fills WHERE ts > ?", (halt_ts,))
    assert not desk.ledger._read("SELECT 1 FROM decisions WHERE ts > ?", (halt_ts,))


def test_resume_clears_the_halt_and_rebases_the_peak(tmp_path):
    desk, _, clock = make_desk(tmp_path)
    desk.tick()
    desk.halt("manual")
    assert desk.risk_state.halted

    desk.resume()
    assert not desk.risk_state.halted
    assert desk.risk_state.peak_equity == pytest.approx(desk.broker.equity(desk._prices))


# ---------------------------------------------------------------- durability
def test_the_book_survives_a_restart(tmp_path):
    """Containers get redeployed; the desk must not forget its open positions."""
    desk, cfg, clock = make_desk(tmp_path)
    run(desk, clock, 600)
    before = desk.state()
    assert before["positions"], "expected open positions to restore"

    restarted = Desk(cfg, feed=SyntheticFeed(seed=9, clock=clock),
                     committee=FixedCommittee(), ledger=Ledger(cfg.db_path), clock=clock)
    assert restarted.restored is True

    after = restarted.state()
    assert after["account"]["cash"] == pytest.approx(before["account"]["cash"])
    assert after["account"]["total_fees"] == pytest.approx(before["account"]["total_fees"])
    assert {p["symbol"]: p["qty"] for p in after["positions"]} == \
           {p["symbol"]: p["qty"] for p in before["positions"]}
    assert {p["symbol"]: p["stop_price"] for p in after["positions"]} == \
           {p["symbol"]: p["stop_price"] for p in before["positions"]}


def test_a_halt_survives_a_restart(tmp_path):
    desk, cfg, clock = make_desk(tmp_path)
    desk.tick()
    desk.halt("manual halt")

    restarted = Desk(cfg, feed=SyntheticFeed(seed=9, clock=clock),
                     committee=FixedCommittee(), ledger=Ledger(cfg.db_path), clock=clock)
    assert restarted.risk_state.halted
    assert restarted.tick()["halted"] is True


# ---------------------------------------------------------------- budget
def test_llm_spend_never_exceeds_the_daily_cap(tmp_path):
    """The cap is what stands between a 24/7 committee and a four-figure bill."""
    committee = FixedCommittee(rating="Buy", cost=0.35, name="fake-llm")
    desk, cfg, clock = make_desk(
        tmp_path, committee=committee, committee_mode="llm",
        llm={"daily_usd_cap": 5.0, "min_minutes_between_calls": 15,
             "scheduled_interval_minutes": 60, "estimated_cost_per_run_usd": 0.35},
    )
    run(desk, clock, 1_500)

    per_day = desk.ledger._read("SELECT day, SUM(cost_usd) AS total FROM llm_spend GROUP BY day")
    assert per_day, "no spend was recorded at all"
    assert all(row["total"] <= cfg.llm.daily_usd_cap + 1e-9 for row in per_day)


def test_feed_failure_on_one_symbol_does_not_stop_the_desk(tmp_path):
    class HalfBrokenFeed(SyntheticFeed):
        name = "half-broken"

        def market(self, symbol, interval="5m", limit=200):
            if symbol == "ETH-USD":
                raise ValueError("simulated venue outage")
            return super().market(symbol, interval, limit)

    desk, _, clock = make_desk(tmp_path)
    desk.feed = HalfBrokenFeed(seed=9, clock=clock)
    run(desk, clock, 200)

    assert "ETH-USD" in desk.feed_errors
    assert "BTC-USD" in desk._prices, "a broken symbol must not blind the others"
    assert desk.state()["account"]["fill_count"] > 0


def test_total_data_loss_is_reported_rather_than_crashing(tmp_path):
    class DeadFeed(SyntheticFeed):
        name = "dead"

        def market(self, symbol, interval="5m", limit=200):
            raise ValueError("no data")

    desk, _, clock = make_desk(tmp_path)
    desk.feed = DeadFeed(seed=9, clock=clock)
    result = desk.tick()
    assert "error" in result and result["actions"] == []


def test_a_winner_is_trimmed_back_when_it_drifts_past_its_weight_cap(tmp_path):
    """The entry cap must not be escapable via mark-to-market appreciation."""
    # A Hold committee keeps the desk from spending the cash this test needs.
    desk, cfg, clock = make_desk(tmp_path,
                                 committee=FixedCommittee(rating="Hold", conviction=0.0))
    desk.tick()

    # Open a position deliberately oversized relative to the cap, then let the
    # desk manage it.
    price = desk._prices["BTC-USD"]
    equity = desk.broker.equity(desk._prices)
    oversized = (equity * 0.5) / price          # 50% weight, cap is 25%
    desk.broker.market_order("BTC-USD", "buy", oversized, price, ts=int(clock()))
    desk.broker.position("BTC-USD").stop_price = price * 0.5   # far away, won't fire

    before = abs(desk.broker.position("BTC-USD").market_value(price)) / equity
    assert before > cfg.risk.max_symbol_weight * (1 + cfg.risk.max_weight_drift)

    desk.tick()

    position = desk.broker.position("BTC-USD")
    price = desk._prices["BTC-USD"]
    after = abs(position.market_value(price)) / desk.broker.equity(desk._prices)
    assert after == pytest.approx(cfg.risk.max_symbol_weight, abs=0.02)


# ---------------------------------------------------------------- committee off the lock
def test_the_committee_runs_with_the_engine_lock_released(tmp_path):
    """state() from another thread must answer while a committee run is in flight."""
    answered = []

    class ProbingCommittee(FixedCommittee):
        def decide(self, symbol, snapshot, trigger_reason="", now=None):
            done = threading.Event()
            probe = threading.Thread(target=lambda: (desk.state(), done.set()), daemon=True)
            probe.start()
            # A held RLock would leave the probe blocked and this False.
            answered.append(done.wait(timeout=5))
            return super().decide(symbol, snapshot, trigger_reason, now)

    desk, _, clock = make_desk(tmp_path, committee=ProbingCommittee(rating="Hold"))
    result = desk.tick()
    assert result["consulted"] == ["BTC-USD", "ETH-USD"]
    assert answered == [True, True]


def test_halt_lands_while_the_committee_is_still_thinking(tmp_path):
    """The kill-switch must not queue behind a five-minute model run."""
    started, release = threading.Event(), threading.Event()

    class BlockingCommittee(FixedCommittee):
        def decide(self, symbol, snapshot, trigger_reason="", now=None):
            started.set()
            assert release.wait(timeout=10), "the test never released the committee"
            return super().decide(symbol, snapshot, trigger_reason, now)

    desk, _, clock = make_desk(tmp_path, committee=BlockingCommittee(rating="Buy"),
                               symbols=["BTC-USD"])
    worker = threading.Thread(target=desk.tick, daemon=True)
    worker.start()
    assert started.wait(timeout=5)

    desk.halt("operator")                      # returns while decide() is blocked
    assert desk.risk_state.halted
    assert desk.busy == "committee BTC-USD"

    release.set()
    worker.join(timeout=10)
    assert not worker.is_alive()
    # The Buy arrived after the halt and must not have been acted on.
    assert desk.broker.positions() == {}
    assert desk.broker.snapshot({})["fill_count"] == 0
    assert desk.ledger.last_decision("BTC-USD")["veto_reason"].startswith("desk halted")
    assert desk.busy is None


def test_orders_fill_at_the_price_after_the_committee_not_before(tmp_path):
    class SlowCommittee(FixedCommittee):
        def decide(self, symbol, snapshot, trigger_reason="", now=None):
            self.asked_at = snapshot.price
            clock.advance(600)                 # two bars: past fast_loop_seconds
            return super().decide(symbol, snapshot, trigger_reason, now)

    committee = SlowCommittee(rating="Buy")
    desk, _, clock = make_desk(tmp_path, committee=committee, symbols=["BTC-USD"])
    start = int(clock())
    result = desk.tick()

    fill = desk.ledger.recent_fills(1)[0]
    refreshed = desk._prices["BTC-USD"]
    assert refreshed != pytest.approx(committee.asked_at), "the market did not move; vacuous"
    assert fill["reference_price"] == pytest.approx(refreshed)
    assert fill["ts"] == start + 600
    assert result["ts"] == start, "the mark keeps the tick-start stamp"
    assert desk.last_tick_ts == start + 600

    decision = desk.ledger.last_decision("BTC-USD")
    assert decision["price"] == pytest.approx(committee.asked_at)
    assert decision["detail"]["applied_price"] == pytest.approx(refreshed)
    assert decision["detail"]["committee_seconds"] == 600
    assert decision["ts"] == start + 600


def test_a_fill_is_on_disk_with_its_book_before_the_next_symbol_is_considered(tmp_path):
    """A crash after the first fill must not restart a desk that forgot it."""
    class CrashingCommittee(FixedCommittee):
        def decide(self, symbol, snapshot, trigger_reason="", now=None):
            if symbol == "ETH-USD":
                raise RuntimeError("committee died")
            return super().decide(symbol, snapshot, trigger_reason, now)

    desk, cfg, clock = make_desk(tmp_path, committee=CrashingCommittee(rating="Buy"))
    with pytest.raises(RuntimeError, match="committee died"):
        desk.tick()
    live = desk.broker.position("BTC-USD")
    assert live is not None and desk.busy is None

    restarted = Desk(cfg, feed=SyntheticFeed(seed=9, clock=clock), committee=FixedCommittee(),
                     ledger=Ledger(cfg.db_path), clock=clock)
    assert restarted.restored
    restored = restarted.broker.position("BTC-USD")
    assert restored is not None
    assert restored.qty == pytest.approx(live.qty)
    assert restored.stop_price == pytest.approx(live.stop_price)
    assert restarted.broker.cash() == pytest.approx(desk.broker.cash())
    fills = restarted.ledger.recent_fills()
    assert [f["symbol"] for f in fills] == ["BTC-USD"]


def test_a_new_position_is_stopped_before_the_ledger_is_touched(tmp_path):
    desk, _, clock = make_desk(tmp_path, symbols=["BTC-USD"])
    real_commit = desk._commit

    def failing_commit(booking):
        if booking.fills:
            raise sqlite3.OperationalError("disk full")
        return real_commit(booking)

    desk._commit = failing_commit
    with pytest.raises(sqlite3.OperationalError):
        desk.tick()

    pos = desk.broker.position("BTC-USD")
    snap = desk._snapshots["BTC-USD"]
    assert pos is not None
    assert pos.stop_price == pytest.approx(snap.price - snap.atr14 * desk.cfg.risk.stop_atr_mult)
    # Nothing half-landed: the blotter and the persisted book agree (both empty).
    assert desk.ledger.recent_fills() == []
    assert desk.ledger.get_state("broker_state") is None


def test_a_paid_committee_that_dies_mid_run_is_still_charged(tmp_path):
    class DyingCommittee(FixedCommittee):
        source = "llm"

        def decide(self, symbol, snapshot, trigger_reason="", now=None):
            raise RuntimeError("provider hung up")

    desk, _, clock = make_desk(tmp_path, committee=DyingCommittee(name="llm"),
                               symbols=["BTC-USD"], llm={"estimated_cost_per_run_usd": 0.35})
    with pytest.raises(RuntimeError, match="hung up"):
        desk.tick()
    assert desk.ledger.spend_total() == pytest.approx(0.35)
    assert desk.busy is None


# ---------------------------------------------------------------- ratings
def test_underweight_never_opens_a_position(tmp_path):
    desk, _, clock = make_desk(tmp_path, committee=FixedCommittee(rating="Underweight",
                                                                  conviction=0.2))
    run(desk, clock, 50)
    assert desk.broker.snapshot({})["fill_count"] == 0
    decisions = desk.ledger.recent_decisions(10)
    assert decisions
    assert all(d["action"] == "none" for d in decisions)
    assert all(d["veto_reason"] == "Underweight with no position: not initiating"
               for d in decisions)


def test_underweight_trims_a_full_position_toward_a_fifth_of_max(tmp_path):
    desk, cfg, clock = make_desk(tmp_path, committee=FixedCommittee(rating="Underweight",
                                                                    conviction=0.2),
                                 symbols=["BTC-USD"])
    candles, price, _ = desk.feed.market("BTC-USD", cfg.candle_interval, cfg.candle_lookback)
    atr = compute_snapshot("BTC-USD", candles, price).atr14
    full = risk.size_position(10_000, price, atr, cfg.risk, 1.0)
    desk.broker.market_order("BTC-USD", "buy", full, price, ts=int(clock()))

    result = desk.tick()

    pos = desk.broker.position("BTC-USD")
    assert pos is not None and pos.qty == pytest.approx(full * 0.2, rel=0.01)
    assert [f["side"] for f in desk.ledger.recent_fills()] == ["sell"]
    action = next(a for a in result["actions"] if a.get("rating") == "Underweight")
    assert action["action"] == "reduced"
    assert action["reason"].startswith("Underweight: trim toward 20% of max")


def test_a_refused_buy_on_an_oversized_position_rebalances_with_an_explicit_reason(tmp_path):
    desk, cfg, clock = make_desk(tmp_path, symbols=["BTC-USD"])
    price = desk.feed.price("BTC-USD")
    # 29% of equity: inside the 30% per-symbol drift band (no drift trim) but
    # well past the 25% the vol target allows, so the Buy is refused as
    # "already at or above target" and the fallback trims.
    desk.broker.market_order("BTC-USD", "buy", 0.29 * 10_000 / price, price, ts=int(clock()))

    result = desk.tick()

    action = next(a for a in result["actions"] if a.get("rating") == "Buy")
    assert action["action"] == "reduced"
    assert action["reason"].startswith("vol-target rebalance: target ")
    assert " < 90% of held " in action["reason"]
    decision = desk.ledger.last_decision("BTC-USD")
    assert decision["action"] == "reduced"
    assert decision["veto_reason"] == action["reason"]
    weight = (desk.broker.position("BTC-USD").market_value(desk._prices["BTC-USD"])
              / desk.broker.equity(desk._prices))
    assert weight == pytest.approx(cfg.risk.max_symbol_weight, abs=0.005)


# ---------------------------------------------------------------- brakes
def test_resume_refuses_a_running_desk_and_keeps_the_days_loss(tmp_path):
    desk, cfg, clock = make_desk(tmp_path, committee=FixedCommittee(rating="Hold"))
    desk.tick()
    with pytest.raises(RuntimeError, match="not halted"):
        desk.resume()

    equity = desk.broker.equity(desk._prices)
    # As if the day had opened 5% higher: past the 3% daily-loss limit.
    desk.risk_state.day_start_equity = equity / 0.95
    desk.halt("maintenance")
    desk.resume()

    assert not desk.risk_state.halted
    assert desk.risk_state.day_start_equity == pytest.approx(equity / 0.95)
    assert desk.risk_state.peak_equity == pytest.approx(equity)
    blocked = triggers.entries_blocked(desk.risk_state, equity, 0.0, cfg.risk, "BTC-USD", clock())
    assert blocked and blocked.startswith("daily loss")


def test_a_flat_symbol_the_risk_state_would_veto_does_not_wake_the_committee(tmp_path):
    desk, cfg, clock = make_desk(tmp_path, committee=FixedCommittee(rating="Hold"))
    desk.tick()                                # spends the cold-start trigger
    equity = desk.broker.equity(desk._prices)
    desk.risk_state.day_start_equity = equity / 0.91      # day is -9%

    paid = FixedCommittee(rating="Buy", cost=0.35, name="fake-llm")
    desk.committee = paid
    run(desk, clock, 100)                      # 8h: the 60m schedule would fire ~8 times
    assert paid.calls == 0
    assert desk.ledger.spend_total() == 0.0
    assert desk.ledger.recent_decisions(100)[0]["rating"] == "Hold"

    desk.risk_state.day_start_equity = desk.broker.equity(desk._prices)
    run(desk, clock, 20)
    assert paid.calls > 0, "with the gate open the committee must run again"


def test_a_halt_keeps_retrying_to_flatten_a_symbol_whose_quote_comes_back(tmp_path):
    clock = Clock()
    feed = ToggleFeed(seed=9, clock=clock)
    desk, cfg, clock = make_desk(tmp_path, committee=FixedCommittee(rating="Hold"), clock=clock)
    desk.feed = feed
    desk.tick()
    desk.broker.market_order("ETH-USD", "buy", 0.5, desk._prices["ETH-USD"], ts=int(clock()))
    desk.tick()

    feed.dark.add("ETH-USD")
    run(desk, clock, 3)
    assert desk.state()["stale_symbols"] == ["ETH-USD"]

    desk.halt("operator")
    assert desk.risk_state.halted
    assert list(desk.broker.positions()) == ["ETH-USD"], "a frozen quote is no price to close at"
    assert desk.state()["risk"]["unflattened"] == ["ETH-USD"]
    assert any(e["kind"] == "halt" and "Could not flatten ETH-USD" in e["message"]
               for e in desk.ledger.recent_events(5))

    result = desk.tick()
    assert result["halted"] and result["unflattened"] == ["ETH-USD"]
    assert result["actions"] == []

    feed.dark.clear()
    result = desk.tick()
    assert result["halted"] and result["unflattened"] == []
    assert desk.broker.positions() == {}
    assert [a["action"] for a in result["actions"]] == ["exited"]
    assert desk.ledger.recent_fills(1)[0]["reason"] == "kill-switch retry"
    assert any("Book flat after the halt" in e["message"] for e in desk.ledger.recent_events(5))
    # Halted, so still no new business — and the flag survived it all.
    assert desk.ledger.get_state("risk_state")["halted"] is True


def test_the_book_is_trimmed_when_gross_exposure_drifts_past_its_cap(tmp_path):
    symbols = ["BTC-USD", "ETH-USD", "SOL-USD", "LINK-USD"]
    desk, cfg, clock = make_desk(tmp_path, committee=FixedCommittee(rating="Hold"),
                                 symbols=symbols)
    desk.tick()
    equity = desk.broker.equity(desk._prices)
    for symbol in symbols:
        price = desk._prices[symbol]
        # 19% each: under the 30% per-symbol drift ceiling, yet 76% in
        # aggregate — past the 72% gross band that per-symbol trims cannot see.
        desk.broker.market_order(symbol, "buy", 0.19 * equity / price, price, ts=int(clock()))
        desk.broker.position(symbol).stop_price = price * 0.5
    band = cfg.risk.max_gross_exposure * (1 + cfg.risk.max_weight_drift)
    assert desk.broker.gross_exposure(desk._prices) > band

    result = desk.tick()

    trims = [a for a in result["actions"] if a.get("reason") == "gross drift"]
    assert len(trims) == 4
    assert desk.broker.gross_exposure(desk._prices) == pytest.approx(
        cfg.risk.max_gross_exposure, abs=0.005)
    equity = desk.broker.equity(desk._prices)
    weights = [abs(pos.market_value(desk._prices[s])) / equity
               for s, pos in desk.broker.positions().items()]
    assert max(weights) - min(weights) < 0.005, "the excess is spread pro-rata"
    assert all(w < cfg.risk.max_symbol_weight for w in weights)
    events = [e for e in desk.ledger.recent_events(20) if e["kind"] == "trim"]
    assert len(events) == 4 and all("gross drift" in e["message"] for e in events)


# ---------------------------------------------------------------- freshness
def test_a_blind_desk_marks_but_neither_trades_nor_thinks(tmp_path):
    clock = Clock()
    feed = ToggleFeed(seed=9, clock=clock)
    desk, cfg, clock = make_desk(tmp_path, committee=FixedCommittee(rating="Hold"), clock=clock)
    desk.feed = feed
    run(desk, clock, 60)
    desk.broker.market_order("BTC-USD", "buy", 0.02, desk._prices["BTC-USD"], ts=int(clock()))
    desk.tick()

    feed.dark.update(cfg.symbols)
    run(desk, clock, 3)
    assert set(desk.state()["stale_symbols"]) == set(cfg.symbols)
    last_prices = dict(desk._prices)
    calls = desk.committee.calls
    fills = desk.broker.snapshot({})["fill_count"]
    decisions = len(desk.ledger.recent_decisions(10_000))
    points = len(desk.ledger.equity_curve(10_000))
    # A stop the frozen quote would trip.
    desk.broker.position("BTC-USD").stop_price = last_prices["BTC-USD"] * 10

    run(desk, clock, 200)                      # ~16 hours dark
    result = desk.tick()

    assert desk.committee.calls == calls
    assert desk.broker.snapshot({})["fill_count"] == fills
    assert len(desk.ledger.recent_decisions(10_000)) == decisions
    assert len(desk.ledger.equity_curve(10_000)) == points, "no fabricated flat-line marks"
    assert desk.broker.position("BTC-USD") is not None, "no stop-out on a frozen quote"
    assert "stop_deferred" in [a["action"] for a in result["actions"]]
    assert result["equity"] == pytest.approx(desk.broker.equity(last_prices))
    assert desk.state()["account"]["equity"] == pytest.approx(desk.broker.equity(last_prices))
    assert desk.state()["positions"][0]["stale"] is True
    stale_events = [e for e in desk.ledger.recent_events(1000) if e["kind"] == "stale"]
    assert len(stale_events) == len(cfg.symbols) + 1, "one per transition, not per tick"

    feed.dark.clear()
    clock.advance(300)
    result = desk.tick()
    assert desk.state()["stale_symbols"] == []
    assert desk.broker.position("BTC-USD") is None, "the armed stop fires on the first fresh quote"
    assert desk.ledger.recent_fills(1)[0]["reason"].startswith("stop hit")
    assert len(desk.ledger.equity_curve(10_000)) == points + 1
    stale_events = [e for e in desk.ledger.recent_events(1000) if e["kind"] == "stale"]
    assert len(stale_events) == 2 * (len(cfg.symbols) + 1)


def test_a_venue_change_is_recorded_and_feed_health_is_exposed(tmp_path):
    clock = Clock()
    primary, backup = ToggleFeed(seed=9, clock=clock), ToggleFeed(seed=9, clock=clock)
    primary.name, backup.name = "primary", "backup"
    desk, cfg, clock = make_desk(tmp_path, committee=FixedCommittee(rating="Hold"),
                                 clock=clock, symbols=["BTC-USD"])
    desk.feed = ChainFeed([primary, backup], clock=clock)
    desk.tick()
    assert desk.state()["feed_venue"] == {"BTC-USD": "primary"}

    primary.dark.add("BTC-USD")
    clock.advance(300)
    desk.tick()

    state = desk.state()
    assert state["feed_venue"] == {"BTC-USD": "backup"}
    assert state["feed_health"]["primary"]["failures"] == 1
    assert state["feed_health"]["backup"]["last_ok_ts"] == int(clock())
    assert state["stale_symbols"] == []
    assert any(e["kind"] == "feed" and "venue changed primary -> backup" in e["message"]
               for e in desk.ledger.recent_events(5))


# ---------------------------------------------------------------- restart
def test_a_restored_position_outside_the_config_is_managed_out_but_never_re_entered(tmp_path):
    desk, cfg, clock = make_desk(tmp_path, committee=FixedCommittee(rating="Hold"))
    desk.tick()
    desk.broker.market_order("ETH-USD", "buy", 0.4, desk._prices["ETH-USD"], ts=int(clock()))
    desk.tick()
    eth_decisions = desk.ledger._read("SELECT COUNT(*) AS n FROM decisions WHERE symbol='ETH-USD'")[0]["n"]

    narrowed = DeskConfig.from_dict({**cfg.to_dict(), "symbols": ["BTC-USD"]})
    restarted = Desk(narrowed, feed=SyntheticFeed(seed=9, clock=clock),
                     committee=FixedCommittee(rating="Buy"), ledger=Ledger(cfg.db_path),
                     clock=clock)
    assert restarted.held_not_configured() == ["ETH-USD"]
    assert any(e["kind"] == "restore" and "ETH-USD" in e["message"] and "exit-only" in e["message"]
               for e in restarted.ledger.recent_events(5))

    restarted.tick()
    state = restarted.state()
    assert state["held_not_configured"] == ["ETH-USD"]
    eth = next(p for p in state["positions"] if p["symbol"] == "ETH-USD")
    assert eth["price"] == pytest.approx(restarted._prices["ETH-USD"]) and eth["stale"] is False
    assert restarted.broker.position("ETH-USD").stop_price is not None

    # Force the exit path: a stop above the market is hit on the next tick.
    restarted.broker.position("ETH-USD").stop_price = restarted._prices["ETH-USD"] * 2
    run(restarted, clock, 300)

    assert restarted.broker.position("ETH-USD") is None
    assert restarted.state()["held_not_configured"] == []
    assert "ETH-USD" not in restarted._prices, "flat and unconfigured: no longer priced"
    eth_fills = [f for f in restarted.ledger.recent_fills(500) if f["symbol"] == "ETH-USD"]
    assert [f["side"] for f in eth_fills] == ["sell"]
    assert eth_fills[0]["reason"].startswith("stop hit")
    after = restarted.ledger._read("SELECT COUNT(*) AS n FROM decisions WHERE symbol='ETH-USD'")[0]["n"]
    assert after == eth_decisions, "the committee was never asked about it again"


def test_trigger_memory_survives_a_restart(tmp_path):
    desk, cfg, clock = make_desk(tmp_path)
    desk.tick()
    assert desk._last_decision_price["BTC-USD"] and desk._last_trend["BTC-USD"]
    # An older row in the LLM committee's shape (trend under 'indicators').
    desk.ledger.record_decision(ts=int(clock()) + 1, symbol="ETH-USD", source="llm",
                                rating="Hold", price=999.0,
                                detail={"indicators": {"trend": "flat"}})

    restarted = Desk(cfg, feed=SyntheticFeed(seed=9, clock=clock), committee=FixedCommittee(),
                     ledger=Ledger(cfg.db_path), clock=clock)
    assert restarted._last_decision_price["BTC-USD"] == desk._last_decision_price["BTC-USD"]
    assert restarted._last_trend["BTC-USD"] == desk._last_trend["BTC-USD"]
    assert restarted._last_decision_price["ETH-USD"] == 999.0
    assert restarted._last_trend["ETH-USD"] == "flat"


def test_the_benchmark_is_marked_even_when_it_is_not_traded(tmp_path):
    clock = Clock()
    feed = ToggleFeed(seed=9, clock=clock)
    desk, cfg, clock = make_desk(tmp_path, clock=clock, symbols=["ETH-USD"],
                                 benchmark_symbol="BTC-USD")
    desk.feed = feed
    run(desk, clock, 5)

    rows = desk.ledger.equity_curve()
    assert len(rows) == 5 and all(r["benchmark_price"] for r in rows)
    assert rows[0]["benchmark_price"] != rows[-1]["benchmark_price"]
    assert "BTC-USD" not in desk._prices
    assert desk.risk_state.first_benchmark_price == pytest.approx(rows[0]["benchmark_price"])
    assert Ledger(cfg.db_path).get_state("risk_state")["first_benchmark_price"] == \
        pytest.approx(rows[0]["benchmark_price"])
    assert desk.state()["benchmark_price"] == pytest.approx(rows[-1]["benchmark_price"])

    # A benchmark outage is tolerated: NULL for that mark, the desk carries on.
    feed.dark.add("BTC-USD")
    result = desk.tick()
    assert "error" not in result
    assert desk.ledger.equity_curve(1)[0]["benchmark_price"] is None
    assert "BTC-USD" in desk.feed_errors and desk.state()["benchmark_price"] is None


# ---------------------------------------------------------------- liveness and shutdown
def test_heartbeat_and_busy_track_the_committee_phase(tmp_path):
    seen = []

    class ObservingCommittee(FixedCommittee):
        def decide(self, symbol, snapshot, trigger_reason="", now=None):
            seen.append((desk.busy, desk.heartbeat_ts, now))
            clock.advance(120)
            return super().decide(symbol, snapshot, trigger_reason, now)

    desk, _, clock = make_desk(tmp_path, committee=ObservingCommittee(rating="Hold"))
    start = int(clock())
    assert desk.heartbeat_ts == start and desk.busy is None

    desk.tick()

    assert seen == [("committee BTC-USD", start, start),
                    ("committee ETH-USD", start + 120, start + 120)]
    assert desk.busy is None
    assert desk.heartbeat_ts == start + 240 == desk.last_tick_ts
    state = desk.state()
    assert state["busy"] is None and state["heartbeat_ts"] == start + 240


def test_stop_joins_the_engine_thread_and_persists_the_book(tmp_path):
    desk, _, clock = make_desk(tmp_path, committee=FixedCommittee(rating="Hold"))
    thread = desk.start_background()
    deadline = time.time() + 10
    while desk.tick_count < 1 and time.time() < deadline:
        time.sleep(0.01)
    assert desk.tick_count >= 1

    # A change the loop has not persisted yet: only the shutdown path can.
    with desk._lock:
        desk.broker.market_order("BTC-USD", "buy", 0.01, desk._prices["BTC-USD"], ts=int(clock()))
        cash = desk.broker.cash()

    desk.stop(timeout=10)
    assert not thread.is_alive()
    persisted = desk.ledger.get_state("broker_state")
    assert persisted["cash"] == pytest.approx(cash)
    assert [p["symbol"] for p in persisted["positions"]] == ["BTC-USD"]


def test_buy_and_hold_anchors_at_the_first_recorded_mark_after_an_upgrade(tmp_path):
    """A ledger written before ``first_benchmark_price`` existed must not
    re-anchor buy-and-hold at the first post-upgrade tick."""
    desk, cfg, clock = make_desk(tmp_path)
    run(desk, clock, 30)
    first_mark = desk.ledger.equity_curve(limit=100000)[0]["benchmark_price"]
    assert first_mark, "the fixture must record a benchmark mark"

    # Simulate the pre-upgrade state: the field is absent from the stored risk state.
    stored = desk.ledger.get_state("risk_state")
    stored.pop("first_benchmark_price", None)
    desk.ledger.set_state("risk_state", stored)

    restarted = Desk(cfg, feed=SyntheticFeed(seed=9, clock=clock),
                     committee=FixedCommittee(), ledger=Ledger(cfg.db_path), clock=clock)
    assert restarted.risk_state.first_benchmark_price == pytest.approx(first_mark)
