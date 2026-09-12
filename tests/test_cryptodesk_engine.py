"""End-to-end desk behaviour over compressed time.

Drives the real engine — real broker, real ledger, real risk logic — against a
deterministic synthetic market, so the tests exercise the code that actually
runs unattended rather than a mock of it.
"""

import pytest

from cryptodesk.config import DeskConfig
from cryptodesk.engine.committee import Proposal
from cryptodesk.engine.ledger import Ledger
from cryptodesk.engine.loop import Desk
from cryptodesk.feeds import SyntheticFeed


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

    def decide(self, symbol, snapshot, trigger_reason=""):
        self.calls += 1
        return Proposal(symbol=symbol, rating=self.rating, conviction=self.conviction,
                        summary=f"always {self.rating}", source=self.source,
                        cost_usd=self.cost, cost_is_estimate=bool(self.cost))


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
        risk={"max_drawdown_halt": 0.02, "stop_atr_mult": 50, "trail_atr_mult": 50,
              "max_symbol_weight": 0.9, "max_gross_exposure": 0.9, "risk_per_trade": 0.9},
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

        def candles(self, symbol, interval="5m", limit=200):
            if symbol == "ETH-USD":
                raise ValueError("simulated venue outage")
            return super().candles(symbol, interval, limit)

    desk, _, clock = make_desk(tmp_path)
    desk.feed = HalfBrokenFeed(seed=9, clock=clock)
    run(desk, clock, 200)

    assert "ETH-USD" in desk.feed_errors
    assert "BTC-USD" in desk._prices, "a broken symbol must not blind the others"
    assert desk.state()["account"]["fill_count"] > 0


def test_total_data_loss_is_reported_rather_than_crashing(tmp_path):
    class DeadFeed(SyntheticFeed):
        name = "dead"

        def candles(self, symbol, interval="5m", limit=200):
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
