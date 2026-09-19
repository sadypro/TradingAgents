"""The committee layer against a stubbed TradingAgents graph.

The graph contract (``propagate(symbol, trade_date, asset_type)`` returning
``(final_state, rating_word)``) and the state keys the desk reads are the most
likely things to drift in future core changes; these tests pin them with a
fake graph so the desk fails CI rather than silently degrading to Hold.
"""

from types import SimpleNamespace

import pytest

from cryptodesk.config import DeskConfig
from cryptodesk.engine.committee import (
    RATINGS,
    FallbackCommittee,
    HeuristicCommittee,
    LLMCommittee,
    Proposal,
    _first_meaningful_line,
    _llm_unavailable_reason,
    _usage_callback_handler,
    build_committee,
    desk_ta_config,
)
from cryptodesk.indicators import Snapshot

NOW = 1_760_000_000            # 2025-10-09 UTC
BAR_TS = 1_700_000_000         # 2023-11-14 UTC

PM_DECISION = (
    "**Rating**: Overweight\n"
    "\n"
    "**Executive Summary**: Scale into BTC on pullbacks toward the 20-day EMA; "
    "keep the stop under the prior swing low.\n"
    "\n"
    "**Investment Thesis**: Funding is neutral and spot demand is absorbing "
    "supply, so the trend has room to continue.\n"
    "\n"
    "**Time Horizon**: 2-4 weeks"
)


def make_snapshot(symbol="BTC-USD", ts=BAR_TS):
    return Snapshot(symbol=symbol, ts=ts, price=100.0, ema_fast=101.0, ema_slow=99.0,
                    rsi14=55.0, atr14=2.0, vol_short=0.02, vol_long=0.02, bars=200)


def make_final_state(pm_text=PM_DECISION):
    """The shape ``TradingAgentsGraph.propagate`` returns for a crypto run."""
    return {
        "company_of_interest": "BTC-USD",
        "trade_date": "2025-10-09",
        "market_report": "Price above both EMAs; RSI 55 with room to run.",
        "sentiment_report": "Overall sentiment: Mildly Bullish (0.3).",
        "news_report": "ETF inflows resumed this week.",
        "fundamentals_report": "",
        "investment_debate_state": {
            "bull_history": "Bull: inflows and trend argue for exposure.",
            "bear_history": "Bear: funding could flip and squeeze longs.",
            "history": "...", "current_response": "...",
            "judge_decision": "Research manager: Overweight on trend with tight risk.",
        },
        "trader_investment_plan": "Trader: buy 60% of the allowed size now, rest on a dip.",
        "risk_debate_state": {
            "judge_decision": "Risk manager: approve at reduced size given event risk.",
        },
        "final_trade_decision": pm_text,
    }


class FakeGraph:
    """Stands in for TradingAgentsGraph: records the call, returns a canned run."""

    def __init__(self, final_state=None, rating="Overweight", on_run=None, raises=None):
        self.final_state = final_state if final_state is not None else make_final_state()
        self.rating = rating
        self.on_run = on_run
        self.raises = raises
        self.calls = []

    def propagate(self, symbol, trade_date, asset_type="stock"):
        self.calls.append({"symbol": symbol, "trade_date": trade_date,
                           "asset_type": asset_type})
        if self.on_run:
            self.on_run()
        if self.raises:
            raise self.raises
        return self.final_state, self.rating


def make_llm_committee(tmp_path, graph=None, **llm_overrides):
    cfg = DeskConfig.from_dict({"home": str(tmp_path), "committee": "llm",
                                "llm": {"estimated_cost_per_run_usd": 0.35}})
    for key, value in llm_overrides.items():
        setattr(cfg.llm, key, value)
    committee = LLMCommittee(cfg)
    committee._graph = graph or FakeGraph()
    return committee


# ------------------------------------------------------------- graph contract
def test_llm_committee_runs_the_graph_in_crypto_mode_on_the_desk_clock(tmp_path):
    graph = FakeGraph()
    committee = make_llm_committee(tmp_path, graph)

    proposal = committee.decide("BTC-USD", make_snapshot(), "scheduled", now=NOW)

    assert graph.calls == [{"symbol": "BTC-USD", "trade_date": "2025-10-09",
                            "asset_type": "crypto"}]
    assert proposal.detail["trade_date"] == "2025-10-09"
    assert proposal.error is None


def test_trade_date_falls_back_to_the_bar_then_the_wall_clock(tmp_path):
    graph = FakeGraph()
    committee = make_llm_committee(tmp_path, graph)

    committee.decide("BTC-USD", make_snapshot(ts=BAR_TS), "scheduled")
    committee.decide("BTC-USD", make_snapshot(ts=0), "scheduled")

    from datetime import datetime, timezone
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    assert [c["trade_date"] for c in graph.calls] == ["2023-11-14", today]


def test_rating_and_summary_come_from_the_pm_decision(tmp_path):
    proposal = make_llm_committee(tmp_path).decide("BTC-USD", make_snapshot(), "scheduled", now=NOW)

    assert proposal.rating == "Overweight"
    assert proposal.conviction == pytest.approx(0.6)
    assert proposal.source == "llm"
    # The one-line 'why' is the executive summary, not the tautology "Rating: X".
    assert proposal.summary.startswith("Executive Summary: Scale into BTC on pullbacks")
    assert "**" not in proposal.summary


def test_rating_word_from_propagate_is_trusted_and_reparsed_otherwise(tmp_path):
    # propagate() already hands back the parsed word; anything else is re-parsed
    # from the PM text rather than passed through.
    trusted = make_llm_committee(tmp_path, FakeGraph(rating="Sell"))
    assert trusted.decide("BTC-USD", make_snapshot(), now=NOW).rating == "Sell"

    reparsed = make_llm_committee(tmp_path, FakeGraph(rating="garbage"))
    assert reparsed.decide("BTC-USD", make_snapshot(), now=NOW).rating == "Overweight"
    assert all(r in RATINGS for r in ("Sell", "Overweight"))


def test_detail_carries_every_agent_section(tmp_path):
    proposal = make_llm_committee(tmp_path).decide("BTC-USD", make_snapshot(), "breakout", now=NOW)
    detail = proposal.detail

    for label in ("Technical analyst", "Sentiment analyst", "News analyst", "Trader",
                  "Portfolio manager", "Bull case", "Bear case", "Research manager",
                  "Risk manager"):
        assert detail[label], label
    assert detail["Portfolio manager"] == PM_DECISION
    assert detail["trigger"] == "breakout"
    assert detail["indicators"]["symbol"] == "BTC-USD"


def test_first_meaningful_line_strips_emphasis_and_skips_the_rating():
    assert _first_meaningful_line(PM_DECISION).startswith("Executive Summary: Scale into")
    assert _first_meaningful_line("Buy") == ""
    assert _first_meaningful_line("**Rating**: Buy") == ""


# ----------------------------------------------------------------- cost
def _llm_end_payload(tokens_in, tokens_out):
    """What langchain hands ``on_llm_end``: generations[0][0].message.usage_metadata."""
    message = SimpleNamespace(usage_metadata={"input_tokens": tokens_in,
                                              "output_tokens": tokens_out})
    return SimpleNamespace(generations=[[SimpleNamespace(message=message)]])


def _simulate_llm_traffic(committee, calls):
    """Return an ``on_run`` hook that drives the real callback handler."""
    handler = _usage_callback_handler(committee._usage)

    def run():
        for tokens_in, tokens_out in calls:
            handler.on_chat_model_start({}, [[]])
            handler.on_llm_end(_llm_end_payload(tokens_in, tokens_out))
    return run


def test_token_usage_is_measured_per_run_and_priced_when_prices_are_configured(tmp_path):
    committee = make_llm_committee(tmp_path, price_per_million_input_usd=2.0,
                                   price_per_million_output_usd=10.0)
    committee._graph.on_run = _simulate_llm_traffic(committee, [(1000, 100), (2000, 300)])

    proposal = committee.decide("BTC-USD", make_snapshot(), now=NOW)

    assert proposal.detail["llm_calls"] == 2
    assert proposal.detail["tokens_in"] == 3000
    assert proposal.detail["tokens_out"] == 400
    assert proposal.cost_usd == pytest.approx(3000 / 1e6 * 2.0 + 400 / 1e6 * 10.0)
    assert proposal.cost_is_estimate is False

    # The next run must not inherit the previous run's counters.
    committee._graph.on_run = _simulate_llm_traffic(committee, [(500, 50)])
    again = committee.decide("BTC-USD", make_snapshot(), now=NOW)
    assert again.detail["tokens_in"] == 500
    assert again.detail["llm_calls"] == 1


def test_without_prices_the_configured_estimate_is_charged_and_labelled(tmp_path):
    committee = make_llm_committee(tmp_path)
    committee._graph.on_run = _simulate_llm_traffic(committee, [(1000, 100)])

    proposal = committee.decide("BTC-USD", make_snapshot(), now=NOW)

    assert proposal.cost_usd == pytest.approx(0.35)
    assert proposal.cost_is_estimate is True
    assert proposal.detail["tokens_in"] == 1000


def test_a_run_that_dies_before_any_llm_call_is_charged_nothing(tmp_path):
    graph = FakeGraph(raises=RuntimeError("API key for provider 'openai' is not set"))
    proposal = make_llm_committee(tmp_path, graph).decide("BTC-USD", make_snapshot(), now=NOW)

    assert proposal.rating == "Hold"
    assert proposal.conviction == 0.0
    assert proposal.error == "API key for provider 'openai' is not set"
    assert proposal.cost_usd == 0.0
    assert proposal.detail["llm_calls"] == 0


def test_a_run_that_dies_mid_way_still_pays_for_the_tokens_it_burned(tmp_path):
    committee = make_llm_committee(tmp_path, price_per_million_input_usd=1.0,
                                   price_per_million_output_usd=1.0)
    committee._graph.raises = TimeoutError("provider timed out")
    committee._graph.on_run = _simulate_llm_traffic(committee, [(4000, 1000)])

    proposal = committee.decide("BTC-USD", make_snapshot(), now=NOW)

    assert proposal.error == "provider timed out"
    assert proposal.cost_usd == pytest.approx(0.005)
    assert proposal.cost_is_estimate is False

    # Same failure without prices falls back to the estimate, honestly labelled.
    estimated = make_llm_committee(tmp_path)
    estimated._graph.raises = TimeoutError("provider timed out")
    estimated._graph.on_run = _simulate_llm_traffic(estimated, [(4000, 1000)])
    failed = estimated.decide("BTC-USD", make_snapshot(), now=NOW)
    assert failed.cost_usd == pytest.approx(0.35)
    assert failed.cost_is_estimate is True


# ----------------------------------------------------------------- fallback
class ErrorCommittee:
    """A primary that fails until told otherwise."""

    source = "llm"
    name = "llm"

    def __init__(self):
        self.failing = True
        self.calls = 0

    def decide(self, symbol, snapshot, trigger_reason="", now=None):
        self.calls += 1
        if self.failing:
            return Proposal(symbol=symbol, rating="Hold", conviction=0.0,
                            summary="boom", source=self.source, error="boom")
        return Proposal(symbol=symbol, rating="Buy", conviction=1.0,
                        summary="fine", source=self.source)


def test_fallback_switches_only_after_the_failure_streak():
    primary = ErrorCommittee()
    committee = FallbackCommittee(primary, HeuristicCommittee(), max_consecutive_failures=3)
    snap = make_snapshot()

    for _ in range(2):
        assert committee.decide("BTC-USD", snap, "t", now=NOW).error == "boom"
    assert committee.active is primary
    assert committee.switched_reason is None
    assert committee.name == "llm"

    third = committee.decide("BTC-USD", snap, "t", now=NOW)
    assert third.error == "boom"                     # the switching call still reports its failure
    assert committee.active is committee.fallback
    assert "3 consecutive llm failures" in committee.switched_reason
    assert committee.name == "heuristic (fallback after 3 llm failures)"
    assert committee.source == "heuristic"

    fourth = committee.decide("BTC-USD", snap, "t", now=NOW)
    assert fourth.source == "heuristic" and fourth.error is None
    assert primary.calls == 3, "the primary must not be retried after switching"


def test_a_success_resets_the_failure_streak():
    primary = ErrorCommittee()
    committee = FallbackCommittee(primary, HeuristicCommittee(), max_consecutive_failures=3)
    snap = make_snapshot()

    for _ in range(2):
        committee.decide("BTC-USD", snap)
    primary.failing = False
    assert committee.decide("BTC-USD", snap).rating == "Buy"
    primary.failing = True
    for _ in range(2):
        committee.decide("BTC-USD", snap)
    assert committee.active is primary


def test_heuristic_accepts_and_ignores_the_clock():
    snap = make_snapshot()
    with_clock = HeuristicCommittee().decide("BTC-USD", snap, "t", now=NOW)
    without = HeuristicCommittee().decide("BTC-USD", snap, "t")
    assert (with_clock.rating, with_clock.summary) == (without.rating, without.summary)


# ------------------------------------------------------- availability & build
def test_availability_checks_the_key_the_configured_provider_needs(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

    reason = _llm_unavailable_reason({"llm_provider": "openai"})
    assert reason and "OPENAI_API_KEY" in reason
    assert _llm_unavailable_reason({"llm_provider": "anthropic"}) is None


def test_keyless_providers_are_available_without_any_key(monkeypatch):
    for var in ("OPENAI_API_KEY", "OPENAI_COMPATIBLE_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    assert _llm_unavailable_reason({"llm_provider": "ollama"}) is None
    assert _llm_unavailable_reason({"llm_provider": "openai_compatible"}) is None
    assert _llm_unavailable_reason({"llm_provider": "bedrock"}) is None


def test_build_committee_auto_wraps_the_llm_in_a_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(LLMCommittee, "_build_graph", lambda self: FakeGraph())
    cfg = DeskConfig.from_dict({"home": str(tmp_path), "committee": "auto"})

    committee = build_committee(cfg, {"llm_provider": "openai"})

    assert isinstance(committee, FallbackCommittee)
    assert isinstance(committee.primary, LLMCommittee)
    assert isinstance(committee.fallback, HeuristicCommittee)
    assert committee.name == "llm"


def test_build_committee_auto_falls_back_when_the_graph_will_not_construct(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    def broken(self):
        raise ValueError("API key for provider 'openai' is not set")
    monkeypatch.setattr(LLMCommittee, "_build_graph", broken)
    cfg = DeskConfig.from_dict({"home": str(tmp_path), "committee": "auto"})

    assert isinstance(build_committee(cfg, {"llm_provider": "openai"}), HeuristicCommittee)


def test_build_committee_auto_falls_back_when_the_provider_key_is_missing(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    cfg = DeskConfig.from_dict({"home": str(tmp_path), "committee": "auto"})

    assert isinstance(build_committee(cfg, {"llm_provider": "openai"}), HeuristicCommittee)


def test_explicit_modes_return_bare_committees(tmp_path):
    llm = build_committee(DeskConfig.from_dict({"home": str(tmp_path), "committee": "llm"}))
    assert type(llm) is LLMCommittee
    heuristic = build_committee(DeskConfig.from_dict({"home": str(tmp_path), "committee": "heuristic"}))
    assert type(heuristic) is HeuristicCommittee


def test_tradingagents_paths_land_under_the_desk_home(tmp_path):
    cfg = DeskConfig.from_dict({"home": str(tmp_path)})
    root = tmp_path / "tradingagents"

    ta = LLMCommittee(cfg).ta_config
    assert ta["results_dir"] == str(root / "logs")
    assert ta["data_cache_dir"] == str(root / "cache")
    assert ta["memory_log_path"] == str(root / "memory" / "trading_memory.md")
    assert "llm_provider" in ta, "the rest of DEFAULT_CONFIG must be carried through"

    assert desk_ta_config(cfg, {"llm_provider": "anthropic"})["llm_provider"] == "anthropic"
    explicit = {"llm_provider": "anthropic", "results_dir": "/elsewhere"}
    assert LLMCommittee(cfg, explicit).ta_config is explicit
