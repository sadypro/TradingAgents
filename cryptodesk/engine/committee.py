"""The decision layer: what the desk thinks it should hold, and why.

Two implementations behind one interface:

* ``HeuristicCommittee`` — a free, deterministic trend/RSI/volatility rule. It
  exists to be the **control group**. If the LLM committee cannot beat this, the
  LLM is costing money for nothing, and that is the single most useful thing the
  desk can tell you. It is also what runs when no API key is configured, so the
  system is fully operable at zero marginal cost.
* ``LLMCommittee`` — the TradingAgents multi-agent graph (analysts, bull/bear
  debate, trader, risk team, portfolio manager) run in crypto mode.

Both return a ``Proposal``: a five-tier rating and a conviction in [0, 1]. A
committee never sees the account's size, cash, or limits, and never returns a
quantity. Sizing belongs to ``engine.risk`` alone.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from ..config import DeskConfig
from ..indicators import Snapshot

logger = logging.getLogger(__name__)

RATINGS = ("Buy", "Overweight", "Hold", "Underweight", "Sell")

# Rating -> fraction of the symbol's maximum allowed weight to target.
# "Hold" means "change nothing" and is handled by the engine, not scaled here.
RATING_CONVICTION = {
    "Buy": 1.0,
    "Overweight": 0.6,
    "Hold": 0.0,
    "Underweight": 0.2,
    "Sell": 0.0,
}


@dataclass
class Proposal:
    """A committee's view on one symbol. Direction and conviction only."""

    symbol: str
    rating: str
    conviction: float
    summary: str
    source: str
    cost_usd: float = 0.0
    latency_ms: int = 0
    detail: dict = field(default_factory=dict)
    # True when the cost figure is an estimate rather than provider-reported
    # usage, so the dashboard can label it honestly.
    cost_is_estimate: bool = False
    error: str | None = None

    @property
    def wants_exit(self) -> bool:
        return self.rating == "Sell"

    @property
    def is_hold(self) -> bool:
        return self.rating == "Hold"

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol, "rating": self.rating, "conviction": self.conviction,
            "summary": self.summary, "source": self.source, "cost_usd": self.cost_usd,
            "latency_ms": self.latency_ms, "detail": self.detail,
            "cost_is_estimate": self.cost_is_estimate, "error": self.error,
        }


class HeuristicCommittee:
    """Free, deterministic baseline: trade with the trend, stand aside in chaos.

    The rules are intentionally boring, because a boring rule is the honest
    benchmark for an expensive one:

    * Trend up (fast EMA over slow) and not yet overbought -> Buy / Overweight.
    * Trend down -> Sell (flat; shorts are off by default).
    * Volatility spiking well past its own baseline -> Hold, regardless of
      trend. Entering a vol explosion is how stops get gapped through.
    """

    source = "heuristic"
    name = "heuristic"

    # Vol ratio past which no new risk is taken.
    vol_panic_ratio = 2.5

    def decide(self, symbol: str, snapshot: Snapshot, trigger_reason: str = "") -> Proposal:
        started = time.perf_counter()
        trend = snapshot.trend
        rsi = snapshot.rsi14
        vol_ratio = snapshot.vol_ratio

        if trend == "unknown" or rsi is None:
            rating, why = "Hold", f"insufficient history ({snapshot.bars} bars)"
        elif vol_ratio is not None and vol_ratio >= self.vol_panic_ratio:
            rating, why = "Hold", f"volatility {vol_ratio:.1f}x baseline; standing aside"
        elif trend == "down":
            rating, why = "Sell", f"fast EMA below slow (RSI {rsi:.0f}); no long exposure"
        elif trend == "up":
            if rsi >= 72:
                rating, why = "Hold", f"uptrend but overbought (RSI {rsi:.0f}); not adding"
            elif rsi < 60:
                rating, why = "Buy", f"uptrend with room to run (RSI {rsi:.0f})"
            else:
                rating, why = "Overweight", f"uptrend, RSI {rsi:.0f} getting extended"
        else:
            rating, why = "Hold", f"no clear trend (RSI {rsi:.0f})"

        return Proposal(
            symbol=symbol, rating=rating, conviction=RATING_CONVICTION[rating],
            summary=why, source=self.source,
            latency_ms=int((time.perf_counter() - started) * 1000),
            detail={
                "rules": "trend + RSI + volatility filter",
                "trend": trend, "rsi14": rsi, "vol_ratio": vol_ratio,
                "atr_pct": snapshot.atr_pct, "trigger": trigger_reason,
            },
        )


class LLMCommittee:
    """Runs the TradingAgents graph for one symbol and extracts a rating.

    The graph is imported lazily so the desk starts (and the heuristic runs)
    even when TradingAgents' heavier dependencies are absent.

    On cost: the graph does not report provider token usage back to callers, so
    ``cost_usd`` here is the configured per-run estimate and is flagged as such
    via ``cost_is_estimate``. Measure one run against your provider's dashboard
    and set ``llm.estimated_cost_per_run_usd`` to the real number — the daily
    cap is only as honest as that figure.
    """

    source = "llm"
    name = "llm"

    def __init__(self, cfg: DeskConfig, ta_config: dict | None = None):
        self.cfg = cfg
        self._ta_config = ta_config
        self._graph = None

    def _build_graph(self):
        """Construct the TradingAgentsGraph on first use."""
        if self._graph is not None:
            return self._graph
        from tradingagents.default_config import DEFAULT_CONFIG
        from tradingagents.graph.trading_graph import TradingAgentsGraph

        ta_config = dict(self._ta_config or DEFAULT_CONFIG)
        # Crypto has no fundamentals to analyse; the analyst would burn tokens
        # producing a report about a company that does not exist.
        analysts = ["market", "social", "news"]
        self._graph = TradingAgentsGraph(
            selected_analysts=analysts, debug=False, config=ta_config
        )
        return self._graph

    def decide(self, symbol: str, snapshot: Snapshot, trigger_reason: str = "") -> Proposal:
        from tradingagents.agents.utils.rating import parse_rating

        started = time.perf_counter()
        trade_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        try:
            graph = self._build_graph()
            final_state, decision = graph.propagate(symbol, trade_date, asset_type="crypto")
        except Exception as exc:  # noqa: BLE001 - any failure must not stop the desk
            logger.exception("LLM committee failed for %s", symbol)
            return Proposal(
                symbol=symbol, rating="Hold", conviction=0.0,
                summary=f"committee failed, holding: {exc}", source=self.source,
                latency_ms=int((time.perf_counter() - started) * 1000),
                error=str(exc),
                # A failed run may still have burned tokens before dying.
                cost_usd=self.cfg.llm.estimated_cost_per_run_usd,
                cost_is_estimate=True,
            )

        rating = parse_rating(str(decision))
        detail = _extract_reports(final_state)
        detail["trigger"] = trigger_reason
        detail["indicators"] = snapshot.to_dict()
        return Proposal(
            symbol=symbol, rating=rating, conviction=RATING_CONVICTION.get(rating, 0.0),
            summary=_first_meaningful_line(str(decision)) or f"Rating: {rating}",
            source=self.source,
            cost_usd=self.cfg.llm.estimated_cost_per_run_usd,
            cost_is_estimate=True,
            latency_ms=int((time.perf_counter() - started) * 1000),
            detail=detail,
        )


# Sections pulled out of the graph's final state for the dashboard, so a human
# can see what each agent argued rather than only the verdict.
_REPORT_KEYS = (
    ("market_report", "Technical analyst"),
    ("sentiment_report", "Sentiment analyst"),
    ("news_report", "News analyst"),
    ("trader_investment_plan", "Trader"),
    ("final_trade_decision", "Portfolio manager"),
)

_MAX_SECTION_CHARS = 4000


def _extract_reports(final_state: dict) -> dict:
    """Pull the per-agent reports out of the graph's final state."""
    detail: dict = {}
    if not isinstance(final_state, dict):
        return detail
    for key, label in _REPORT_KEYS:
        text = final_state.get(key)
        if text:
            detail[label] = str(text)[:_MAX_SECTION_CHARS]
    debate = final_state.get("investment_debate_state") or {}
    if isinstance(debate, dict):
        for key, label in (("bull_history", "Bull case"),
                           ("bear_history", "Bear case"),
                           ("judge_decision", "Research manager")):
            text = debate.get(key)
            if text:
                detail[label] = str(text)[:_MAX_SECTION_CHARS]
    risk = final_state.get("risk_debate_state") or {}
    if isinstance(risk, dict) and risk.get("judge_decision"):
        detail["Risk manager"] = str(risk["judge_decision"])[:_MAX_SECTION_CHARS]
    return detail


def _first_meaningful_line(text: str, limit: int = 240) -> str:
    """First non-heading, non-empty line — a usable one-line summary."""
    for line in text.splitlines():
        stripped = line.strip().lstrip("#*->").strip()
        if len(stripped) > 30 and not stripped.lower().startswith("rating"):
            return stripped[:limit]
    return ""


def build_committee(cfg: DeskConfig, ta_config: dict | None = None):
    """Pick a committee per config, falling back to the heuristic when needed.

    ``committee="auto"`` uses the LLM only when TradingAgents imports *and* a
    provider key is present. Falling back rather than crashing is deliberate: a
    desk that stops trading because a key expired is worse than one that keeps
    running a documented baseline and says so on the dashboard.
    """
    mode = (cfg.committee or "auto").lower()
    if mode == "heuristic":
        return HeuristicCommittee()
    if mode == "llm":
        return LLMCommittee(cfg, ta_config)
    if mode != "auto":
        raise ValueError(f"Unknown committee mode {cfg.committee!r}; "
                         "expected 'auto', 'llm' or 'heuristic'")

    reason = _llm_unavailable_reason()
    if reason:
        logger.warning("Committee falling back to heuristic: %s", reason)
        return HeuristicCommittee()
    return LLMCommittee(cfg, ta_config)


# Provider keys recognised by TradingAgents; any one is enough to try the LLM.
_PROVIDER_KEYS = (
    "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "XAI_API_KEY",
    "DEEPSEEK_API_KEY", "DASHSCOPE_API_KEY", "ZHIPU_API_KEY", "MINIMAX_API_KEY",
    "OPENROUTER_API_KEY", "OPENAI_COMPATIBLE_API_KEY", "AWS_ACCESS_KEY_ID",
)


def _llm_unavailable_reason() -> str | None:
    """Return why the LLM committee cannot run, or None when it can."""
    import importlib.util
    import os

    if importlib.util.find_spec("langgraph") is None:
        return "TradingAgents dependencies are not installed (langgraph missing)"
    if not any(os.environ.get(key) for key in _PROVIDER_KEYS):
        return "no LLM provider API key found in the environment"
    return None
