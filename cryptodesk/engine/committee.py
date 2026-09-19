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
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

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

    def decide(self, symbol: str, snapshot: Snapshot, trigger_reason: str = "",
               now: float | None = None) -> Proposal:
        # ``now`` is part of the shared committee signature; the rule reads
        # only the snapshot, so the clock is irrelevant here.
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


class UsageTracker:
    """Token and call counts for one graph run, fed by the LLM callbacks.

    Kept free of langchain imports so the counters (and their pricing) can be
    exercised without the graph's dependencies; ``_usage_callback_handler``
    wraps it in a real ``BaseCallbackHandler`` when the graph is built.
    """

    def __init__(self) -> None:
        # Callbacks can arrive from worker threads when the graph fans out.
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self.llm_calls = 0
            self.tokens_in = 0
            self.tokens_out = 0

    def record_start(self) -> None:
        with self._lock:
            self.llm_calls += 1

    def record_end(self, response: Any) -> None:
        """Read ``usage_metadata`` off the first generation, as cli/stats_handler does."""
        try:
            generation = response.generations[0][0]
        except (IndexError, TypeError, AttributeError):
            return
        usage = getattr(getattr(generation, "message", None), "usage_metadata", None)
        if usage:
            with self._lock:
                self.tokens_in += int(usage.get("input_tokens", 0) or 0)
                self.tokens_out += int(usage.get("output_tokens", 0) or 0)

    def to_dict(self) -> dict:
        return {"llm_calls": self.llm_calls, "tokens_in": self.tokens_in,
                "tokens_out": self.tokens_out}


def _usage_callback_handler(tracker: UsageTracker):
    """Build the langchain callback that feeds ``tracker``; imported lazily."""
    from langchain_core.callbacks import BaseCallbackHandler

    class _UsageHandler(BaseCallbackHandler):
        def on_llm_start(self, serialized, prompts, **kwargs):
            tracker.record_start()

        def on_chat_model_start(self, serialized, messages, **kwargs):
            tracker.record_start()

        def on_llm_end(self, response, **kwargs):
            tracker.record_end(response)

    return _UsageHandler()


def desk_ta_config(cfg: DeskConfig, overrides: dict | None = None) -> dict:
    """TradingAgents config rooted under the desk home.

    DEFAULT_CONFIG points results, cache and the memory log at
    ``~/.tradingagents``, which the desk's Docker volume does not cover; the
    reflection loop would then be reset on every image rebuild. Keying them
    under ``cfg.home_path`` keeps everything the desk learns in one place.
    """
    from tradingagents.default_config import DEFAULT_CONFIG

    root = cfg.home_path / "tradingagents"
    return {
        **DEFAULT_CONFIG,
        "results_dir": str(root / "logs"),
        "data_cache_dir": str(root / "cache"),
        "memory_log_path": str(root / "memory" / "trading_memory.md"),
        **(overrides or {}),
    }


class LLMCommittee:
    """Runs the TradingAgents graph for one symbol and extracts a rating.

    The graph is imported lazily so the desk starts (and the heuristic runs)
    even when TradingAgents' heavier dependencies are absent.

    On cost: a callback handler on the graph's LLMs counts provider-reported
    tokens per run. When the desk config carries per-million-token prices
    (``llm.price_per_million_input_usd`` / ``..._output_usd``) the run is priced
    from those counts; otherwise ``cost_usd`` is the configured per-run estimate
    and is flagged via ``cost_is_estimate``. A run that died before its first
    LLM call is charged nothing.
    """

    source = "llm"
    name = "llm"

    def __init__(self, cfg: DeskConfig, ta_config: dict | None = None):
        self.cfg = cfg
        self._ta_config = ta_config
        self._graph = None
        self._usage = UsageTracker()

    @property
    def ta_config(self) -> dict:
        """The TradingAgents config this committee runs with (resolved lazily)."""
        if self._ta_config is None:
            self._ta_config = desk_ta_config(self.cfg)
        return self._ta_config

    def _build_graph(self):
        """Construct the TradingAgentsGraph on first use."""
        if self._graph is not None:
            return self._graph
        from tradingagents.graph.trading_graph import TradingAgentsGraph

        # Crypto has no fundamentals to analyse; the analyst would burn tokens
        # producing a report about a company that does not exist.
        analysts = ["market", "social", "news"]
        self._graph = TradingAgentsGraph(
            selected_analysts=analysts, debug=False, config=dict(self.ta_config),
            callbacks=[_usage_callback_handler(self._usage)],
        )
        return self._graph

    def decide(self, symbol: str, snapshot: Snapshot, trigger_reason: str = "",
               now: float | None = None) -> Proposal:
        from tradingagents.agents.utils.rating import parse_rating

        started = time.perf_counter()
        trade_date = _trade_date(now, snapshot)
        # Counters are cumulative across the graph's life; zero them so the
        # cost below is this run's alone.
        self._usage.reset()
        try:
            graph = self._build_graph()
            final_state, decision = graph.propagate(symbol, trade_date, asset_type="crypto")
        except Exception as exc:  # noqa: BLE001 - any failure must not stop the desk
            logger.exception("LLM committee failed for %s", symbol)
            cost, is_estimate = self._price_run(ok=False)
            return Proposal(
                symbol=symbol, rating="Hold", conviction=0.0,
                summary=f"committee failed, holding: {exc}", source=self.source,
                latency_ms=int((time.perf_counter() - started) * 1000),
                error=str(exc), cost_usd=cost, cost_is_estimate=is_estimate,
                detail={**self._usage.to_dict(), "trigger": trigger_reason,
                        "trade_date": trade_date},
            )

        # propagate() already returns the parsed rating word; the PM's rendered
        # markdown is where the reasoning lives, so the summary comes from there.
        pm_text = str((final_state or {}).get("final_trade_decision") or decision)
        rating = decision if decision in RATINGS else parse_rating(pm_text)
        detail = _extract_reports(final_state)
        detail["trigger"] = trigger_reason
        detail["trade_date"] = trade_date
        detail["indicators"] = snapshot.to_dict()
        detail.update(self._usage.to_dict())
        cost, is_estimate = self._price_run(ok=True)
        return Proposal(
            symbol=symbol, rating=rating, conviction=RATING_CONVICTION.get(rating, 0.0),
            summary=_first_meaningful_line(pm_text) or f"Rating: {rating}",
            source=self.source, cost_usd=cost, cost_is_estimate=is_estimate,
            latency_ms=int((time.perf_counter() - started) * 1000),
            detail=detail,
        )

    def _price_run(self, ok: bool) -> tuple[float, bool]:
        """Return ``(cost_usd, cost_is_estimate)`` for the run just finished."""
        usage = self._usage
        price_in = getattr(self.cfg.llm, "price_per_million_input_usd", None)
        price_out = getattr(self.cfg.llm, "price_per_million_output_usd", None)
        if price_in is not None and price_out is not None \
                and (usage.tokens_in or usage.tokens_out):
            cost = usage.tokens_in / 1e6 * price_in + usage.tokens_out / 1e6 * price_out
            return round(cost, 6), False
        # Nothing reached a provider, so nothing was billed. Only on failure:
        # a successful run with no callback traffic means usage went
        # unreported, and over-booking the estimate is the safer error.
        if not ok and usage.llm_calls == 0:
            return 0.0, False
        return self.cfg.llm.estimated_cost_per_run_usd, True


def _trade_date(now: float | None, snapshot: Snapshot) -> str:
    """The date the graph analyses: the desk clock, else the bar, else today.

    ``simulate --replay`` drives the desk with a clock set in the past; using
    the wall clock there would have the committee reason about today's market
    while the fills come from the replayed one.
    """
    ts = now if now is not None else (snapshot.ts or None)
    when = datetime.fromtimestamp(ts, tz=timezone.utc) if ts else datetime.now(timezone.utc)
    return when.strftime("%Y-%m-%d")


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


_EMPHASIS_RE = re.compile(r"\*\*|__")


def _first_meaningful_line(text: str, limit: int = 240) -> str:
    """First non-heading, non-empty line — a usable one-line summary."""
    for line in text.splitlines():
        # Drop markdown emphasis so "**Executive Summary**: ..." reads cleanly.
        stripped = _EMPHASIS_RE.sub("", line).strip().lstrip("#*->").strip()
        if len(stripped) > 30 and not stripped.lower().startswith("rating"):
            return stripped[:limit]
    return ""


class FallbackCommittee:
    """Runs ``primary`` until it fails repeatedly, then ``fallback`` for good.

    A single failed run is noise (rate limit, transient network); a streak
    means the LLM path is broken — expired key, deleted model — and every
    further attempt would only burn the daily cap on Hold. Switching is
    permanent for the process: whoever fixed the key restarts the desk, and
    the name says which committee actually produced the track record.
    """

    def __init__(self, primary, fallback, max_consecutive_failures: int = 3):
        self.primary = primary
        self.fallback = fallback
        self.max_consecutive_failures = max_consecutive_failures
        self.active = primary
        self.consecutive_failures = 0
        self.switched_reason: str | None = None

    @property
    def name(self) -> str:
        if self.switched_reason is None:
            return self.active.name
        return (f"{self.fallback.name} (fallback after "
                f"{self.max_consecutive_failures} {self.primary.name} failures)")

    @property
    def source(self) -> str:
        return self.active.source

    def decide(self, symbol: str, snapshot: Snapshot, trigger_reason: str = "",
               now: float | None = None) -> Proposal:
        proposal = self.active.decide(symbol, snapshot, trigger_reason, now=now)
        if self.active is not self.primary:
            return proposal
        if proposal.error is None:
            self.consecutive_failures = 0
            return proposal
        self.consecutive_failures += 1
        if self.consecutive_failures >= self.max_consecutive_failures:
            self.switch(f"{self.consecutive_failures} consecutive {self.primary.name} "
                        f"failures; last: {proposal.error}")
        return proposal

    def switch(self, reason: str) -> None:
        """Route every further decision to the fallback and remember why."""
        self.active = self.fallback
        self.switched_reason = reason
        logger.error("Committee switched to %s: %s", self.fallback.name, reason)


def build_committee(cfg: DeskConfig, ta_config: dict | None = None):
    """Pick a committee per config, falling back to the heuristic when needed.

    ``committee="auto"`` uses the LLM only when TradingAgents imports, the key
    its configured provider needs is present, *and* the graph constructs; at
    runtime it is wrapped so a streak of failed runs drops to the heuristic
    too. Falling back rather than crashing is deliberate: a desk that stops
    trading because a key expired is worse than one that keeps running a
    documented baseline and says so on the dashboard.
    """
    mode = (cfg.committee or "auto").lower()
    if mode == "heuristic":
        return HeuristicCommittee()
    if mode == "llm":
        return LLMCommittee(cfg, ta_config)
    if mode != "auto":
        raise ValueError(f"Unknown committee mode {cfg.committee!r}; "
                         "expected 'auto', 'llm' or 'heuristic'")

    reason = _llm_unavailable_reason(ta_config)
    if reason:
        logger.warning("Committee falling back to heuristic: %s", reason)
        return HeuristicCommittee()
    llm = LLMCommittee(cfg, ta_config)
    try:
        # Construct now rather than on the first trigger: a provider that
        # rejects its key at client construction should fall back at startup,
        # not after burning the failure streak.
        llm._build_graph()
    except Exception as exc:  # noqa: BLE001 - construction errors are provider-specific
        logger.warning("Committee falling back to heuristic: graph construction failed: %s", exc)
        return HeuristicCommittee()
    return FallbackCommittee(llm, HeuristicCommittee())


def _llm_unavailable_reason(ta_config: dict | None = None) -> str | None:
    """Return why the LLM committee cannot run, or None when it can.

    The key checked is the one the *configured* provider needs — a desk with
    only ``ANTHROPIC_API_KEY`` set and the default (openai) provider would
    otherwise pass here and fail on every run.
    """
    import importlib.util
    import os

    if importlib.util.find_spec("langgraph") is None:
        return "TradingAgents dependencies are not installed (langgraph missing)"

    from tradingagents.llm_clients.api_key_env import get_api_key_env

    if ta_config is None:
        from tradingagents.default_config import DEFAULT_CONFIG
        ta_config = DEFAULT_CONFIG
    provider = str(ta_config.get("llm_provider") or "openai")
    key_env = get_api_key_env(provider)
    if key_env is None or _key_is_optional(provider):
        # Bedrock uses the AWS credential chain; local runtimes do not
        # authenticate; unknown providers cannot be checked here.
        return None
    if not os.environ.get(key_env):
        return f"{key_env} is not set (llm_provider={provider!r})"
    return None


def _key_is_optional(provider: str) -> bool:
    """Whether the provider registry marks ``provider`` as runnable keyless."""
    try:
        from tradingagents.llm_clients.openai_client import OPENAI_COMPATIBLE_PROVIDERS
    except ImportError:
        return False
    spec = OPENAI_COMPATIBLE_PROVIDERS.get(provider.lower())
    return bool(spec is not None and spec.key_optional)
