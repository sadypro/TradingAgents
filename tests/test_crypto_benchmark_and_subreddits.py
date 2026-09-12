"""Crypto-specific correctness in the reflection layer and the sentiment feed.

Three defects this covers, all crypto-only:

1. ``_resolve_benchmark`` matched equity exchange suffixes only, so a pair like
   ``BTC-USD`` fell through to the empty-suffix default and had its alpha
   measured against SPY.
2. ``_fetch_returns`` indexed the asset and the benchmark series by the same
   row number. Crypto trades 7 days a week and equity benchmarks 5, so the two
   legs of the alpha subtraction covered different holding periods.
3. Reddit sentiment searched the equity subreddits for crypto tickers.
"""

import pandas as pd
import pytest

from tradingagents.dataflows.reddit import (
    CRYPTO_SUBREDDITS,
    DEFAULT_SUBREDDITS,
    subreddits_for,
)
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph


class _Graph:
    """Bare carrier for config; ``_resolve_benchmark`` does not touch the rest."""

    def __init__(self, config=None):
        self.config = dict(config or DEFAULT_CONFIG)


def resolve(ticker, config=None):
    return TradingAgentsGraph._resolve_benchmark(_Graph(config), ticker)


# ------------------------------------------------------------- benchmark
@pytest.mark.parametrize("ticker", ["BTC-USD", "ETH-USD", "btc-usd", "SOLUSDT", "DOGE-USDC"])
def test_crypto_is_benchmarked_against_crypto_not_the_sp500(ticker):
    assert resolve(ticker) == "BTC-USD"


@pytest.mark.parametrize("ticker, expected", [
    ("AAPL", "SPY"), ("SPY", "SPY"), ("BRK.B", "SPY"),
    ("7203.T", "^N225"), ("0700.HK", "^HSI"), ("RELIANCE.NS", "^NSEI"),
])
def test_equity_benchmarks_are_unchanged(ticker, expected):
    assert resolve(ticker) == expected


def test_the_crypto_benchmark_is_configurable():
    config = dict(DEFAULT_CONFIG, crypto_benchmark="ETH-USD")
    assert resolve("SOL-USD", config) == "ETH-USD"


def test_an_explicit_benchmark_still_overrides_everything():
    config = dict(DEFAULT_CONFIG, benchmark_ticker="QQQ")
    assert resolve("BTC-USD", config) == "QQQ"


# ------------------------------------------------------------- alignment
def _frame(start, days, step, price_fn):
    """A price frame on a fixed calendar: ``days`` bars every ``step`` days."""
    index = pd.date_range(start, periods=days, freq=f"{step}D")
    return pd.DataFrame({"Close": [price_fn(i) for i in range(days)]}, index=index)


def test_benchmark_is_aligned_by_date_not_by_row(monkeypatch):
    """The crypto/equity calendar mismatch must not shift the benchmark window.

    The asset trades daily; the benchmark trades every other day (standing in
    for weekends). Row 5 of each is a different calendar date, so a positional
    comparison silently measured 5 days of asset return against 10 days of
    benchmark return.
    """
    asset = _frame("2026-01-05", 15, 1, lambda i: 100.0 * (1.01 ** i))
    bench = _frame("2026-01-05", 15, 2, lambda i: 100.0 * (1.01 ** i))

    class _Ticker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, start, end):
            return asset if self.symbol == "BTC-USD" else bench

    monkeypatch.setattr("tradingagents.graph.trading_graph.yf.Ticker", _Ticker)

    raw, alpha, days = TradingAgentsGraph._fetch_returns(
        _Graph(), "BTC-USD", "2026-01-05", holding_days=5, benchmark="SPY")

    assert days == 5
    assert raw == pytest.approx(1.01 ** 5 - 1)
    # Both series rise 1% per bar, but the benchmark's bars are two days apart,
    # so over the same five calendar days it has only advanced two bars.
    assert alpha == pytest.approx((1.01 ** 5 - 1) - (1.01 ** 2 - 1))


def test_a_shared_calendar_gives_zero_alpha_for_identical_series(monkeypatch):
    frame = _frame("2026-01-05", 15, 1, lambda i: 100.0 * (1.01 ** i))

    monkeypatch.setattr("tradingagents.graph.trading_graph.yf.Ticker",
                        lambda symbol: type("T", (), {"history": lambda self, start, end: frame})())

    raw, alpha, _ = TradingAgentsGraph._fetch_returns(
        _Graph(), "BTC-USD", "2026-01-05", holding_days=5, benchmark="BTC-USD")
    assert alpha == pytest.approx(0.0)
    assert raw == pytest.approx(1.01 ** 5 - 1)


def test_a_short_benchmark_series_does_not_index_out_of_range(monkeypatch):
    asset = _frame("2026-01-05", 15, 1, lambda i: 100.0 + i)
    bench = _frame("2026-01-05", 2, 1, lambda i: 100.0 + i)

    class _Ticker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, start, end):
            return asset if self.symbol == "BTC-USD" else bench

    monkeypatch.setattr("tradingagents.graph.trading_graph.yf.Ticker", _Ticker)
    raw, alpha, days = TradingAgentsGraph._fetch_returns(
        _Graph(), "BTC-USD", "2026-01-05", holding_days=5, benchmark="SPY")
    assert raw is not None and alpha is not None


# ------------------------------------------------------------- subreddits
@pytest.mark.parametrize("ticker", ["AAPL", "SPY", "7203.T"])
def test_equities_keep_the_finance_subreddits(ticker):
    assert subreddits_for(ticker) == DEFAULT_SUBREDDITS


@pytest.mark.parametrize("ticker, leading", [
    ("BTC-USD", "Bitcoin"), ("ETH-USD", "ethereum"), ("SOL-USD", "solana"),
    ("DOGE-USD", "dogecoin"), ("BTCUSDT", "Bitcoin"),
])
def test_crypto_searches_its_own_communities_first(ticker, leading):
    subs = subreddits_for(ticker)
    assert subs[0] == leading
    assert not set(subs) & set(DEFAULT_SUBREDDITS)
    assert set(CRYPTO_SUBREDDITS).issubset(set(subs))


def test_an_unlisted_coin_still_gets_the_general_crypto_rooms():
    assert subreddits_for("LTC-USD")[0] == "litecoin"
    # crypto_base only recognises a fixed set of bases; anything else is
    # treated as an equity rather than guessed at.
    assert subreddits_for("XMR-USD") == DEFAULT_SUBREDDITS


def test_an_explicit_subreddit_list_still_wins(monkeypatch):
    seen = []
    monkeypatch.setattr("tradingagents.dataflows.reddit._fetch_subreddit",
                        lambda ticker, sub, limit, timeout: seen.append(sub) or [])
    from tradingagents.dataflows.reddit import fetch_reddit_posts
    fetch_reddit_posts("BTC-USD", subreddits=["mycustomsub"], inter_request_delay=0)
    assert seen == ["mycustomsub"]


def test_both_legs_truncate_to_the_last_shared_date(monkeypatch):
    """A short benchmark must shorten the asset's window too, not just its own.

    Otherwise a 5-day asset return would be compared against a 2-day benchmark
    return and the difference reported as alpha.
    """
    asset = _frame("2026-01-05", 15, 1, lambda i: 100.0 * (1.02 ** i))
    bench = _frame("2026-01-05", 3, 1, lambda i: 100.0 * (1.02 ** i))

    class _Ticker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, start, end):
            return asset if self.symbol == "BTC-USD" else bench

    monkeypatch.setattr("tradingagents.graph.trading_graph.yf.Ticker", _Ticker)
    raw, alpha, days = TradingAgentsGraph._fetch_returns(
        _Graph(), "BTC-USD", "2026-01-05", holding_days=5, benchmark="SPY")

    assert days == 2, "the asset window should truncate to the benchmark's last date"
    assert raw == pytest.approx(1.02 ** 2 - 1)
    assert alpha == pytest.approx(0.0), "identical series over the same window have no alpha"
