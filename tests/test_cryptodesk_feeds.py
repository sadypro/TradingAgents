"""Feeds: symbol mapping, venue-error handling, and the offline generators."""


import pytest
import requests

from cryptodesk.feeds import (
    BinanceFeed,
    ChainFeed,
    CryptoComFeed,
    ReplayFeed,
    SyntheticFeed,
    build_feed,
)
from cryptodesk.feeds.base import Candle, FeedError, split_symbol


# ---------------------------------------------------------------- symbols
@pytest.mark.parametrize("raw, expected", [
    ("BTC-USD", ("BTC", "USD")),
    ("btc-usd", ("BTC", "USD")),
    ("BTCUSD", ("BTC", "USD")),
    ("ETH-USDT", ("ETH", "USDT")),
    ("SOLUSDC", ("SOL", "USDC")),
])
def test_symbols_parse_in_every_form_users_paste(raw, expected):
    assert split_symbol(raw) == expected


@pytest.mark.parametrize("raw", ["", "   ", "NOTASYMBOL"])
def test_unparseable_symbols_raise_rather_than_guess(raw):
    with pytest.raises(FeedError):
        split_symbol(raw)


def test_venue_symbol_conventions():
    assert CryptoComFeed.venue_symbol("BTC-USD") == "BTC_USD"
    # Binance lists USDT, not USD; the USD request is served from the USDT pair.
    assert BinanceFeed.venue_symbol("BTC-USD") == "BTCUSDT"
    assert BinanceFeed.venue_symbol("ETH-USDT") == "ETHUSDT"


# ---------------------------------------------------------------- synthetic
def test_synthetic_prices_are_deterministic_across_instances():
    a = SyntheticFeed(seed=3).candles("BTC-USD", "5m", 50)
    b = SyntheticFeed(seed=3).candles("BTC-USD", "5m", 50)
    assert [c.close for c in a] == [c.close for c in b]


def test_synthetic_history_does_not_revise_when_more_bars_are_requested():
    """A feed whose past changes under you invalidates every indicator."""
    feed = SyntheticFeed(seed=3)
    short = feed.candles("BTC-USD", "5m", 50)
    long = feed.candles("BTC-USD", "5m", 200)
    assert [c.close for c in short] == [c.close for c in long[-50:]]


def test_synthetic_price_agrees_with_the_last_close():
    feed = SyntheticFeed(seed=3)
    assert feed.price("BTC-USD") == pytest.approx(feed.candles("BTC-USD", "5m", 10)[-1].close)


def test_synthetic_candles_are_well_formed():
    candles = SyntheticFeed(seed=3).candles("ETH-USD", "5m", 200)
    assert all(c.low <= min(c.open, c.close) for c in candles)
    assert all(c.high >= max(c.open, c.close) for c in candles)
    assert all(c.low > 0 and c.volume >= 0 for c in candles)
    assert all(candles[i].ts < candles[i + 1].ts for i in range(len(candles) - 1))


def test_synthetic_withholds_the_in_progress_bar():
    """Handing a partial bar to a live loop is how look-ahead gets baked in."""
    now = 1_700_000_000
    feed = SyntheticFeed(seed=3, clock=lambda: now)
    last = feed.candles("BTC-USD", "5m", 10)[-1]
    assert last.ts < (now // 300) * 300


# ---------------------------------------------------------------- replay
def test_replay_reads_csv_and_respects_the_cursor(tmp_path):
    (tmp_path / "BTC-USD.csv").write_text(
        "ts,open,high,low,close,volume\n" +
        "".join(f"{1_700_000_000 + i * 300},{100 + i},{101 + i},{99 + i},{100.5 + i},10\n"
                for i in range(10)),
        encoding="utf-8")

    feed = ReplayFeed(tmp_path, cursor=3)
    assert len(feed.candles("BTC-USD")) == 3
    assert feed.price("BTC-USD") == pytest.approx(102.5)

    feed.advance(2)
    assert len(feed.candles("BTC-USD")) == 5


@pytest.mark.parametrize("ts_value, expected", [
    ("1700000000", 1_700_000_000),
    ("1700000000000", 1_700_000_000),          # milliseconds
    ("2023-11-14T22:13:20+00:00", 1_700_000_000),
])
def test_replay_accepts_the_common_timestamp_formats(tmp_path, ts_value, expected):
    (tmp_path / "X-USD.csv").write_text(
        f"ts,open,high,low,close,volume\n{ts_value},1,2,0.5,1.5,10\n", encoding="utf-8")
    assert ReplayFeed(tmp_path).candles("X-USD")[0].ts == expected


def test_replay_reports_a_missing_or_malformed_file(tmp_path):
    with pytest.raises(FeedError, match="No replay CSV"):
        ReplayFeed(tmp_path).candles("NOPE-USD")

    (tmp_path / "BAD-USD.csv").write_text("ts,open\n1,2\n", encoding="utf-8")
    with pytest.raises(FeedError, match="Malformed row"):
        ReplayFeed(tmp_path).candles("BAD-USD")


# ---------------------------------------------------------------- chain
class _Boom:
    name = "boom"

    def candles(self, symbol, interval="5m", limit=200):
        raise FeedError("venue down")

    def price(self, symbol):
        raise FeedError("venue down")


def test_chain_falls_back_to_the_next_feed():
    chain = ChainFeed([_Boom(), SyntheticFeed(seed=1)])
    assert len(chain.candles("BTC-USD", "5m", 10)) == 10
    assert chain.last_used == "synthetic"


def test_chain_raises_only_when_every_feed_fails():
    chain = ChainFeed([_Boom(), _Boom()])
    with pytest.raises(FeedError, match="All feeds failed"):
        chain.price("BTC-USD")


def test_build_feed_rejects_unknown_names():
    with pytest.raises(FeedError, match="Unknown feed"):
        build_feed(["nasdaq"])


def test_build_feed_returns_a_chain_for_several_names():
    assert isinstance(build_feed(["cryptocom", "binance"]), ChainFeed)


# ---------------------------------------------------------------- venue errors
class _FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")

    def json(self):
        if isinstance(self._payload, str):
            raise ValueError("not json")
        return self._payload


class _FakeSession:
    def __init__(self, payload, status=200):
        self._payload, self._status = payload, status

    def get(self, *args, **kwargs):
        return _FakeResponse(self._payload, self._status)


def test_cryptocom_surfaces_an_in_band_error_code():
    """The venue returns application errors inside a 200, past raise_for_status."""
    feed = CryptoComFeed(session=_FakeSession({"code": 10004, "message": "bad instrument"}))
    with pytest.raises(FeedError, match="10004"):
        feed.candles("BTC-USD")


def test_cryptocom_parses_a_well_formed_payload():
    payload = {"code": 0, "result": {"data": [
        {"t": 1_700_000_000_000, "o": "1", "h": "2", "l": "0.5", "c": "1.5", "v": "10"},
    ]}}
    candle = CryptoComFeed(session=_FakeSession(payload)).candles("BTC-USD")[0]
    assert candle == Candle(ts=1_700_000_000, open=1.0, high=2.0, low=0.5, close=1.5, volume=10.0)


def test_empty_venue_responses_raise_rather_than_returning_nothing():
    with pytest.raises(FeedError, match="no candles"):
        CryptoComFeed(session=_FakeSession({"code": 0, "result": {"data": []}})).candles("BTC-USD")
    with pytest.raises(FeedError, match="no candles"):
        BinanceFeed(session=_FakeSession([])).candles("BTC-USD")


def test_non_json_venue_responses_are_reported_clearly():
    with pytest.raises(FeedError, match="non-JSON"):
        BinanceFeed(session=_FakeSession("<html>502</html>")).candles("BTC-USD")


def test_unsupported_intervals_are_refused():
    with pytest.raises(FeedError, match="interval"):
        CryptoComFeed().candles("BTC-USD", "7m")
