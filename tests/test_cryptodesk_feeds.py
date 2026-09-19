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
from cryptodesk.feeds.base import (
    INTERVAL_SECONDS,
    Candle,
    FeedError,
    canonical_symbol,
    interval_seconds,
    split_symbol,
)


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
    ("2023-11-14T22:13:20Z", 1_700_000_000),
    ("2023-11-14T22:13:20", 1_700_000_000),     # naive => UTC, not local
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
        self.calls: list[tuple[str, dict, dict]] = []

    def get(self, url, params=None, **kwargs):
        self.calls.append((url, params or {}, kwargs))
        return _FakeResponse(self._payload, self._status)


class _RoutedSession:
    """One payload per endpoint, keyed by a substring of the URL path."""

    def __init__(self, routes: dict):
        self._routes = routes

    def get(self, url, params=None, **kwargs):
        for key, payload in self._routes.items():
            if key in url:
                return _FakeResponse(payload)
        raise AssertionError(f"unrouted {url}")


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


# ---------------------------------------------------------------- intervals / canonical symbols
def test_interval_table_is_shared_and_unknown_tokens_raise():
    assert interval_seconds("5m") == 300
    assert interval_seconds("1d") == 86400
    assert set(INTERVAL_SECONDS) == {"1m", "5m", "15m", "30m", "1h", "4h", "1d"}
    with pytest.raises(FeedError, match="interval"):
        interval_seconds("7m")


@pytest.mark.parametrize("raw", ["BTCUSD", "btc-usd", "BTC-USD", " btcusd "])
def test_canonical_symbol_collapses_every_user_form(raw):
    assert canonical_symbol(raw) == "BTC-USD"


def test_canonical_symbol_keeps_stablecoin_quotes():
    assert canonical_symbol("ETHUSDT") == "ETH-USDT"


# ---------------------------------------------------------------- live feeds: closed bars only
_NOW = 1_700_000_000 + 120  # two minutes into a 5m bar


def _cc_rows(*ts_list):
    return [{"t": ts * 1000, "o": "1", "h": "2", "l": "0.5", "c": "1.5", "v": "10"}
            for ts in ts_list]


def _binance_rows(*ts_list):
    return [[ts * 1000, "1", "2", "0.5", "1.5", "10", ts * 1000 + 299_999, "0", 1, "0", "0", "0"]
            for ts in ts_list]


def test_cryptocom_drops_the_in_progress_candle():
    """Vendor rows end with the forming bar; a live desk must never see it."""
    rows = _cc_rows(1_700_000_000 - 600, 1_700_000_000 - 300, 1_700_000_000)
    feed = CryptoComFeed(session=_FakeSession({"code": 0, "result": {"data": rows}}),
                         clock=lambda: _NOW)
    out = feed.candles("BTC-USD", "5m", 10)
    assert [c.ts for c in out] == [1_700_000_000 - 600, 1_700_000_000 - 300]
    # A bar that closes exactly now is closed.
    feed = CryptoComFeed(session=_FakeSession({"code": 0, "result": {"data": rows}}),
                         clock=lambda: 1_700_000_000 + 300)
    assert feed.candles("BTC-USD", "5m", 10)[-1].ts == 1_700_000_000


def test_binance_drops_the_in_progress_candle():
    rows = _binance_rows(1_700_000_000 - 600, 1_700_000_000 - 300, 1_700_000_000)
    feed = BinanceFeed(session=_FakeSession(rows), clock=lambda: _NOW)
    assert [c.ts for c in feed.candles("BTC-USD", "5m", 10)] == [
        1_700_000_000 - 600, 1_700_000_000 - 300]


def test_a_payload_holding_only_the_open_bar_is_an_error_not_an_empty_list():
    rows = _cc_rows(1_700_000_000)
    feed = CryptoComFeed(session=_FakeSession({"code": 0, "result": {"data": rows}}),
                         clock=lambda: _NOW)
    with pytest.raises(FeedError, match="no closed candles"):
        feed.candles("BTC-USD")
    feed = BinanceFeed(session=_FakeSession(_binance_rows(1_700_000_000)), clock=lambda: _NOW)
    with pytest.raises(FeedError, match="no closed candles"):
        feed.candles("BTC-USD")


def test_live_feeds_use_a_short_connect_timeout():
    """A black-holed venue should cost seconds, not a full read timeout."""
    session = _FakeSession({"code": 0, "result": {"data": _cc_rows(1_700_000_000 - 300)}})
    CryptoComFeed(session=session, timeout=7.0, clock=lambda: _NOW).candles("BTC-USD")
    assert session.calls[0][2]["timeout"] == (3.0, 7.0)
    session = _FakeSession(_binance_rows(1_700_000_000 - 300))
    BinanceFeed(session=session, timeout=7.0, clock=lambda: _NOW).candles("BTC-USD")
    assert session.calls[0][2]["timeout"] == (3.0, 7.0)


# ---------------------------------------------------------------- cryptocom count cap
def test_cryptocom_refuses_more_than_300_candles_instead_of_truncating():
    session = _FakeSession({"code": 0, "result": {"data": _cc_rows(1_700_000_000 - 300)}})
    feed = CryptoComFeed(session=session, clock=lambda: _NOW)
    with pytest.raises(FeedError, match="at most 300"):
        feed.candles("BTC-USD", "5m", 301)
    assert session.calls == []  # refused before any request went out
    feed.candles("BTC-USD", "5m", 300)
    assert session.calls[0][1]["count"] == 300


# ---------------------------------------------------------------- cryptocom price
def _ticker(**fields):
    row = {"a": None, "b": None, "k": None, "l": "76986.77", "h": "78000"}
    row.update(fields)
    return {"code": 0, "result": {"data": [row]}}


def test_cryptocom_price_prefers_the_last_trade():
    feed = CryptoComFeed(session=_FakeSession(_ticker(a="77257.76", b="77250", k="77260")))
    assert feed.price("BTC-USD") == pytest.approx(77257.76)


def test_cryptocom_price_uses_the_midpoint_when_last_trade_is_null():
    """'l' is the 24h low, never a price — marking there fabricates a gap-down."""
    feed = CryptoComFeed(session=_FakeSession(_ticker(a=None, b="77260", k="77269.39")))
    assert feed.price("BTC-USD") == pytest.approx(77264.695)


def test_cryptocom_price_raises_when_no_trade_and_no_quotes():
    feed = CryptoComFeed(session=_FakeSession(_ticker(a=None, b=None, k=None)))
    with pytest.raises(FeedError, match="no usable price"):
        feed.price("BTC-USD")


# ---------------------------------------------------------------- venue payload parse errors
def test_malformed_venue_rows_become_feed_errors():
    """Schema drift must be a venue failure the chain can route around."""
    bad = {"code": 0, "result": {"data": [{"t": 1_700_000_000_000, "o": "1"}]}}
    with pytest.raises(FeedError, match="malformed"):
        CryptoComFeed(session=_FakeSession(bad), clock=lambda: _NOW).candles("BTC-USD")
    with pytest.raises(FeedError, match="malformed"):
        BinanceFeed(session=_FakeSession([[1_700_000_000_000, "1"]]),
                    clock=lambda: _NOW).candles("BTC-USD")
    with pytest.raises(FeedError, match="malformed"):
        CryptoComFeed(session=_FakeSession({"code": 0, "result": {"data": [{"a": "x"}]}})
                      ).price("BTC-USD")


def test_live_feeds_market_returns_candles_price_and_venue():
    session = _RoutedSession({
        "get-candlestick": {"code": 0, "result": {"data": _cc_rows(1_700_000_000 - 300)}},
        "get-tickers": _ticker(a="1.7"),
    })
    candles, price, venue = CryptoComFeed(session=session, clock=lambda: _NOW).market("BTC-USD")
    assert (len(candles), price, venue) == (1, 1.7, "cryptocom")
    session = _RoutedSession({"klines": _binance_rows(1_700_000_000 - 300),
                              "ticker/price": {"price": "1.8"}})
    candles, price, venue = BinanceFeed(session=session, clock=lambda: _NOW).market("BTC-USD")
    assert (len(candles), price, venue) == (1, 1.8, "binance")


# ---------------------------------------------------------------- chain: market() and health
class _Scripted:
    """Feed whose candles/price each succeed or fail on demand, counting calls."""

    def __init__(self, name, candles_ok=True, price_ok=True):
        self.name = name
        self.candles_ok, self.price_ok = candles_ok, price_ok
        self.calls = 0

    def candles(self, symbol, interval="5m", limit=200):
        self.calls += 1
        if not self.candles_ok:
            raise FeedError(f"{self.name} candles down")
        return [Candle(ts=1, open=1, high=1, low=1, close=100.0, volume=1)]

    def price(self, symbol):
        self.calls += 1
        if not self.price_ok:
            raise FeedError(f"{self.name} ticker down")
        return 200.0

    def market(self, symbol, interval="5m", limit=200):
        return self.candles(symbol, interval, limit), self.price(symbol), self.name


def test_chain_market_never_mixes_candles_and_price_from_different_venues():
    primary = _Scripted("primary", candles_ok=True, price_ok=False)
    fallback = _Scripted("fallback")
    chain = ChainFeed([primary, fallback], clock=lambda: 1_000)
    candles, price, venue = chain.market("BTC-USD", "5m", 10)
    assert venue == "fallback" and chain.last_used == "fallback"
    assert candles[0].close == 100.0 and price == 200.0
    assert chain.health["primary"]["failures"] == 1
    assert "ticker down" in chain.health["primary"]["last_error"]
    assert chain.health["fallback"] == {"failures": 0, "last_error": None, "last_ok_ts": 1_000}


def test_chain_market_uses_the_primary_when_it_is_healthy():
    primary, fallback = _Scripted("primary"), _Scripted("fallback")
    chain = ChainFeed([primary, fallback])
    assert chain.market("BTC-USD")[2] == "primary"
    assert fallback.calls == 0


def test_chain_market_raises_when_every_venue_fails():
    chain = ChainFeed([_Scripted("a", price_ok=False), _Scripted("b", candles_ok=False)])
    with pytest.raises(FeedError, match="All feeds failed"):
        chain.market("BTC-USD")


def test_chain_only_catches_feed_errors():
    """A parser bug is a programming error; hiding it behind the fallback
    leaves the primary dead for days with nothing but a log line."""
    class Buggy:
        name = "buggy"

        def candles(self, symbol, interval="5m", limit=200):
            raise KeyError("t")

        def price(self, symbol):
            raise KeyError("price")

    chain = ChainFeed([Buggy(), SyntheticFeed(seed=1)])
    with pytest.raises(KeyError):
        chain.candles("BTC-USD")
    with pytest.raises(KeyError):
        chain.market("BTC-USD")


def test_chain_breaker_skips_a_failing_feed_then_reprobes_after_the_cooldown():
    now = {"t": 1_000.0}
    dead, live = _Scripted("dead", candles_ok=False), _Scripted("live")
    chain = ChainFeed([dead, live], clock=lambda: now["t"])

    for _ in range(3):
        chain.candles("BTC-USD")
    assert dead.calls == 3 and chain.health["dead"]["failures"] == 3

    # Breaker open: the dead venue is not even asked, so its timeout is not paid.
    now["t"] += 100
    chain.candles("BTC-USD")
    chain.price("BTC-USD")
    assert dead.calls == 3 and chain.last_used == "live"

    # Cooldown elapsed: probe again, and a success closes the breaker.
    now["t"] += 200
    dead.candles_ok = True
    assert chain.market("BTC-USD")[2] == "dead"
    assert dead.calls == 5  # 3 failed candles + the probe's candles and price
    assert chain.health["dead"]["failures"] == 0
    assert chain.health["dead"]["last_ok_ts"] == 1_300


def test_chain_breaker_reopens_when_the_probe_fails_again():
    now = {"t": 1_000.0}
    dead, live = _Scripted("dead", candles_ok=False), _Scripted("live")
    chain = ChainFeed([dead, live], clock=lambda: now["t"])
    for _ in range(3):
        chain.candles("BTC-USD")
    now["t"] += 300
    chain.candles("BTC-USD")            # probe fails -> reopened for another 300 s
    assert dead.calls == 4
    now["t"] += 299
    chain.candles("BTC-USD")
    assert dead.calls == 4
    now["t"] += 1
    chain.candles("BTC-USD")
    assert dead.calls == 5


def test_chain_still_probes_when_every_breaker_is_open():
    """Skipping everything would turn a shared blip into a 300 s blackout."""
    a, b = _Scripted("a", candles_ok=False), _Scripted("b", candles_ok=False)
    chain = ChainFeed([a, b], clock=lambda: 1_000)
    for _ in range(3):
        with pytest.raises(FeedError):
            chain.candles("BTC-USD")
    a.candles_ok = True
    assert len(chain.candles("BTC-USD")) == 1
    assert chain.health["a"]["failures"] == 0


# ---------------------------------------------------------------- build_feed kwargs
def test_build_feed_forwards_interval_and_clock():
    clock = lambda: 1_700_000_000  # noqa: E731
    feed = build_feed("synthetic", interval="1h", clock=clock, timeout=99)
    assert feed.interval == "1h" and feed._clock is clock
    chain = build_feed(["cryptocom", "binance"], clock=clock, interval="1h")
    assert chain._clock is clock
    assert all(f._clock is clock for f in chain.feeds)
    with pytest.raises(FeedError, match="Unknown feed"):
        build_feed(["nasdaq"], interval="1h", clock=clock)


# ---------------------------------------------------------------- synthetic: interval + symbols
@pytest.mark.parametrize("interval", ["1m", "15m", "1h", "1d"])
def test_synthetic_price_agrees_with_candles_at_the_configured_interval(interval):
    feed = SyntheticFeed(seed=3, interval=interval)
    assert feed.price("BTC-USD") == pytest.approx(
        feed.candles("BTC-USD", interval, 10)[-1].close)
    candles, price, venue = feed.market("BTC-USD", interval, 10)
    assert price == pytest.approx(candles[-1].close) and venue == "synthetic"


def test_synthetic_market_marks_on_the_series_it_served():
    """Even a mis-configured feed cannot hand back candles from one walk and a
    price from another."""
    feed = SyntheticFeed(seed=3, interval="5m")
    candles, price, _ = feed.market("BTC-USD", "1h", 10)
    assert price == pytest.approx(candles[-1].close)
    assert candles[1].ts - candles[0].ts == 3600


def test_synthetic_rejects_unknown_intervals_like_the_live_feeds():
    with pytest.raises(FeedError, match="interval"):
        SyntheticFeed(seed=3).candles("BTC-USD", "7m")
    with pytest.raises(FeedError, match="interval"):
        SyntheticFeed(seed=3, interval="7m")


def test_synthetic_treats_every_symbol_form_as_the_same_asset():
    feed = SyntheticFeed(seed=3)
    dashed = feed.candles("BTC-USD", "5m", 5)
    assert feed.candles("BTCUSD", "5m", 5) == dashed
    assert feed.candles("btc-usd", "5m", 5) == dashed
    # The first primed bar opens at the BTC anchor, not a hashed pseudo-price.
    assert feed.candles("BTCUSD", "5m", 400)[0].open == pytest.approx(60_000.0)
    assert set(feed._walks) == {("BTC-USD", "5m")}


# ---------------------------------------------------------------- replay: exhaustion, symbols, UTC
def _write_series(tmp_path, name="BTC-USD", bars=10):
    (tmp_path / f"{name}.csv").write_text(
        "ts,open,high,low,close,volume\n" +
        "".join(f"{1_700_000_000 + i * 300},{100 + i},{101 + i},{99 + i},{100.5 + i},10\n"
                for i in range(bars)),
        encoding="utf-8")


def test_replay_reports_exhaustion_and_clamps_the_cursor(tmp_path):
    _write_series(tmp_path, bars=10)
    feed = ReplayFeed(tmp_path, cursor=8)
    assert feed.total_bars("BTC-USD") == 10
    assert not feed.exhausted("BTC-USD")
    feed.advance(1)
    assert not feed.exhausted("BTC-USD")
    feed.advance(5)
    assert feed.cursor == 10 and feed.exhausted("BTC-USD")
    feed.advance(1)
    assert feed.cursor == 10
    # An exhausted feed keeps serving the file's final close as its price.
    assert feed.price("BTC-USD") == pytest.approx(109.5)
    assert len(feed.candles("BTC-USD")) == 10


def test_replay_exhaustion_is_visible_without_any_prior_candles_call(tmp_path):
    _write_series(tmp_path, bars=10)
    assert ReplayFeed(tmp_path, cursor=10).exhausted("BTC-USD")
    assert not ReplayFeed(tmp_path, cursor=9).exhausted("BTC-USD")
    # A cursor-less feed has nothing left to reveal.
    assert ReplayFeed(tmp_path).exhausted("BTC-USD")


def test_replay_opens_the_canonical_file_for_any_symbol_form(tmp_path):
    _write_series(tmp_path, bars=3)
    feed = ReplayFeed(tmp_path)
    assert feed.candles("BTCUSD") == feed.candles("btc-usd") == feed.candles("BTC-USD")
    candles, price, venue = feed.market("BTCUSD")
    assert price == pytest.approx(candles[-1].close) and venue == "replay"


def test_replay_naive_iso_timestamps_are_utc_regardless_of_local_zone(tmp_path, monkeypatch):
    import time as _time
    (tmp_path / "X-USD.csv").write_text(
        "ts,open,high,low,close,volume\n2023-11-14T22:13:20,1,2,0.5,1.5,10\n",
        encoding="utf-8")
    monkeypatch.setenv("TZ", "America/New_York")
    _time.tzset()
    try:
        assert ReplayFeed(tmp_path).candles("X-USD")[0].ts == 1_700_000_000
    finally:
        monkeypatch.undo()
        _time.tzset()
