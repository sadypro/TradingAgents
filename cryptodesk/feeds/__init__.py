"""Market data feeds.

Each feed maps the desk's canonical ``BASE-QUOTE`` symbol (``BTC-USD``) to its
own venue convention and returns plain ``Candle`` objects, so the rest of the
desk never learns which venue it is talking to. ``build_feed`` assembles a
fallback chain, because a single exchange API being briefly unreachable should
degrade the desk to a second source rather than stop it trading.
"""

from .base import Candle, Feed, FeedError
from .binance import BinanceFeed
from .chain import ChainFeed
from .cryptocom import CryptoComFeed
from .replay import ReplayFeed
from .synthetic import SyntheticFeed

_REGISTRY = {
    "cryptocom": CryptoComFeed,
    "binance": BinanceFeed,
    "synthetic": SyntheticFeed,
    "replay": ReplayFeed,
}


def build_feed(names, **kwargs) -> Feed:
    """Build a single feed, or a fallback chain when several names are given."""
    if isinstance(names, str):
        names = [names]
    unknown = [n for n in names if n not in _REGISTRY]
    if unknown:
        raise FeedError(
            f"Unknown feed(s): {', '.join(unknown)}. "
            f"Available: {', '.join(sorted(_REGISTRY))}"
        )
    feeds = [_REGISTRY[n](**_kwargs_for(n, kwargs)) for n in names]
    if not feeds:
        raise FeedError("No feeds configured")
    return feeds[0] if len(feeds) == 1 else ChainFeed(feeds)


def _kwargs_for(name: str, kwargs: dict) -> dict:
    """Pass only the kwargs a given feed accepts."""
    import inspect

    accepted = inspect.signature(_REGISTRY[name].__init__).parameters
    return {k: v for k, v in kwargs.items() if k in accepted}


__all__ = [
    "Candle", "Feed", "FeedError", "ChainFeed", "CryptoComFeed",
    "BinanceFeed", "SyntheticFeed", "ReplayFeed", "build_feed",
]
