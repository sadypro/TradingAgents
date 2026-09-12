"""Command line entry point: ``python -m cryptodesk <command>``.

Commands
--------
``run``       Start the desk and the dashboard, and keep trading until stopped.
``simulate``  Fast-forward days or weeks of paper trading in seconds, against a
              synthetic or replayed market, to get a track record to look at
              before committing to a long live run.
``doctor``    Check config, feed reachability, and LLM credentials before a run.
``report``    Print a performance summary from an existing ledger.
``init``      Write a default config file to edit.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from .api.server import build_app, performance
from .config import DeskConfig
from .engine.committee import HeuristicCommittee, _llm_unavailable_reason, build_committee
from .engine.ledger import Ledger
from .engine.loop import Desk
from .feeds import build_feed
from .feeds.base import FeedError


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)-28s %(message)s",
        datefmt="%H:%M:%S",
    )
    # yfinance/urllib3 chatter drowns the desk's own log lines.
    for noisy in ("urllib3", "yfinance", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


class _Clock:
    """A manually advanced clock, for simulation."""

    def __init__(self, start: float):
        self.t = float(start)

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


# ---------------------------------------------------------------- commands
def cmd_run(args) -> int:
    """Start the engine in a background thread and serve the dashboard."""
    import uvicorn

    cfg = DeskConfig.load(args.config)
    if args.symbols:
        cfg.symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if args.feed:
        cfg.feeds = [args.feed]
    if args.committee:
        cfg.committee = args.committee

    desk = Desk(cfg)
    print(f"CryptoDesk: {', '.join(cfg.symbols)}")
    print(f"  feed      {desk.feed.name}")
    print(f"  committee {desk.committee.name}")
    print(f"  equity    ${cfg.starting_equity:,.2f} (paper)")
    print(f"  ledger    {cfg.db_path}")
    print(f"  dashboard http://{cfg.api_host}:{cfg.api_port}")
    if desk.committee.name == "heuristic" and cfg.committee != "heuristic":
        print(f"  note: LLM committee unavailable — {_llm_unavailable_reason()}")

    desk.start_background()
    app = build_app(desk)
    try:
        uvicorn.run(app, host=cfg.api_host, port=cfg.api_port, log_level="warning")
    except KeyboardInterrupt:  # pragma: no cover - interactive
        pass
    finally:
        desk.stop()
        print("\nDesk stopped.")
    return 0


def cmd_simulate(args) -> int:
    """Run the desk over compressed time, then optionally serve the result.

    The point is speed of feedback: a month of five-minute bars is ~8,600 ticks,
    which runs in seconds with the free heuristic committee. It is the honest way
    to exercise the machinery (stops, caps, the kill-switch, the ledger) before
    letting it run for a month in real time.

    Cost note: with ``--committee llm`` this makes real, paid API calls on every
    trigger — thousands of them over a long window. The default is the free
    heuristic for that reason.
    """
    cfg = DeskConfig.load(args.config)
    if args.symbols:
        cfg.symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    cfg.committee = args.committee or "heuristic"
    # Simulated runs get their own ledger so they never pollute a live record.
    cfg.home = args.home or str(Path(cfg.home_path) / "simulations" / time.strftime("%Y%m%d-%H%M%S"))
    cfg.ensure_dirs()

    step = args.step_minutes * 60
    ticks = int((args.days * 24 * 60) / args.step_minutes)
    clock = _Clock(time.time() - ticks * step)

    if args.replay:
        from .feeds import ReplayFeed
        feed = ReplayFeed(directory=args.replay, cursor=args.warmup)
    else:
        from .feeds import SyntheticFeed
        feed = SyntheticFeed(seed=args.seed, drift_per_year=args.drift,
                             annual_vol=args.vol, clock=clock)

    committee = HeuristicCommittee() if cfg.committee == "heuristic" else build_committee(cfg)
    desk = Desk(cfg, feed=feed, committee=committee, clock=clock)

    label = f"replay:{args.replay}" if args.replay else f"synthetic(seed={args.seed}, vol={args.vol}, drift={args.drift})"
    print(f"Simulating {args.days} days over {label}")
    print(f"  {ticks} ticks at {args.step_minutes}m · committee={committee.name} · ledger={cfg.db_path}")
    if args.replay is None:
        print("  NOTE: synthetic prices are a random walk. Results here test the")
        print("        machinery, not the strategy. They are not evidence of edge.")

    started = time.time()
    for i in range(ticks):
        desk.tick()
        clock.advance(step)
        if args.replay:
            feed.advance(1)
        if args.progress and i % max(1, ticks // 20) == 0:
            state = desk.state()
            print(f"    {i:6d}/{ticks}  equity ${state['account']['equity']:,.2f}  "
                  f"dd {state['risk']['drawdown']:.1%}"
                  f"{'  HALTED' if state['risk']['halted'] else ''}")

    elapsed = time.time() - started
    print(f"  done in {elapsed:.1f}s ({ticks / max(elapsed, 1e-9):,.0f} ticks/s)\n")
    _print_report(desk.ledger, desk.broker.starting_equity, cfg)

    if args.serve:
        import uvicorn
        print(f"\nServing the simulated result at http://{cfg.api_host}:{cfg.api_port}")
        print("(the desk is not ticking — this is the finished run)")
        uvicorn.run(build_app(desk), host=cfg.api_host, port=cfg.api_port, log_level="warning")
    return 0


def cmd_doctor(args) -> int:
    """Check everything a live run needs, and say what is wrong if anything is."""
    cfg = DeskConfig.load(args.config)
    problems = 0

    print("Config")
    print(f"  symbols        {', '.join(cfg.symbols)}")
    print(f"  starting cash  ${cfg.starting_equity:,.2f} (paper)")
    print(f"  home           {cfg.home_path}")
    print(f"  risk/trade     {cfg.risk.risk_per_trade:.2%} · max/symbol "
          f"{cfg.risk.max_symbol_weight:.0%} · gross cap {cfg.risk.max_gross_exposure:.0%}")
    print(f"  halt at        {cfg.risk.max_drawdown_halt:.0%} drawdown · "
          f"daily stop {cfg.risk.daily_loss_limit:.0%}")
    print(f"  llm budget     ${cfg.llm.daily_usd_cap:.2f}/day, "
          f"min {cfg.llm.min_minutes_between_calls}m between runs per symbol")

    print("\nFeeds")
    for name in cfg.feeds:
        try:
            feed = build_feed(name)
            candles = feed.candles(cfg.symbols[0], cfg.candle_interval, 20)
            price = feed.price(cfg.symbols[0])
            print(f"  [ok]   {name:12} {len(candles)} candles, last price "
                  f"${price:,.2f} for {cfg.symbols[0]}")
        except (FeedError, Exception) as exc:  # noqa: BLE001 - report, never raise
            problems += 1
            print(f"  [FAIL] {name:12} {exc}")
    print("  (a feed can fail because the venue blocks your network or region;")
    print("   the chain falls back to the next one, and 'synthetic' always works)")

    print("\nCommittee")
    reason = _llm_unavailable_reason()
    if reason:
        print(f"  [warn] LLM committee unavailable: {reason}")
        print("         The desk will run the free heuristic baseline instead.")
    else:
        print("  [ok]   TradingAgents importable and a provider key is present")
    print(f"  configured mode: {cfg.committee}")

    print("\nLedger")
    try:
        ledger = Ledger(cfg.db_path)
        curve = ledger.equity_curve(limit=1)
        print(f"  [ok]   {cfg.db_path} ({len(curve)} equity marks so far)")
        ledger.close()
    except Exception as exc:  # noqa: BLE001
        problems += 1
        print(f"  [FAIL] {cfg.db_path}: {exc}")

    print(f"\n{'No blocking problems.' if problems == 0 else f'{problems} problem(s) above.'}")
    return 1 if problems else 0


def cmd_report(args) -> int:
    """Print a performance summary from an existing ledger."""
    cfg = DeskConfig.load(args.config)
    path = Path(args.db) if args.db else cfg.db_path
    if not path.exists():
        print(f"No ledger at {path}. Run the desk first.", file=sys.stderr)
        return 1
    ledger = Ledger(path)
    _print_report(ledger, cfg.starting_equity, cfg)
    ledger.close()
    return 0


def cmd_init(args) -> int:
    """Write a config file pre-filled with the defaults."""
    cfg = DeskConfig.load(None)
    path = Path(args.path).expanduser()
    if path.exists() and not args.force:
        print(f"{path} already exists; pass --force to overwrite.", file=sys.stderr)
        return 1
    cfg.save(path)
    print(f"Wrote {path}")
    print(f"Edit it, then run: CRYPTODESK_CONFIG={path} python -m cryptodesk run")
    return 0


def _print_report(ledger: Ledger, starting_equity: float, cfg: DeskConfig) -> None:
    curve = ledger.equity_curve(limit=1_000_000)
    stats = ledger.trade_stats()
    perf = performance(curve, starting_equity, llm_spend=ledger.spend_total(),
                       bar_minutes=max(cfg.fast_loop_seconds / 60.0, 1 / 60.0))
    if not curve:
        print("No equity history recorded yet.")
        return

    def pct(value):
        return "—" if value is None else f"{value * 100:+.2f}%"

    print("Performance")
    print(f"  window            {time.strftime('%Y-%m-%d %H:%M', time.gmtime(perf['start_ts']))}"
          f" → {time.strftime('%Y-%m-%d %H:%M', time.gmtime(perf['end_ts']))} UTC"
          f"  ({perf['points']:,} marks)")
    print(f"  equity            ${starting_equity:,.2f} → ${perf['final_equity']:,.2f}"
          f"   ({pct(perf['total_return'])})")
    print(f"  buy & hold        {pct(perf['benchmark_return'])}"
          f"   ({cfg.benchmark_symbol})")
    print(f"  alpha vs hold     {pct(perf['alpha_vs_benchmark'])}"
          "   <- the number that decides whether this was worth running")
    print(f"  max drawdown      {perf['max_drawdown'] * 100:.2f}%")
    if perf["sharpe"] is not None:
        print(f"  annualised vol    {perf['annualised_vol'] * 100:.1f}%"
              f"   ·  Sharpe {perf['sharpe']:.2f} (rf=0)")
    print("\nCosts")
    print(f"  LLM spend         ${perf['llm_spend']:,.2f}")
    print(f"  net of LLM        ${perf['net_equity_after_llm']:,.2f}   ({pct(perf['net_return_after_llm'])})")
    print("\nTrades")
    print(f"  closed            {stats['closed_trades']}  "
          f"({stats['wins']}W / {stats['losses']}L, {stats['win_rate'] * 100:.1f}% win rate)")
    if stats["closed_trades"]:
        pf = stats["profit_factor"]
        print(f"  avg win / loss    ${stats['avg_win']:,.2f} / ${stats['avg_loss']:,.2f}"
              f"   ·  profit factor {'—' if pf is None else f'{pf:.2f}'}")
        print(f"  net realised      ${stats['net_realized']:,.2f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cryptodesk",
        description="A 24/7 crypto paper-trading desk driven by TradingAgents.",
    )
    parser.add_argument("--config", help="path to a JSON config file")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="start the desk and dashboard")
    run.add_argument("--symbols", help="comma-separated, e.g. BTC-USD,ETH-USD")
    run.add_argument("--feed", help="cryptocom | binance | synthetic | replay")
    run.add_argument("--committee", choices=["auto", "llm", "heuristic"])
    run.set_defaults(func=cmd_run)

    sim = sub.add_parser("simulate", help="fast-forward paper trading over compressed time")
    sim.add_argument("--days", type=float, default=30.0)
    sim.add_argument("--step-minutes", type=float, default=5.0)
    sim.add_argument("--symbols")
    sim.add_argument("--committee", choices=["auto", "llm", "heuristic"], default="heuristic")
    sim.add_argument("--seed", type=int, default=7)
    sim.add_argument("--vol", type=float, default=0.65, help="annualised vol for synthetic prices")
    sim.add_argument("--drift", type=float, default=0.0, help="annualised drift for synthetic prices")
    sim.add_argument("--replay", help="directory of <SYMBOL>.csv candle files to replay instead")
    sim.add_argument("--warmup", type=int, default=200, help="replay bars visible before tick 1")
    sim.add_argument("--home", help="where to write this simulation's ledger")
    sim.add_argument("--serve", action="store_true", help="serve the dashboard when done")
    sim.add_argument("--progress", action="store_true", default=True)
    sim.set_defaults(func=cmd_simulate)

    doc = sub.add_parser("doctor", help="check config, feeds and credentials")
    doc.set_defaults(func=cmd_doctor)

    rep = sub.add_parser("report", help="print a performance summary")
    rep.add_argument("--db", help="ledger path (defaults to the configured one)")
    rep.set_defaults(func=cmd_report)

    ini = sub.add_parser("init", help="write a default config file")
    ini.add_argument("path", nargs="?", default="cryptodesk.json")
    ini.add_argument("--force", action="store_true")
    ini.set_defaults(func=cmd_init)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
