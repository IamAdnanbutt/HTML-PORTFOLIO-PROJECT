"""Command-line interface for ictbot.

Examples
--------
    # generate sample data and back-test the model on it
    python -m ictbot.cli demo --symbol NQ --days 60

    # generate and save synthetic data to a CSV
    python -m ictbot.cli gen --days 60 --out data/sample_NQ_1m.csv

    # back-test against your own CSV (columns: time,open,high,low,close,volume)
    python -m ictbot.cli backtest --symbol NQ --csv data/my_data.csv

    # paper-trade by streaming a CSV bar-by-bar (no live broker required)
    python -m ictbot.cli paper --symbol NQ --csv data/sample_NQ_1m.csv
"""

from __future__ import annotations

import argparse
import sys
from typing import List

from .backtest import Backtester
from .config import Config
from .data import generate_sessions, load_csv, write_csv
from .models import Candle, Trade


# Representative starting price per product, so synthetic data is scaled
# realistically (NQ/MNQ trade ~18,000; ES/MES ~5,000).
DEFAULT_START_PRICE = {"NQ": 18_000.0, "MNQ": 18_000.0,
                       "ES": 5_000.0, "MES": 5_000.0}


def _print_trades(trades: List[Trade]) -> None:
    if not trades:
        print("\nNo trades were taken.")
        return
    print("\n date       time   model         dir   entry     stop    target"
          "   exit      R     result")
    print(" " + "-" * 86)
    for t in trades:
        s = t.signal
        print(f" {t.entry_time:%Y-%m-%d %H:%M}  {s.model:<12} "
              f"{s.direction.name:<5} {s.entry:8.2f} {s.stop:8.2f} "
              f"{s.target:8.2f} {('-' if t.exit_price is None else f'{t.exit_price:8.2f}')} "
              f"{t.r_multiple:+5.2f}  {t.exit_reason.value if t.exit_reason else '-'}")


def _run_backtest(cfg: Config, candles: List[Candle], show_trades: bool) -> int:
    if not candles:
        print("No candles to back-test.", file=sys.stderr)
        return 1
    result = Backtester(cfg).run(candles)
    print(result.summary())
    if show_trades:
        _print_trades(result.trades)
    return 0


def _paper_stream(cfg: Config, candles: List[Candle]) -> int:
    """Replay candles one at a time through the live runner (paper mode)."""
    from .live import LiveRunner
    runner = LiveRunner(cfg)
    for c in candles:
        for msg in runner.on_candle(c):
            print(msg)
    print("\nPaper session complete.")
    print(runner.result_summary())
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="ictbot",
                                description="ICT 9:30 Open Model trading system")
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--symbol", default="NQ", help="instrument (NQ/ES/MNQ/MES)")

    g = sub.add_parser("gen", parents=[common], help="generate synthetic data")
    g.add_argument("--days", type=int, default=40)
    g.add_argument("--seed", type=int, default=7)
    g.add_argument("--out", required=True)

    d = sub.add_parser("demo", parents=[common],
                       help="generate data and back-test in one step")
    d.add_argument("--days", type=int, default=60)
    d.add_argument("--seed", type=int, default=7)
    d.add_argument("--no-trades", action="store_true", help="hide trade list")

    b = sub.add_parser("backtest", parents=[common], help="back-test a CSV")
    b.add_argument("--csv", required=True)
    b.add_argument("--no-trades", action="store_true")

    pp = sub.add_parser("paper", parents=[common],
                        help="paper-trade by streaming a CSV")
    pp.add_argument("--csv", required=True)

    args = p.parse_args(argv)
    cfg = Config.for_symbol(args.symbol)
    tick = cfg.instrument.tick_size

    if args.cmd == "gen":
        candles = generate_sessions(args.days, args.seed,
            start_price=DEFAULT_START_PRICE.get(args.symbol.upper(), 18_000.0),
            tick=tick)
        write_csv(args.out, candles)
        print(f"Wrote {len(candles)} candles -> {args.out}")
        return 0

    if args.cmd == "demo":
        candles = generate_sessions(args.days, args.seed,
            start_price=DEFAULT_START_PRICE.get(args.symbol.upper(), 18_000.0),
            tick=tick)
        return _run_backtest(cfg, candles, show_trades=not args.no_trades)

    if args.cmd == "backtest":
        candles = load_csv(args.csv)
        return _run_backtest(cfg, candles, show_trades=not args.no_trades)

    if args.cmd == "paper":
        candles = load_csv(args.csv)
        return _paper_stream(cfg, candles)

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
