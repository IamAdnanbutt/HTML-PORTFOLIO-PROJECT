"""Command-line interface for ictbot — all 5 ICT systems.

Commands
--------
    # ── Cat 1: 9:30 AM Open Model ──────────────────────────────────────────
    python -m ictbot demo --symbol NQ --days 60          # synthetic backtest
    python -m ictbot backtest --symbol NQ --csv data/my_data.csv
    python -m ictbot paper --symbol NQ --csv data/sample_NQ_1m.csv

    # ── All 5 systems in one run ────────────────────────────────────────────
    python -m ictbot multidemo --symbol NQ --days 60     # synthetic multi-system
    python -m ictbot multibacktest --symbol NQ --csv data/my_data.csv

    # ── Utilities ───────────────────────────────────────────────────────────
    python -m ictbot gen --days 60 --out data/sample_NQ_1m.csv
"""

from __future__ import annotations

import argparse
import sys
from typing import List

from .backtest import Backtester
from .config import Config
from .data import generate_sessions, load_csv, write_csv
from .models import Candle, Trade


# Representative starting prices so synthetic data is realistically scaled.
DEFAULT_START_PRICE = {"NQ": 18_000.0, "MNQ": 18_000.0,
                       "ES": 5_000.0, "MES": 5_000.0}


def _start_price(symbol: str) -> float:
    return DEFAULT_START_PRICE.get(symbol.upper(), 18_000.0)


# ------------------------------------------------------------------- helpers
def _print_trades(trades: List[Trade]) -> None:
    if not trades:
        print("\n  No trades were taken.")
        return
    print("\n date       time   system               dir   entry     stop    target"
          "   exit      R     result")
    print(" " + "-" * 96)
    for t in trades:
        s = t.signal
        sys_tag = t.tags.get("system", s.model)
        print(f" {t.entry_time:%Y-%m-%d %H:%M}  {sys_tag:<20} "
              f"{s.direction.name:<5} {s.entry:8.2f} {s.stop:8.2f} "
              f"{s.target:8.2f} "
              f"{('-' if t.exit_price is None else f'{t.exit_price:8.2f}')} "
              f"{t.r_multiple:+5.2f}  "
              f"{t.exit_reason.value if t.exit_reason else '-'}")


def _run_single_backtest(cfg: Config, candles: List[Candle],
                         show_trades: bool) -> int:
    if not candles:
        print("No candles to back-test.", file=sys.stderr)
        return 1
    result = Backtester(cfg).run(candles)
    print(result.summary())
    if show_trades:
        _print_trades(result.trades)
    return 0


def _run_multi_backtest(cfg: Config, candles: List[Candle],
                        show_trades: bool) -> int:
    if not candles:
        print("No candles to back-test.", file=sys.stderr)
        return 1
    from .runner import MultiSystemRunner
    runner = MultiSystemRunner(cfg)
    result = runner.run(candles)
    print(result.summary())
    if show_trades:
        all_trades = sorted(
            [t for sys in result.systems.values() for t in sys.trades],
            key=lambda t: t.entry_time,
        )
        _print_trades(all_trades)
    return 0


def _paper_stream(cfg: Config, candles: List[Candle]) -> int:
    from .live import LiveRunner
    runner = LiveRunner(cfg)
    for c in candles:
        for msg in runner.on_candle(c):
            print(msg)
    for msg in runner.finalize():   # close any position still open at stream end
        print(msg)
    print("\nPaper session complete.")
    print(runner.result_summary())
    return 0


# ------------------------------------------------------------------ CLI
def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="ictbot",
                                description="ICT automated trading systems")
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--symbol", default="NQ",
                        help="instrument (NQ / ES / MNQ / MES)")

    # ---- gen ----
    g = sub.add_parser("gen", parents=[common],
                       help="generate and save synthetic 1-min data")
    g.add_argument("--days", type=int, default=40)
    g.add_argument("--seed", type=int, default=7)
    g.add_argument("--out", required=True)
    g.add_argument("--full-day", action="store_true",
                   help="include overnight + Asia windows (all 5 systems)")

    # ---- Cat 1 single-system commands ----
    d = sub.add_parser("demo", parents=[common],
                       help="[Cat1] generate + backtest 9:30 Open Model")
    d.add_argument("--days", type=int, default=60)
    d.add_argument("--seed", type=int, default=7)
    d.add_argument("--no-trades", action="store_true")

    b = sub.add_parser("backtest", parents=[common],
                       help="[Cat1] backtest 9:30 Open Model against a CSV")
    b.add_argument("--csv", required=True)
    b.add_argument("--no-trades", action="store_true")

    pp = sub.add_parser("paper", parents=[common],
                        help="[Cat1] paper-trade 9:30 Open Model from a CSV")
    pp.add_argument("--csv", required=True)

    # ---- All-5-system commands ----
    md = sub.add_parser("multidemo", parents=[common],
                        help="[ALL 5] generate + backtest all systems")
    md.add_argument("--days", type=int, default=60)
    md.add_argument("--seed", type=int, default=7)
    md.add_argument("--no-trades", action="store_true")

    mb = sub.add_parser("multibacktest", parents=[common],
                        help="[ALL 5] backtest all systems against a CSV")
    mb.add_argument("--csv", required=True)
    mb.add_argument("--no-trades", action="store_true")

    args = p.parse_args(argv)
    cfg = Config.for_symbol(args.symbol)
    tick = cfg.instrument.tick_size

    if args.cmd == "gen":
        candles = generate_sessions(args.days, args.seed,
                                    start_price=_start_price(args.symbol),
                                    tick=tick, full_day=args.full_day)
        write_csv(args.out, candles)
        print(f"Wrote {len(candles)} candles -> {args.out}")
        return 0

    if args.cmd == "demo":
        candles = generate_sessions(args.days, args.seed,
                                    start_price=_start_price(args.symbol),
                                    tick=tick)
        return _run_single_backtest(cfg, candles, not args.no_trades)

    if args.cmd == "backtest":
        return _run_single_backtest(cfg, load_csv(args.csv), not args.no_trades)

    if args.cmd == "paper":
        return _paper_stream(cfg, load_csv(args.csv))

    if args.cmd == "multidemo":
        # Full-day data so the overnight + Asia systems get their windows.
        candles = generate_sessions(args.days, args.seed,
                                    start_price=_start_price(args.symbol),
                                    tick=tick, full_day=True)
        return _run_multi_backtest(cfg, candles, not args.no_trades)

    if args.cmd == "multibacktest":
        return _run_multi_backtest(cfg, load_csv(args.csv), not args.no_trades)

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
