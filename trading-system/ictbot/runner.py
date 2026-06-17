"""Unified multi-system runner.

Feeds every closed candle through all active trading systems in parallel and
collects signals/trades from each.  This lets you compare systems on the same
data or run them together in paper / live mode.

Systems:
  Cat1: NineThirtyOpenModel         (9:30 AM Open Model)
  Cat2: SetupForLife                (ICT 2024 "The Setup For Life")
  Cat4: OpeningRangeModel × 5      (Midnight / London / NYKillZone / AM / PM)
  Cat5: AsiaKillzoneModel           (Asia KZ NDOG / NWOG)

(Cat3 PD Arrays is the shared entry toolkit, not a standalone signal emitter.)
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from typing import Dict, List, Optional

from .asia_killzone import ASIA_OPEN, AsiaKillzoneModel
from .bias import htf_bias
from .broker import PaperBroker, position_size
from .config import Config
from .liquidity import build_liquidity_pools, group_by_day
from .model_2024 import SetupForLife
from .models import Candle, Direction, ExitReason, Signal, Trade
from .opening_range import OR_DEFINITIONS, build_all_or_models
from .strategy import NineThirtyOpenModel


@dataclass
class SystemResult:
    """Per-system trade log and summary stats."""

    name: str
    trades: List[Trade] = field(default_factory=list)

    @property
    def n_trades(self) -> int:
        return len(self.trades)

    @property
    def win_rate(self) -> float:
        wins = [t for t in self.trades if t.r_multiple > 0]
        return len(wins) / self.n_trades if self.n_trades else 0.0

    @property
    def total_r(self) -> float:
        return sum(t.r_multiple for t in self.trades)

    @property
    def expectancy(self) -> float:
        return self.total_r / self.n_trades if self.n_trades else 0.0

    def summary_line(self) -> str:
        return (f"  {self.name:<20}: {self.n_trades:3d} trades  "
                f"WR {self.win_rate * 100:4.0f}%  "
                f"{self.total_r:+.2f}R  "
                f"E={self.expectancy:+.2f}R")


@dataclass
class MultiSystemResult:
    systems: Dict[str, SystemResult] = field(default_factory=dict)
    config: Optional[Config] = None

    def summary(self) -> str:
        lines = [
            "=" * 66,
            " ICT Multi-System Backtest Summary",
            f" Instrument: {self.config.instrument.symbol if self.config else '?'}",
            "=" * 66,
        ]
        total_trades = sum(s.n_trades for s in self.systems.values())
        for name, sys_res in self.systems.items():
            if sys_res.n_trades:
                lines.append(sys_res.summary_line())
        lines.append("-" * 66)
        lines.append(f"  {'TOTAL':<20}: {total_trades:3d} trades combined")
        lines.append("=" * 66)
        return "\n".join(lines)


class MultiSystemRunner:
    """Run all systems in parallel on the same candle stream."""

    def __init__(self, config: Config):
        self.cfg = config
        self.cat1 = NineThirtyOpenModel(config)
        self.cat2 = SetupForLife(config)
        self.cat4 = build_all_or_models(config)
        self.cat5 = AsiaKillzoneModel(config)
        self.broker = PaperBroker(config.instrument)
        self.result = MultiSystemResult(config=config)
        for name in (["Cat1_930Open", "Cat2_SetupForLife"] +
                     [f"Cat4_OR_{n}" for n in self.cat4] +
                     ["Cat5_AsiaKZ"]):
            self.result.systems[name] = SystemResult(name)

    def run(self, candles: List[Candle]) -> MultiSystemResult:
        candles = sorted(candles, key=lambda c: c.time)
        self._candles = candles
        self._times = [c.time for c in candles]
        by_day = group_by_day(candles)
        days = sorted(by_day)
        s = self.cfg.session

        for i, day in enumerate(days):
            day_candles = by_day[day]
            session = [c for c in day_candles
                       if s.open_time <= c.time.time() <= s.close_time]
            premarket = [c for c in day_candles
                         if s.premarket_start <= c.time.time() < s.open_time]
            prev_day = by_day[days[i - 1]] if i >= 1 else []
            prev_session = [c for c in prev_day
                            if s.open_time <= c.time.time() <= s.close_time] if prev_day else []

            if not session:
                continue

            pools = build_liquidity_pools(premarket, prev_session, prev_day)

            # HTF bias as of the RTH open (Cat1 / Cat2 / AM&PM ORs).
            open_dt = next((c.time for c in session
                            if c.time.time() >= s.open_time), session[0].time)
            rth_bias = self._bias_as_of(open_dt)

            # Per-window bias for every Opening Range, computed as of the moment
            # that window closes (no look-ahead — entries come later still).
            or_biases: Dict[str, Optional[Direction]] = {}
            for name, win in OR_DEFINITIONS.items():
                or_biases[name] = self._bias_as_of(
                    datetime.combine(day, win.end))

            # Asia KZ bias as of the 6 PM open; NDOG anchors on *today's* RTH
            # close (the gap the evening session opens against).
            asia_bias = self._bias_as_of(datetime.combine(day, ASIA_OPEN))
            rth_close = session[-1].close

            # Arm each system for the new session.
            self.cat1.start_session(day, pools, rth_bias)
            self.cat2.start_session(pools, rth_bias)
            for or_model in self.cat4.values():
                or_model._reset()
            self.cat5.start_session(prev_4pm_close=rth_close, htf_bias=asia_bias)

            # RTH-only running range feeds Cat2's premium/discount read.
            rth_high = float("-inf")
            rth_low = float("inf")
            trades_today: Dict[str, int] = {k: 0 for k in self.result.systems}

            # Feed the *entire* calendar day so the overnight and early-morning
            # systems see their windows; each model self-gates on its own clock.
            # Trades are flattened at session boundaries (early -> RTH ->
            # evening) so a position never bleeds across a price discontinuity.
            prev_seg: Optional[str] = None
            prev_candle: Optional[Candle] = None
            for candle in day_candles:
                t = candle.time.time()
                seg = self._segment(t)
                if prev_seg is not None and seg != prev_seg and prev_candle:
                    for trade in self.broker.flatten(prev_candle,
                                                     ExitReason.SESSION_CLOSE):
                        self._record(trade)
                prev_seg, prev_candle = seg, candle

                if s.premarket_start <= t <= s.close_time:
                    rth_high = max(rth_high, candle.high)
                    rth_low = min(rth_low, candle.low)

                # Manage open positions.
                for trade in self.broker.update(candle):
                    self._record(trade)

                max_t = self.cfg.risk.max_trades_per_day
                in_rth = s.premarket_start <= t <= s.close_time

                # Cat1: 9:30 Open Model (RTH only).
                if in_rth and trades_today["Cat1_930Open"] < max_t:
                    sig = self.cat1.on_candle(candle)
                    if sig:
                        self._submit(sig, "Cat1_930Open", trades_today)

                # Cat2: Setup For Life (RTH only).
                if in_rth and trades_today["Cat2_SetupForLife"] < max_t:
                    sig = self.cat2.on_candle(candle, rth_high, rth_low)
                    if sig:
                        self._submit(sig, "Cat2_SetupForLife", trades_today)

                # Cat4: the 5 Opening Ranges, each with its own window bias.
                for name, or_model in self.cat4.items():
                    sys_key = f"Cat4_OR_{name}"
                    if trades_today[sys_key] < max_t:
                        sig = or_model.on_candle(candle, or_biases[name])
                        if sig:
                            self._submit(sig, sys_key, trades_today)

                # Cat5: Asia KZ (only acts on the 6 PM-9:30 PM evening bars).
                if trades_today["Cat5_AsiaKZ"] < max_t:
                    sig = self.cat5.on_candle(candle)
                    if sig:
                        self._submit(sig, "Cat5_AsiaKZ", trades_today)

            # Flatten anything still open at the end of the calendar day.
            if day_candles:
                for trade in self.broker.flatten(day_candles[-1],
                                                 ExitReason.SESSION_CLOSE):
                    self._record(trade)

        return self.result

    def _segment(self, t: time) -> str:
        """Classify a clock time into a session segment for boundary flushing."""
        s = self.cfg.session
        if t < s.premarket_start:
            return "early"        # overnight + early-morning Opening Ranges
        if t <= s.close_time:
            return "rth"          # regular trading hours
        return "evening"          # Asia Killzone

    def _bias_as_of(self, as_of: datetime) -> Optional[Direction]:
        """HTF bias using only candles strictly before ``as_of``.

        Uses ``bisect`` over the pre-sorted candle times so each call slices the
        relevant ~72-hour window in O(log n) rather than rescanning the stream.
        """
        hi = bisect.bisect_left(self._times, as_of)
        lo = bisect.bisect_left(self._times, as_of - timedelta(hours=72))
        return htf_bias(self._candles[lo:hi], as_of,
                        swing_lookback=self.cfg.strategy.swing_lookback)

    def _record(self, trade: Trade) -> None:
        tag = trade.tags.get("system", "unknown")
        if tag in self.result.systems:
            self.result.systems[tag].trades.append(trade)

    def _submit(self, sig: Signal, sys_key: str,
                trades_today: Dict[str, int]) -> None:
        size = position_size(sig, self.cfg.instrument, self.cfg.risk)
        if size > 0:
            trade = self.broker.submit(sig, size)
            trade.tags["system"] = sys_key
            trades_today[sys_key] += 1
