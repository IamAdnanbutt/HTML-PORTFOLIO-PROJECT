"""Live / paper trading loop.

:class:`LiveRunner` drives the exact same :class:`NineThirtyOpenModel` and
:class:`PaperBroker` used in back-testing, but bar-by-bar, so it can sit behind
a real-time feed. It manages session rollover: at the first bar of a new day it
rebuilds the liquidity pools and recomputes the HTF bias from the rolling
history it has accumulated.

To go truly live you would:
  1. replace the candle source with your broker/data-feed websocket, and
  2. swap :class:`PaperBroker` for a real :class:`Broker` implementation that
     places actual bracket orders.
Everything else - the signal logic and risk sizing - stays identical.
"""

from __future__ import annotations

from datetime import date
from typing import List, Optional

from .bias import htf_bias
from .broker import PaperBroker, position_size
from .config import Config
from .liquidity import build_liquidity_pools
from .models import Candle, ExitReason, Trade
from .strategy import NineThirtyOpenModel


class LiveRunner:
    def __init__(self, config: Config):
        self.cfg = config
        self.model = NineThirtyOpenModel(config)
        self.broker = PaperBroker(config.instrument)
        self.history: List[Candle] = []
        self.closed: List[Trade] = []
        self._cur_day: Optional[date] = None
        self._trades_today = 0
        self._day_premarket: List[Candle] = []
        self._prev_day: List[Candle] = []
        self._prev_session: List[Candle] = []
        self._today: List[Candle] = []

    def on_candle(self, candle: Candle) -> List[str]:
        msgs: List[str] = []
        s = self.cfg.session

        # ---- session rollover -------------------------------------------
        if candle.time.date() != self._cur_day:
            # flatten anything left from yesterday at the previous close
            if self._today:
                for tr in self.broker.flatten(self._today[-1],
                                              ExitReason.SESSION_CLOSE):
                    self.closed.append(tr)
                    msgs.append(self._fmt_exit(tr))
            self._prev_day = self._today
            self._prev_session = [c for c in self._prev_day
                                  if s.open_time <= c.time.time() <= s.close_time]
            self._cur_day = candle.time.date()
            self._today = []
            self._day_premarket = []
            self._trades_today = 0
            # session is primed at the open (below), once pre-market is known

        self.history.append(candle)
        self._today.append(candle)
        t = candle.time.time()

        if t < s.open_time:
            self._day_premarket.append(candle)

        # Prime the model exactly once, on the first bar at/after the open.
        if t >= s.open_time and self.model.session_date != self._cur_day:
            pools = build_liquidity_pools(self._day_premarket,
                                          self._prev_session, self._prev_day)
            bias = htf_bias(self.history, candle.time,
                            swing_lookback=self.cfg.strategy.swing_lookback)
            self.model.start_session(self._cur_day, pools, bias)
            if bias is not None:
                msgs.append(f"[{candle.time:%Y-%m-%d}] open: HTF bias = "
                            f"{bias.name}; armed.")

        # ---- manage open positions on this bar --------------------------
        for tr in self.broker.update(candle):
            self.closed.append(tr)
            msgs.append(self._fmt_exit(tr))

        # ---- ask for a new signal ---------------------------------------
        if self._trades_today < self.cfg.risk.max_trades_per_day:
            signal = self.model.on_candle(candle)
            if signal is not None:
                size = position_size(signal, self.cfg.instrument, self.cfg.risk)
                if size > 0:
                    self.broker.submit(signal, size)
                    self._trades_today += 1
                    msgs.append(
                        f"[{candle.time:%Y-%m-%d %H:%M}] ENTRY {signal.model} "
                        f"{signal.direction.name} {size}x @ {signal.entry:.2f} "
                        f"stop {signal.stop:.2f} target {signal.target:.2f} "
                        f"(RR {signal.rr:.1f})  -> {signal.reason}")
        return msgs

    def _fmt_exit(self, tr: Trade) -> str:
        return (f"[{tr.exit_time:%Y-%m-%d %H:%M}] EXIT  {tr.signal.model} "
                f"{tr.direction.name} @ {tr.exit_price:.2f} "
                f"({tr.exit_reason.value}) {tr.r_multiple:+.2f}R")

    def result_summary(self) -> str:
        from .backtest import BacktestResult
        res = BacktestResult(trades=self.closed, config=self.cfg)
        return res.summary()
