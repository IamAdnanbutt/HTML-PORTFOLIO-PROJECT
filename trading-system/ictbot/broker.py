"""Broker abstraction.

A real deployment would implement :class:`Broker` against a live API (e.g. a
futures broker's REST/FIX gateway). The :class:`PaperBroker` fills orders
against the candle stream so the exact same strategy code can run in backtests,
paper trading and (with a real adapter) live trading.

NOTE: This project ships only a paper broker. Wiring a live broker means money
is at risk - do that deliberately, with your own credentials and your own
risk checks, never on synthetic data.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from .config import Instrument, RiskConfig
from .models import Candle, Direction, ExitReason, Signal, Trade


def position_size(signal: Signal, instrument: Instrument,
                  risk: RiskConfig) -> float:
    """Size a position so a stop-out loses ~``risk_per_trade`` of the account.

    Returns whole units for futures (contracts) and equities (shares), and a
    fractional quantity for crypto. Equity and crypto sizes are additionally
    capped so a single position's notional stays under ``max_notional``.
    """
    risk_dollars = risk.account_size * risk.risk_per_trade
    stop_points = abs(signal.entry - signal.stop)
    if stop_points <= 0:
        return 0
    per_unit_loss = stop_points * instrument.point_value
    if per_unit_loss <= 0:
        return 0
    raw = risk_dollars / per_unit_loss

    if instrument.asset_class == "crypto":
        # Fractional units, capped by the per-position notional ceiling.
        size = raw
        if signal.entry > 0:
            size = min(size, risk.max_notional / signal.entry)
        return round(max(0.0, size), 6)

    # Whole units: futures contracts or equity shares.
    cap = risk.max_shares if instrument.asset_class == "equity" \
        else risk.max_contracts
    n = int(raw)
    if instrument.asset_class == "equity" and signal.entry > 0:
        n = min(n, int(risk.max_notional / signal.entry))
    return max(0, min(n, cap))


class Broker(ABC):
    """Minimal broker interface used by the strategy runner."""

    @abstractmethod
    def submit(self, signal: Signal, size: int) -> Trade: ...

    @abstractmethod
    def update(self, candle: Candle) -> List[Trade]:
        """Feed a new bar; return any trades that closed on this bar."""

    @abstractmethod
    def flatten(self, candle: Candle, reason: ExitReason) -> List[Trade]: ...


class PaperBroker(Broker):
    """Fills bracket orders (entry already touched, stop + target) on bars.

    The strategy emits a signal only once price has *touched* the entry, so we
    treat the position as filled at the signal's entry price on the signal bar.
    From then on each incoming bar is checked against stop and target. If a bar
    straddles both levels we conservatively assume the stop hit first.
    """

    def __init__(self, instrument: Instrument):
        self.instrument = instrument
        self.open_trades: List[Trade] = []
        self.closed_trades: List[Trade] = []

    def submit(self, signal: Signal, size: int) -> Trade:
        trade = Trade(
            signal=signal,
            entry_time=signal.time,
            entry_price=signal.entry,
            size=size,
            fees=self.instrument.commission * size,
        )
        self.open_trades.append(trade)
        return trade

    def update(self, candle: Candle) -> List[Trade]:
        just_closed: List[Trade] = []
        still_open: List[Trade] = []
        for tr in self.open_trades:
            # Don't evaluate the entry bar itself for an exit.
            if candle.time <= tr.entry_time:
                still_open.append(tr)
                continue
            exit_price, reason = self._check_exit(tr, candle)
            if reason is not None:
                tr.exit_time = candle.time
                tr.exit_price = exit_price
                tr.exit_reason = reason
                tr.fees += self.instrument.commission * tr.size
                self.closed_trades.append(tr)
                just_closed.append(tr)
            else:
                still_open.append(tr)
        self.open_trades = still_open
        return just_closed

    def _check_exit(self, tr: Trade, candle: Candle):
        sig = tr.signal
        if sig.direction is Direction.LONG:
            hit_stop = candle.low <= sig.stop
            hit_target = candle.high >= sig.target
            if hit_stop:                       # stop-first when ambiguous
                return sig.stop, ExitReason.STOP
            if hit_target:
                return sig.target, ExitReason.TARGET
        else:
            hit_stop = candle.high >= sig.stop
            hit_target = candle.low <= sig.target
            if hit_stop:
                return sig.stop, ExitReason.STOP
            if hit_target:
                return sig.target, ExitReason.TARGET
        return None, None

    def flatten(self, candle: Candle, reason: ExitReason) -> List[Trade]:
        closed: List[Trade] = []
        for tr in self.open_trades:
            tr.exit_time = candle.time
            tr.exit_price = candle.close
            tr.exit_reason = reason
            tr.fees += self.instrument.commission * tr.size
            self.closed_trades.append(tr)
            closed.append(tr)
        self.open_trades = []
        return closed
