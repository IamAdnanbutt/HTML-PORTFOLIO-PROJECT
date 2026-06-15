"""Plain data structures shared across the system.

These are deliberately dumb containers (no behaviour beyond a couple of
convenience properties) so they are easy to construct in tests and to serialise.
All prices are floats; all timestamps are timezone-naive ``datetime`` objects
that the rest of the system treats as New York / Eastern time (the reference
clock for the ICT session model).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class Direction(Enum):
    """Trade / bias direction."""

    LONG = 1
    SHORT = -1

    @property
    def sign(self) -> int:
        return self.value

    def opposite(self) -> "Direction":
        return Direction.SHORT if self is Direction.LONG else Direction.LONG


@dataclass(frozen=True)
class Candle:
    """A single OHLCV bar."""

    time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

    @property
    def is_up(self) -> bool:
        return self.close >= self.open

    @property
    def body_high(self) -> float:
        return max(self.open, self.close)

    @property
    def body_low(self) -> float:
        return min(self.open, self.close)

    @property
    def range(self) -> float:
        return self.high - self.low


@dataclass
class FVG:
    """A Fair Value Gap (3-candle imbalance) - a PD Array used for entries.

    For a bullish FVG the gap sits between ``low`` and ``high`` where
    ``low`` is candle-1's high and ``high`` is candle-3's low. A bearish FVG is
    the mirror image. ``mid`` (consequent encroachment) is the 50% level.
    """

    direction: Direction
    top: float
    bottom: float
    created_at: datetime
    # index of the middle (displacement) candle within the session series
    index: int
    filled: bool = False

    @property
    def mid(self) -> float:
        return (self.top + self.bottom) / 2.0

    @property
    def size(self) -> float:
        return self.top - self.bottom

    def contains(self, price: float) -> bool:
        return self.bottom <= price <= self.top


@dataclass
class LiquidityPool:
    """Resting liquidity - a price level where stops are expected to cluster."""

    name: str           # e.g. "PML", "PSH", "PDH"
    price: float
    side: Direction     # LONG  => buyside (highs)   SHORT => sellside (lows)
    created_at: Optional[datetime] = None
    swept: bool = False
    swept_at: Optional[datetime] = None

    @property
    def is_buyside(self) -> bool:
        return self.side is Direction.LONG


@dataclass
class Signal:
    """A fully-formed trade idea emitted by the strategy."""

    time: datetime
    direction: Direction
    entry: float
    stop: float
    target: float
    model: str                      # "Reversal" or "Continuation"
    reason: str = ""
    pd_array: Optional[FVG] = None
    target_pool: Optional[LiquidityPool] = None

    @property
    def risk(self) -> float:
        return abs(self.entry - self.stop)

    @property
    def reward(self) -> float:
        return abs(self.target - self.entry)

    @property
    def rr(self) -> float:
        return self.reward / self.risk if self.risk else 0.0


class ExitReason(Enum):
    TARGET = "target"
    STOP = "stop"
    SESSION_CLOSE = "session_close"


@dataclass
class Trade:
    """A signal that has been filled and (eventually) closed."""

    signal: Signal
    entry_time: datetime
    entry_price: float
    size: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[ExitReason] = None
    fees: float = 0.0
    tags: dict = field(default_factory=dict)

    @property
    def direction(self) -> Direction:
        return self.signal.direction

    @property
    def is_open(self) -> bool:
        return self.exit_price is None

    @property
    def pnl_points(self) -> float:
        if self.exit_price is None:
            return 0.0
        return (self.exit_price - self.entry_price) * self.direction.sign

    @property
    def r_multiple(self) -> float:
        """Profit/loss expressed in units of initial risk (R)."""
        risk = abs(self.entry_price - self.signal.stop)
        if not risk:
            return 0.0
        return self.pnl_points / risk
