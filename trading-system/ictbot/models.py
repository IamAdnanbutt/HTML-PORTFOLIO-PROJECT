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


def round_to_tick(price: float, tick: float) -> float:
    """Snap a price to the nearest valid tick.

    Entry/stop/target prices are derived from FVG midpoints and SD projections,
    which can land on a fraction of a tick (e.g. an 0.125 entry on a 0.25-tick
    instrument). A real futures broker rejects such limit prices, and a
    back-test that fills at them reports prices that could never have traded, so
    every order price is snapped to the instrument's tick before use.
    """
    if tick <= 0:
        return price
    return round(round(price / tick) * tick, 10)


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


@dataclass
class OrderBlock:
    """ICT Order Block — last opposing candle before a displacement.

    Bullish OB: last bearish (down-close) candle before a bullish displacement.
    Bearish OB: last bullish (up-close) candle before a bearish displacement.

    MTH (Mean Threshold) = 50% of the body (open-to-close).  Notes say
    "candle bodies respect mean threshold [reaction off opening price]".
    Wicks may trade through the OB but body-closes through MTH invalidate it.
    """

    direction: Direction          # LONG = bullish OB; SHORT = bearish OB
    top: float                    # higher of open/close
    bottom: float                 # lower of open/close
    wick_high: float              # full candle high (including wick)
    wick_low: float               # full candle low (including wick)
    created_at: datetime
    index: int
    mitigated: bool = False
    mitigated_at: Optional[datetime] = None

    @property
    def mth(self) -> float:
        return (self.top + self.bottom) / 2.0

    @property
    def size(self) -> float:
        return self.top - self.bottom

    def contains(self, price: float) -> bool:
        return self.bottom <= price <= self.top

    def body_closed_through(self, candle: "Candle") -> bool:
        if self.direction is Direction.LONG:
            return candle.body_low < self.bottom
        return candle.body_high > self.top


@dataclass
class RejectionBlock:
    """ICT Rejection Block — a significant-wick candle used as a PD array.

    Bullish RB: down-close candle with a long lower wick; buyers rejected lows.
    Bearish RB: up-close candle with a long upper wick; sellers rejected highs.

    CE (Consequent Encroachment) = 50% of the wick range (wick tip to body edge).
    Notes: "CE = 50% wick range".  Price bodies should respect CE; a body-close
    through CE makes the swing point "vulnerable".
    """

    direction: Direction
    wick_tip: float               # extreme wick: low (bull RB) or high (bear RB)
    body_edge: float              # body edge near the wick: body_low (bull) or body_high (bear)
    created_at: datetime
    index: int
    mitigated: bool = False

    @property
    def ce(self) -> float:
        return (self.wick_tip + self.body_edge) / 2.0

    @property
    def wick_size(self) -> float:
        return abs(self.body_edge - self.wick_tip)


@dataclass
class BreakerBlock:
    """ICT Breaker Block — a failed OB swept and flipped to the opposite role.

    When price closes THROUGH an OB body, that OB converts to a Breaker:
      Bullish Breaker: was a bearish OB, price closed above it → now support.
      Bearish Breaker: was a bullish OB, price closed below it → now resistance.

    Notes: "up close candle = bullish, down close candle = bearish"
    "candle close through breaker block [change in state of delivery]"
    CE = 50% of the breaker body range.
    """

    direction: Direction          # LONG = bullish breaker (support)
    top: float
    bottom: float
    created_at: datetime
    index: int                    # bar that closed through the original OB
    mitigated: bool = False

    @property
    def ce(self) -> float:
        return (self.top + self.bottom) / 2.0

    @property
    def size(self) -> float:
        return self.top - self.bottom

    def contains(self, price: float) -> bool:
        return self.bottom <= price <= self.top


@dataclass
class OpeningRange:
    """A 30-minute session Opening Range that anchors the session's narrative.

    The algorithm establishes three key reference points during the window:
    Opening Price, Session High, Session Low.  These become the blueprint for
    how price delivers for the rest of that session.

    Standard Deviation (SD) targets are projected from the OR extremes:
      Bullish: or_high + sd_mult * range   (0.5, 1.0, 1.5, 2.5 multiples)
      Bearish: or_low  - sd_mult * range   (0.5, 1.0, 1.5, 2.5 multiples)

    The quadrant / label system (from the charts):
      0    = OR Low
      0.25 = lower quadrant
      0.5  = Midpoint / CE (Consequent Encroachment)
      0.75 = upper quadrant
      1    = OR High
      1.5 … = bullish SD extensions above
      -0.5 … = bearish SD extensions below
    """

    label: str                    # "Midnight", "London", "NYKillZone", "AM", "PM"
    or_open: float                # opening price of the first candle in the window
    or_high: float
    or_low: float
    formed_at: datetime           # timestamp of the last OR candle
    first_pfvg: Optional["FVG"] = None   # first FVG formed inside the OR window

    @property
    def range(self) -> float:
        return self.or_high - self.or_low

    @property
    def midpoint(self) -> float:
        return (self.or_high + self.or_low) / 2.0

    @property
    def upper_quadrant(self) -> float:
        return self.or_low + 0.75 * self.range

    @property
    def lower_quadrant(self) -> float:
        return self.or_low + 0.25 * self.range

    def sd_target(self, direction: Direction, mult: float) -> float:
        """Project a standard-deviation target from the OR extreme.

        LONG: above OR_HIGH = or_high + mult * range
        SHORT: below OR_LOW = or_low  - mult * range
        """
        if direction is Direction.LONG:
            return self.or_high + mult * self.range
        return self.or_low - mult * self.range

    def normalized_level(self, price: float) -> float:
        """Express price as a normalized OR level (0=low, 1=high)."""
        if self.range == 0:
            return 0.0
        return (price - self.or_low) / self.range


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
