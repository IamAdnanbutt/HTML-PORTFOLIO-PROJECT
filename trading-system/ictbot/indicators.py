"""Price-action primitives used to read 'delivery': FVGs, swings, displacement.

These functions are pure and operate on plain lists of :class:`Candle`, which
keeps them trivial to unit-test and free of any third-party dependency.
"""

from __future__ import annotations

from typing import List, Optional

from .models import Candle, Direction, FVG


def true_range(prev: Candle, cur: Candle) -> float:
    return max(
        cur.high - cur.low,
        abs(cur.high - prev.close),
        abs(cur.low - prev.close),
    )


def atr(candles: List[Candle], period: int) -> float:
    """Simple average true range over the last ``period`` bars.

    Returns 0.0 if there is not enough data yet.
    """
    if len(candles) < 2:
        return 0.0
    trs = [true_range(candles[i - 1], candles[i])
           for i in range(1, len(candles))]
    window = trs[-period:]
    return sum(window) / len(window) if window else 0.0


def find_fvg(candles: List[Candle], index: int) -> Optional[FVG]:
    """Detect a 3-candle Fair Value Gap ending at ``index``.

    Bullish FVG: candle[index].low > candle[index-2].high  (gap left unfilled
    by the displacement candle in the middle).
    Bearish FVG: candle[index].high < candle[index-2].low.

    ``index`` must point at the third candle of the pattern.
    """
    if index < 2 or index >= len(candles):
        return None
    c1, _c2, c3 = candles[index - 2], candles[index - 1], candles[index]

    # Bullish imbalance: third candle's low sits above first candle's high.
    if c3.low > c1.high:
        return FVG(Direction.LONG, top=c3.low, bottom=c1.high,
                   created_at=c3.time, index=index)
    # Bearish imbalance: third candle's high sits below first candle's low.
    if c3.high < c1.low:
        return FVG(Direction.SHORT, top=c1.low, bottom=c3.high,
                   created_at=c3.time, index=index)
    return None


def is_displacement(candles: List[Candle], index: int,
                    atr_mult: float, atr_period: int) -> bool:
    """True if the bar at ``index`` is an expansion (displacement) candle.

    Displacement is the energetic move that creates an imbalance and signals
    intent. We require the bar's range to exceed ``atr_mult`` x ATR and the bar
    to close in the top/bottom third of its range (a decisive close).
    """
    if index < 1 or index >= len(candles):
        return False
    a = atr(candles[: index], atr_period)
    if a <= 0:
        return False
    cur = candles[index]
    if cur.range < atr_mult * a:
        return False
    # decisive close: in the upper third (up) or lower third (down) of range
    if cur.range == 0:
        return False
    close_pos = (cur.close - cur.low) / cur.range
    return close_pos >= 0.66 or close_pos <= 0.34


def swing_highs(candles: List[Candle], lookback: int) -> List[int]:
    """Indices of confirmed swing-high pivots (fractal of width ``lookback``)."""
    out: List[int] = []
    n = len(candles)
    for i in range(lookback, n - lookback):
        h = candles[i].high
        if all(candles[i].high > candles[i - j].high for j in range(1, lookback + 1)) and \
           all(candles[i].high > candles[i + j].high for j in range(1, lookback + 1)):
            out.append(i)
    return out


def swing_lows(candles: List[Candle], lookback: int) -> List[int]:
    """Indices of confirmed swing-low pivots."""
    out: List[int] = []
    n = len(candles)
    for i in range(lookback, n - lookback):
        if all(candles[i].low < candles[i - j].low for j in range(1, lookback + 1)) and \
           all(candles[i].low < candles[i + j].low for j in range(1, lookback + 1)):
            out.append(i)
    return out


def latest_fvgs(candles: List[Candle], direction: Direction,
                up_to: Optional[int] = None) -> List[FVG]:
    """All FVGs of a given direction in the series, oldest first.

    ``up_to`` limits detection to candles[: up_to] (exclusive) so the strategy
    only ever 'sees' bars that have already closed - no look-ahead.
    """
    end = len(candles) if up_to is None else min(up_to, len(candles))
    found: List[FVG] = []
    for i in range(2, end):
        fvg = find_fvg(candles, i)
        if fvg and fvg.direction is direction:
            found.append(fvg)
    return found
