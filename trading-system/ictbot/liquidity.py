"""Liquidity pools and sweep ('manipulation') detection.

The first leg of the model is *Liquidity*: identifying the resting pools that
the algorithm is likely to target. The second leg is *Manipulation*: the raid
that clears one of those pools before delivery in the opposite direction.
"""

from __future__ import annotations

from datetime import date
from typing import Dict, List, Optional

from .models import Candle, Direction, LiquidityPool


def session_extremes(candles: List[Candle]) -> Optional[tuple]:
    """Return (high, low) across a list of candles, or None if empty."""
    if not candles:
        return None
    return max(c.high for c in candles), min(c.low for c in candles)


def build_liquidity_pools(
    premarket: List[Candle],
    prev_session: List[Candle],
    prev_day: List[Candle],
) -> List[LiquidityPool]:
    """Construct the standard pools referenced in the notes.

    * PMH / PML - pre-market high / low (today, before 09:30)
    * PSH / PSL - previous regular-session high / low
    * PDH / PDL - previous *day* high / low (full day incl. overnight)
    """
    pools: List[LiquidityPool] = []

    pm = session_extremes(premarket)
    if pm:
        hi, lo = pm
        pools.append(LiquidityPool("PMH", hi, Direction.LONG))
        pools.append(LiquidityPool("PML", lo, Direction.SHORT))

    ps = session_extremes(prev_session)
    if ps:
        hi, lo = ps
        pools.append(LiquidityPool("PSH", hi, Direction.LONG))
        pools.append(LiquidityPool("PSL", lo, Direction.SHORT))

    pd = session_extremes(prev_day)
    if pd:
        hi, lo = pd
        pools.append(LiquidityPool("PDH", hi, Direction.LONG))
        pools.append(LiquidityPool("PDL", lo, Direction.SHORT))

    return pools


def detect_sweep(candle: Candle, pool: LiquidityPool) -> bool:
    """Has ``candle`` raided ``pool`` (a stop run / manipulation)?

    Buyside pool (a high): swept when the candle trades *above* it.
    Sellside pool (a low): swept when the candle trades *below* it.

    We use the wick (high/low), because a stop run only needs price to *trade*
    through the level, not to close beyond it.
    """
    if pool.is_buyside:
        return candle.high > pool.price
    return candle.low < pool.price


def nearest_target_pool(
    pools: List[LiquidityPool],
    direction: Direction,
    reference_price: float,
    min_distance: float = 0.0,
) -> Optional[LiquidityPool]:
    """Pick the draw-on-liquidity target in the direction of the trade.

    For a long we want an un-swept buyside pool above price; for a short, an
    un-swept sellside pool below price. We prefer the *nearest* pool that is at
    least ``min_distance`` away (so the reward:risk filter can be met); if none
    is far enough, fall back to the furthest available pool.
    """
    candidates = [
        p for p in pools
        if not p.swept and p.side is direction and (
            (direction is Direction.LONG and p.price > reference_price) or
            (direction is Direction.SHORT and p.price < reference_price)
        )
    ]
    if not candidates:
        return None
    far_enough = [p for p in candidates
                  if abs(p.price - reference_price) >= min_distance]
    if far_enough:
        far_enough.sort(key=lambda p: abs(p.price - reference_price))
        return far_enough[0]
    # nothing far enough: take the furthest draw we have
    candidates.sort(key=lambda p: abs(p.price - reference_price), reverse=True)
    return candidates[0]


def group_by_day(candles: List[Candle]) -> "Dict[date, List[Candle]]":
    """Bucket a flat candle stream into day -> candles (preserving order)."""
    out: Dict[date, List[Candle]] = {}
    for c in candles:
        out.setdefault(c.time.date(), []).append(c)
    return out
