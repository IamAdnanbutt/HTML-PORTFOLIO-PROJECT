"""Cat 3: PD Arrays (Premium/Discount Arrays) — the ICT entry toolkit.

Every system in this package uses these four array types to frame entries.
They share one concept: a 50% "consequent encroachment" (CE / MTH) level
where price bodies are expected to respect before continuation.

Four array types (FVG is already in indicators.py):
  1. OrderBlock  — last opposing candle before a displacement (MTH = 50% body)
  2. RejectionBlock — significant-wick candle (CE = 50% of the wick range)
  3. BreakerBlock — a failed OB that has been swept and flipped
  4. FVG — see indicators.py (CE = fvg.mid = 50% of the gap)

Detection functions here are no-lookahead: they only examine candles that have
already closed.  Each returns None when the pattern is not yet confirmed.

Invalidation rules (from the notes):
  - No displacement: there is no energetic expansion candle creating the array
  - Weak FVG: gap size < min_gap_mult * ATR
  - Body closed through: OB / RB / Breaker is mitigated when a body closes
    beyond its 50% level (MTH / CE)
"""

from __future__ import annotations

from typing import List, Optional

from .indicators import atr, find_fvg, is_displacement
from .models import BreakerBlock, Candle, Direction, FVG, OrderBlock, RejectionBlock


# ----------------------------------------------------------------- Order Block
def find_order_block(candles: List[Candle], index: int,
                     atr_period: int = 14,
                     atr_mult: float = 1.5) -> Optional[OrderBlock]:
    """Return the Order Block ending at the displacement candle at ``index``.

    ``index`` points to the displacement candle (the energetic expansion bar).
    We look backward for the last opposing candle immediately before ``index``.

    Bullish OB: the last down-close candle before an upward displacement.
    Bearish OB: the last up-close candle before a downward displacement.

    The displacement at ``index`` must meet the ATR threshold so we do not label
    every small push as an OB.
    """
    if index < 1 or index >= len(candles):
        return None
    disp = candles[index]
    if not is_displacement(candles, index, atr_mult, atr_period):
        return None

    # Decide which candle type we are looking for based on displacement direction.
    disp_is_up = disp.close > disp.open

    # Scan backward to find the last opposing candle.
    ob_idx = index - 1
    while ob_idx >= 0:
        c = candles[ob_idx]
        is_opposing = (not c.is_up) if disp_is_up else c.is_up
        if is_opposing:
            direction = Direction.LONG if disp_is_up else Direction.SHORT
            return OrderBlock(
                direction=direction,
                top=c.body_high,
                bottom=c.body_low,
                wick_high=c.high,
                wick_low=c.low,
                created_at=c.time,
                index=ob_idx,
            )
        ob_idx -= 1
    return None


def scan_order_blocks(candles: List[Candle], direction: Direction,
                      atr_period: int = 14, atr_mult: float = 1.5,
                      up_to: Optional[int] = None) -> List[OrderBlock]:
    """All valid (non-mitigated) OBs of the given direction, oldest first."""
    end = len(candles) if up_to is None else min(up_to, len(candles))
    obs: List[OrderBlock] = []
    for i in range(1, end):
        ob = find_order_block(candles, i, atr_period, atr_mult)
        if ob and ob.direction is direction:
            # Mark as mitigated if a subsequent body closed through MTH.
            mitigated = False
            for j in range(i + 1, end):
                if ob.body_closed_through(candles[j]):
                    ob.mitigated = True
                    ob.mitigated_at = candles[j].time
                    mitigated = True
                    break
            if not mitigated:
                obs.append(ob)
    return obs


def nearest_order_block(candles: List[Candle], direction: Direction,
                        reference_price: float,
                        atr_period: int = 14, atr_mult: float = 1.5,
                        up_to: Optional[int] = None) -> Optional[OrderBlock]:
    """The most recent non-mitigated OB that price can retrace into."""
    obs = scan_order_blocks(candles, direction, atr_period, atr_mult, up_to)
    if direction is Direction.LONG:
        candidates = [o for o in obs if o.top < reference_price]
    else:
        candidates = [o for o in obs if o.bottom > reference_price]
    return candidates[-1] if candidates else None


# -------------------------------------------------------------- Rejection Block
def find_rejection_block(candle: Candle, index: int,
                         min_wick_ratio: float = 0.4) -> Optional[RejectionBlock]:
    """Detect a Rejection Block in ``candle``.

    A bullish RB: down-close candle with a lower wick >= ``min_wick_ratio``
    of the total candle range.
    A bearish RB: up-close candle with an upper wick >= ``min_wick_ratio``.

    CE = 50% of the wick (from the wick tip to the body edge).
    """
    if candle.range == 0:
        return None

    lower_wick = candle.body_low - candle.low
    upper_wick = candle.high - candle.body_high

    if not candle.is_up and lower_wick >= min_wick_ratio * candle.range:
        return RejectionBlock(
            direction=Direction.LONG,
            wick_tip=candle.low,
            body_edge=candle.body_low,
            created_at=candle.time,
            index=index,
        )
    if candle.is_up and upper_wick >= min_wick_ratio * candle.range:
        return RejectionBlock(
            direction=Direction.SHORT,
            wick_tip=candle.high,
            body_edge=candle.body_high,
            created_at=candle.time,
            index=index,
        )
    return None


def scan_rejection_blocks(candles: List[Candle], direction: Direction,
                          min_wick_ratio: float = 0.4,
                          up_to: Optional[int] = None) -> List[RejectionBlock]:
    """All non-mitigated Rejection Blocks of the given direction, oldest first."""
    end = len(candles) if up_to is None else min(up_to, len(candles))
    rbs: List[RejectionBlock] = []
    for i in range(end):
        rb = find_rejection_block(candles[i], i, min_wick_ratio)
        if rb and rb.direction is direction:
            # Mitigated when body closes beyond CE.
            mit = False
            for j in range(i + 1, end):
                c = candles[j]
                if direction is Direction.LONG and c.body_low < rb.ce:
                    rb.mitigated = True
                    mit = True
                    break
                if direction is Direction.SHORT and c.body_high > rb.ce:
                    rb.mitigated = True
                    mit = True
                    break
            if not mit:
                rbs.append(rb)
    return rbs


# --------------------------------------------------------------- Breaker Block
def scan_breaker_blocks(candles: List[Candle], direction: Direction,
                        atr_period: int = 14, atr_mult: float = 1.5,
                        up_to: Optional[int] = None) -> List[BreakerBlock]:
    """Detect all BreakerBlocks of the given direction.

    Steps:
      1. Find an OB in the OPPOSITE direction.
      2. Detect the bar where price body closes THROUGH that OB (the
         "change in state of delivery" candle — up-close = bullish breaker,
         down-close = bearish breaker).
      3. That bar and the OB range define the BreakerBlock.

    A bullish Breaker: bearish OB → price closes ABOVE its body → now support.
    A bearish Breaker: bullish OB → price closes BELOW its body → now resistance.
    """
    end = len(candles) if up_to is None else min(up_to, len(candles))
    breakers: List[BreakerBlock] = []

    opp = direction.opposite()
    # Scan for OBs in the opposite direction first.
    for i in range(1, end):
        ob = find_order_block(candles, i, atr_period, atr_mult)
        if ob is None or ob.direction is not opp:
            continue

        # Now scan forward from that OB for the candle that closes through it.
        for j in range(i + 1, end):
            c = candles[j]
            if direction is Direction.LONG:
                swept = c.close > ob.top   # close above bearish OB body
            else:
                swept = c.close < ob.bottom  # close below bullish OB body

            if swept:
                bb = BreakerBlock(
                    direction=direction,
                    top=ob.top,
                    bottom=ob.bottom,
                    created_at=c.time,
                    index=j,
                )
                # Mark as mitigated if subsequent price trades back beyond CE.
                for k in range(j + 1, end):
                    ck = candles[k]
                    if direction is Direction.LONG and ck.low < bb.bottom:
                        bb.mitigated = True
                        break
                    if direction is Direction.SHORT and ck.high > bb.top:
                        bb.mitigated = True
                        break
                if not bb.mitigated:
                    breakers.append(bb)
                break  # one breaker per OB

    return breakers


def nearest_breaker(candles: List[Candle], direction: Direction,
                    reference_price: float,
                    atr_period: int = 14, atr_mult: float = 1.5,
                    up_to: Optional[int] = None) -> Optional[BreakerBlock]:
    """The most recently formed, non-mitigated Breaker that price can retrace to."""
    bbs = scan_breaker_blocks(candles, direction, atr_period, atr_mult, up_to)
    if direction is Direction.LONG:
        candidates = [b for b in bbs if b.top < reference_price]
    else:
        candidates = [b for b in bbs if b.bottom > reference_price]
    return candidates[-1] if candidates else None


# ------------------------------------------------ unified PD array entry level
def best_pd_entry(candles: List[Candle], direction: Direction,
                  reference_price: float,
                  atr_period: int = 14, atr_mult: float = 1.5,
                  up_to: Optional[int] = None) -> Optional[float]:
    """Return the CE / MTH of the highest-quality PD array near ``reference_price``.

    Priority (per the notes — stack confluence): FVG → OB → Breaker → RB.
    Returns None if no valid array is found.
    """
    end = up_to or len(candles)

    # 1. FVG (already identified by the calling strategy)
    from .indicators import latest_fvgs
    fvgs = [f for f in latest_fvgs(candles, direction, end)
            if f.contains(reference_price)]
    if fvgs:
        return fvgs[-1].mid

    # 2. Order Block
    ob = nearest_order_block(candles, direction, reference_price,
                             atr_period, atr_mult, end)
    if ob and ob.contains(reference_price):
        return ob.mth

    # 3. Breaker Block
    bb = nearest_breaker(candles, direction, reference_price,
                         atr_period, atr_mult, end)
    if bb and bb.contains(reference_price):
        return bb.ce

    # 4. Rejection Block
    rbs = scan_rejection_blocks(candles, direction, up_to=end)
    if rbs:
        rb = rbs[-1]
        if (direction is Direction.LONG and rb.wick_tip <= reference_price <= rb.body_edge) or \
           (direction is Direction.SHORT and rb.body_edge <= reference_price <= rb.wick_tip):
            return rb.ce

    return None
