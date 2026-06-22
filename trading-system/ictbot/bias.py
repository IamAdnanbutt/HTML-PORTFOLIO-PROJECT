"""Higher-time-frame (H1) order-flow bias.

The notes stress aligning entries with the higher-time-frame draw and order
flow ("Wait for Alignment w/ HTF Bias & Draw"). We approximate HTF order flow
by resampling 1-minute candles into 1-hour candles and reading the swing
structure: a sequence of higher highs *and* higher lows is bullish; lower highs
*and* lower lows is bearish. We also reward price trading into an H1 FVG in the
direction of that structure (the "H1 FVG" PD Array from the notes).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import List, Optional

from .indicators import latest_fvgs, swing_highs, swing_lows
from .models import Candle, Direction


def resample(candles: List[Candle], minutes: int) -> List[Candle]:
    """Aggregate 1-minute candles into ``minutes``-minute candles.

    Buckets are aligned to the clock (e.g. 09:00, 10:00 for 60-minute bars).
    """
    if not candles:
        return []
    buckets: dict = {}
    order: List[datetime] = []
    for c in candles:
        # floor the timestamp to the bucket boundary
        epoch_min = int(c.time.timestamp() // 60)
        bucket_min = epoch_min - (epoch_min % minutes)
        key = datetime.fromtimestamp(bucket_min * 60)
        if key not in buckets:
            buckets[key] = [c.open, c.high, c.low, c.close, c.volume]
            order.append(key)
        else:
            agg = buckets[key]
            agg[1] = max(agg[1], c.high)
            agg[2] = min(agg[2], c.low)
            agg[3] = c.close
            agg[4] += c.volume
    out = []
    for key in order:
        o, h, l, cl, v = buckets[key]
        out.append(Candle(key, o, h, l, cl, v))
    return out


def structure_bias(htf: List[Candle], lookback: int = 2) -> Optional[Direction]:
    """Read swing structure on HTF candles into a directional bias."""
    highs = swing_highs(htf, lookback)
    lows = swing_lows(htf, lookback)
    if len(highs) < 2 or len(lows) < 2:
        return None
    hh = htf[highs[-1]].high > htf[highs[-2]].high
    hl = htf[lows[-1]].low > htf[lows[-2]].low
    lh = htf[highs[-1]].high < htf[highs[-2]].high
    ll = htf[lows[-1]].low < htf[lows[-2]].low
    if hh and hl:
        return Direction.LONG
    if lh and ll:
        return Direction.SHORT
    return None


def htf_bias(history: List[Candle], as_of: datetime,
             hours: int = 72, swing_lookback: int = 2) -> Optional[Direction]:
    """Compute the H1 order-flow bias as of ``as_of``.

    Uses only candles strictly before ``as_of`` (no look-ahead) and looks back
    ``hours`` hours so the read reflects the recent few sessions. The default
    spans roughly three sessions, which keeps enough H1 structure to read even
    across the overnight gap (when only regular-hours data is available).
    """
    window_start = as_of - timedelta(hours=hours)
    recent = [c for c in history if window_start <= c.time < as_of]
    if len(recent) < 120:  # need at least a couple of hours of 1m data
        return None
    h1 = resample(recent, 60)
    bias = structure_bias(h1, swing_lookback)
    if bias is None:
        return None

    # Confluence: confirm an H1 FVG exists in the same direction (the PD Array
    # the notes reference for "bullish continuation"). If structure says long
    # but there is no supportive H1 imbalance at all, stand down.
    supportive = latest_fvgs(h1, bias)
    if not supportive:
        return None
    return bias
