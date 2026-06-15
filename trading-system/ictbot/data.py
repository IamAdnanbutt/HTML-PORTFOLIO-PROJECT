"""Data loading and a synthetic intraday generator.

Two entry points:

* :func:`load_csv` - read OHLCV bars from a CSV file (your own historical data).
* :func:`generate_sessions` - produce synthetic 1-minute sessions that *exhibit*
  the 9:30 Open Model so the whole pipeline can be demonstrated offline.

The synthetic data is deliberately shaped to contain the Liquidity ->
Manipulation -> Delivery pattern on most days (with noise, losers and no-trade
days mixed in). It is for demonstration and testing ONLY - it is not a market
simulation and tells you nothing about real-world profitability. Always
re-validate on real historical data before trusting any number here.
"""

from __future__ import annotations

import csv
import random
from datetime import datetime, time, timedelta
from typing import List, Optional

from .models import Candle


# ----------------------------------------------------------------- CSV loader
def load_csv(path: str, dt_format: str = "%Y-%m-%d %H:%M:%S") -> List[Candle]:
    """Load candles from a CSV with columns: time,open,high,low,close[,volume].

    The ``time`` column is parsed with ``dt_format`` and treated as Eastern.
    """
    out: List[Candle] = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            out.append(Candle(
                time=datetime.strptime(row["time"], dt_format),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row.get("volume", 0) or 0),
            ))
    out.sort(key=lambda c: c.time)
    return out


def write_csv(path: str, candles: List[Candle],
              dt_format: str = "%Y-%m-%d %H:%M:%S") -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time", "open", "high", "low", "close", "volume"])
        for c in candles:
            w.writerow([c.time.strftime(dt_format),
                        f"{c.open:.2f}", f"{c.high:.2f}",
                        f"{c.low:.2f}", f"{c.close:.2f}", f"{c.volume:.0f}"])


# --------------------------------------------------- synthetic data generator
def _round_tick(p: float, tick: float) -> float:
    return round(round(p / tick) * tick, 2)


def _walk(rng: random.Random, start_dt: datetime, start_price: float,
          target: float, n_bars: int, vol: float, tick: float,
          wick: float = 0.6) -> List[Candle]:
    """Generate ``n_bars`` 1-minute candles drifting from start to target.

    ``wick`` scales the random wick size relative to ``vol``. A small wick with a
    strong drift produces 'displacement' candles that leave Fair Value Gaps.
    """
    candles: List[Candle] = []
    prev_close = start_price
    for i in range(n_bars):
        frac = (i + 1) / n_bars
        mid = start_price + (target - start_price) * frac
        o = prev_close
        c = mid + rng.gauss(0, vol)
        hi = max(o, c) + abs(rng.gauss(0, vol * wick))
        lo = min(o, c) - abs(rng.gauss(0, vol * wick))
        o, hi, lo, c = (_round_tick(x, tick) for x in (o, hi, lo, c))
        candles.append(Candle(start_dt + timedelta(minutes=i), o, hi, lo, c,
                              volume=rng.randint(200, 1500)))
        prev_close = c
    return candles


def _first_fvg_mid(candles: List[Candle], start_index: int,
                   direction: int) -> Optional[float]:
    """Mid-price of the first FVG (in ``direction``) at/after ``start_index``."""
    from .indicators import find_fvg
    from .models import Direction
    want = Direction.LONG if direction > 0 else Direction.SHORT
    for i in range(max(start_index, 2), len(candles)):
        fvg = find_fvg(candles, i)
        if fvg and fvg.direction is want:
            return fvg.mid
    return None


def _build_day(rng: random.Random, day: datetime, prior_extreme: float,
               direction: int, model: str, is_setup: bool,
               tick: float, scale: float = 1.0) -> List[Candle]:
    """Build one full session (08:00-15:59) for the given direction.

    ``direction`` is +1 for a long-bias day, -1 for short. ``prior_extreme`` is
    yesterday's session high (long regime) or low (short regime) - it becomes
    today's draw on liquidity once we open with a pullback against it. ``scale``
    multiplies every price magnitude so the data tracks the instrument (NQ-scale
    at 1.0, ES-scale at ~0.3) which keeps stop distances and position sizes
    sensible across products.
    """
    sign = direction
    s = scale
    # Open with a pullback *against* the regime so the prior extreme sits beyond
    # us as a draw on liquidity (long: prior high above; short: prior low below).
    gap = rng.uniform(80, 130) * s
    open_price = prior_extreme - sign * gap

    candles: List[Candle] = []
    t0 = datetime.combine(day.date(), time(8, 0))

    # --- pre-market (08:00 - 09:29): build a tight range around the open ----
    pm = _walk(rng, t0, open_price, open_price + rng.uniform(-10, 10) * s,
               89, 6.0 * s, tick)
    candles += pm
    pm_high = max(c.high for c in pm)
    pm_low = min(c.low for c in pm)
    last = pm[-1].close
    cur = pm[-1].time + timedelta(minutes=1)

    if not is_setup:
        # Choppy day: low-vol drift in the regime direction, no clean setup.
        rest = _walk(rng, cur, last, last + sign * rng.uniform(10, 30) * s,
                     480 - len(pm), 9.0 * s, tick)
        return candles + rest

    # The manipulation extreme: a shallow raid just beyond the pre-market level
    # on the side *opposite* the regime (long: sweep the low; short: the high).
    if sign > 0:
        manip_extreme = pm_low - rng.uniform(8, 16) * s
    else:
        manip_extreme = pm_high + rng.uniform(8, 16) * s

    if model == "continuation":
        # Liquidity is cleared *before* the open: drop into the sweep during the
        # last pre-market minutes, then displace from the open.
        sweep = _walk(rng, candles[-5].time, last, manip_extreme, 5, 7.0 * s, tick)
        candles = candles[:-5] + sweep
        cur = candles[-1].time + timedelta(minutes=1)
        last = candles[-1].close
    else:
        # Reversal: the raid happens just after 09:30.
        manip = _walk(rng, cur, last, manip_extreme, 6, 8.0 * s, tick)
        candles += manip
        cur = manip[-1].time + timedelta(minutes=1)
        last = manip[-1].close

    # --- displacement: a sharp, low-wick thrust that leaves FVGs ------------
    thrust_start = len(candles)
    thrust_to = last + sign * rng.uniform(55, 80) * s
    thrust = _walk(rng, cur, last, thrust_to, 10, 7.0 * s, tick, wick=0.18)
    candles += thrust
    cur = thrust[-1].time + timedelta(minutes=1)
    last = thrust[-1].close

    # Retrace into the *first* FVG of the thrust (the 1st PFVG entry), but never
    # beyond the manipulation extreme. This lands inside the 09:50 macro window.
    fvg_mid = _first_fvg_mid(candles, thrust_start, sign)
    if fvg_mid is None:
        retrace_to = last - sign * rng.uniform(25, 35) * s
    else:
        retrace_to = fvg_mid - sign * rng.uniform(0, 4) * s  # dip into the gap
    if sign > 0:
        retrace_to = max(retrace_to, manip_extreme + 6 * s)
    else:
        retrace_to = min(retrace_to, manip_extreme - 6 * s)
    retrace = _walk(rng, cur, last, retrace_to, 6, 5.0 * s, tick)
    candles += retrace
    cur = retrace[-1].time + timedelta(minutes=1)
    last = retrace[-1].close

    # delivery: run to (and just past) the draw on liquidity
    draw = prior_extreme + sign * rng.uniform(30, 70) * s
    deliver = _walk(rng, cur, last, draw, 25, 10.0 * s, tick)
    candles += deliver
    cur = deliver[-1].time + timedelta(minutes=1)
    last = deliver[-1].close

    # --- afternoon: drift to close near the day's extreme (sets tomorrow's draw)
    remaining = 480 - len(candles)
    if remaining > 0:
        close_to = last + sign * rng.uniform(0, 20) * s
        candles += _walk(rng, cur, last, close_to, remaining, 8.0 * s, tick)
    return candles[:480]


def generate_sessions(n_days: int = 40, seed: int = 7,
                      start_price: float = 18_000.0,
                      tick: float = 0.25) -> List[Candle]:
    """Generate ``n_days`` weekday sessions of synthetic 1-minute candles.

    Price magnitudes scale with ``start_price`` (relative to NQ's ~18,000) so
    that ES-scale data (~5,000) produces proportionally smaller swings and stop
    distances - keeping position sizing sensible across instruments.
    """
    rng = random.Random(seed)
    candles: List[Candle] = []
    scale = start_price / 18_000.0

    day = datetime(2026, 1, 5)          # a Monday
    regime = 1
    regime_left = rng.randint(4, 7)
    prior_extreme = start_price

    produced = 0
    while produced < n_days:
        if day.weekday() >= 5:          # skip weekends
            day += timedelta(days=1)
            continue

        if regime_left == 0:
            regime *= -1
            regime_left = rng.randint(4, 7)
        regime_left -= 1

        model = "continuation" if rng.random() < 0.30 else "reversal"
        is_setup = rng.random() < 0.82

        day_candles = _build_day(rng, day, prior_extreme, regime,
                                 model, is_setup, tick, scale)
        candles += day_candles

        # tomorrow's draw = today's extreme on the regime side
        if regime > 0:
            prior_extreme = max(c.high for c in day_candles)
        else:
            prior_extreme = min(c.low for c in day_candles)

        produced += 1
        day += timedelta(days=1)

    return candles
