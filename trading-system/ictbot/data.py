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


def _walk_to(rng: random.Random, cur: datetime, last: float, target: float,
             n: int, vol: float, tick: float, wick: float = 0.6):
    """Convenience wrapper: walk ``n`` bars to ``target`` and return
    (candles, next_time, last_close).  ``n`` is clamped to >= 1."""
    n = max(1, n)
    seg = _walk(rng, cur, last, target, n, vol, tick, wick=wick)
    return seg, seg[-1].time + timedelta(minutes=1), seg[-1].close


def _or_band(rng: random.Random, start: datetime, base: float, half: float,
             scale: float, tick: float):
    """Build a 31-bar Opening-Range band around ``base`` (+/- ``half``).

    Returns (candles, next_time, last_close, or_low, or_high)."""
    seg1, cur, last = _walk_to(rng, start, base, base + half, 15, 2.2 * scale, tick)
    seg2, cur, last = _walk_to(rng, cur, last, base - half, 16, 2.2 * scale, tick)
    band = seg1 + seg2
    return band, cur, last, min(c.low for c in band), max(c.high for c in band)


def _setup_event(rng: random.Random, cur: datetime, last: float,
                 ref_low: float, ref_high: float, sign: int,
                 scale: float, tick: float, target: float,
                 n_sweep: int = 4, n_thrust: int = 5,
                 n_retrace: int = 2, n_deliver: int = 8):
    """One sweep -> displacement -> macro entry -> delivery event.

    Sweeps the reference extreme *against* the trade (``ref_low`` for a long,
    ``ref_high`` for a short), prints a low-wick displacement that leaves a Fair
    Value Gap, retraces to that FVG midpoint (the macro entry the OR / Asia
    models trigger on), then delivers cleanly to ``target``.  The delivery uses
    a strong drift and a small wick so price escapes the stop on the very first
    bar instead of wicking straight back through it.  Returns (candles, cur,
    last).
    """
    s = scale
    out: List[Candle] = []
    if sign > 0:
        sweep_to = ref_low - rng.uniform(7, 12) * s
    else:
        sweep_to = ref_high + rng.uniform(7, 12) * s
    seg, cur, last = _walk_to(rng, cur, last, sweep_to, n_sweep, 3.0 * s, tick)
    out += seg
    tf = len(out)
    seg, cur, last = _walk_to(rng, cur, last, last + sign * rng.uniform(42, 58) * s,
                              n_thrust, 4.5 * s, tick, wick=0.15)
    out += seg
    fvg_mid = _first_fvg_mid(out, tf, sign) or (last - sign * 14 * s)
    seg, cur, last = _walk_to(rng, cur, last, fvg_mid, n_retrace, 1.6 * s, tick)
    out += seg
    seg, cur, last = _walk_to(rng, cur, last, target, n_deliver, 2.0 * s, tick,
                              wick=0.12)
    out += seg
    return out, cur, last


def _full_overnight(rng: random.Random, day: datetime, prev_close: float,
                    sign: int, scale: float, tick: float,
                    is_setup: bool) -> List[Candle]:
    """Build the early-morning windows (00:00 - 07:59) for one calendar day.

    The three early Opening Ranges share one continuous, in-band price path —
    exactly as they would on a real chart, where the Midnight, London and NY
    Kill Zone ranges all sit on the same overnight tape.  The path keeps price
    inside a single band so every OR locks a similar high/low, then prints two
    chained sweep -> displacement -> macro-entry events (one timed into the
    Midnight 02:50 macro, one into the London 04:00 macro) before a dedicated
    NY-Kill-Zone setup into the 07:50 macro.  Each trade hits its target before
    the next price discontinuity, so nothing is left dangling into the RTH open.
    """
    s = scale
    d = day.date()

    def at(h, m):
        return datetime.combine(d, time(h, m))

    def span(a, b):
        return max(1, int((b - a).total_seconds() // 60))

    base = prev_close + sign * rng.uniform(2, 10) * s
    half = 12.0 * s
    out: List[Candle] = []

    # ---- 00:00-00:30  Midnight OR band -----------------------------------
    band, cur, last, or_low, or_high = _or_band(rng, at(0, 0), base, half, s, tick)
    out += band
    or_range = or_high - or_low

    if not is_setup:
        # Quiet, setup-free night: drift gently in-band to ~07:59.
        seg, cur, last = _walk_to(rng, cur, last, base + sign * rng.uniform(2, 8) * s,
                                  span(cur, at(8, 0)), 3.5 * s, tick)
        out += seg
        return [c for c in out if c.time < at(8, 0)]

    # ---- 00:31-01:29 hold, then form the London OR band 01:30-02:00 ------
    seg, cur, last = _walk_to(rng, cur, last, base, span(cur, at(1, 30)), 2.0 * s, tick)
    out += seg
    lon_band, cur, last, lon_low, lon_high = _or_band(rng, at(1, 30), last, half, s, tick)
    out += lon_band
    lon_range = lon_high - lon_low
    # hold in-band until just before the Midnight macro (02:50)
    seg, cur, last = _walk_to(rng, cur, last, base, span(cur, at(2, 41)), 2.0 * s, tick)
    out += seg

    # ---- Midnight + London shared sweep, then displacement into 02:50 -----
    # Sweep both OR lows (against a long / both highs against a short).
    sweep_to = (or_low - rng.uniform(8, 13) * s) if sign > 0 \
        else (or_high + rng.uniform(8, 13) * s)
    seg, cur, last = _walk_to(rng, cur, last, sweep_to, 5, 3.0 * s, tick)
    out += seg                                              # 02:41-02:45 sweep
    thrust_from = len(out)
    seg, cur, last = _walk_to(rng, cur, last, last + sign * rng.uniform(46, 60) * s,
                              7, 4.5 * s, tick, wick=0.15)
    out += seg                                              # 02:46-02:52 displacement; macro 02:50
    fvg_mid = _first_fvg_mid(out, thrust_from, sign) or (last - sign * 16 * s)
    seg, cur, last = _walk_to(rng, cur, last, fvg_mid, 2, 1.6 * s, tick)
    out += seg                                              # 02:53-02:54 entry -> Midnight fills
    # First rally: hit the Midnight 1.0/1.5-SD target by ~03:05.
    rally_to = (or_high + 1.9 * or_range) if sign > 0 else (or_low - 1.9 * or_range)
    seg, cur, last = _walk_to(rng, cur, last, rally_to, span(cur, at(3, 6)),
                              2.0 * s, tick, wick=0.12)
    out += seg

    # ---- revert into the FVG during London's 03:15-03:45 macro -----------
    revert_to = fvg_mid - sign * 4 * s
    seg, cur, last = _walk_to(rng, cur, last, revert_to, span(cur, at(3, 23)),
                              2.2 * s, tick)
    out += seg                                              # crosses fvg_mid ~03:20 -> London fills
    # Second rally: hit London's target by ~04:20.
    lon_target = (lon_high + 1.9 * lon_range) if sign > 0 else (lon_low - 1.9 * lon_range)
    seg, cur, last = _walk_to(rng, cur, last, lon_target, span(cur, at(4, 20)),
                              2.0 * s, tick, wick=0.12)
    out += seg

    # ---- drift to 07:00, then a self-contained NY Kill Zone setup ---------
    seg, cur, last = _walk_to(rng, cur, last, base + sign * rng.uniform(4, 14) * s,
                              span(cur, at(7, 0)), 3.0 * s, tick)
    out += seg
    nyk_band, cur, last, nyk_low, nyk_high = _or_band(rng, at(7, 0), last, half, s, tick)
    out += nyk_band                                        # NY Kill Zone OR 07:00-07:30
    nyk_range = nyk_high - nyk_low
    seg, cur, last = _walk_to(rng, cur, last, last, span(cur, at(7, 42)), 2.0 * s, tick)
    out += seg                                              # hold to 07:41
    # Fast event so the trade closes before the 08:00 RTH discontinuity.
    nyk_target = (nyk_high + 1.7 * nyk_range) if sign > 0 else (nyk_low - 1.7 * nyk_range)
    event, cur, last = _setup_event(rng, cur, last, nyk_low, nyk_high, sign, s,
                                    tick, nyk_target, n_sweep=4, n_thrust=5,
                                    n_retrace=2, n_deliver=6)
    out += event                                           # 07:42-07:58

    # Trim to strictly before the RTH pre-market open so there is no clock
    # collision with _build_day (which starts its own series at 08:00).
    return [c for c in out if c.time < at(8, 0)]


def _build_asia(rng: random.Random, day: datetime, rth_close: float,
                sign: int, scale: float, tick: float,
                is_setup: bool) -> List[Candle]:
    """Build the Asia Killzone evening session (18:00 - 21:30).

    Opens with a New Day Opening Gap (NDOG) from this day's 4 PM RTH close,
    accumulates inside the gap, prints the 7:50 PM "Judas Swing" that raids the
    NDOG extreme against the bias, then displaces and delivers from the 8:50 PM
    macro toward a standard-deviation projection of the NDOG range.
    """
    s = scale
    d = day.date()

    def at(h, m):
        return datetime.combine(d, time(h, m))

    def span(a, b):
        return int((b - a).total_seconds() // 60)

    # NDOG: 6 PM open gaps from the 4 PM close in the regime direction.
    gap = rng.uniform(22, 34) * s
    ndog_open = rth_close + sign * gap
    ndog_low = min(ndog_open, rth_close)
    ndog_high = max(ndog_open, rth_close)
    ndog_range = ndog_high - ndog_low

    out: List[Candle] = []
    # 18:00 open exactly at ndog_open so the model's NDOG matches ours.
    first = Candle(at(18, 0), _round_tick(ndog_open, tick),
                   _round_tick(ndog_open + 1.5 * s, tick),
                   _round_tick(ndog_open - 1.5 * s, tick),
                   _round_tick(ndog_open, tick), volume=rng.randint(200, 1200))
    out.append(first)
    cur = at(18, 1)
    last = first.close

    if not is_setup:
        seg, cur, last = _walk_to(rng, cur, last, ndog_open, span(cur, at(21, 30)),
                                  3.5 * s, tick)
        out += seg
        return [c for c in out if c.time <= at(21, 30)]

    # ---- accumulate inside the NDOG until the 7:50 PM Judas window --------
    seg, cur, last = _walk_to(rng, cur, last, ndog_open, span(cur, at(19, 50)),
                              2.5 * s, tick)
    out += seg

    # ---- 7:50 PM Judas Swing: raid the NDOG extreme against the bias ------
    if sign > 0:
        judas_to = ndog_low - rng.uniform(7, 12) * s     # sweep sellside
    else:
        judas_to = ndog_high + rng.uniform(7, 12) * s    # sweep buyside
    seg, cur, last = _walk_to(rng, cur, last, judas_to, 6, 3.5 * s, tick)
    out += seg                                            # 19:50-19:55

    # drift back inside the gap to close the Judas window (~20:09)
    seg, cur, last = _walk_to(rng, cur, last,
                              ndog_low + 0.4 * ndog_range if sign > 0
                              else ndog_high - 0.4 * ndog_range,
                              span(cur, at(20, 10)), 2.0 * s, tick)
    out += seg

    # ---- 8:50 PM delivery: sweep -> displacement -> entry -> run to target -
    # The displacement + entry land in the 20:50 macro; the target is a 1.0-SD
    # projection of the NDOG range, matching the model's draw-on-liquidity.
    # Hold quietly until just before the delivery macro.
    seg, cur, last = _walk_to(rng, cur, last,
                              ndog_low + 0.4 * ndog_range if sign > 0
                              else ndog_high - 0.4 * ndog_range,
                              span(cur, at(20, 44)), 2.0 * s, tick)
    out += seg
    target = (ndog_high + 1.2 * ndog_range) if sign > 0 \
        else (ndog_low - 1.2 * ndog_range)
    event, cur, last = _setup_event(rng, cur, last, ndog_low, ndog_high, sign, s,
                                    tick, target, n_sweep=3, n_thrust=5,
                                    n_retrace=2, n_deliver=14)
    out += event                                          # 20:44-21:08
    return [c for c in out if c.time <= at(21, 30)]


def _build_rth_full(rng: random.Random, day: datetime, prior_extreme: float,
                    sign: int, model: str, is_setup: bool,
                    tick: float, scale: float) -> List[Candle]:
    """RTH session for full-day mode: the standard 9:30 setup in the morning,
    with a purpose-built PM Opening Range setup engineered into the afternoon.

    The morning (through ~13:10, which contains the 9:30 manipulation,
    displacement, entry and delivery) is reused verbatim from :func:`_build_day`
    so Cat1 is unchanged.  The afternoon is then replaced with a clean PM OR
    band (13:30-14:00) and a sweep -> displacement -> 14:50-macro entry ->
    delivery event so the PM Opening Range has a designed setup to trade.
    """
    base = _build_day(rng, day, prior_extreme, sign, model, is_setup, tick, scale)
    if not is_setup:
        return base

    s = scale
    d = day.date()

    def at(h, m):
        return datetime.combine(d, time(h, m))

    def span(a, b):
        return max(1, int((b - a).total_seconds() // 60))

    cutoff = at(13, 10)
    out = [c for c in base if c.time < cutoff]
    cur = out[-1].time + timedelta(minutes=1)
    last = out[-1].close
    pm_base = last

    # drift into the PM OR window, build the band, then hold to ~14:44
    seg, cur, last = _walk_to(rng, cur, last, pm_base, span(cur, at(13, 30)), 3.0 * s, tick)
    out += seg
    band, cur, last, pm_low, pm_high = _or_band(rng, at(13, 30), last, 12.0 * s, s, tick)
    out += band
    pm_range = pm_high - pm_low
    seg, cur, last = _walk_to(rng, cur, last, pm_base, span(cur, at(14, 44)), 2.0 * s, tick)
    out += seg

    # PM OR event: sweep -> displacement -> 14:50 entry -> delivery
    pm_target = (pm_high + 1.7 * pm_range) if sign > 0 else (pm_low - 1.7 * pm_range)
    event, cur, last = _setup_event(rng, cur, last, pm_low, pm_high, sign, s,
                                    tick, pm_target, n_sweep=3, n_thrust=5,
                                    n_retrace=2, n_deliver=8)
    out += event                                          # 14:44-15:01

    # drift to the close
    seg, cur, last = _walk_to(rng, cur, last, last + sign * rng.uniform(0, 12) * s,
                              span(cur, at(16, 0)), 3.5 * s, tick)
    out += seg
    return [c for c in out if c.time < at(16, 0)]


def generate_sessions(n_days: int = 40, seed: int = 7,
                      start_price: float = 18_000.0,
                      tick: float = 0.25,
                      full_day: bool = False) -> List[Candle]:
    """Generate ``n_days`` weekday sessions of synthetic 1-minute candles.

    Price magnitudes scale with ``start_price`` (relative to NQ's ~18,000) so
    that ES-scale data (~5,000) produces proportionally smaller swings and stop
    distances - keeping position sizing sensible across instruments.

    With ``full_day=True`` each calendar day also includes the overnight and
    early-morning windows (the Midnight / London / NY-Kill-Zone Opening Ranges
    and the 6 PM-9:30 PM Asia Killzone), so every one of the five systems
    receives data in its own session.  The default (RTH only, 08:00-15:59)
    keeps the single-system commands and their fixtures unchanged.
    """
    rng = random.Random(seed)
    candles: List[Candle] = []
    scale = start_price / 18_000.0

    day = datetime(2026, 1, 5)          # a Monday
    regime = 1
    regime_left = rng.randint(4, 7)
    prior_extreme = start_price
    prev_rth_close = start_price

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

        day_candles: List[Candle] = []
        if full_day:
            day_candles += _full_overnight(rng, day, prev_rth_close, regime,
                                           scale, tick, is_setup)
            rth = _build_rth_full(rng, day, prior_extreme, regime,
                                  model, is_setup, tick, scale)
        else:
            rth = _build_day(rng, day, prior_extreme, regime,
                             model, is_setup, tick, scale)
        day_candles += rth

        if full_day:
            day_candles += _build_asia(rng, day, rth[-1].close, regime,
                                       scale, tick, is_setup)

        day_candles.sort(key=lambda c: c.time)
        candles += day_candles

        # tomorrow's draw = today's RTH extreme on the regime side
        if regime > 0:
            prior_extreme = max(c.high for c in rth)
        else:
            prior_extreme = min(c.low for c in rth)
        prev_rth_close = rth[-1].close

        produced += 1
        day += timedelta(days=1)

    return candles
