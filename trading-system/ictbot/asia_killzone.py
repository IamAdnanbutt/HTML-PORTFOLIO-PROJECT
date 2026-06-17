"""Cat 5: Asia Killzone — NDOG / NWOG overnight session model.

Source: IMG_0110 (Asia KZ Delivery), IMG_0111–0113 (Chart Examples #1–#4).

Active window: 6:00 PM – ~9:30 PM New York time (the overnight / Asia session).

Key reference levels:
  NDOG (New Day Opening Gap)  — gap between yesterday's 4 PM close and
                                today's 6 PM open.  Price accumulates inside
                                the NDOG then delivers away from it.
  NWOG (New Week Opening Gap) — gap between Friday's 4 PM close and Monday's
                                6 PM open (weekly-scale draw).  Treated as the
                                higher-timeframe draw when price is at the weekly.
  PWH / PWL                  — Previous Weekly High / Low (bigger targets).

Quadrant levels (divide NDOG High–Low into 4):
  0.00 = NDOG Low
  0.25 = Lower Quadrant  (accumulation zone for bullish days)
  0.50 = NDOG Midpoint / CE (Consequent Encroachment)
  0.75 = Upper Quadrant  (accumulation zone for bearish days)
  1.00 = NDOG High

Asia macro time windows (NY time):
  7:50–8:10 PM  — Judas Swing / manipulation window
  8:50–9:10 PM  — Delivery window (primary entry time)

IFVG (Inverted Fair Value Gap):
  A FVG that has been fully mitigated (price traded through the gap) and now
  acts as resistance / support in the OPPOSITE direction.

Sequence (from the chart examples):
  1. After 6 PM: price opens and establishes the NDOG range.
  2. Price accumulates into one quadrant (lower for bullish, upper for bearish).
  3. 7:50 PM Macro: the "Judas Swing" — price raids liquidity in the OPPOSITE
     direction to the intended delivery (e.g. for a long: sweeps below NDOG Low).
  4. Price rebounds; the sweep extreme becomes the stop reference.
  5. 8:50 PM Macro: displacement + 1st PFVG / IFVG formation.
  6. Entry: FVG CE rebalance inside the 8:50 macro.
  7. Target: NDOG High / NWOG level / PWH (buyside) or NDOG Low / PWL (sellside).

Model variants (from the examples):
  Variant A — Reversal: sweep below NDOG Low → long to NDOG High / BSL target.
  Variant B — Continuation: price trades through NDOG High (buystops) → short
    back to NDOG Low / sellside target.  IFVG + NDOG CE = entry (IMG_0113).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from enum import Enum, auto
from typing import List, Optional, Tuple

from .config import Config
from .indicators import atr, find_fvg
from .models import Candle, Direction, FVG, LiquidityPool, Signal


# ----------------------------------------------------------------- time bounds
ASIA_OPEN = time(18, 0)     # 6:00 PM — futures session opens
ASIA_CLOSE = time(21, 30)   # 9:30 PM — approximate Asia session end
FRIDAY_CLOSE = time(16, 0)  # 4:00 PM Friday → sets NWOG on Sunday open

ASIA_MACROS: List[Tuple[time, time]] = [
    (time(19, 50), time(20, 10)),   # 7:50–8:10 PM — Judas Swing window
    (time(20, 50), time(21, 10)),   # 8:50–9:10 PM — Delivery window
]


def _in_asia_macro(t: time) -> bool:
    return any(s <= t <= e for s, e in ASIA_MACROS)


def _normalize(price: float, low: float, high: float) -> float:
    r = high - low
    return (price - low) / r if r else 0.0


# ---------------------------------------------------------------- data classes
@dataclass
class NDOG:
    """New Day Opening Gap."""

    open_price: float   # 6 PM open of the new session
    prev_close: float   # 4 PM close of the prior session
    formed_at: datetime

    @property
    def high(self) -> float:
        return max(self.open_price, self.prev_close)

    @property
    def low(self) -> float:
        return min(self.open_price, self.prev_close)

    @property
    def range(self) -> float:
        return self.high - self.low

    @property
    def midpoint(self) -> float:
        return (self.high + self.low) / 2.0

    @property
    def upper_quadrant(self) -> float:
        return self.low + 0.75 * self.range

    @property
    def lower_quadrant(self) -> float:
        return self.low + 0.25 * self.range

    def quadrant(self, price: float) -> float:
        """Normalized quadrant level (0=low, 1=high)."""
        return _normalize(price, self.low, self.high)


# ---------------------------------------------------------------- state machine
class AsiaState(Enum):
    ACCUMULATION = auto()       # inside NDOG range; watching for Judas Swing
    JUDAS_SWING = auto()        # 7:50 macro: manipulation in progress
    WAIT_DISPLACEMENT = auto()  # sweep done; wait for the delivery displacement
    WAIT_ENTRY = auto()         # FVG identified; wait for price to rebalance
    DONE = auto()


class AsiaKillzoneModel:
    """State machine for the Asia Killzone NDOG/NWOG model.

    Usage::

        model = AsiaKillzoneModel(config)
        # at 6 PM, tell it the prior session's close:
        model.start_session(prev_4pm_close=18500.0, nwog_high=18600.0, nwog_low=18400.0,
                            htf_bias=Direction.LONG)
        for candle in overnight_candles:
            sig = model.on_candle(candle)
            if sig: handle(sig)
    """

    def __init__(self, config: Config):
        self.cfg = config
        self._reset()

    def start_session(self, prev_4pm_close: float,
                      nwog_high: Optional[float] = None,
                      nwog_low: Optional[float] = None,
                      htf_bias: Optional[Direction] = None) -> None:
        self._reset()
        self._prev_close = prev_4pm_close
        self._nwog_high = nwog_high
        self._nwog_low = nwog_low
        self.htf_bias = htf_bias

    def on_candle(self, candle: Candle) -> Optional[Signal]:
        """Process one closed candle; return a Signal on entry."""
        t = candle.time.time()
        self._series.append(candle)
        idx = len(self._series) - 1

        if self.state is AsiaState.DONE:
            return None

        # Build the NDOG from the very first candle of the session.
        if self._ndog is None and t >= ASIA_OPEN:
            self._ndog = NDOG(
                open_price=candle.open,
                prev_close=self._prev_close or candle.open,
                formed_at=candle.time,
            )

        if self._ndog is None:
            return None

        # Hard cutoff: stop looking for entries after Asia session ends.
        if t > ASIA_CLOSE:
            self.state = AsiaState.DONE
            return None

        if self.state is AsiaState.ACCUMULATION:
            self._update_session_range(candle)
            if _in_asia_macro(t):
                self.state = AsiaState.JUDAS_SWING
            return None

        if self.state is AsiaState.JUDAS_SWING:
            self._update_session_range(candle)
            if not _in_asia_macro(t):
                # Check whether a sweep happened during the macro window.
                if self._sweep_extreme is not None:
                    self.state = AsiaState.WAIT_DISPLACEMENT
                else:
                    # No sweep yet; try the next macro window.
                    self.state = AsiaState.ACCUMULATION
                return None
            self._detect_judas_sweep(candle)
            return None

        if self.state is AsiaState.WAIT_DISPLACEMENT:
            return self._await_displacement(candle, idx)

        if self.state is AsiaState.WAIT_ENTRY:
            return self._await_entry(candle, idx)

        return None

    # ----------------------------------------------------------------- helpers
    def _reset(self) -> None:
        self.state = AsiaState.ACCUMULATION
        self._series: List[Candle] = []
        self._ndog: Optional[NDOG] = None
        self._prev_close: Optional[float] = None
        self._nwog_high: Optional[float] = None
        self._nwog_low: Optional[float] = None
        self.htf_bias: Optional[Direction] = None
        self._session_high: float = float("-inf")
        self._session_low: float = float("inf")
        self._sweep_side: Optional[Direction] = None
        self._sweep_extreme: Optional[float] = None
        self._entry_fvg: Optional[FVG] = None
        self._trade_direction: Optional[Direction] = None
        self._judas_swing_high: Optional[float] = None
        self._judas_swing_low: Optional[float] = None

    def _update_session_range(self, c: Candle) -> None:
        self._session_high = max(self._session_high, c.high)
        self._session_low = min(self._session_low, c.low)

    def _detect_judas_sweep(self, candle: Candle) -> None:
        """Detect a sweep during the 7:50 PM Judas Swing macro window."""
        ndog = self._ndog
        if ndog is None:
            return

        # Track the extremes reached during the Judas window.
        self._judas_swing_high = max(self._judas_swing_high or candle.high, candle.high)
        self._judas_swing_low = min(self._judas_swing_low or candle.low, candle.low)

        swept_above_ndog = candle.high > ndog.high
        swept_below_ndog = candle.low < ndog.low

        # Variant B (IMG_0113): price sweeps above NDOG High (buystops cleared)
        # then expected to drop → SHORT narrative.
        if swept_above_ndog:
            if self.htf_bias is Direction.SHORT or self.htf_bias is None:
                self._sweep_side = Direction.LONG  # swept buyside
                self._sweep_extreme = max(self._sweep_extreme or candle.high, candle.high)

        # Variant A (reversal): price sweeps below NDOG Low (sellstops cleared)
        # then expected to rally → LONG narrative.
        if swept_below_ndog:
            if self.htf_bias is Direction.LONG or self.htf_bias is None:
                self._sweep_side = Direction.SHORT  # swept sellside
                self._sweep_extreme = min(self._sweep_extreme or candle.low, candle.low)

    def _await_displacement(self, candle: Candle, idx: int) -> Optional[Signal]:
        """After the Judas Swing, wait for the displacement that creates the FVG."""
        if self._sweep_side is None:
            return None

        # Delivery direction is OPPOSITE to what was swept.
        if self._sweep_side is Direction.LONG:
            direction = Direction.SHORT   # buyside swept → short delivery
        else:
            direction = Direction.LONG    # sellside swept → long delivery

        # Bias conflict check.
        if self.htf_bias is not None and self.htf_bias is not direction:
            self.state = AsiaState.DONE
            return None

        a = atr(self._series[:idx], self.cfg.strategy.atr_period)
        fvg = find_fvg(self._series, idx)
        if fvg and fvg.direction is direction:
            if a <= 0 or fvg.size >= 0.25 * a:
                self._entry_fvg = fvg
                self._trade_direction = direction
                self.state = AsiaState.WAIT_ENTRY
        return None

    def _await_entry(self, candle: Candle, idx: int) -> Optional[Signal]:
        """Trigger when price rebalances into the FVG during the 8:50 PM macro."""
        if self._entry_fvg is None or self._trade_direction is None:
            return None

        # Must be inside the 8:50 PM delivery macro window.
        if not _in_asia_macro(candle.time.time()):
            return None

        direction = self._trade_direction
        level = self._entry_fvg.mid

        triggered = (candle.low <= level if direction is Direction.LONG
                     else candle.high >= level)
        if not triggered:
            return None

        return self._build_signal(candle, direction, level)

    def _build_signal(self, candle: Candle, direction: Direction,
                      entry: float) -> Optional[Signal]:
        if self._ndog is None:
            return None

        tick = self.cfg.instrument.tick_size
        buf = self.cfg.strategy.stop_buffer_ticks * tick
        ndog = self._ndog

        # Draw on liquidity, in priority order:
        #   1. NWOG / PWH-PWL when supplied and it sits beyond the entry, then
        #   2. the opposite NDOG extreme, then
        #   3. a 1.0-SD projection of the NDOG range past that extreme.
        # We take whichever is *furthest* in the trade direction so the draw is
        # a genuine target rather than a level price has already reached.
        rng_ = ndog.range
        if direction is Direction.LONG:
            stop = (self._sweep_extreme or ndog.low) - buf
            candidates = [ndog.high, ndog.high + rng_]
            if self._nwog_high:
                candidates.append(self._nwog_high)
            target = max(c for c in candidates if c > entry) \
                if any(c > entry for c in candidates) else ndog.high + rng_
        else:
            stop = (self._sweep_extreme or ndog.high) + buf
            candidates = [ndog.low, ndog.low - rng_]
            if self._nwog_low:
                candidates.append(self._nwog_low)
            target = min(c for c in candidates if c < entry) \
                if any(c < entry for c in candidates) else ndog.low - rng_

        # Sanity: stop and target must each sit on the correct side of entry.
        if direction is Direction.LONG and (stop >= entry or target <= entry):
            self.state = AsiaState.DONE
            return None
        if direction is Direction.SHORT and (stop <= entry or target >= entry):
            self.state = AsiaState.DONE
            return None

        # Reject degenerate, near-zero-risk entries (stop too close to entry).
        if ndog.range > 0 and abs(entry - stop) < 0.25 * ndog.range:
            self.state = AsiaState.DONE
            return None

        risk = abs(entry - stop)
        if risk == 0:
            return None
        reward = abs(target - entry)
        if reward / risk < self.cfg.risk.min_rr:
            self.state = AsiaState.DONE
            return None

        sweep_name = "sellside" if direction is Direction.LONG else "buyside"
        signal = Signal(
            time=candle.time,
            direction=direction,
            entry=entry,
            stop=stop,
            target=target,
            model="AsiaKZ",
            reason=(f"Asia KZ: Judas Swing swept {sweep_name} below NDOG "
                    f"-> 8:50 macro displacement -> 1st PFVG CE rebalance "
                    f"-> target NDOG {'high' if direction is Direction.LONG else 'low'}/"
                    f"{'NWOG' if (self._nwog_high or self._nwog_low) else 'DOL'}"),
            pd_array=self._entry_fvg,
        )
        self.state = AsiaState.DONE
        return signal
