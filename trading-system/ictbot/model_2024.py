"""Cat 2: ICT 2024 Model — "The Setup For Life".

A simplified, beginner-friendly distillation of the full ICT framework.
Source: IMG_1056 ("The Setup For Life BSL") and IMG_0093 (invalidation rules).

Checklist (from the notes):
  □ Micro time validation: ONLY valid during macro time windows
  □ Trading sessions: 9:50–10:10 AM and 10:50–11:10 AM (+ PM macros)
  □ Premium/Discount PD Array: price must hit a premium OR discount PD array
    - For shorts: price must be at a Premium PD Array
    - For longs:  price must be at a Discount PD Array
  □ Stop Hunt → 1st Entry FVG → 2nd Entry Breaker Block

Macro times (NY Time) from the notes:
  AM: 7:50–8:10, 8:50–9:10, 9:50–10:10, 10:50–12:10
  PM: 1:20–1:40, 2:50–3:10, 3:15–3:45, 3:50–4:10

Flow:
  1. Establish HTF bias and identify the session Draw on Liquidity (DOL)
  2. Wait for a macro time window
  3. Price reaches a Premium (for SHORT) or Discount (for LONG) PD Array
     → the HFT Premium PD Array reference from the notes
  4. A Stop Hunt (liquidity sweep) occurs within or just before the macro window
  5. Displacement occurs away from the swept level
  6. 1st Entry: FVG from the displacement (FVG CE = entry)
     OR
     2nd Entry: Breaker Block from the displacement (Breaker CE = entry)
  7. Target = session Draw on Liquidity (nearest liquidity pool in bias direction)

Invalidation (from IMG_0093 — what invalidates an opening range / setup):
  - No displacement
  - HTF conflict (bias doesn't support the trade)
  - News events (not coded; operator's responsibility)
  - Choppy delivery (detected via low displacement ATR mult)
  - No liquidity sweep
  - Weak FVG (gap < min size threshold)
  - Failed MSS (market structure shift doesn't confirm)
"""

from __future__ import annotations

from datetime import time
from enum import Enum, auto
from typing import List, Optional, Tuple

from .config import Config
from .indicators import atr, find_fvg, is_displacement
from .models import Candle, Direction, FVG, LiquidityPool, Signal
from .pd_arrays import nearest_breaker, nearest_order_block


# ---------------------------------------------------------------- macro clocks
AM_MACROS: List[Tuple[time, time]] = [
    (time(7, 50), time(8, 10)),
    (time(8, 50), time(9, 10)),
    (time(9, 50), time(10, 10)),
    (time(10, 50), time(12, 10)),
]
PM_MACROS: List[Tuple[time, time]] = [
    (time(13, 20), time(13, 40)),
    (time(14, 50), time(15, 10)),
    (time(15, 15), time(15, 45)),
    (time(15, 50), time(16, 10)),
]
ALL_MACROS = AM_MACROS + PM_MACROS


def in_any_macro(t: time) -> bool:
    return any(s <= t <= e for s, e in ALL_MACROS)


# ------------------------------------------------------------------- constants
# A PD Array is "premium" (for shorts) when normalized above this midpoint.
PREMIUM_THRESHOLD = 0.618
DISCOUNT_THRESHOLD = 0.382


def _is_premium(price: float, session_high: float,
                session_low: float) -> bool:
    if session_high == session_low:
        return False
    normalized = (price - session_low) / (session_high - session_low)
    return normalized >= PREMIUM_THRESHOLD


def _is_discount(price: float, session_high: float,
                 session_low: float) -> bool:
    if session_high == session_low:
        return False
    normalized = (price - session_low) / (session_high - session_low)
    return normalized <= DISCOUNT_THRESHOLD


# ---------------------------------------------------------------- state machine
class SetupState(Enum):
    IDLE = auto()              # no macro window yet
    IN_MACRO = auto()          # inside a macro window; watching for sweep
    WAIT_SWEEP = auto()        # macro window passed; sweep detected; wait for disp
    WAIT_DISPLACEMENT = auto()
    WAIT_ENTRY = auto()
    DONE = auto()


class SetupForLife:
    """ICT 2024 "Setup For Life" state machine.

    Feed closed candles with the session's running high/low and a HTF bias.
    Returns a Signal on the first valid entry, then resets to DONE for the day.

    Usage::

        model = SetupForLife(cfg)
        model.start_session(pools, htf_bias)
        for candle in session_candles:
            sig = model.on_candle(candle, session_high, session_low)
            if sig: handle(sig)
    """

    def __init__(self, config: Config):
        self.cfg = config
        self._reset()

    def start_session(self, pools: List[LiquidityPool],
                      bias: Optional[Direction]) -> None:
        self._reset()
        self.pools = pools
        self.bias = bias

    def on_candle(self, candle: Candle,
                  session_high: float,
                  session_low: float) -> Optional[Signal]:
        """Process one closed candle; return a Signal on entry."""
        self._series.append(candle)
        idx = len(self._series) - 1
        t = candle.time.time()

        if self.state is SetupState.DONE:
            return None

        # Hard cutoff: no new setups after 3:45 PM.
        if t > time(15, 45):
            self.state = SetupState.DONE
            return None

        # Enter macro state.
        if in_any_macro(t) and self.state is SetupState.IDLE:
            self.state = SetupState.IN_MACRO
            self._macro_high = candle.high
            self._macro_low = candle.low
            self._session_high = session_high
            self._session_low = session_low
            return None

        if self.state is SetupState.IN_MACRO:
            if not in_any_macro(t):
                # Macro window closed.
                self.state = SetupState.IDLE
                self._macro_high = None
                self._macro_low = None
                return None
            self._macro_high = max(self._macro_high or candle.high, candle.high)
            self._macro_low = min(self._macro_low or candle.low, candle.low)
            # Check for stop hunt during the macro window.
            if self._detect_sweep(candle, session_high, session_low):
                self.state = SetupState.WAIT_DISPLACEMENT
            return None

        if self.state is SetupState.WAIT_DISPLACEMENT:
            return self._await_displacement(candle, idx)

        if self.state is SetupState.WAIT_ENTRY:
            return self._await_entry(candle, idx)

        return None

    # ---------------------------------------------------------------- helpers
    def _detect_sweep(self, candle: Candle,
                      session_high: float, session_low: float) -> bool:
        """Was a liquidity pool raided during this bar?"""
        for pool in self.pools:
            if pool.swept:
                continue
            if pool.is_buyside and candle.high > pool.price:
                pool.swept = True
                pool.swept_at = candle.time
                self._swept_pool = pool
                self._sweep_extreme = candle.high
                # For a short: sweep must be at a premium PD array.
                if self.bias is Direction.SHORT or self.bias is None:
                    if _is_premium(pool.price, session_high, session_low) or \
                       self.bias is None:
                        return True
            elif not pool.is_buyside and candle.low < pool.price:
                pool.swept = True
                pool.swept_at = candle.time
                self._swept_pool = pool
                self._sweep_extreme = candle.low
                if self.bias is Direction.LONG or self.bias is None:
                    if _is_discount(pool.price, session_high, session_low) or \
                       self.bias is None:
                        return True
        return False

    def _await_displacement(self, candle: Candle, idx: int) -> Optional[Signal]:
        """Look for the displacement candle + FVG after the stop hunt."""
        if self._swept_pool is None:
            return None

        # Determine expected delivery direction from the sweep.
        if self._swept_pool.is_buyside:
            direction = Direction.SHORT   # swept highs → expect drop
        else:
            direction = Direction.LONG    # swept lows → expect rise

        # Confirm HTF bias alignment; if bias conflicts → invalidate.
        if self.bias is not None and self.bias is not direction:
            self.state = SetupState.DONE
            return None

        a = atr(self._series[:idx], self.cfg.strategy.atr_period)
        fvg = find_fvg(self._series, idx)
        if fvg and fvg.direction is direction:
            if a <= 0 or fvg.size >= 0.3 * a:
                self._entry_fvg = fvg
                self._trade_direction = direction
                self.state = SetupState.WAIT_ENTRY
        return None

    def _await_entry(self, candle: Candle, idx: int) -> Optional[Signal]:
        """Trigger when price rebalances into the FVG during a macro window."""
        if self._entry_fvg is None or self._trade_direction is None:
            return None

        # Entry must occur inside a macro window.
        if not in_any_macro(candle.time.time()):
            return None

        direction = self._trade_direction
        level = self._entry_fvg.mid

        triggered = (candle.low <= level if direction is Direction.LONG
                     else candle.high >= level)
        if not triggered:
            # 2nd entry attempt: Breaker Block retest.
            bb = nearest_breaker(self._series, direction,
                                 candle.close, up_to=idx + 1)
            if bb:
                bb_entry = bb.ce
                triggered = (candle.low <= bb_entry if direction is Direction.LONG
                             else candle.high >= bb_entry)
                if triggered:
                    level = bb_entry
            if not triggered:
                return None

        return self._build_signal(candle, direction, level)

    def _build_signal(self, candle: Candle, direction: Direction,
                      entry: float) -> Optional[Signal]:
        tick = self.cfg.instrument.tick_size
        buf = self.cfg.strategy.stop_buffer_ticks * tick

        if direction is Direction.LONG:
            stop = (self._sweep_extreme or candle.low) - buf
        else:
            stop = (self._sweep_extreme or candle.high) + buf

        # Target: nearest draw-on-liquidity in the direction of the trade.
        from .liquidity import nearest_target_pool
        risk = abs(entry - stop)
        target_pool = nearest_target_pool(self.pools, direction, entry,
                                          min_distance=risk * self.cfg.risk.min_rr)
        if target_pool is None:
            self.state = SetupState.DONE
            return None
        target = target_pool.price

        # Sanity check: stop must be strictly on the correct side of entry.
        if direction is Direction.LONG and stop >= entry:
            self.state = SetupState.DONE
            return None
        if direction is Direction.SHORT and stop <= entry:
            self.state = SetupState.DONE
            return None

        signal = Signal(
            time=candle.time,
            direction=direction,
            entry=entry,
            stop=stop,
            target=target,
            model="SetupForLife",
            reason=(f"ICT 2024: stop hunt on {self._swept_pool.name if self._swept_pool else '?'} "
                    f"-> displacement -> FVG CE rebalance -> draw on {target_pool.name}"),
            pd_array=self._entry_fvg,
            target_pool=target_pool,
        )
        if signal.rr < self.cfg.risk.min_rr:
            self.state = SetupState.DONE
            return None
        self.state = SetupState.DONE
        return signal

    def _reset(self) -> None:
        self.state = SetupState.IDLE
        self.pools: List[LiquidityPool] = []
        self.bias: Optional[Direction] = None
        self._series: List[Candle] = []
        self._macro_high: Optional[float] = None
        self._macro_low: Optional[float] = None
        self._session_high: Optional[float] = None
        self._session_low: Optional[float] = None
        self._swept_pool: Optional[LiquidityPool] = None
        self._sweep_extreme: Optional[float] = None
        self._entry_fvg: Optional[FVG] = None
        self._trade_direction: Optional[Direction] = None
