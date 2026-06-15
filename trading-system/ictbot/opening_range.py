"""Cat 4: The 5 ICT Opening Range systems.

Every session has its own 30-minute window that sets the blueprint for delivery.
The algorithm establishes 3 key reference points: Opening Price, Session High,
Session Low.  Everything that follows — liquidity raids, FVGs, price runs —
is built around those reference points.

The 5 Opening Ranges (all times New York / Eastern):

  Name            Window           Purpose
  ──────────────  ───────────────  ───────────────────────────────────────
  Midnight OR     12:00–12:30 AM   Foundation of the full trading day
  London OR       01:30–02:00 AM   First real directional move of the day
  NY Kill Zone    07:00–07:30 AM   Pre-market setup; covers all instruments
  AM Session OR   09:30–10:00 AM   Most powerful — sets the AM session
  PM Session OR   01:30–02:00 PM   Afternoon push; aligns with PM macros

All five share the same state machine:
  BUILDING → ARMED → WAIT_SWEEP → WAIT_DISPLACEMENT → WAIT_ENTRY → DONE

Standard Deviation (SD) targets from the notes (AM OR):
  0.5  SD = first target
  1.0  SD = standard target
  1.5  SD = extended target
  2.5  SD = full blow-off target

Projection labels (from the charts):
  0 = OR Low, 0.5 = OR Midpoint/CE, 1 = OR High
  -0.5 / -1 / -1.5 / -2 / -2.5 / -3  = bearish extensions below OR Low
  1.5 / 2 / 2.5 / 3                   = bullish extensions above OR High

Invalidation conditions (from the notes):
  no displacement, HTF conflict, no liquidity sweep, weak FVG, choppy delivery,
  failed MSS
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time
from enum import Enum, auto
from typing import List, Optional, Tuple

from .config import Config
from .indicators import atr, find_fvg, is_displacement
from .models import (Candle, Direction, FVG, LiquidityPool, OpeningRange,
                     Signal)
from .pd_arrays import nearest_order_block, scan_breaker_blocks


# SD target multipliers used as exits, in ascending order.
SD_TARGETS = (0.5, 1.0, 1.5, 2.5)


class ORState(Enum):
    BUILDING = auto()       # still inside the opening range window
    ARMED = auto()          # range is locked; watching for a sweep
    WAIT_SWEEP = auto()     # synonymous with ARMED (alias for clarity)
    WAIT_DISPLACEMENT = auto()
    WAIT_ENTRY = auto()
    DONE = auto()


@dataclass
class ORWindow:
    """Definition of one of the 5 opening-range windows."""

    label: str
    start: time   # inclusive
    end: time     # inclusive (the last bar of the window)
    # Macro windows immediately after the OR that are valid entry times.
    macro_windows: List[Tuple[time, time]]
    # How many macro-window bars we allow the entry to take.
    # A sweep outside the window that is detected after OR closes still counts.
    sweep_cutoff_bars: int = 60  # bars after OR end before we give up


# The 5 canonical OR windows (all times NY / Eastern).
OR_DEFINITIONS: dict[str, ORWindow] = {
    "Midnight": ORWindow(
        label="Midnight",
        start=time(0, 0),
        end=time(0, 30),
        macro_windows=[(time(2, 50), time(3, 10)), (time(3, 15), time(3, 45))],
    ),
    "London": ORWindow(
        label="London",
        start=time(1, 30),
        end=time(2, 0),
        macro_windows=[(time(3, 15), time(3, 45)), (time(4, 0), time(4, 30))],
    ),
    "NYKillZone": ORWindow(
        label="NYKillZone",
        start=time(7, 0),
        end=time(7, 30),
        macro_windows=[(time(7, 50), time(8, 10)), (time(8, 50), time(9, 10))],
    ),
    "AM": ORWindow(
        label="AM",
        start=time(9, 30),
        end=time(10, 0),
        macro_windows=[(time(9, 50), time(10, 10)), (time(10, 50), time(11, 10))],
    ),
    "PM": ORWindow(
        label="PM",
        start=time(13, 30),
        end=time(14, 0),
        macro_windows=[(time(14, 50), time(15, 10)), (time(15, 15), time(15, 45))],
    ),
}


def _in_macro(t: time, windows: List[Tuple[time, time]]) -> bool:
    return any(s <= t <= e for s, e in windows)


class OpeningRangeModel:
    """State machine for one of the 5 ICT Opening Range strategies.

    Instantiate with one of the :data:`OR_DEFINITIONS` and a :class:`Config`,
    then call :meth:`on_candle` for every closed bar.  The model resets itself
    for the next session automatically.
    """

    def __init__(self, window: ORWindow, config: Config):
        self.window = window
        self.cfg = config
        self._reset()

    # ------------------------------------------------------------------ public
    def on_candle(self, candle: Candle,
                  htf_bias: Optional[Direction] = None) -> Optional[Signal]:
        """Process one closed candle.  Returns a Signal on entry, else None."""
        t = candle.time.time()
        self._series.append(candle)
        idx = len(self._series) - 1

        # ---- building the OR ------------------------------------------------
        if self.window.start <= t <= self.window.end:
            if self.state is not ORState.BUILDING:
                self._reset()
            self._build_or_candle(candle, idx)
            return None

        # ---- transition: OR just closed, arm the model ----------------------
        if self.state is ORState.BUILDING and self._or is not None:
            self.state = ORState.ARMED
            self.htf_bias = htf_bias
            self._bars_since_or = 0
            return None

        if self.state in (ORState.DONE, ORState.BUILDING):
            return None

        self._bars_since_or += 1
        if self._bars_since_or > self.window.sweep_cutoff_bars:
            self.state = ORState.DONE
            return None

        if self.state is ORState.ARMED:
            self._check_sweep(candle, idx)
            return None
        if self.state is ORState.WAIT_DISPLACEMENT:
            self._check_displacement(candle, idx)
            return None
        if self.state is ORState.WAIT_ENTRY:
            return self._check_entry(candle, idx)
        return None

    @property
    def opening_range(self) -> Optional[OpeningRange]:
        return self._or

    # ----------------------------------------------------------------- private
    def _reset(self) -> None:
        self.state = ORState.BUILDING
        self._or: Optional[OpeningRange] = None
        self._or_candles: List[Candle] = []
        self._or_high = float("-inf")
        self._or_low = float("inf")
        self._or_open: Optional[float] = None
        self._series: List[Candle] = []
        self.htf_bias: Optional[Direction] = None
        self._sweep_side: Optional[Direction] = None
        self._sweep_extreme: Optional[float] = None
        self._entry_fvg: Optional[FVG] = None
        self._bars_since_or: int = 0

    def _build_or_candle(self, c: Candle, idx: int) -> None:
        if self._or_open is None:
            self._or_open = c.open
        self._or_high = max(self._or_high, c.high)
        self._or_low = min(self._or_low, c.low)
        self._or_candles.append(c)

        # Detect the first FVG formed inside the OR window.
        if len(self._or_candles) >= 3:
            fvg = find_fvg(self._or_candles, len(self._or_candles) - 1)
            if fvg is not None:
                existing_pfvg = self._or.first_pfvg if self._or else None
                if existing_pfvg is None:
                    # Build the OR object now (even though the window isn't closed)
                    # so we can store the PFVG reference.
                    pass
        # (Re)build the OpeningRange each time to keep it current.
        if self._or_open is not None and self._or_high > float("-inf"):
            pfvg = self._or.first_pfvg if self._or else None
            if pfvg is None and len(self._or_candles) >= 3:
                fvg = find_fvg(self._or_candles, len(self._or_candles) - 1)
                if fvg:
                    pfvg = fvg
            self._or = OpeningRange(
                label=self.window.label,
                or_open=self._or_open,
                or_high=self._or_high,
                or_low=self._or_low,
                formed_at=c.time,
                first_pfvg=pfvg,
            )

    def _check_sweep(self, candle: Candle, idx: int) -> None:
        """Detect a liquidity sweep of one side of the OR."""
        if self._or is None:
            return
        swept_high = candle.high > self._or.or_high
        swept_low = candle.low < self._or.or_low

        # If we have a HTF bias, we want the sweep to run AGAINST it
        # (manipulation), then expect delivery IN the bias direction.
        if self.htf_bias is Direction.LONG and swept_low:
            self._sweep_side = Direction.SHORT  # swept the low (sellside)
            self._sweep_extreme = min(self._sweep_extreme or candle.low, candle.low)
            self.state = ORState.WAIT_DISPLACEMENT
        elif self.htf_bias is Direction.SHORT and swept_high:
            self._sweep_side = Direction.LONG
            self._sweep_extreme = max(self._sweep_extreme or candle.high, candle.high)
            self.state = ORState.WAIT_DISPLACEMENT
        elif self.htf_bias is None:
            # No bias: accept the first sweep in either direction.
            if swept_high and not swept_low:
                self._sweep_side = Direction.LONG  # swept high → expect drop
                self._sweep_extreme = candle.high
                self.state = ORState.WAIT_DISPLACEMENT
            elif swept_low and not swept_high:
                self._sweep_side = Direction.SHORT
                self._sweep_extreme = candle.low
                self.state = ORState.WAIT_DISPLACEMENT

    def _check_displacement(self, candle: Candle, idx: int) -> None:
        """Wait for displacement + FVG after the sweep."""
        if self._or is None or self._sweep_side is None:
            return

        # The expected delivery direction is OPPOSITE to what was swept.
        delivery = self._sweep_side.opposite()
        a = atr(self._series[:idx], self.cfg.strategy.atr_period)
        middle = self._series[idx - 1] if idx >= 1 else candle
        fvg = find_fvg(self._series, idx)

        # Accept the first PFVG that aligns with delivery direction.
        if fvg and fvg.direction is delivery:
            # Confirm the middle candle was expansive enough.
            if a <= 0 or fvg.size >= 0.3 * a:
                self._entry_fvg = fvg
                self._sweep_side = delivery  # repurpose: now stores trade direction
                self.state = ORState.WAIT_ENTRY

    def _check_entry(self, candle: Candle, idx: int) -> Optional[Signal]:
        """Trigger entry when price rebalances into the FVG during a macro window."""
        if self._or is None or self._entry_fvg is None:
            return None

        direction = self._sweep_side  # repurposed in _check_displacement

        # Entries only valid inside a macro time window.
        if not _in_macro(candle.time.time(), self.window.macro_windows):
            return None

        level = self._entry_fvg.mid
        triggered = (candle.low <= level if direction is Direction.LONG
                     else candle.high >= level)
        if not triggered:
            return None

        return self._build_signal(candle, direction, level)

    def _build_signal(self, candle: Candle, direction: Direction,
                      entry: float) -> Optional[Signal]:
        if self._or is None:
            return None
        tick = self.cfg.instrument.tick_size
        buf = self.cfg.strategy.stop_buffer_ticks * tick

        # Stop beyond the sweep extreme.
        if direction is Direction.LONG:
            stop = (self._sweep_extreme or self._or.or_low) - buf
            target = self._or.sd_target(direction, 1.0)
        else:
            stop = (self._sweep_extreme or self._or.or_high) + buf
            target = self._or.sd_target(direction, 1.0)

        # Escalate target to 1.5 SD if HTF draw aligns.
        if self.htf_bias is direction:
            target = self._or.sd_target(direction, 1.5)

        # Sanity: stop must be strictly on the losing side of entry.
        if direction is Direction.SHORT and stop <= entry:
            self.state = ORState.DONE
            return None
        if direction is Direction.LONG and stop >= entry:
            self.state = ORState.DONE
            return None

        from .models import Signal as Sig
        signal = Sig(
            time=candle.time,
            direction=direction,
            entry=entry,
            stop=stop,
            target=target,
            model=f"OR_{self.window.label}",
            reason=(f"{self.window.label} OR: swept {'low' if direction is Direction.LONG else 'high'} "
                    f"-> 1st PFVG rebalance -> target {self._or.normalized_level(target):.1f} SD"),
            pd_array=self._entry_fvg,
        )
        if signal.rr >= self.cfg.risk.min_rr:
            self.state = ORState.DONE
            return signal
        self.state = ORState.DONE
        return None


def build_all_or_models(config: Config) -> dict[str, OpeningRangeModel]:
    """Instantiate all 5 OR models for the given instrument config."""
    return {name: OpeningRangeModel(win, config)
            for name, win in OR_DEFINITIONS.items()}
