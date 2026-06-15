"""The ICT 9:30 A.M. Open Model, implemented as a per-session state machine.

The flow mirrors the flowchart in the notes:

    Define Liquidity -> Manipulation -> Wait for Alignment w/ HTF Bias & Draw
                     -> Entry Model -> Targets

State machine (one pass per session):

    WAIT_OPEN          accumulate the pre-market range; if a counter-bias pool
                       is raided *before* 09:30 we are in the CONTINUATION model
    WAIT_MANIPULATION  after 09:30 wait for the raid on resting liquidity that
                       runs *against* the HTF bias (the REVERSAL model)
    WAIT_DISPLACEMENT  wait for the first expansion leg back in the bias
                       direction that prints an FVG - the "1st PFVG" PD Array
    WAIT_ENTRY         inside a macro window, wait for price to rebalance into
                       that FVG (the IOFED entry) and emit the trade signal
    DONE               one setup per session (anchor on the first PFVG)

The class is fed one *closed* candle at a time via :meth:`on_candle`, so it is
equally usable for backtesting and for live trading.
"""

from __future__ import annotations

from datetime import date
from enum import Enum, auto
from typing import List, Optional

from .config import Config
from .indicators import atr, find_fvg, is_displacement
from .liquidity import detect_sweep, nearest_target_pool
from .models import Candle, Direction, FVG, LiquidityPool, Signal


class State(Enum):
    WAIT_OPEN = auto()
    WAIT_MANIPULATION = auto()
    WAIT_DISPLACEMENT = auto()
    WAIT_ENTRY = auto()
    DONE = auto()


class NineThirtyOpenModel:
    """Stateful detector for a single trading session."""

    def __init__(self, config: Config):
        self.cfg = config
        self._reset()

    # ------------------------------------------------------------------ setup
    def start_session(self, session_date: date,
                      pools: List[LiquidityPool],
                      bias: Optional[Direction]) -> None:
        """Prime the model for a new session."""
        self._reset()
        self.session_date = session_date
        self.pools = pools
        self.bias = bias

    def _reset(self) -> None:
        self.state = State.WAIT_OPEN
        self.session_date = None
        self.pools: List[LiquidityPool] = []
        self.bias: Optional[Direction] = None
        self.model_label: Optional[str] = None
        self.swept_pool: Optional[LiquidityPool] = None
        self.manip_extreme: Optional[float] = None
        self.manip_time = None
        self.entry_fvg: Optional[FVG] = None
        self._series: List[Candle] = []
        self._open_seen = False
        self._open_index: Optional[int] = None

    # --------------------------------------------------------------- main loop
    def on_candle(self, candle: Candle) -> Optional[Signal]:
        """Process one closed candle; return a Signal if an entry triggers."""
        self._series.append(candle)
        t = candle.time.time()
        s = self.cfg.session

        # Nothing to do without a directional read.
        if self.bias is None:
            return None

        # ---- pre-market: only watch for an early (continuation) raid --------
        if t < s.open_time:
            self._track_premarket_sweep(candle)
            return None

        # First bar at/after the open: decide Reversal vs Continuation.
        if not self._open_seen:
            self._open_seen = True
            self._open_index = len(self._series) - 1
            if self.swept_pool is not None:
                # Liquidity was already cleared pre-open -> CONTINUATION.
                self.model_label = "Continuation"
                self.state = State.WAIT_DISPLACEMENT
            else:
                self.model_label = "Reversal"
                self.state = State.WAIT_MANIPULATION

        # Hard stop: flatten the search after the setup cutoff.
        if t > s.setup_cutoff and self.state != State.DONE:
            self.state = State.DONE
            return None

        if self.state is State.WAIT_MANIPULATION:
            self._await_manipulation(candle)
            return None
        if self.state is State.WAIT_DISPLACEMENT:
            self._await_displacement(candle)
            return None
        if self.state is State.WAIT_ENTRY:
            return self._await_entry(candle)
        return None

    # ----------------------------------------------------------- sub-routines
    def _counter_bias_pools(self) -> List[LiquidityPool]:
        """Pools whose raid runs *against* the HTF bias (the manipulation)."""
        want = self.bias.opposite()  # long bias -> raid sellside (lows)
        return [p for p in self.pools if p.side is want and not p.swept]

    def _track_premarket_sweep(self, candle: Candle) -> None:
        for pool in self._counter_bias_pools():
            if detect_sweep(candle, pool):
                pool.swept = True
                pool.swept_at = candle.time
                self.swept_pool = pool
                self._update_manip_extreme(candle)

    def _update_manip_extreme(self, candle: Candle) -> None:
        """Track the extreme of the manipulation leg (stop reference)."""
        if self.bias is Direction.LONG:
            self.manip_extreme = (candle.low if self.manip_extreme is None
                                  else min(self.manip_extreme, candle.low))
        else:
            self.manip_extreme = (candle.high if self.manip_extreme is None
                                  else max(self.manip_extreme, candle.high))
        self.manip_time = candle.time

    def _await_manipulation(self, candle: Candle) -> None:
        bars_since_open = len(self._series) - 1 - (self._open_index or 0)
        if bars_since_open > self.cfg.strategy.manipulation_window_bars:
            self.state = State.DONE
            return
        for pool in self._counter_bias_pools():
            if detect_sweep(candle, pool):
                pool.swept = True
                pool.swept_at = candle.time
                self.swept_pool = pool
                self._update_manip_extreme(candle)
                self.state = State.WAIT_DISPLACEMENT
                return

    def _await_displacement(self, candle: Candle) -> None:
        # Keep tracking the manipulation extreme until price turns.
        if self.bias is Direction.LONG and candle.low < (self.manip_extreme or candle.low):
            self._update_manip_extreme(candle)
        elif self.bias is Direction.SHORT and candle.high > (self.manip_extreme or candle.high):
            self._update_manip_extreme(candle)

        idx = len(self._series) - 1
        fvg = find_fvg(self._series, idx)
        if fvg is None or fvg.direction is not self.bias:
            return
        # An FVG is the footprint of displacement (the middle candle expands and
        # leaves the imbalance). Confirm that middle candle was a genuine
        # expansion, or that the resulting gap itself is meaningfully large.
        a = atr(self._series[:idx], self.cfg.strategy.atr_period)
        middle_displaced = is_displacement(
            self._series, idx - 1,
            self.cfg.strategy.displacement_atr_mult,
            self.cfg.strategy.atr_period)
        if not middle_displaced and (a <= 0 or fvg.size < 0.3 * a):
            return
        # The 1st PFVG must sit on the correct side of the manipulation extreme.
        if self.bias is Direction.LONG and fvg.bottom < (self.manip_extreme or fvg.bottom):
            return
        if self.bias is Direction.SHORT and fvg.top > (self.manip_extreme or fvg.top):
            return
        self.entry_fvg = fvg
        self.state = State.WAIT_ENTRY

    def _entry_level(self) -> float:
        f = self.entry_fvg
        depth = self.cfg.strategy.fvg_entry_depth
        # Retrace *into* the gap: long enters as price drops from the top edge
        # toward the mid; short enters as price rises from the bottom edge.
        if self.bias is Direction.LONG:
            return f.top - depth * f.size
        return f.bottom + depth * f.size

    def _await_entry(self, candle: Candle) -> Optional[Signal]:
        f = self.entry_fvg
        # Invalidation: price violates the manipulation extreme before entry.
        if self.bias is Direction.LONG and candle.low < (self.manip_extreme or candle.low):
            self.state = State.DONE
            return None
        if self.bias is Direction.SHORT and candle.high > (self.manip_extreme or candle.high):
            self.state = State.DONE
            return None

        # Entries are only valid inside a macro window (per the notes).
        if not self.cfg.session.in_macro(candle.time.time()):
            return None

        level = self._entry_level()
        triggered = (candle.low <= level if self.bias is Direction.LONG
                     else candle.high >= level)
        if not triggered:
            return None

        return self._build_signal(candle, level)

    def _build_signal(self, candle: Candle, entry: float) -> Optional[Signal]:
        tick = self.cfg.instrument.tick_size
        buf = self.cfg.strategy.stop_buffer_ticks * tick
        if self.bias is Direction.LONG:
            stop = (self.manip_extreme or candle.low) - buf
        else:
            stop = (self.manip_extreme or candle.high) + buf

        risk = abs(entry - stop)
        min_dist = risk * self.cfg.risk.min_rr
        target_pool = nearest_target_pool(self.pools, self.bias, entry,
                                          min_distance=min_dist)
        if target_pool is None:
            self.state = State.DONE
            return None
        target = target_pool.price

        signal = Signal(
            time=candle.time,
            direction=self.bias,
            entry=entry,
            stop=stop,
            target=target,
            model=self.model_label or "Reversal",
            reason=(f"{self.model_label}: raid on {self.swept_pool.name} -> "
                    f"realigned with {self.bias.name} HTF bias -> entry at 1st "
                    f"PFVG, draw on {target_pool.name}"),
            pd_array=self.entry_fvg,
            target_pool=target_pool,
        )

        # Respect the minimum reward:risk filter.
        if signal.rr < self.cfg.risk.min_rr:
            self.state = State.DONE
            return None

        self.state = State.DONE
        return signal
