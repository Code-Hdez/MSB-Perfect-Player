"""
Batter controller — state machine that decides WHERE to move and WHEN to
swing based on ball trajectory prediction and current batter/strikezone
position.

No ML model — pure deterministic control logic with tunable parameters.

State machine

  IDLE          →  waiting for a pitch (no ball tracked)
  TRACKING      →  ball detected, building trajectory (not enough data yet)
  ALIGNING      →  prediction available, moving batter toward crossing point
  SWINGING      →  swing command issued, holding A for a few frames
  COOLDOWN      →  postswing recovery, ignore inputs briefly

Inputs (per frame):
  predicted_crossing: (x, y) where the ball will cross target_y
  predicted_frame: frame index of the crossing
  current_frame: current frame index
  strike_zone_center: (x, y) current aim point from MovementTracker
  batter_ready: bool — True when BatterState == NORMAL

Outputs:
  stick_x, stick_y: analog stick target (0.0–1.0, 0.5 = center)
  swing: bool — whether to press A this frame

Usage::

    from msb.batter_controller import BatterController

    ctrl = BatterController()
    ctrl.update(crossing, pred_frame, cur_frame, sz_center, True)
    print(ctrl.stick_x, ctrl.stick_y, ctrl.swing, ctrl.state)
"""

from __future__ import annotations

import time
from enum import Enum, auto
from typing import Optional, Tuple


# STATE ENUM

class ControlState(Enum):
    """Battercontroller lifecycle states."""
    IDLE      = auto()   # no ball tracked
    TRACKING  = auto()   # ball tracked, prediction not ready yet
    ALIGNING  = auto()   # prediction available, moving stick
    SWINGING  = auto()   # swing command active (holding A)
    COOLDOWN  = auto()   # postswing dead time


# BATTER CONTROLLER

class BatterController:
    """Closedloop batter positioning + swing timing.

    All coordinates are in **ROIrelative pixels** (same space as the
    detector / tracker / predictor / MovementTracker).

    Parameters
   
    position_threshold_px : float
        Deadzone in pixels.  If the error between predicted crossing and
        strikezone centre is smaller than this, no movement command is
        issued.  Prevents oscillation / jitter.
    swing_anticipation_frames : int
        How many frames *before* predicted impact to press A.
        Compensates for inputpipeline latency + swing animation startup.
        Start with 4–6 and tune by observation.
    proportional_gain : float
        Gain for mapping pixel error → stick deflection (0–0.5 range added
        to the 0.5 centre).  Higher = faster corrections, but more
        overshoot.  ``1.0 / error_for_full_tilt`` is a good starting heuristic.
    stick_min_deflection : float
        Minimum stick offset from centre (below this, snap to centre to
        avoid microdrifts the emulator might ignore).
    swing_hold_frames : int
        How many frames to keep A pressed after issuing a swing.
    cooldown_frames : int
        Frames to stay in COOLDOWN after swing release before accepting
        a new pitch.
    max_frames_without_update : int
        If we go this many frames without a new prediction while ALIGNING,
        fall back to centre stick (lost ball).
    """

    def __init__(
        self,
        position_threshold_px: float = 15.0,
        swing_anticipation_frames: int = 5,
        proportional_gain: float = 0.008,
        stick_min_deflection: float = 0.05,
        swing_hold_frames: int = 3,
        cooldown_frames: int = 2,
        max_frames_without_update: int = 30,
    )-> None:
        self.position_threshold_px = position_threshold_px
        self.swing_anticipation_frames = swing_anticipation_frames
        self.proportional_gain = proportional_gain
        self.stick_min_deflection = stick_min_deflection
        self.swing_hold_frames = swing_hold_frames
        self.cooldown_frames = cooldown_frames
        self.max_no_update = max_frames_without_update

        self.stick_x: float = 0.5      # 0.0 = full left, 1.0 = full right
        self.stick_y: float = 0.5      # 0.0 = full down,  1.0 = full up
        self.swing: bool = False        # True → press A this frame

        # Internal state
        self.state: ControlState = ControlState.IDLE
        self._frames_to_impact: Optional[int] = None
        self._last_crossing: Optional[Tuple[int, int]] = None
        self._last_pred_frame: Optional[int] = None
        self._error_px: Tuple[float, float] = (0.0, 0.0)
        self._swing_counter: int = 0
        self._cooldown_counter: int = 0
        self._stale_counter: int = 0  # frames since last valid prediction
        self._stale_ts: float = time.perf_counter()  # wall-clock time of last valid prediction

        # MAIN UPDATE
    
    def update(
        self,
        predicted_crossing: Optional[Tuple[int, int]],
        predicted_frame: Optional[int],
        current_frame: int,
        strike_zone_center: Optional[Tuple[int, int]],
        batter_ready: bool = True,
    )-> None:
        """One control step.  Sets ``stick_x``, ``stick_y``, ``swing``.

        Parameters
       
        predicted_crossing :
            (x, y) where the ball will cross target_y.  ``None`` if the
            predictor has no estimate yet.
        predicted_frame :
            Frame index when crossing occurs.  ``None`` if unknown.
        current_frame :
            Current frame index (monotonically increasing).
        strike_zone_center :
            Current aim/strikezone centre in pixels.  ``None`` if the
            MovementTracker has not locked on yet.
        batter_ready :
            ``True`` when BatterStateClassifier says NORMAL.  When
            ``False``, the controller centres the stick and does NOT swing.
        """
        self.stick_x = 0.5
        self.stick_y = 0.5
        self.swing = False

        if not batter_ready:
            self._go_safe("batter not ready")
            return

        # COOLDOWN
        if self.state == ControlState.COOLDOWN:
            self._cooldown_counter += 1
            if self._cooldown_counter >= self.cooldown_frames:
                self._last_crossing = None
                self._last_pred_frame = None
                self._frames_to_impact = None
                self._stale_counter = 0
                self._transition(ControlState.IDLE)
            return  # stick centred, no swing

        # SWINGING
        if self.state == ControlState.SWINGING:
            self.swing = True
            self._swing_counter += 1
            # Keep last stick position during swing
            self.stick_x, self.stick_y = self._error_to_stick(
                self._error_px[0], self._error_px[1])
            if self._swing_counter >= self.swing_hold_frames:
                self.swing = False
                self._cooldown_counter = 0
                self._transition(ControlState.COOLDOWN)
            return

        # Determine available data
        has_prediction = (
            predicted_crossing is not None
            and predicted_frame is not None
        )

        if has_prediction:
            self._last_crossing = predicted_crossing
            self._last_pred_frame = predicted_frame
            self._stale_counter = 0
            self._stale_ts = time.perf_counter()
        else:
            self._stale_counter += 1

        # IDLE / TRACKING
        if self.state in (ControlState.IDLE, ControlState.TRACKING):
            if has_prediction:
                self._transition(ControlState.ALIGNING)
                # Fall through to ALIGNING logic below
            else:
                if self.state != ControlState.IDLE:
                    self._transition(ControlState.IDLE)
                return

        # ALIGNING
        if self.state == ControlState.ALIGNING:
            crossing = self._last_crossing
            pred_f = self._last_pred_frame
            if crossing is None or pred_f is None:
                self._go_safe("no prediction")
                return

            frames_to_impact = pred_f - current_frame

            # Impact has passed or is imminent?
            if frames_to_impact <= 0:
                grace = self.swing_anticipation_frames + 3
                if frames_to_impact >= -grace:
                    if strike_zone_center is not None:
                        dx = float(crossing[0] - strike_zone_center[0])
                        dy = float(crossing[1] - strike_zone_center[1])
                        self._error_px = (dx, dy)
                        self.stick_x, self.stick_y = self._error_to_stick(
                            dx, dy)
                    self.swing = True
                    self._swing_counter = 1
                    self._transition(ControlState.SWINGING)
                    return
                self._go_safe("impact passed (too late)")
                return

            if frames_to_impact > 600:
                self._go_safe("prediction unrealistic")
                return

            committed = (frames_to_impact
                         <= self.swing_anticipation_frames * 5)

            stale_wall = (time.perf_counter() - self._stale_ts) > 2.0
            stale_frames = self._stale_counter > self.max_no_update
            if not committed and (stale_wall or stale_frames):
                self._go_safe("prediction stale (far from impact)")
                return

            self._frames_to_impact = frames_to_impact

            if strike_zone_center is not None:
                dx = float(crossing[0] - strike_zone_center[0])
                dy = float(crossing[1] - strike_zone_center[1])
            else:
                dx, dy = 0.0, 0.0  # no correction, swing only
            self._error_px = (dx, dy)

            # Proportional stick command
            self.stick_x, self.stick_y = self._error_to_stick(dx, dy)

            # Swing timing check
            if frames_to_impact <= self.swing_anticipation_frames:
                self.swing = True
                self._swing_counter = 1
                self._transition(ControlState.SWINGING)

        # HELPERS
    
    def _error_to_stick(self, dx: float, dy: float)-> Tuple[float, float]:
        """Convert pixel error to stick values (0.0–1.0).

        Mapping convention (Dolphin GCPad pipe):
          stick_x: 0.0 = full left,  0.5 = centre, 1.0 = full right
          stick_y: 0.0 = full down,  0.5 = centre, 1.0 = full up

        Screen coordinates: +X = right, +Y = **down** (OpenCV convention).
        So positive dy (ball below strike zone) → stick_y < 0.5 (push down).
        """
        # Apply deadzone
        if abs(dx) < self.position_threshold_px:
            sx = 0.5
        else:
            offset = dx * self.proportional_gain
            offset = max(-0.5, min(0.5, offset))  # clamp to ±0.5
            if 0 < abs(offset) < self.stick_min_deflection:
                offset = self.stick_min_deflection * (1 if offset > 0 else -1)
            sx = 0.5 + offset

        if abs(dy) < self.position_threshold_px:
            sy = 0.5
        else:
            # Invert Y: screendown → stickdown (lower value)
            offset = -dy * self.proportional_gain
            offset = max(-0.5, min(0.5, offset))  # clamp to ±0.5
            if 0 < abs(offset) < self.stick_min_deflection:
                offset = self.stick_min_deflection * (1 if offset > 0 else -1)
            sy = 0.5 + offset

        return (_clamp01(sx), _clamp01(sy))

    def _go_safe(self, reason: str = "")-> None:
        """Return to IDLE with neutral outputs."""
        self.stick_x = 0.5
        self.stick_y = 0.5
        self.swing = False
        self._frames_to_impact = None
        self._error_px = (0.0, 0.0)
        self._stale_counter = 0
        self._stale_ts = time.perf_counter()
        if self.state != ControlState.IDLE:
            print(f"[CTRL] {self.state.name} → IDLE  (reason: {reason})")
            self._transition(ControlState.IDLE)

    def _transition(self, new: ControlState)-> None:
        self.state = new

        # PUBLIC PROPERTIES
    
    @property
    def frames_to_impact(self)-> Optional[int]:
        return self._frames_to_impact

    @property
    def error_px(self)-> Tuple[float, float]:
        """(dx, dy) pixel error: crossing − strike_zone_center."""
        return self._error_px

    @property
    def last_crossing(self)-> Optional[Tuple[int, int]]:
        return self._last_crossing

        # RESET
    
    def reset(self)-> None:
        """Full reset — call when the user presses R or between pitches."""
        self.state = ControlState.IDLE
        self.stick_x = 0.5
        self.stick_y = 0.5
        self.swing = False
        self._frames_to_impact = None
        self._last_crossing = None
        self._last_pred_frame = None
        self._error_px = (0.0, 0.0)
        self._swing_counter = 0
        self._cooldown_counter = 0
        self._stale_counter = 0
        self._stale_ts = time.perf_counter()


# MODULE-LEVEL HELPERS

def _clamp01(v: float)-> float:
    """Clamp a value to [0.0, 1.0]."""
    return max(0.0, min(1.0, v))
