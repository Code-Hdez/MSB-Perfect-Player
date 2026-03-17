"""
Swing controller — PD + feedforward control for precise batter alignment
and FPS-invariant swing timing.

Replaces the proportional-only ``BatterController`` with a proper
closed-loop controller that uses wall-clock time and derivative damping.

State machine
-------------
  IDLE       → No pitch in progress.  Neutral stick.
  TRACKING   → Ball detected, waiting for enough trajectory data.
  ALIGNING   → Prediction available, PD + feedforward stick control.
  COMMITTED  → Within swing decision window — target position frozen.
  SWINGING   → Swing button held for ``swing_hold_sec``.
  COOLDOWN   → Post-swing recovery period.

The controller uses **wall-clock seconds** everywhere — no frame
counting for timing decisions.

Usage::

    from msb.swing_controller import SwingController, SwingState

    ctrl = SwingController()
    ctrl.update(
        predicted_crossing=(400, 700),
        time_to_crossing_sec=0.18,
        strike_zone_center=(410, 700),
        batter_ready=True,
        predicted_vx=0.0,
    )
    print(ctrl.stick_x, ctrl.stick_y, ctrl.swing, ctrl.state)
"""

from __future__ import annotations

import time
from enum import Enum, auto
from typing import Optional, Tuple


# State enum

class SwingState(Enum):
    IDLE      = auto()
    TRACKING  = auto()
    ALIGNING  = auto()
    COMMITTED = auto()
    SWINGING  = auto()
    COOLDOWN  = auto()


# Swing controller

class SwingController:
    """PD + feedforward batter controller with wall-clock swing timing.

    Parameters
    ----------
    Kp : float
        Proportional gain (stick offset per pixel of error).
    Kd : float
        Derivative gain (stick offset per pixel/sec of error rate).
    Kff : float
        Feedforward gain.  Applied to the predicted ball lateral velocity
        so the batter leads the pitch instead of chasing.
    deadzone_px : float
        Error below this threshold → no stick correction.
    stick_min_deflection : float
        Minimum meaningful stick offset from centre (below this,
        small deflections are rounded to zero to avoid micro-noise).
    pipeline_latency_sec : float
        Estimated seconds of input → screen latency (capture + inference
        + input propagation).  Shifts the effective swing trigger earlier.
    swing_startup_sec : float
        Seconds the in-game swing animation takes to reach the contact
        zone.  The A press must happen this much earlier than crossing.
    swing_hold_sec : float
        How long to hold A after pressing.
    swing_window_sec : float
        The crossing is treated as a *time window*, not a point.  If
        the ball is within ±window of the predicted crossing, the swing
        can trigger.
    cooldown_sec : float
        Post-swing dead time before accepting a new pitch.
    max_stale_sec : float
        If prediction hasn't been updated for this long and we're far
        from impact, return stick to neutral.
    """

    def __init__(
        self,
        Kp: float = 0.006,
        Kd: float = 0.0015,
        Kff: float = 0.0008,
        deadzone_px: float = 12.0,
        stick_min_deflection: float = 0.04,
        pipeline_latency_sec: float = 0.040,
        swing_startup_sec: float = 0.100,
        swing_hold_sec: float = 0.100,
        swing_window_sec: float = 0.050,
        cooldown_sec: float = 0.400,
        max_stale_sec: float = 1.5,
    ) -> None:
        # Gains
        self.Kp = Kp
        self.Kd = Kd
        self.Kff = Kff
        self.deadzone_px = deadzone_px
        self.stick_min_deflection = stick_min_deflection

        # Timing
        self.pipeline_latency_sec = pipeline_latency_sec
        self.swing_startup_sec = swing_startup_sec
        self.swing_hold_sec = swing_hold_sec
        self.swing_window_sec = swing_window_sec
        self.cooldown_sec = cooldown_sec
        self.max_stale_sec = max_stale_sec

        # Outputs
        self.stick_x: float = 0.5
        self.stick_y: float = 0.5
        self.swing: bool = False

        # State
        self.state: SwingState = SwingState.IDLE
        self._prev_error: Tuple[float, float] = (0.0, 0.0)
        self._prev_ts: float = time.perf_counter()
        self._last_prediction_ts: float = 0.0
        self._swing_start_ts: float = 0.0
        self._cooldown_start_ts: float = 0.0
        self._committed_crossing: Optional[Tuple[int, int]] = None
        self._last_crossing: Optional[Tuple[int, int]] = None
        self._last_time_to_cross: Optional[float] = None
        self._error_px: Tuple[float, float] = (0.0, 0.0)

        # Diagnostics
        self._frames_to_impact: Optional[int] = None

    # Main update

    def update(
        self,
        predicted_crossing: Optional[Tuple[int, int]] = None,
        time_to_crossing_sec: Optional[float] = None,
        strike_zone_center: Optional[Tuple[int, int]] = None,
        batter_ready: bool = True,
        predicted_vx: float = 0.0,
        fps_estimate: float = 30.0,
    ) -> None:
        """One control step.  Sets ``stick_x``, ``stick_y``, ``swing``.

        Parameters
        ----------
        predicted_crossing : (x, y) where the ball will cross target_y.
        time_to_crossing_sec : seconds until the ball crosses target_y.
        strike_zone_center : current batter aim point in pixels.
        batter_ready : True when the batter is in NORMAL stance.
        predicted_vx : predicted horizontal velocity (px/sec) at crossing.
        fps_estimate : current FPS estimate (for legacy frame readout only).
        """
        now = time.perf_counter()
        dt = max(now - self._prev_ts, 1e-6)
        self._prev_ts = now

        self.stick_x = 0.5
        self.stick_y = 0.5
        self.swing = False

        if not batter_ready:
            self._go_idle("batter not ready")
            return

        # COOLDOWN
        if self.state == SwingState.COOLDOWN:
            elapsed = now - self._cooldown_start_ts
            if elapsed >= self.cooldown_sec:
                self._full_reset()
            return

        # SWINGING
        if self.state == SwingState.SWINGING:
            self.swing = True
            # Hold last stick position during swing
            self.stick_x, self.stick_y = self._error_to_stick(
                self._error_px[0], self._error_px[1])
            elapsed = now - self._swing_start_ts
            if elapsed >= self.swing_hold_sec:
                self.swing = False
                self._cooldown_start_ts = now
                self._transition(SwingState.COOLDOWN)
            return

        # Update prediction freshness
        has_prediction = (
            predicted_crossing is not None
            and time_to_crossing_sec is not None
        )
        if has_prediction:
            self._last_crossing = predicted_crossing
            self._last_time_to_cross = time_to_crossing_sec
            self._last_prediction_ts = now
        else:
            # Age the time-to-crossing estimate
            if self._last_time_to_cross is not None:
                self._last_time_to_cross -= dt

        # IDLE / TRACKING → advance when prediction available
        if self.state in (SwingState.IDLE, SwingState.TRACKING):
            if has_prediction:
                self._transition(SwingState.ALIGNING)
            else:
                if self.state != SwingState.IDLE:
                    self._transition(SwingState.IDLE)
                return

        # Effective time to crossing (compensated for latency)
        ttc = self._last_time_to_cross
        if ttc is None:
            self._go_idle("no timing data")
            return

        effective_ttc = ttc - self.pipeline_latency_sec

        # Legacy frame estimate
        self._frames_to_impact = max(0, int(ttc * fps_estimate))

        # COMMITTED
        if self.state == SwingState.COMMITTED:
            crossing = self._committed_crossing or self._last_crossing
            if crossing is not None and strike_zone_center is not None:
                dx = float(crossing[0] - strike_zone_center[0])
                dy = float(crossing[1] - strike_zone_center[1])
                self._error_px = (dx, dy)
                self.stick_x, self.stick_y = self._error_to_stick(dx, dy)

            # Fire swing
            if effective_ttc <= self.swing_startup_sec:
                self.swing = True
                self._swing_start_ts = now
                self._transition(SwingState.SWINGING)
            return

        # ALIGNING
        if self.state == SwingState.ALIGNING:
            crossing = self._last_crossing
            if crossing is None:
                self._go_idle("lost prediction")
                return

            # Staleness check (only when far from impact)
            stale = (now - self._last_prediction_ts) > self.max_stale_sec
            far_away = effective_ttc > 1.0
            if stale and far_away:
                self._go_idle("prediction stale")
                return

            # Impact already passed (well beyond recovery)
            if effective_ttc < -0.5:
                self._go_idle("impact passed")
                return

            # Unrealistic prediction (> 5 seconds away)
            if effective_ttc > 5.0:
                self._go_idle("prediction too far")
                return

            # Compute error
            if strike_zone_center is not None:
                dx = float(crossing[0] - strike_zone_center[0])
                dy = float(crossing[1] - strike_zone_center[1])
            else:
                dx, dy = 0.0, 0.0

            # PD control
            d_dx = (dx - self._prev_error[0]) / dt
            d_dy = (dy - self._prev_error[1]) / dt
            self._prev_error = (dx, dy)
            self._error_px = (dx, dy)

            # Feedforward from predicted lateral velocity
            ff_x = predicted_vx * self.Kff

            stick_offset_x = dx * self.Kp + d_dx * self.Kd + ff_x
            stick_offset_y = -dy * self.Kp + (-d_dy) * self.Kd  # invert Y

            self.stick_x = _clamp01(0.5 + self._apply_deadzone(stick_offset_x))
            self.stick_y = _clamp01(0.5 + self._apply_deadzone(stick_offset_y))

            # Enter commit zone
            swing_lead = self.swing_startup_sec + self.swing_window_sec
            if effective_ttc <= swing_lead:
                self._committed_crossing = crossing
                self._transition(SwingState.COMMITTED)

    # Helpers

    def _error_to_stick(self, dx: float, dy: float) -> Tuple[float, float]:
        """Simple proportional error → stick for SWINGING/COMMITTED phases."""
        sx_offset = dx * self.Kp
        sy_offset = -dy * self.Kp
        sx = _clamp01(0.5 + self._apply_deadzone(sx_offset))
        sy = _clamp01(0.5 + self._apply_deadzone(sy_offset))
        return (sx, sy)

    def _apply_deadzone(self, offset: float) -> float:
        """Clamp offset to ±0.5 and enforce minimum deflection."""
        offset = max(-0.5, min(0.5, offset))
        if 0 < abs(offset) < self.stick_min_deflection:
            return 0.0
        return offset

    def _go_idle(self, reason: str = "") -> None:
        self.stick_x = 0.5
        self.stick_y = 0.5
        self.swing = False
        self._error_px = (0.0, 0.0)
        self._prev_error = (0.0, 0.0)
        self._frames_to_impact = None
        if self.state != SwingState.IDLE:
            self._transition(SwingState.IDLE)

    def _full_reset(self) -> None:
        """Full reset for next pitch."""
        self._go_idle()
        self._last_crossing = None
        self._last_time_to_cross = None
        self._committed_crossing = None
        self._last_prediction_ts = 0.0

    def _transition(self, new: SwingState) -> None:
        self.state = new

    # Public properties

    @property
    def frames_to_impact(self) -> Optional[int]:
        """Legacy compatibility: estimated frames to impact."""
        return self._frames_to_impact

    @property
    def error_px(self) -> Tuple[float, float]:
        return self._error_px

    @property
    def last_crossing(self) -> Optional[Tuple[int, int]]:
        return self._last_crossing

    @property
    def time_to_impact_sec(self) -> Optional[float]:
        return self._last_time_to_cross

    # Reset

    def reset(self) -> None:
        """Full reset — call between pitches or on user [R]."""
        self.state = SwingState.IDLE
        self.stick_x = 0.5
        self.stick_y = 0.5
        self.swing = False
        self._prev_error = (0.0, 0.0)
        self._prev_ts = time.perf_counter()
        self._last_prediction_ts = 0.0
        self._swing_start_ts = 0.0
        self._cooldown_start_ts = 0.0
        self._committed_crossing = None
        self._last_crossing = None
        self._last_time_to_cross = None
        self._error_px = (0.0, 0.0)
        self._frames_to_impact = None


# Module helpers

def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))
