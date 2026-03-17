"""
Game-phase finite state machine — the single most impactful robustness
improvement for the MSB batter.

Detects whether the game is showing the batter view (pitch incoming) or
a different camera (fielding, cut-scene, replay) and gates every
downstream subsystem accordingly.

Detection is based on lightweight SSIM scene similarity against a small
downscaled reference frame captured during confirmed batter view, plus
optional ball-motion cues.  No ML model required.

States
------
  PRE_PITCH         Waiting for a new pitch.  Stick neutral.
  PITCH_ACTIVE      Ball is in flight; detection/tracking/control active.
  SWING_COMMIT      Inside the swing decision window — target frozen.
  BALL_IN_PLAY      Camera changed away from batter view.  Freeze all.
  TRANSITION        Camera returning toward batter.  Still frozen.
  RESET_TO_BATTER   Hard-reset downstream systems, reacquire batter.

Usage::

    from msb.game_phase import GamePhaseManager, GamePhase

    gpm = GamePhaseManager()
    gpm.update(frame, ball_tracked=True, ball_vy=12.3, swung=False)
    if gpm.phase == GamePhase.PITCH_ACTIVE:
        ...  # run detection / tracking / control
"""

from __future__ import annotations

import time
from enum import Enum, auto
from typing import Optional, Tuple

import cv2
import numpy as np


# Phase enum

class GamePhase(Enum):
    PRE_PITCH       = auto()
    PITCH_ACTIVE    = auto()
    SWING_COMMIT    = auto()
    BALL_IN_PLAY    = auto()
    TRANSITION      = auto()
    RESET_TO_BATTER = auto()


# Configuration defaults

_DEFAULT_CFG = dict(
    # Scene-similarity thresholds (SSIM on small grey thumbnails)
    ssim_batter_thresh=0.72,       # above this → probably batter view
    ssim_deviation_thresh=0.55,    # below this → definitely NOT batter view

    # Hysteresis: consecutive frames required before switching
    hysteresis_enter_bip=5,        # frames of low similarity → BALL_IN_PLAY
    hysteresis_exit_bip=8,         # frames of high similarity → TRANSITION done

    # Reference thumbnail size (width, height)
    ref_thumb_size=(160, 120),

    # How often to refresh the reference frame (in confirmed batter frames)
    ref_refresh_interval=120,

    # After swing, minimum wall-clock seconds before re-entering PRE_PITCH
    post_swing_lockout_sec=0.8,

    # Maximum seconds to stay in RESET_TO_BATTER before forcing PRE_PITCH
    reset_timeout_sec=0.5,

    # Minimum ball velocity (pixels/sec) to trigger PITCH_ACTIVE from PRE_PITCH
    min_pitch_vy_trigger=40.0,
)


# GamePhaseManager

class GamePhaseManager:
    """Finite state machine controlling what the pipeline processes each
    frame.

    Call ``update()`` once per frame with the current frame and
    contextual signals (ball tracking state, velocity, swing events).

    Read ``phase`` afterwards to determine which subsystems should run.
    """

    def __init__(self, cfg: Optional[dict] = None, **kwargs) -> None:
        overrides = cfg if isinstance(cfg, dict) else {}
        overrides.update(kwargs)
        self._cfg = {**_DEFAULT_CFG, **overrides}

        self.phase: GamePhase = GamePhase.PRE_PITCH
        self._prev_phase: GamePhase = GamePhase.PRE_PITCH

        # Scene similarity state
        self._ref_thumb: Optional[np.ndarray] = None  # grey, small
        self._ref_age: int = 0           # frames since last ref update
        self._ssim_score: float = 1.0    # latest similarity
        self._low_sim_streak: int = 0    # consecutive low-similarity frames
        self._high_sim_streak: int = 0   # consecutive high-similarity frames

        # Timing
        self._phase_enter_ts: float = time.perf_counter()
        self._last_swing_ts: float = 0.0
        self._last_transition_ts: float = 0.0

        # Counters
        self._batter_frames: int = 0     # confirmed batter-view frames
        self._total_frames: int = 0

        # Ball velocity sign-change detection (hit signal)
        self._prev_ball_vy: float = 0.0
        self._hit_detected: bool = False
        self._batter_visible: bool = True

    # Public API

    def update(
        self,
        frame: np.ndarray,
        *,
        ball_tracked: bool = False,
        ball_vy: float = 0.0,
        ball_y: float = 0.0,
        target_y: Optional[float] = None,
        swung: bool = False,
        swing_committed: bool = False,
        batter_visible: bool = True,
    ) -> GamePhase:
        """Advance the FSM by one frame.

        Parameters
        ----------
        frame : current BGR game frame.
        ball_tracked : True if the ball tracker has an active track.
        ball_vy : vertical velocity of the ball in pixels/sec (positive = downward).
        ball_y : current ball Y position (pixels).
        target_y : strike-zone Y position (pixels), for proximity check.
        swung : True the frame the swing was executed.
        swing_committed : True while the swing controller is in COMMITTED/SWINGING.
        batter_visible : True if the batter tracker found the batter this frame.

        Returns the new GamePhase.
        """
        self._total_frames += 1
        now = time.perf_counter()
        self._batter_visible = batter_visible

        self._hit_detected = False
        if ball_tracked and self._prev_ball_vy > 0 and ball_vy < 0:
            near_target = True
            if target_y is not None and target_y > 0:
                near_target = abs(ball_y - target_y) < 80  # px tolerance
            if near_target:
                self._hit_detected = True
        self._prev_ball_vy = ball_vy if ball_tracked else 0.0

        # Compute scene similarity
        is_batter_view = self._classify_scene(frame, batter_visible)

        # Track streak counters
        if is_batter_view:
            self._high_sim_streak += 1
            self._low_sim_streak = 0
        else:
            self._low_sim_streak += 1
            self._high_sim_streak = 0

        # Record swing events
        if swung:
            self._last_swing_ts = now

        # FSM transitions
        prev = self.phase

        if self.phase == GamePhase.PRE_PITCH:
            self._handle_pre_pitch(ball_tracked, ball_vy, is_batter_view)

        elif self.phase == GamePhase.PITCH_ACTIVE:
            self._handle_pitch_active(
                swung, swing_committed, is_batter_view, now)

        elif self.phase == GamePhase.SWING_COMMIT:
            self._handle_swing_commit(swung, is_batter_view, now)

        elif self.phase == GamePhase.BALL_IN_PLAY:
            self._handle_ball_in_play(is_batter_view, now)

        elif self.phase == GamePhase.TRANSITION:
            self._handle_transition(is_batter_view, now)

        elif self.phase == GamePhase.RESET_TO_BATTER:
            self._handle_reset(is_batter_view, now)

        if self.phase != prev:
            self._prev_phase = prev
            self._phase_enter_ts = now

        return self.phase

    def _set_phase(self, phase: GamePhase) -> None:
        """Internal helper — set phase and record timestamp."""
        self._prev_phase = self.phase
        self.phase = phase
        self._phase_enter_ts = time.perf_counter()

    def force_phase(self, phase: GamePhase) -> None:
        """Force a specific phase (for manual reset / testing)."""
        self._set_phase(phase)

    def reset(self) -> None:
        """Full reset to PRE_PITCH."""
        self.phase = GamePhase.PRE_PITCH
        self._prev_phase = GamePhase.PRE_PITCH
        self._ref_thumb = None
        self._ref_age = 0
        self._ssim_score = 1.0
        self._low_sim_streak = 0
        self._high_sim_streak = 0
        self._batter_frames = 0
        self._phase_enter_ts = time.perf_counter()
        self._prev_ball_vy = 0.0
        self._hit_detected = False

    def notify_swing_complete(self) -> None:
        """Called by the orchestrator when a pitch cycle completes.

        Transitions FSM through SWING_COMMIT → RESET_TO_BATTER so it
        will naturally settle back to PRE_PITCH.
        """
        if self.phase in (GamePhase.PITCH_ACTIVE, GamePhase.SWING_COMMIT):
            self._set_phase(GamePhase.RESET_TO_BATTER)

    @property
    def ssim_score(self) -> float:
        return self._ssim_score

    @property
    def phase_duration(self) -> float:
        """Seconds since entering the current phase."""
        return time.perf_counter() - self._phase_enter_ts

    @property
    def is_batter_view(self) -> bool:
        """True if the FSM believes we're in batter camera view."""
        return self.phase in (
            GamePhase.PRE_PITCH,
            GamePhase.PITCH_ACTIVE,
            GamePhase.SWING_COMMIT,
        )

    @property
    def should_track_ball(self) -> bool:
        return self.phase in (GamePhase.PITCH_ACTIVE, GamePhase.SWING_COMMIT)

    @property
    def should_track_batter(self) -> bool:
        return self.phase in (
            GamePhase.PRE_PITCH,
            GamePhase.PITCH_ACTIVE,
            GamePhase.SWING_COMMIT,
        )

    @property
    def hit_detected(self) -> bool:
        """True the frame the ball's vy sign-changed near target_y."""
        return self._hit_detected

    @property
    def should_send_control(self) -> bool:
        return self.phase in (GamePhase.PITCH_ACTIVE, GamePhase.SWING_COMMIT)

    # Scene classification

    def _classify_scene(self, frame: np.ndarray,
                         batter_visible: bool = True) -> bool:
        """Return True if *frame* looks like the batter camera view."""
        tw, th = self._cfg["ref_thumb_size"]
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thumb = cv2.resize(grey, (tw, th), interpolation=cv2.INTER_AREA)

        if self._ref_thumb is None:
            if not batter_visible:
                self._ssim_score = 0.5  # indeterminate
                return False
            self._ref_thumb = thumb
            self._ssim_score = 1.0
            return True

        # Fast structural similarity (mean SSIM on small thumb)
        self._ssim_score = self._fast_ssim(self._ref_thumb, thumb)

        is_batter = self._ssim_score >= self._cfg["ssim_batter_thresh"]

        if is_batter and self.is_batter_view:
            self._batter_frames += 1
            self._ref_age += 1
            if self._ref_age >= self._cfg["ref_refresh_interval"]:
                self._ref_thumb = thumb
                self._ref_age = 0

        return is_batter

    @staticmethod
    def _fast_ssim(a: np.ndarray, b: np.ndarray) -> float:
        """Compute a fast approximation of structural similarity."""
        a = a.astype(np.float32)
        b = b.astype(np.float32)

        mu_a = np.mean(a)
        mu_b = np.mean(b)
        sigma_a2 = np.var(a)
        sigma_b2 = np.var(b)
        sigma_ab = np.mean((a - mu_a) * (b - mu_b))

        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        num = (2 * mu_a * mu_b + C1) * (2 * sigma_ab + C2)
        den = (mu_a**2 + mu_b**2 + C1) * (sigma_a2 + sigma_b2 + C2)

        return float(num / den) if den > 0 else 0.0

    # Per-state transition handlers

    def _handle_pre_pitch(
        self, ball_tracked: bool, ball_vy: float, is_batter: bool,
    ) -> None:
        # Camera changed before pitch even started
        if (self._low_sim_streak >= self._cfg["hysteresis_enter_bip"]
                and not self._batter_visible):
            self.phase = GamePhase.BALL_IN_PLAY
            return

        # Pitch detection: meaningful downward ball motion
        if (ball_tracked
                and ball_vy >= self._cfg["min_pitch_vy_trigger"]):
            self.phase = GamePhase.PITCH_ACTIVE

    def _handle_pitch_active(
        self,
        swung: bool,
        swing_committed: bool,
        is_batter: bool,
        now: float,
    ) -> None:
        # Hit detected via velocity sign-change
        if self._hit_detected:
            self.phase = GamePhase.BALL_IN_PLAY
            return

        # Swing commitment
        if swing_committed:
            self.phase = GamePhase.SWING_COMMIT
            return

        # Camera change during pitch (e.g., ball hit immediately)
        if self._low_sim_streak >= self._cfg["hysteresis_enter_bip"]:
            self.phase = GamePhase.BALL_IN_PLAY
            return

    def _handle_swing_commit(
        self, swung: bool, is_batter: bool, now: float,
    ) -> None:
        if swung:
            self._last_swing_ts = now

        # Hit detected via velocity sign-change
        if self._hit_detected:
            self.phase = GamePhase.BALL_IN_PLAY
            return

        # Camera changed → ball in play
        if self._low_sim_streak >= self._cfg["hysteresis_enter_bip"]:
            self.phase = GamePhase.BALL_IN_PLAY
            return

        # Post-swing lockout elapsed and camera still batter view
        post_swing_elapsed = now - self._last_swing_ts
        if (self._last_swing_ts > 0
                and post_swing_elapsed > self._cfg["post_swing_lockout_sec"]
                and is_batter):
            self.phase = GamePhase.RESET_TO_BATTER

    def _handle_ball_in_play(
        self, is_batter: bool, now: float,
    ) -> None:
        # Camera returning to batter view
        if self._high_sim_streak >= self._cfg["hysteresis_exit_bip"]:
            self.phase = GamePhase.TRANSITION

    def _handle_transition(
        self, is_batter: bool, now: float,
    ) -> None:
        # Stable batter view confirmed
        phase_dur = now - self._phase_enter_ts
        if is_batter and phase_dur >= 0.15:
            self.phase = GamePhase.RESET_TO_BATTER
            return

        # Flicker: camera went away again
        if (self._low_sim_streak >= self._cfg["hysteresis_enter_bip"]
            and not self._batter_visible):
            self.phase = GamePhase.BALL_IN_PLAY

    def _handle_reset(
        self, is_batter: bool, now: float,
    ) -> None:
        # Auto-advance to PRE_PITCH after brief reset window
        phase_dur = now - self._phase_enter_ts
        if phase_dur >= self._cfg["reset_timeout_sec"]:
            self.phase = GamePhase.PRE_PITCH

        # Camera changed again during reset
        if (self._low_sim_streak >= self._cfg["hysteresis_enter_bip"]
            and not self._batter_visible):
            self.phase = GamePhase.BALL_IN_PLAY
