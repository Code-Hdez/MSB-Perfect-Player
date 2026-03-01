"""
Ball tracker — Kalman-filtered state machine that maintains ball
identity across frames.

Lifecycle: IDLE → TENTATIVE → CONFIRMED → LOST

Pitcher-animation rejection is enforced via layered rules in
``_kill_tentative_if_bad``.
"""

from __future__ import annotations

from collections import deque
from enum import Enum, auto
from typing import List, Optional, Tuple

import cv2
import numpy as np

from msb.config import Config
from msb.detector import BallCandidate, BallDetector
from msb.utils import in_rect


#  TRACK STATE

class TrackState(Enum):
    IDLE = auto()
    TENTATIVE = auto()
    CONFIRMED = auto()
    LOST = auto()


#  BALL TRACK  (single tracked object)

class BallTrack:
    """Ball state tracked across frames with a Kalman filter.

    Kalman model: constant-velocity  [x, y, vx, vy].
    """

    def __init__(self, pos: Tuple[int, int], frame_idx: int,
                 area: float = 0.0,
                 cfg: Optional[Config] = None) -> None:
        self.cfg = cfg or Config()

        # Kalman filter setup
        self._kf = cv2.KalmanFilter(4, 2, 0)
        self._kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]], dtype=np.float32)
        self._kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]], dtype=np.float32)
        self._kf.processNoiseCov = (
            np.eye(4, dtype=np.float32) * self.cfg.kf_process_noise)
        self._kf.measurementNoiseCov = (
            np.eye(2, dtype=np.float32) * self.cfg.kf_measurement_noise)
        self._kf.errorCovPost = np.eye(4, dtype=np.float32)
        self._kf.statePost = np.array(
            [[pos[0]], [pos[1]], [0], [0]], dtype=np.float32)

        # History
        self.positions: deque = deque(maxlen=self.cfg.trajectory_history)
        self.positions.append((pos[0], pos[1], frame_idx))
        self.frames_since_seen: int = 0
        self.total_frames: int = 1
        self.active: bool = True
        self.confirmed: bool = False

        # Track lifecycle helpers
        self._start_pos: Tuple[int, int] = pos
        self._start_frame: int = frame_idx
        self._frames_in_pitcher_zone: int = (
            1 if in_rect(pos[0], pos[1], self.cfg.pitcher_zone) else 0)
        self._left_pitcher_zone: bool = (
            not in_rect(pos[0], pos[1], self.cfg.pitcher_zone))
        self._vy_sign_changes: int = 0
        self._prev_vy_sign: int = 0
        self._positive_vy_count: int = 0

        # Area tracking
        self.avg_area: float = area if area > 0 else 20.0
        self._area_alpha: float = 0.3

    # Properties

    @property
    def velocity(self) -> Tuple[float, float]:
        s = self._kf.statePost
        return (float(s[2, 0]), float(s[3, 0]))

    @property
    def predicted_next(self) -> Tuple[int, int]:
        vx, vy = self.velocity
        p = self.last_pos
        return (int(p[0] + vx), int(p[1] + vy))

    @property
    def last_pos(self) -> Tuple[int, int]:
        p = self.positions[-1]
        return (p[0], p[1])

    @property
    def last_frame(self) -> int:
        return self.positions[-1][2]

    # Kalman helpers

    def kf_predict(self) -> Tuple[int, int]:
        pred = self._kf.predict()
        return (int(pred[0, 0]), int(pred[1, 0]))

    def kf_correct(self, x: int, y: int) -> None:
        meas = np.array([[np.float32(x)], [np.float32(y)]])
        self._kf.correct(meas)

    def kf_accept_prediction(self) -> None:
        self._kf.statePost = self._kf.statePre.copy()
        self._kf.errorCovPost = self._kf.errorCovPre.copy()

    # High-level interface

    def update(self, pos: Tuple[int, int], frame_idx: int,
               area: float = 0.0) -> None:
        """Register a new detection and update the Kalman filter."""
        self.kf_correct(pos[0], pos[1])
        cfg = self.cfg

        # Track vy sign changes (oscillation detection)
        _vx, vy = self.velocity
        sign = 1 if vy > 1.0 else (-1 if vy < -1.0 else 0)
        if (sign != 0 and self._prev_vy_sign != 0
                and sign != self._prev_vy_sign):
            self._vy_sign_changes += 1
        if sign != 0:
            self._prev_vy_sign = sign
        if vy > 0:
            self._positive_vy_count += 1

        self.positions.append((pos[0], pos[1], frame_idx))
        self.frames_since_seen = 0
        self.total_frames += 1

        if area > 0:
            self.avg_area = (self._area_alpha * area
                             + (1 - self._area_alpha) * self.avg_area)

        in_pz = in_rect(pos[0], pos[1], cfg.pitcher_zone)
        if in_pz:
            self._frames_in_pitcher_zone += 1
        else:
            self._left_pitcher_zone = True

        self._check_confirmation()

    def _check_confirmation(self) -> None:
        if self.confirmed:
            return
        cfg = self.cfg
        pts = list(self.positions)
        if len(pts) < cfg.track_min_confirmations:
            return

        # 1) Enough positive-vy frames
        if self._positive_vy_count < cfg.track_min_confirmations:
            return

        # 2) Must have left pitcher zone
        cur = self.last_pos
        dist = np.hypot(cur[0] - self._start_pos[0],
                        cur[1] - self._start_pos[1])
        if not self._left_pitcher_zone and dist < cfg.min_departure_dist:
            return

        # 3) Average forward velocity
        dy = pts[-1][1] - pts[0][1]
        df = pts[-1][2] - pts[0][2]
        if df <= 0:
            return
        avg_vy = dy / df
        if avg_vy < cfg.min_pitch_vy:
            return

        # 4) Minimum net Y displacement — pitcher body moves a little
        # but the ball must travel significantly downward
        net_dy = pts[-1][1] - pts[0][1]
        if net_dy < 120:
            return

        # 5) Trajectory consistency (>= 40% steps must show forward Y)
        inc_count = sum(1 for i in range(1, len(pts))
                        if pts[i][1] > pts[i - 1][1])
        frac = inc_count / max(len(pts) - 1, 1)
        if frac < 0.4:
            return

        # 6) Not too many oscillations
        if self._vy_sign_changes > cfg.max_vy_sign_changes:
            return

        self.confirmed = True

    def predict(self, n_frames: int = 1) -> Tuple[int, int]:
        """Simple linear extrapolation for *n_frames*."""
        vx, vy = self.velocity
        p = self.last_pos
        return (int(p[0] + vx * n_frames), int(p[1] + vy * n_frames))

    def mark_missed(self) -> None:
        """Called when no detection matched this frame."""
        self.frames_since_seen += 1
        self.kf_accept_prediction()
        cfg = self.cfg
        max_gap = (cfg.track_lost_frames if self.confirmed
                   else cfg.track_tentative_lost)
        if self.frames_since_seen >= max_gap:
            self.active = False

    def avg_velocity_y(self) -> float:
        pts = list(self.positions)
        if len(pts) < 2:
            return 0.0
        dy = pts[-1][1] - pts[0][1]
        df = pts[-1][2] - pts[0][2]
        return dy / df if df > 0 else 0.0


# Backward-compat alias
TrackedBall = BallTrack


#  BALL TRACKER  (lifecycle management)

class BallTracker:
    """Track the pitched ball across frames.

    State machine: IDLE → TENTATIVE → CONFIRMED → LOST

    Pitcher-animation rejection layers
    -----------------------------------
    1) Only start tracks from pitcher zone + in-motion + small area.
    2) Require departure from pitcher zone within max frames.
    3) Require sustained positive-Y velocity.
    4) Require minimum travel distance.
    5) Reject oscillating vy (too many sign flips).
    6) Penalise candidates near large foreground blobs.
    """

    def __init__(self, cfg: Optional[Config] = None) -> None:
        self.cfg = cfg or Config()
        self.track: Optional[BallTrack] = None
        self.state: TrackState = TrackState.IDLE
        self._frame_idx: int = 0
        self.selected: Optional[BallCandidate] = None
        self._last_track_end_frame: int = -999

    def reset(self) -> None:
        self.track = None
        self.state = TrackState.IDLE
        self._frame_idx = 0
        self.selected = None
        self._last_track_end_frame = -999

    # Size consistency

    def _is_size_consistent(self, candidate: BallCandidate) -> bool:
        if self.track is None or self.track.avg_area <= 0:
            return True
        avg = self.track.avg_area
        ratio = max(candidate.area, avg) / max(min(candidate.area, avg), 1)
        return ratio <= self.cfg.track_size_ratio

    # Nearest-candidate matching

    def _find_nearest_candidate(
        self,
        candidates: List[BallCandidate],
        pred: Tuple[int, int],
    ) -> Optional[BallCandidate]:
        """Match closest candidate to Kalman-predicted position."""
        cfg = self.cfg
        if self.track is None:
            return None

        speed = np.hypot(self.track.velocity[0], self.track.velocity[1])
        # Wider radius during tentative phase
        if self.track.confirmed:
            min_d = cfg.track_max_dist_min
        else:
            min_d = max(40.0, cfg.track_max_dist_min)
        base_d = max(min_d, cfg.track_max_dist_speed_k * speed)
        max_d = base_d * (1.0 + cfg.track_gap_expand *
                          self.track.frames_since_seen)

        best_dist = float("inf")
        chosen: Optional[BallCandidate] = None

        for c in candidates:
            if c.area > cfg.ball_flight_max_area:
                continue
            # Relax size consistency during tentative (area varies a lot
            # near the pitcher)
            if self.track.confirmed and not self._is_size_consistent(c):
                continue

            d_pred = np.hypot(c.center[0] - pred[0],
                              c.center[1] - pred[1])

            if self.track.confirmed:
                d = d_pred
            else:
                last = self.track.last_pos
                d_last = np.hypot(c.center[0] - last[0],
                                  c.center[1] - last[1])
                d = min(d_pred, d_last)

            if d > max_d:
                continue

            # Direction bonus — prefer forward motion (for both phases)
            dir_penalty = 1.0
            if len(self.track.positions) >= 2:
                _vx, vy = self.track.velocity
                if vy > 2.0:
                    dy_cand = c.center[1] - self.track.last_pos[1]
                    if dy_cand < -5:
                        dir_penalty = 2.0

            # Brightness bonus — prefer brighter candidates when close
            bri_bonus = 1.0
            if hasattr(c, 'brightness_score') and c.brightness_score > 0.6:
                bri_bonus = 0.8  # slight advantage

            effective_d = d * dir_penalty * bri_bonus
            if effective_d < best_dist:
                best_dist = effective_d
                chosen = c

        return chosen

    # Tentative / confirmed validation

    def _kill_tentative_if_bad(self) -> None:
        """Apply validation rules to kill bad tracks."""
        t = self.track
        if t is None:
            return
        cfg = self.cfg

        if not t.confirmed:
            # Rule 1: stayed in pitcher zone too long
            if t._frames_in_pitcher_zone > cfg.max_pitcher_zone_frames:
                self._drop_track(TrackState.IDLE)
                return
            # Rule 2: too many vy oscillations
            if t._vy_sign_changes > cfg.max_vy_sign_changes:
                self._drop_track(TrackState.IDLE)
                return
            # Rule 3: stalled too long without confirming
            if t.total_frames > cfg.track_min_confirmations + 20:
                self._drop_track(TrackState.IDLE)
                return
            # Rule 4: moving away from batter (clearly wrong direction)
            if t.total_frames >= 6 and t.avg_velocity_y() <= -3.0:
                self._drop_track(TrackState.IDLE)
                return
            # Rule 5: hovering
            if len(t.positions) >= 6:
                pts = list(t.positions)
                recent = pts[-6:]
                dx = abs(recent[-1][0] - recent[0][0])
                dy = abs(recent[-1][1] - recent[0][1])
                if dx < 15 and dy < 15:
                    self._drop_track(TrackState.IDLE)
                    return
            # Rule 6: all detections stuck in pitcher zone — no progress
            if (t.total_frames >= 10 and not t._left_pitcher_zone
                    and t.total_frames == t._frames_in_pitcher_zone):
                self._drop_track(TrackState.IDLE)
                return
        else:
            # Confirmed: kill if vy reverses significantly
            if t.total_frames >= 8 and t.avg_velocity_y() <= -5.0:
                self._drop_track(TrackState.LOST)
                return
            # Confirmed: kill if stalled
            if len(t.positions) >= 8:
                pts = list(t.positions)
                recent = pts[-8:]
                dx = abs(recent[-1][0] - recent[0][0])
                dy = abs(recent[-1][1] - recent[0][1])
                if dx < 20 and dy < 20:
                    self._drop_track(TrackState.LOST)
                    return

    def _drop_track(self, new_state: TrackState) -> None:
        self.track = None
        self.state = new_state
        self.selected = None

    # Main update

    def update(
        self,
        candidates: List[BallCandidate],
        best_candidate: Optional[BallCandidate] = None,
        detector: Optional[BallDetector] = None,
    ) -> Optional[BallTrack]:
        """Run one tracker step with ALL detector candidates.

        Two-phase strategy:
          Phase 1: Ball detected in pitcher zone → tentative track.
          Blind  : Ball dims near pitcher → rescue detection.
          Phase 2: Ball brightens below pitcher → re-acquisition.
        """
        cfg = self.cfg
        self._frame_idx += 1
        self.selected = None
        had_track = (self.track is not None and self.track.active)

        # Active track: predict + match
        if self.track is not None and self.track.active:
            pred = self.track.kf_predict()
            chosen = self._find_nearest_candidate(candidates, pred)

            if chosen is not None:
                self.track.update(chosen.center, self._frame_idx,
                                  area=chosen.area)
                self.selected = chosen
                self.state = (TrackState.CONFIRMED if self.track.confirmed
                              else TrackState.TENTATIVE)
                self._kill_tentative_if_bad()
            else:
                # Try rescue detection before marking missed
                rescued = False
                if detector is not None:
                    rescue_c = detector.rescue_near(pred, radius=60)
                    if rescue_c is not None:
                        self.track.update(rescue_c.center, self._frame_idx,
                                          area=rescue_c.area)
                        self.selected = rescue_c
                        self.state = (TrackState.CONFIRMED
                                      if self.track.confirmed
                                      else TrackState.TENTATIVE)
                        self._kill_tentative_if_bad()
                        rescued = True

                if not rescued:
                    self.track.mark_missed()
                    if not self.track.active:
                        self.state = TrackState.LOST
                        self.track = None

        # Record when a track just died
        if had_track and self.track is None:
            self._last_track_end_frame = self._frame_idx

        # No active track: try to start one
        if self.track is None:
            self.state = TrackState.IDLE
            best_start: Optional[BallCandidate] = None
            best_start_score = -1.0

            # Priority 1: pitcher zone
            for c in candidates:
                if not c.in_motion_mask:
                    continue
                if c.area > cfg.ball_flight_max_area:
                    continue
                if not in_rect(c.center[0], c.center[1], cfg.pitcher_zone):
                    continue
                if c.score > best_start_score:
                    best_start_score = c.score
                    best_start = c

            # Priority 2: reacquisition zone (after recent track death)
            if (best_start is None
                    and (self._frame_idx - self._last_track_end_frame)
                    <= cfg.reacq_window):
                best_rq: Optional[BallCandidate] = None
                best_rq_score = -1.0
                for c in candidates:
                    if not c.in_motion_mask:
                        continue
                    if c.area > cfg.ball_flight_max_area:
                        continue
                    if not in_rect(c.center[0], c.center[1], cfg.reacq_zone):
                        continue
                    if c.score > best_rq_score:
                        best_rq_score = c.score
                        best_rq = c
                best_start = best_rq

            if best_start is not None:
                self.track = BallTrack(
                    best_start.center, self._frame_idx,
                    area=best_start.area, cfg=cfg)
                self.state = TrackState.TENTATIVE
                self.selected = best_start

        return self.track
