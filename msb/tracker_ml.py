"""
Simplified ball tracker for use with the ML (YOLO) detector.

The classical tracker (``tracker.py``) has many heuristic rules to
reject pitcher-animation false positives.  With ML detection, those
rules are unnecessary — the model itself discriminates ball vs. not-ball.

This tracker keeps only:
  - Kalman filter for smooth trajectory
  - Simple nearest-neighbour matching
  - Track lifecycle: IDLE → ACTIVE → LOST
  - Gap bridging (coast through short detection dropouts)

v2: Each position entry stores a **wall-clock timestamp** alongside the
frame index — ``(x, y, frame_idx, timestamp)``.  This allows the
predictor to use real time instead of frame counts.
"""

from __future__ import annotations

import time
from collections import deque
from enum import Enum, auto
from typing import List, Optional, Tuple

import cv2
import numpy as np

from msb.config import Config
from msb.detector import BallCandidate


# TRACK STATE

class MLTrackState(Enum):
    IDLE = auto()
    ACTIVE = auto()
    LOST = auto()


# ML BALL TRACK

class MLBallTrack:
    """Single tracked ball with Kalman filter (constant-velocity model).

    Position entries are ``(x, y, frame_idx, timestamp)`` where
    *timestamp* is ``time.perf_counter()`` at the moment of the
    detection.
    """

    def __init__(self, pos: Tuple[int, int], frame_idx: int,
                 conf: float = 0.0,
                 timestamp: Optional[float] = None) -> None:
        ts = timestamp if timestamp is not None else time.perf_counter()

        # Kalman: state = [x, y, vx, vy]
        self._kf = cv2.KalmanFilter(4, 2, 0)
        self._kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]], dtype=np.float32)
        self._kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]], dtype=np.float32)
        self._kf.processNoiseCov = np.eye(4, dtype=np.float32) * 16.0
        self._kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 4.0
        self._kf.errorCovPost = np.eye(4, dtype=np.float32)
        self._kf.statePost = np.array(
            [[pos[0]], [pos[1]], [0], [0]], dtype=np.float32)

        self.positions: deque = deque(maxlen=120)
        self.positions.append((pos[0], pos[1], frame_idx, ts))
        self.frames_since_seen: int = 0
        self.total_frames: int = 1
        self.active: bool = True
        self.avg_conf: float = conf

        # Always considered "confirmed" with ML (model already filtered)
        self.confirmed: bool = True

    @property
    def velocity(self) -> Tuple[float, float]:
        s = self._kf.statePost
        return (float(s[2, 0]), float(s[3, 0]))

    @property
    def last_pos(self) -> Tuple[int, int]:
        p = self.positions[-1]
        return (p[0], p[1])

    @property
    def last_frame(self) -> int:
        return self.positions[-1][2]

    @property
    def last_timestamp(self) -> float:
        """Wall-clock timestamp of the most recent position."""
        return self.positions[-1][3]

    @property
    def predicted_next(self) -> Tuple[int, int]:
        vx, vy = self.velocity
        p = self.last_pos
        return (int(p[0] + vx), int(p[1] + vy))

    @property
    def _frames_in_pitcher_zone(self) -> int:
        return 0

    @property
    def avg_area(self) -> float:
        return 20.0

    def kf_predict(self) -> Tuple[int, int]:
        pred = self._kf.predict()
        return (int(pred[0, 0]), int(pred[1, 0]))

    def kf_correct(self, x: int, y: int) -> None:
        meas = np.array([[np.float32(x)], [np.float32(y)]])
        self._kf.correct(meas)

    def update(self, pos: Tuple[int, int], frame_idx: int,
               conf: float = 0.0,
               timestamp: Optional[float] = None) -> None:
        ts = timestamp if timestamp is not None else time.perf_counter()
        self.kf_correct(pos[0], pos[1])
        self.positions.append((pos[0], pos[1], frame_idx, ts))
        self.frames_since_seen = 0
        self.total_frames += 1
        # Running average confidence
        self.avg_conf = 0.8 * self.avg_conf + 0.2 * conf

    def mark_missed(self, max_gap: int = 10) -> None:
        """Coast one frame using Kalman prediction."""
        self.frames_since_seen += 1
        # Accept the prediction as the state
        self._kf.statePost = self._kf.statePre.copy()
        self._kf.errorCovPost = self._kf.errorCovPre.copy()
        if self.frames_since_seen >= max_gap:
            self.active = False

    def predict(self, n_frames: int = 1) -> Tuple[int, int]:
        vx, vy = self.velocity
        p = self.last_pos
        return (int(p[0] + vx * n_frames), int(p[1] + vy * n_frames))

    def avg_velocity_y(self) -> float:
        pts = list(self.positions)
        if len(pts) < 2:
            return 0.0
        dy = pts[-1][1] - pts[0][1]
        df = pts[-1][2] - pts[0][2]
        return dy / df if df > 0 else 0.0

    def velocity_y_px_per_sec(self) -> float:
        """Vertical velocity in pixels/second using wall-clock time."""
        pts = list(self.positions)
        if len(pts) < 2:
            return 0.0
        dy = pts[-1][1] - pts[0][1]
        dt = pts[-1][3] - pts[0][3]
        return dy / dt if dt > 1e-6 else 0.0


# ML BALL TRACKER

class MLBallTracker:
    """Simplified tracker for ML-detected ball candidates.

    No pitcher-animation heuristics needed — the model handles that.
    Just: match nearest candidate → Kalman smooth → gap bridging.
    """

    def __init__(self, cfg: Optional[Config] = None,
                 max_match_dist: float = 80.0,
                 max_gap_frames: int = 10,
                 min_conf_start: float = 0.3) -> None:
        self.cfg = cfg or Config()
        self.track: Optional[MLBallTrack] = None
        self.state: MLTrackState = MLTrackState.IDLE
        self.selected: Optional[BallCandidate] = None
        self._frame_idx: int = 0
        self._max_match_dist = max_match_dist
        self._max_gap = max_gap_frames
        self._min_conf_start = min_conf_start
        self._killed_positions: list = []

    def reset(self) -> None:
        self.track = None
        self.state = MLTrackState.IDLE
        self.selected = None
        self._frame_idx = 0
        self._killed_positions.clear()

    def update(
        self,
        candidates: List[BallCandidate],
        best_candidate: Optional[BallCandidate] = None,
        detector=None,  # unused, kept for interface compat
        timestamp: Optional[float] = None,
    ) -> Optional[MLBallTrack]:
        """One tracker step.

        Parameters
        ----------
        candidates : list of BallCandidate
            All ML detections for this frame (sorted by confidence).
        best_candidate : BallCandidate, optional
            Ignored (kept for API compatibility).
        detector : optional
            Ignored.
        timestamp : float, optional
            Wall-clock timestamp for this frame.  If None, uses
            ``time.perf_counter()`` at call time.
        """
        ts = timestamp if timestamp is not None else time.perf_counter()
        self._frame_idx += 1
        self.selected = None

        # Active track: predict + match
        if self.track is not None and self.track.active:
            pred = self.track.kf_predict()
            chosen = self._match_nearest(candidates, pred)

            if chosen is not None:
                self.track.update(chosen.center, self._frame_idx,
                                  conf=chosen.score, timestamp=ts)
                self.selected = chosen
                self.state = MLTrackState.ACTIVE
            else:
                self.track.mark_missed(self._max_gap)
                if not self.track.active:
                    self.state = MLTrackState.LOST
                    self.track = None

        # No active track: start from best candidate
        if self.track is None and candidates:
            # Take highest-confidence candidate above threshold
            for c in candidates:
                if c.score >= self._min_conf_start:
                    self.track = MLBallTrack(
                        c.center, self._frame_idx, conf=c.score,
                        timestamp=ts)
                    self.state = MLTrackState.ACTIVE
                    self.selected = c
                    break

        if self.track is None:
            self.state = MLTrackState.IDLE

        return self.track

    def kill_track(self) -> None:
        """Kill the current track and remember its position."""
        if self.track is not None and self.track.active:
            self._killed_positions.append(
                (self.track.last_pos[0], self.track.last_pos[1],
                 self._frame_idx))
        self.track = None
        self.state = MLTrackState.IDLE
        self.selected = None

    def expire_killed(self, max_age: int = 90) -> None:
        """Drop killed positions older than *max_age* frames."""
        cutoff = self._frame_idx - max_age
        self._killed_positions = [
            (kx, ky, kf) for kx, ky, kf in self._killed_positions
            if kf >= cutoff
        ]

    def is_near_killed(self, x: int, y: int, radius: int = 30) -> bool:
        """Return True if (x, y) is near a recently-killed position."""
        return any(
            abs(x - kx) < radius and abs(y - ky) < radius
            for kx, ky, _ in self._killed_positions
        )

    def _match_nearest(
        self,
        candidates: List[BallCandidate],
        pred: Tuple[int, int],
    ) -> Optional[BallCandidate]:
        """Find the nearest candidate to the Kalman prediction."""
        if not candidates:
            return None

        # Dynamic match radius: base + speed-scaled
        speed = 0.0
        if self.track is not None:
            vx, vy = self.track.velocity
            speed = np.hypot(vx, vy)
        max_d = max(self._max_match_dist, 2.0 * speed)
        max_d = min(max_d, 150.0)
        # Expand radius if we've missed recent frames
        if self.track is not None:
            max_d *= (1.0 + 0.2 * self.track.frames_since_seen)
            max_d = min(max_d, 180.0)

        last_pos = pred
        recent_dy = 0.0
        if self.track is not None:
            last_pos = self.track.last_pos
            pts = list(self.track.positions)
            if len(pts) >= 2:
                recent_dy = float(pts[-1][1] - pts[-2][1])

        best_d = float("inf")
        chosen: Optional[BallCandidate] = None

        for c in candidates:
            if recent_dy > 8.0 and c.center[1] < last_pos[1] - 20:
                continue

            d = np.hypot(c.center[0] - pred[0], c.center[1] - pred[1])
            if d < max_d:
                score_d = d - 8.0 * float(c.score)
                if score_d < best_d:
                    best_d = score_d
                    chosen = c

        return chosen
