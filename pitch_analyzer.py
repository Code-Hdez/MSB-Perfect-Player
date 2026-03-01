"""
Pitch / Ball Detection & Trajectory Analyzer

Detects the pitched ball as it travels from pitcher to batter, tracks its
position across frames, and predicts where it will cross the strike zone.

Phase 1 — Capture & Detection

  • Live-captures frames at 60 fps via dxcam.
  • Press [SPACE] to start recording a pitch sequence (auto-stops after
    ``RECORD_MAX_FRAMES`` or press [SPACE] again).
  • Ball detection:  frame-differencing (motion isolation) + white/bright
    colour threshold + contour filter (area, circularity) + ball-candidate
    scoring.
  • Saves recorded pitch sequences to ``pitches/<timestamp>/`` with every
    frame + metadata JSON.

Phase 2 — Tracking & Trajectory

  • Nearest-neighbour tracking across frames (with velocity prediction).
  • Trajectory history stored per pitch.
  • Linear & quadratic extrapolation to predict strike-zone crossing.

Controls

  SPACE  = Start / stop recording a pitch
  D      = Toggle debug panel
  C      = Click to define ball search ROI (optional, limits search area)
  S      = Save current pitch recording to disk
  Q/ESC  = Quit

Usage

  python pitch_analyzer.py
  python pitch_analyzer.py --source folder --input ./pitches/20260227_120000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

#  CONFIGURATION  (shared with msb_hitbox_detector.py)

SCREEN_ROI: Tuple[int, int, int, int] = (378, 127, 1542, 1019)
"""(left, top, right, bottom) pixel coords of the Dolphin game window."""

MONITOR_INDEX: int = 0

TARGET_FPS: int = 60

# Ball detection (HSV)

# The ball is white/bright in-game.  Capture the bright-white range.
BALL_HSV_LOWER: np.ndarray = np.array([0, 0, 200])
BALL_HSV_UPPER: np.ndarray = np.array([180, 50, 255])
"""Very low saturation + very high value → white / near-white."""

# Also detect the motion-blur trail (slightly dimmer / more spread)
TRAIL_HSV_LOWER: np.ndarray = np.array([0, 0, 160])
TRAIL_HSV_UPPER: np.ndarray = np.array([180, 70, 255])

# Contour filtering

BALL_MIN_AREA: int = 8
"""Minimum contour area (px²) for a ball candidate."""

BALL_MAX_AREA: int = 300
"""Maximum contour area — the ball is small even up close."""

BALL_FLIGHT_MAX_AREA: int = 120
"""Maximum area for a ball in flight (tighter filter once tracking)."""

BALL_MIN_CIRCULARITY: float = 0.20
"""Minimum circularity (4π·area / perimeter²).  Low because motion blur
elongates the shape.  A perfect circle = 1.0."""

# Frame differencing

DIFF_THRESHOLD: int = 30
"""Absolute grey-level difference to count as motion."""

DIFF_DILATE_ITER: int = 2
"""Dilation iterations on the motion mask to group nearby blobs."""

# Ball tracker

TRACK_MAX_DIST_MIN: float = 30.0
"""Minimum search radius (px) when matching a candidate to the tracked
ball.  Used when the ball speed is very low (just released)."""

TRACK_MAX_DIST_SPEED_K: float = 2.0
"""Multiplier on ball speed to compute search radius.
Effective max-dist = max(TRACK_MAX_DIST_MIN, speed * K)."""

TRACK_GAP_EXPAND: float = 0.15
"""Per-missed-frame expansion factor on the search radius."""

TRACK_LOST_FRAMES: int = 5
"""Frames without a match before a *confirmed* track is frozen/lost.
Kept short to avoid re-acquiring noise after the ball leaves the
detectable zone."""

TRACK_TENTATIVE_LOST: int = 3
"""Frames without a match before a *tentative* (unconfirmed) track is
killed.  Shorter than confirmed because tentative tracks are likely
false starts."""

TRAJECTORY_HISTORY: int = 60
"""How many past positions to keep for trajectory fitting."""

TRACK_MIN_CONFIRMATIONS: int = 3
"""Number of consecutive forward-moving (positive ΔY) detections needed
to *confirm* a track as a real pitch.  Until confirmed the track is
tentative and will be killed easily."""

TRACK_SIZE_RATIO: float = 4.0
"""Max ratio between candidate area and running average tracked area.
Prevents jumping from a small ball to a large character blob."""

# Pitcher zone
# Approximate screen rectangle (in ROI coords) where the ball is first
# visible after the pitcher releases it.  Tracks may ONLY start here.

PITCHER_ZONE: Tuple[int, int, int, int] = (540, 100, 660, 220)
"""(x1, y1, x2, y2) — tight region around where the pitched ball first
appears (after release, NOT the pitcher body).  Intentionally excludes
the pitcher's lower body (y>220) to prevent false starts."""

MIN_PITCH_VY: float = 10.0
"""Minimum average positive Y-velocity (pixels/frame) over the first
confirmed frames to accept a track as a real pitch.  Set high enough
(10+) to reject pitcher body animation (avg vy ≈ 5–7) while accepting
the ball (avg vy ≈ 11+ once in flight)."""

# Static element suppression

STATIC_CELL_SIZE: int = 12
"""Grid cell size (px) for static element detection."""

STATIC_HIT_THRESHOLD: int = 8
"""A grid cell seen this many times is considered static and suppressed."""

# Pitch recording

RECORD_MAX_FRAMES: int = 120
"""Auto-stop recording after this many frames (~2 s at 60 fps)."""

PITCHES_DIR: Path = Path("./pitches")

# Display

DISPLAY_SCALE: float = 0.85

WINDOW_MAIN: str = "MSB Pitch Analyzer"
WINDOW_DEBUG: str = "MSB Pitch Debug"

# Colours (BGR)

COL_GREEN   = (0, 255, 0)
COL_RED     = (0, 0, 255)
COL_YELLOW  = (0, 255, 255)
COL_CYAN    = (255, 255, 0)
COL_WHITE   = (255, 255, 255)
COL_BLACK   = (0, 0, 0)
COL_MAGENTA = (255, 0, 255)
COL_ORANGE  = (0, 165, 255)
FONT        = cv2.FONT_HERSHEY_SIMPLEX


#  UTILITY HELPERS

def crop(frame: np.ndarray, rect: Tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = rect
    return frame[max(0, y1):y2, max(0, x1):x2]


def put_text(img: np.ndarray, text: str, org: Tuple[int, int],
             scale: float = 0.6, color: Tuple[int, ...] = COL_GREEN,
             thickness: int = 2) -> None:
    x, y = org
    cv2.putText(img, text, (x + 1, y + 1), FONT, scale,
                COL_BLACK, thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), FONT, scale,
                color, thickness, cv2.LINE_AA)


#  BALL CANDIDATE

class BallCandidate:
    """One detected blob that might be the ball."""

    __slots__ = ("center", "area", "circularity", "bbox", "contour",
                 "in_motion_mask", "score")

    def __init__(self) -> None:
        self.center: Tuple[int, int] = (0, 0)
        self.area: float = 0.0
        self.circularity: float = 0.0
        self.bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)  # x,y,w,h
        self.contour: Optional[np.ndarray] = None
        self.in_motion_mask: bool = False
        self.score: float = 0.0

    def __repr__(self) -> str:
        return (f"Ball({self.center}, area={self.area:.0f}, "
                f"circ={self.circularity:.2f}, score={self.score:.2f})")


#  BALL DETECTOR

class BallDetector:
    """Detect the baseball in a game frame.

    Pipeline
    --------
    1. **Frame differencing** — absolute difference of current vs previous
       grey frame → threshold → dilate.  Produces a *motion mask*.
    2. **White/bright colour threshold** — HSV mask for white ball pixels.
    3. **AND combine** — only keep white-ish blobs that overlap with motion.
    4. **Contour filter** — area, circularity.
    5. **Static suppression** — suppress candidates at positions that
       appear repeatedly across many frames (HUD, field markings).
    6. **Scoring** — rank candidates (smaller area scores higher when inside
       motion, higher circularity preferred).
    """

    def __init__(self) -> None:
        self._prev_grey: Optional[np.ndarray] = None
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self._kernel_small = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (3, 3))

        # Static element map: grid_cell -> hit count
        self._static_map: Dict[Tuple[int, int], int] = {}

        # Cached masks for debug
        self.motion_mask: Optional[np.ndarray] = None
        self.white_mask: Optional[np.ndarray] = None
        self.combined_mask: Optional[np.ndarray] = None
        self.candidates: List[BallCandidate] = []
        self.best: Optional[BallCandidate] = None

    def reset(self) -> None:
        """Call when starting a new pitch / scene change."""
        self._prev_grey = None
        self.motion_mask = None
        self.white_mask = None
        self.combined_mask = None
        self.candidates.clear()
        self.best = None
        # Note: do NOT reset _static_map — it should persist across pitches

    def _cell(self, x: int, y: int) -> Tuple[int, int]:
        """Map pixel (x, y) to a grid cell for static suppression."""
        return (x // STATIC_CELL_SIZE, y // STATIC_CELL_SIZE)

    def _is_static(self, x: int, y: int) -> bool:
        """Return True if this position is a known static element."""
        cell = self._cell(x, y)
        return self._static_map.get(cell, 0) >= STATIC_HIT_THRESHOLD

    def _record_static(self, x: int, y: int, in_motion: bool) -> None:
        """Record a detection for static-element tracking.
        ALL detections contribute — quasi-static elements near home plate
        show up as 'in_motion' due to character animations but should
        still be suppressed when they appear too frequently."""
        cell = self._cell(x, y)
        self._static_map[cell] = self._static_map.get(cell, 0) + 1

    def detect(self, frame: np.ndarray,
               search_roi: Optional[Tuple[int, int, int, int]] = None
               ) -> Optional[BallCandidate]:
        """Detect the ball in *frame* (BGR).

        Parameters
        ----------
        search_roi : optional (x1, y1, x2, y2) to limit the search area.
                     If None, searches the full frame.

        Returns
        -------
        The best BallCandidate, or None if no ball found.
        """
        if search_roi is not None:
            roi_img = crop(frame, search_roi)
            ox, oy = search_roi[0], search_roi[1]
        else:
            roi_img = frame
            ox, oy = 0, 0

        if roi_img.size == 0:
            self.best = None
            return None

        grey = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)

        # 1) Frame differencing → motion mask
        if self._prev_grey is not None and self._prev_grey.shape == grey.shape:
            diff = cv2.absdiff(grey, self._prev_grey)
            _, motion = cv2.threshold(diff, DIFF_THRESHOLD, 255,
                                      cv2.THRESH_BINARY)
            motion = cv2.dilate(motion, self._kernel,
                                iterations=DIFF_DILATE_ITER)
        else:
            motion = np.zeros_like(grey)
        self._prev_grey = grey.copy()
        self.motion_mask = motion

        # 2) White / bright colour mask
        white = cv2.inRange(hsv, BALL_HSV_LOWER, BALL_HSV_UPPER)
        # Light morphology to clean noise
        white = cv2.morphologyEx(white, cv2.MORPH_OPEN, self._kernel_small,
                                 iterations=1)
        white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, self._kernel_small,
                                 iterations=1)
        self.white_mask = white

        # 3) Combine: white blobs that overlap with motion
        #    But also keep pure-white blobs as lower-confidence candidates
        combined = cv2.bitwise_and(white, motion)
        combined = cv2.dilate(combined, self._kernel_small, iterations=1)
        self.combined_mask = combined

        # 4) Find contours on the combined mask
        self.candidates.clear()
        self.best = None

        # Primary: combined (motion + white)
        candidates_motion = self._extract_candidates(
            combined, ox, oy, in_motion=True)
        # Secondary: white-only (for when the ball just appeared and there's
        # no prev-frame diff yet, or near the edge of the frame)
        candidates_white = self._extract_candidates(
            white, ox, oy, in_motion=False)

        # Merge, de-duplicate by proximity
        all_cands = candidates_motion[:]
        for wc in candidates_white:
            dup = False
            for mc in candidates_motion:
                d = np.hypot(wc.center[0] - mc.center[0],
                             wc.center[1] - mc.center[1])
                if d < 20:
                    dup = True
                    break
            if not dup:
                all_cands.append(wc)

        if not all_cands:
            return None

        # 5) Suppress known static elements (HUD, field markings)
        filtered: List[BallCandidate] = []
        for c in all_cands:
            self._record_static(c.center[0], c.center[1], c.in_motion_mask)
            if self._is_static(c.center[0], c.center[1]):
                continue  # skip static element
            filtered.append(c)

        if not filtered:
            # Fallback: keep all if every candidate was suppressed
            filtered = all_cands

        # 6) Score candidates
        for c in filtered:
            # Prefer: in-motion, smaller area (faraway ball), high circularity
            motion_bonus = 2.0 if c.in_motion_mask else 0.5
            # Normalise area to [0, 1] where smaller is better
            area_norm = 1.0 - min(c.area / BALL_MAX_AREA, 1.0)
            circ_norm = min(c.circularity / 1.0, 1.0)
            c.score = motion_bonus * (0.5 * circ_norm + 0.5 * area_norm)

        filtered.sort(key=lambda c: c.score, reverse=True)
        self.candidates = filtered
        self.best = filtered[0]
        return self.best

    def _extract_candidates(self, mask: np.ndarray,
                            ox: int, oy: int,
                            in_motion: bool) -> List[BallCandidate]:
        """Extract ball candidates from a binary mask."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        result: List[BallCandidate] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (BALL_MIN_AREA <= area <= BALL_MAX_AREA):
                continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter < 1:
                continue
            circularity = 4.0 * np.pi * area / (perimeter * perimeter)
            if circularity < BALL_MIN_CIRCULARITY:
                continue

            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"]) + ox
                cy = int(M["m01"] / M["m00"]) + oy
            else:
                bx, by, bw, bh = cv2.boundingRect(cnt)
                cx, cy = bx + bw // 2 + ox, by + bh // 2 + oy

            bx, by, bw, bh = cv2.boundingRect(cnt)

            c = BallCandidate()
            c.center = (cx, cy)
            c.area = area
            c.circularity = circularity
            c.bbox = (bx + ox, by + oy, bw, bh)
            # Offset contour to full-frame coords
            cnt_full = cnt.copy()
            cnt_full[:, :, 0] += ox
            cnt_full[:, :, 1] += oy
            c.contour = cnt_full
            c.in_motion_mask = in_motion
            result.append(c)
        return result


#  BALL TRACKER

def _in_pitcher_zone(x: int, y: int) -> bool:
    """Return True if (x, y) is inside the pitcher zone."""
    zx1, zy1, zx2, zy2 = PITCHER_ZONE
    return zx1 <= x <= zx2 and zy1 <= y <= zy2


class TrackedBall:
    """State for a tracked ball across frames."""

    def __init__(self, pos: Tuple[int, int], frame_idx: int,
                 area: float = 0.0) -> None:
        self.positions: deque = deque(maxlen=TRAJECTORY_HISTORY)
        self.positions.append((pos[0], pos[1], frame_idx))
        self.velocity: Tuple[float, float] = (0.0, 0.0)
        self.predicted_next: Tuple[int, int] = pos
        self.frames_since_seen: int = 0
        self.total_frames: int = 1
        self.active: bool = True
        self.confirmed: bool = False   # True once we have consistent forward motion
        # Running average of tracked area (for size consistency)
        self.avg_area: float = area if area > 0 else 20.0
        self._area_alpha: float = 0.3  # EMA smoothing factor

    @property
    def last_pos(self) -> Tuple[int, int]:
        p = self.positions[-1]
        return (p[0], p[1])

    @property
    def last_frame(self) -> int:
        return self.positions[-1][2]

    def update(self, pos: Tuple[int, int], frame_idx: int,
               area: float = 0.0) -> None:
        """Register a new detection."""
        old = self.last_pos
        dt = frame_idx - self.last_frame
        if dt > 0:
            self.velocity = ((pos[0] - old[0]) / dt,
                             (pos[1] - old[1]) / dt)
        self.positions.append((pos[0], pos[1], frame_idx))
        self.predicted_next = (
            int(pos[0] + self.velocity[0]),
            int(pos[1] + self.velocity[1]))
        self.frames_since_seen = 0
        self.total_frames += 1
        if area > 0:
            self.avg_area = (self._area_alpha * area +
                             (1 - self._area_alpha) * self.avg_area)
        # Check confirmation: need TRACK_MIN_CONFIRMATIONS points with
        # an overall positive Y-trend (ball moving toward batter).
        if not self.confirmed and self.total_frames >= TRACK_MIN_CONFIRMATIONS:
            pts = list(self.positions)
            if len(pts) >= TRACK_MIN_CONFIRMATIONS:
                dy_total = pts[-1][1] - pts[0][1]
                df_total = pts[-1][2] - pts[0][2]
                if df_total > 0:
                    avg_vy = dy_total / df_total
                    if avg_vy >= MIN_PITCH_VY:
                        self.confirmed = True

    def predict(self, n_frames: int = 1) -> Tuple[int, int]:
        """Predict position n_frames into the future using velocity."""
        p = self.last_pos
        return (int(p[0] + self.velocity[0] * n_frames),
                int(p[1] + self.velocity[1] * n_frames))

    def mark_missed(self) -> None:
        """Called when no detection matched this frame."""
        self.frames_since_seen += 1
        # Advance predicted position using velocity (extrapolate)
        self.predicted_next = self.predict(self.frames_since_seen + 1)
        # Different timeout for confirmed vs tentative tracks
        max_gap = TRACK_LOST_FRAMES if self.confirmed else TRACK_TENTATIVE_LOST
        if self.frames_since_seen >= max_gap:
            self.active = False

    def avg_velocity_y(self) -> float:
        """Average Y-velocity over the whole track."""
        pts = list(self.positions)
        if len(pts) < 2:
            return 0.0
        dy = pts[-1][1] - pts[0][1]
        df = pts[-1][2] - pts[0][2]
        return dy / df if df > 0 else 0.0


class BallTracker:
    """Track the pitched ball across frames.

    Design principles
    -----------------
    1. **Pitcher-zone gating**: New tracks may ONLY start from candidates
       inside ``PITCHER_ZONE``.  This eliminates 90 %+ of false starts
       from HUD elements, home-plate noise, and character animations.

    2. **Confirmation**: A new track is *tentative* until it has
       ``TRACK_MIN_CONFIRMATIONS`` detections with consistent forward
       (positive-Y) motion.  Tentative tracks that stall are killed.

    3. **Proximity-to-prediction matching**: Once a track is active,
       the tracker picks the candidate *closest to the predicted position*
       (not just the detector's top score).

    4. **Gap tolerance**: Allows up to ``TRACK_LOST_FRAMES`` missed frames
       so the ball can survive the cluttered batter/catcher zone.
    """

    def __init__(self) -> None:
        self.track: Optional[TrackedBall] = None
        self._frame_idx: int = 0
        self.selected: Optional[BallCandidate] = None

    def reset(self) -> None:
        self.track = None
        self._frame_idx = 0
        self.selected = None

    # ------------------------------------------------------------------
    def _is_size_consistent(self, candidate: BallCandidate) -> bool:
        """Reject candidates whose area is wildly different from
        what we've been tracking."""
        if self.track is None or self.track.avg_area <= 0:
            return True
        avg = self.track.avg_area
        ratio = max(candidate.area, avg) / max(min(candidate.area, avg), 1)
        return ratio <= TRACK_SIZE_RATIO

    # ------------------------------------------------------------------
    def _find_nearest_candidate(
        self, candidates: List[BallCandidate]
    ) -> Optional[BallCandidate]:
        """Find the candidate closest to the predicted position of the
        active track, respecting distance and size constraints."""
        if self.track is None:
            return None
        pred = self.track.predicted_next
        last = self.track.last_pos
        best_dist = float("inf")
        chosen: Optional[BallCandidate] = None
        for c in candidates:
            if c.area > BALL_FLIGHT_MAX_AREA:
                continue
            if not self._is_size_consistent(c):
                continue
            d_pred = np.hypot(c.center[0] - pred[0],
                              c.center[1] - pred[1])
            # For confirmed tracks use prediction only;
            # for tentative also consider distance to last position
            if self.track.confirmed:
                d = d_pred
            else:
                d_last = np.hypot(c.center[0] - last[0],
                                  c.center[1] - last[1])
                d = min(d_pred, d_last)
            # Speed-proportional search radius
            speed = np.hypot(self.track.velocity[0],
                             self.track.velocity[1])
            base_d = max(TRACK_MAX_DIST_MIN,
                         TRACK_MAX_DIST_SPEED_K * speed)
            max_d = base_d * (1.0 + TRACK_GAP_EXPAND *
                              self.track.frames_since_seen)
            if d < best_dist and d <= max_d:
                best_dist = d
                chosen = c
        return chosen

    # ------------------------------------------------------------------
    def update(self, candidates: List[BallCandidate],
               best_candidate: Optional[BallCandidate] = None
               ) -> Optional[TrackedBall]:
        """Update tracker with ALL candidates from the detector."""
        self._frame_idx += 1
        self.selected = None

        if self.track is not None and self.track.active:
            # ACTIVE TRACK: match closest candidate to prediction
            chosen = self._find_nearest_candidate(candidates)

            if chosen is not None:
                self.track.update(chosen.center, self._frame_idx,
                                  area=chosen.area)
                self.selected = chosen

                if not self.track.confirmed:
                    # Kill tentative tracks moving AWAY from the batter
                    # (negative avg vy = probably pitcher body / hand).
                    if (self.track.total_frames >= 2
                            and self.track.avg_velocity_y() <= 0):
                        self.track = None
                        self.selected = None
                    # Kill tentative tracks that stalled without confirming
                    elif self.track.total_frames >= TRACK_MIN_CONFIRMATIONS + 2:
                        self.track = None
                        self.selected = None
            else:
                self.track.mark_missed()
                if not self.track.active:
                    self.track = None

        # START A NEW TRACK IF NONE ACTIVE
        # NOTE: intentionally *not* an ``else`` — when a tentative track
        # dies above we can immediately try to start a fresh one in the
        # same frame.
        if self.track is None:
            best_pz: Optional[BallCandidate] = None
            best_score = -1.0
            for c in candidates:
                if not c.in_motion_mask:
                    continue
                if c.area > BALL_FLIGHT_MAX_AREA:
                    continue
                if not _in_pitcher_zone(c.center[0], c.center[1]):
                    continue
                if c.score > best_score:
                    best_score = c.score
                    best_pz = c

            if best_pz is not None:
                self.track = TrackedBall(best_pz.center, self._frame_idx,
                                         area=best_pz.area)
                self.selected = best_pz

        return self.track


#  TRAJECTORY PREDICTOR

class TrajectoryPredictor:
    """Predict where the ball will cross a target Y-coordinate (the strike
    zone horizontal line) using the tracked trajectory.

    Uses quadratic regression on the last N positions for better curve-ball
    handling, falls back to linear if too few points.
    """

    def __init__(self, target_y: Optional[int] = None) -> None:
        self.target_y = target_y
        self.predicted_x: Optional[int] = None
        self.predicted_frame: Optional[int] = None
        self.fit_type: str = ""   # "linear" or "quadratic"
        self.confidence: float = 0.0

    def set_target_y(self, y: int) -> None:
        """Set the Y-coordinate of the strike zone (where the ball arrives)."""
        self.target_y = y

    def predict(self, track: TrackedBall) -> Optional[Tuple[int, int]]:
        """Predict (x, y) where the ball crosses target_y.

        Returns None if prediction isn't possible yet.
        """
        self.predicted_x = None
        self.predicted_frame = None
        self.confidence = 0.0

        if self.target_y is None:
            return None

        positions = list(track.positions)
        n = len(positions)
        if n < 3:
            return None

        # Use frame index as the independent variable
        frames = np.array([p[2] for p in positions], dtype=np.float64)
        xs = np.array([p[0] for p in positions], dtype=np.float64)
        ys = np.array([p[1] for p in positions], dtype=np.float64)

        # Normalise frames to start at 0
        f0 = frames[0]
        frames_n = frames - f0

        # Check if ball is moving toward target_y
        if n >= 2:
            dy = ys[-1] - ys[0]
            target_dir = self.target_y - ys[-1]
            if dy * target_dir < 0:
                # Ball moving away from target
                return None

        try:
            if n >= 5:
                # Quadratic fit
                cx = np.polyfit(frames_n, xs, 2)
                cy = np.polyfit(frames_n, ys, 2)
                self.fit_type = "quadratic"
            else:
                # Linear fit
                cx = np.polyfit(frames_n, xs, 1)
                cy = np.polyfit(frames_n, ys, 1)
                self.fit_type = "linear"
        except (np.linalg.LinAlgError, ValueError):
            return None

        # Solve for frame where y = target_y
        # For quadratic: cy[0]*f² + cy[1]*f + cy[2] = target_y
        # For linear: cy[0]*f + cy[1] = target_y
        poly_y = np.poly1d(cy)
        poly_y_shifted = poly_y - self.target_y
        roots = np.roots(poly_y_shifted)

        # Find the smallest positive real root (future frame)
        current_f = frames_n[-1]
        best_f: Optional[float] = None
        for r in roots:
            if np.isreal(r):
                rv = float(np.real(r))
                if rv > current_f:
                    if best_f is None or rv < best_f:
                        best_f = rv

        if best_f is None:
            return None

        poly_x = np.poly1d(cx)
        pred_x = int(poly_x(best_f))
        pred_frame = int(best_f + f0)

        # Confidence based on how many points and how far away
        frames_to_target = best_f - current_f
        self.confidence = min(1.0, n / 15.0) * max(0.1, 1.0 - frames_to_target / 60.0)
        self.predicted_x = pred_x
        self.predicted_frame = pred_frame

        return (pred_x, self.target_y)

    def get_trajectory_points(self, track: TrackedBall,
                              n_future: int = 30
                              ) -> List[Tuple[int, int]]:
        """Return fitted trajectory points (past + future) for drawing."""
        positions = list(track.positions)
        n = len(positions)
        if n < 3:
            return [(p[0], p[1]) for p in positions]

        frames = np.array([p[2] for p in positions], dtype=np.float64)
        xs = np.array([p[0] for p in positions], dtype=np.float64)
        ys = np.array([p[1] for p in positions], dtype=np.float64)
        f0 = frames[0]
        frames_n = frames - f0

        try:
            deg = 2 if n >= 5 else 1
            cx = np.polyfit(frames_n, xs, deg)
            cy = np.polyfit(frames_n, ys, deg)
        except (np.linalg.LinAlgError, ValueError):
            return [(p[0], p[1]) for p in positions]

        poly_x = np.poly1d(cx)
        poly_y = np.poly1d(cy)

        pts: List[Tuple[int, int]] = []
        f_start = frames_n[0]
        f_end = frames_n[-1] + n_future
        for f in np.linspace(f_start, f_end, int(f_end - f_start) + 1):
            px = int(poly_x(f))
            py = int(poly_y(f))
            pts.append((px, py))

        return pts


#  PITCH RECORDER

class PitchRecorder:
    """Record a pitch sequence (frames + detections) to disk."""

    def __init__(self, root: Path = PITCHES_DIR) -> None:
        self.root = root
        self.recording: bool = False
        self.frames: List[np.ndarray] = []
        self.detections: List[Optional[Dict[str, Any]]] = []
        self._start_time: float = 0.0

    def start(self) -> None:
        self.recording = True
        self.frames.clear()
        self.detections.clear()
        self._start_time = time.time()
        print("[REC] Recording started...")

    def stop(self) -> None:
        self.recording = False
        print(f"[REC] Stopped. {len(self.frames)} frames captured.")

    def add_frame(self, frame: np.ndarray,
                  ball: Optional[BallCandidate]) -> bool:
        """Add a frame to the recording. Returns False if max reached."""
        self.frames.append(frame.copy())
        if ball is not None:
            self.detections.append({
                "center": list(ball.center),
                "area": ball.area,
                "circularity": ball.circularity,
                "bbox": list(ball.bbox),
                "in_motion": ball.in_motion_mask,
                "score": ball.score,
            })
        else:
            self.detections.append(None)

        if len(self.frames) >= RECORD_MAX_FRAMES:
            self.stop()
            return False
        return True

    def save(self) -> Optional[str]:
        """Save the recorded pitch to disk."""
        if not self.frames:
            print("[REC] Nothing to save.")
            return None

        ts = time.strftime("%Y%m%d_%H%M%S")
        pitch_dir = self.root / ts
        pitch_dir.mkdir(parents=True, exist_ok=True)

        for i, frame in enumerate(self.frames):
            cv2.imwrite(str(pitch_dir / f"frame_{i:04d}.png"), frame)

        meta = {
            "timestamp": ts,
            "n_frames": len(self.frames),
            "fps": TARGET_FPS,
            "screen_roi": list(SCREEN_ROI),
            "detections": self.detections,
        }
        with open(pitch_dir / "pitch_meta.json", "w") as fh:
            json.dump(meta, fh, indent=2)

        print(f"[REC] Saved {len(self.frames)} frames to {pitch_dir}")
        return str(pitch_dir)


#  VISUALISER

class PitchVisualiser:
    """Draw overlays for ball detection + tracking + prediction."""

    @staticmethod
    def overlay(frame: np.ndarray,
                detector: BallDetector,
                tracker: BallTracker,
                predictor: TrajectoryPredictor,
                search_roi: Optional[Tuple[int, int, int, int]],
                recording: bool,
                fps: float,
                frame_num: int = 0) -> np.ndarray:
        vis = frame.copy()
        h, w = vis.shape[:2]

        # Frame number
        put_text(vis, f"Frame {frame_num}", (10, 20), 0.45, COL_WHITE, 1)

        # Pitcher zone (where tracks can start)
        pzx1, pzy1, pzx2, pzy2 = PITCHER_ZONE
        cv2.rectangle(vis, (pzx1, pzy1), (pzx2, pzy2), (128, 128, 0), 1)
        put_text(vis, "PITCHER ZONE", (pzx1, pzy1 - 6), 0.30,
                 (128, 128, 0), 1)

        # Search ROI
        if search_roi is not None:
            x1, y1, x2, y2 = search_roi
            cv2.rectangle(vis, (x1, y1), (x2, y2), COL_CYAN, 1)
            put_text(vis, "SEARCH ROI", (x1, y1 - 8), 0.35, COL_CYAN, 1)

        # All candidates (small dots)
        for c in detector.candidates:
            col = COL_YELLOW if c.in_motion_mask else (128, 128, 128)
            cv2.circle(vis, c.center, 3, col, -1)

        # Tracker's selected candidate (the one actually matched to track)
        sel = tracker.selected
        if sel is not None:
            cv2.circle(vis, sel.center, 10, COL_GREEN, 2)
            cv2.circle(vis, sel.center, 2, COL_RED, -1)
            bx, by, bw, bh = sel.bbox
            cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), COL_GREEN, 1)
            put_text(vis, f"TRACKED  a={sel.area:.0f} c={sel.circularity:.2f}",
                     (bx, by - 8), 0.38, COL_GREEN, 1)
        elif detector.best is not None:
            # No active track — show detector's best pick dimmer
            b = detector.best
            cv2.circle(vis, b.center, 8, COL_ORANGE, 2)
            cv2.circle(vis, b.center, 2, COL_ORANGE, -1)
            bx, by, bw, bh = b.bbox
            cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), COL_ORANGE, 1)
            put_text(vis, f"CAND  a={b.area:.0f} c={b.circularity:.2f}",
                     (bx, by - 8), 0.38, COL_ORANGE, 1)

        # Tracked ball trajectory
        track = tracker.track
        if track is not None and track.active and len(track.positions) >= 2:
            # Draw past positions as a fading trail
            positions = list(track.positions)
            n = len(positions)
            for i in range(1, n):
                alpha = i / n
                col = (0, int(255 * alpha), int(255 * (1 - alpha)))
                pt1 = (positions[i - 1][0], positions[i - 1][1])
                pt2 = (positions[i][0], positions[i][1])
                cv2.line(vis, pt1, pt2, col, 2)

            # Velocity arrow
            last = track.last_pos
            vx, vy = track.velocity
            if abs(vx) + abs(vy) > 1:
                tip = (int(last[0] + vx * 3), int(last[1] + vy * 3))
                cv2.arrowedLine(vis, last, tip, COL_MAGENTA, 2,
                                tipLength=0.3)

            # Predicted trajectory line
            traj_pts = predictor.get_trajectory_points(track, n_future=20)
            if len(traj_pts) >= 2:
                for i in range(len(traj_pts) - 1):
                    # Future points in dashed yellow
                    if i > n:
                        if i % 2 == 0:
                            cv2.line(vis, traj_pts[i], traj_pts[i + 1],
                                     COL_YELLOW, 1)

            # Prediction crosshair at target_y
            if predictor.predicted_x is not None and predictor.target_y is not None:
                px = predictor.predicted_x
                py = predictor.target_y
                # Crosshair
                cv2.drawMarker(vis, (px, py), COL_RED,
                               cv2.MARKER_CROSS, 20, 2)
                cv2.circle(vis, (px, py), 12, COL_RED, 2)
                put_text(vis,
                         f"PREDICT ({px},{py})  "
                         f"{predictor.fit_type}  "
                         f"conf={predictor.confidence:.0%}",
                         (px + 16, py - 6), 0.38, COL_RED, 1)

            # Target Y line
            if predictor.target_y is not None:
                cv2.line(vis, (0, predictor.target_y),
                         (w, predictor.target_y), COL_RED, 1)
                put_text(vis, "STRIKE Y",
                         (w - 100, predictor.target_y - 6),
                         0.35, COL_RED, 1)

            # Track info
            status = "CONFIRMED" if track.confirmed else "tentative"
            put_text(vis,
                     f"TRACK [{status}]  frames={track.total_frames}  "
                     f"vel=({track.velocity[0]:.1f},{track.velocity[1]:.1f})  "
                     f"missed={track.frames_since_seen}",
                     (10, h - 55), 0.40, COL_CYAN, 1)

        # Recording indicator
        if recording:
            cv2.circle(vis, (w - 30, 30), 12, COL_RED, -1)
            put_text(vis, "REC", (w - 65, 36), 0.50, COL_RED, 2)

        # FPS + hints
        put_text(vis, f"FPS {fps:.0f}", (w - 110, h - 8), 0.45,
                 COL_GREEN, 1)
        hints = ("[SPACE] Record  [D] Debug  [C] Set ROI  "
                 "[Y] Set strike-Y  [S] Save  [Q] Quit")
        put_text(vis, hints, (10, h - 8), 0.35, COL_WHITE, 1)

        return vis

    @staticmethod
    def debug_panel(detector: BallDetector,
                    frame: np.ndarray,
                    search_roi: Optional[Tuple[int, int, int, int]]
                    ) -> np.ndarray:
        PW, PH = 320, 240
        panels: List[np.ndarray] = []

        def _tile(img: Optional[np.ndarray], label: str,
                  col: Tuple[int, ...] = COL_GREEN) -> np.ndarray:
            if img is None or img.size == 0:
                t = np.zeros((PH, PW, 3), np.uint8)
            else:
                t = cv2.resize(img, (PW, PH))
            if t.ndim == 2:
                t = cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)
            put_text(t, label, (4, 18), 0.45, col, 1)
            return t

        # Motion mask
        panels.append(_tile(detector.motion_mask, "Motion mask", COL_CYAN))

        # White mask
        panels.append(_tile(detector.white_mask, "White mask", COL_YELLOW))

        # Combined mask
        panels.append(_tile(detector.combined_mask,
                            "Combined (motion+white)", COL_GREEN))

        # ROI with all candidates drawn
        if search_roi is not None:
            roi_vis = crop(frame, search_roi).copy()
        else:
            roi_vis = frame.copy()
        if roi_vis.size > 0:
            for c in detector.candidates:
                col = COL_GREEN if c.in_motion_mask else COL_YELLOW
                # Adjust to ROI coords if needed
                if search_roi is not None:
                    cx = c.center[0] - search_roi[0]
                    cy = c.center[1] - search_roi[1]
                else:
                    cx, cy = c.center
                cv2.circle(roi_vis, (cx, cy), 5, col, 2)
                cv2.putText(roi_vis, f"{c.score:.1f}",
                            (cx + 6, cy - 4), FONT, 0.30, col, 1)
            panels.append(_tile(roi_vis,
                                f"Candidates ({len(detector.candidates)})",
                                COL_MAGENTA))
        else:
            panels.append(_tile(None, "No ROI", COL_RED))

        # Arrange 2×2
        while len(panels) < 4:
            panels.append(np.zeros((PH, PW, 3), np.uint8))
        row1 = np.hstack(panels[:2])
        row2 = np.hstack(panels[2:4])
        return np.vstack([row1, row2])


#  CLICK-TO-SET ROI (simple 2-click)

class ROISelector:
    """Simple two-click ROI selector for the ball search area."""

    def __init__(self, display_scale: float = 1.0) -> None:
        self.scale = display_scale
        self.active: bool = False
        self.rect: Optional[Tuple[int, int, int, int]] = None
        self._pt1: Optional[Tuple[int, int]] = None
        self._frozen: Optional[np.ndarray] = None

    def start(self, frame: np.ndarray) -> None:
        self.active = True
        self._pt1 = None
        self._frozen = frame.copy()

    def cancel(self) -> None:
        self.active = False
        self._pt1 = None
        self._frozen = None

    def mouse_callback(self, event: int, x: int, y: int,
                       flags: int, param: Any) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if not self.active:
            return

        ox = int(x / self.scale)
        oy = int(y / self.scale)

        if self._pt1 is None:
            self._pt1 = (ox, oy)
        else:
            x1 = min(self._pt1[0], ox)
            y1 = min(self._pt1[1], oy)
            x2 = max(self._pt1[0], ox)
            y2 = max(self._pt1[1], oy)
            self.rect = (x1, y1, x2, y2)
            self.active = False
            self._frozen = None
            print(f"[ROI] Ball search area: {self.rect}")

    def draw_frozen(self) -> Optional[np.ndarray]:
        if self._frozen is None:
            return None
        vis = self._frozen.copy()
        overlay = np.zeros_like(vis)
        vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)
        put_text(vis, "Click TOP-LEFT then BOTTOM-RIGHT of ball search area",
                 (10, 30), 0.55, COL_YELLOW, 2)
        if self._pt1 is not None:
            cv2.drawMarker(vis, self._pt1, COL_CYAN,
                           cv2.MARKER_CROSS, 20, 2)
            put_text(vis, "TL",
                     (self._pt1[0] + 5, self._pt1[1] - 5),
                     0.40, COL_CYAN, 1)
        if self.rect is not None:
            x1, y1, x2, y2 = self.rect
            cv2.rectangle(vis, (x1, y1), (x2, y2), COL_CYAN, 2)
        return vis


#  CLICK-TO-SET STRIKE-Y

class StrikeYSelector:
    """Click anywhere to set the Y-coordinate of the strike zone line."""

    def __init__(self, display_scale: float = 1.0) -> None:
        self.scale = display_scale
        self.active: bool = False
        self.target_y: Optional[int] = None
        self._frozen: Optional[np.ndarray] = None

    def start(self, frame: np.ndarray) -> None:
        self.active = True
        self._frozen = frame.copy()

    def cancel(self) -> None:
        self.active = False
        self._frozen = None

    def mouse_callback(self, event: int, x: int, y: int,
                       flags: int, param: Any) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if not self.active:
            return

        oy = int(y / self.scale)
        self.target_y = oy
        self.active = False
        self._frozen = None
        print(f"[STRIKE-Y] Target Y = {oy}")

    def draw_frozen(self) -> Optional[np.ndarray]:
        if self._frozen is None:
            return None
        vis = self._frozen.copy()
        overlay = np.zeros_like(vis)
        vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)
        put_text(vis, "Click the Y-level where the ball reaches the batter",
                 (10, 30), 0.55, COL_YELLOW, 2)
        # Draw horizontal guide line at mouse position (can't track live,
        # so just show instruction)
        return vis


#  FRAME SOURCES

def source_live(fps: int = TARGET_FPS,
                roi: Tuple[int, int, int, int] = SCREEN_ROI):
    try:
        import dxcam
    except ImportError:
        print("[ERROR] dxcam not installed.  pip install dxcam")
        sys.exit(1)

    cam = dxcam.create(output_idx=MONITOR_INDEX, output_color="BGR")
    if cam is None:
        print("[ERROR] dxcam.create() returned None.")
        sys.exit(1)
    cam.start(target_fps=fps, region=roi)
    time.sleep(0.2)
    try:
        while True:
            yield cam.get_latest_frame()
    finally:
        try:
            cam.stop()
        except Exception:
            pass


def source_folder(folder: str):
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    files = sorted(f for f in Path(folder).iterdir()
                   if f.suffix.lower() in exts)
    if not files:
        print(f"[ERROR] No images in {folder}")
        sys.exit(1)
    print(f"[INFO] {len(files)} images from {folder}")
    for f in files:
        img = cv2.imread(str(f))
        if img is not None:
            yield img


#  MAIN

def main() -> None:
    ap = argparse.ArgumentParser(
        description="MSB Pitch / Ball Detection & Trajectory Analyzer")
    ap.add_argument("--source", choices=["live", "folder"],
                    default="live")
    ap.add_argument("--input", default=None,
                    help="Folder of pitch frames (for folder source)")
    args = ap.parse_args()

    # Components
    detector  = BallDetector()
    tracker   = BallTracker()
    predictor = TrajectoryPredictor()
    recorder  = PitchRecorder()
    vis       = PitchVisualiser()
    roi_sel   = ROISelector(display_scale=DISPLAY_SCALE)
    sy_sel    = StrikeYSelector(display_scale=DISPLAY_SCALE)

    show_debug: bool = False
    search_roi: Optional[Tuple[int, int, int, int]] = None
    frame_num: int = 0
    is_folder: bool = (args.source == "folder")

    # Frame source
    print(f"[INFO] source={args.source}")
    if args.source == "live":
        src = source_live()
    else:
        if not args.input:
            print("[ERROR] --input required for folder source")
            sys.exit(1)
        src = source_folder(args.input)

    prev_t = time.perf_counter()
    fps_ema = float(TARGET_FPS)

    print("\n")
    print(" CONTROLS")
    print("  SPACE = Start / stop recording a pitch")
    print("  D     = Toggle debug panel")
    print("  C     = Click to define ball search ROI")
    print("  Y     = Click to set strike-zone Y level")
    print("  S     = Save recorded pitch to disk")
    print("  R     = Reset tracker (clear trajectory)")
    print("  Q/ESC = Quit")
    print("\n")

    cv2.namedWindow(WINDOW_MAIN, cv2.WINDOW_NORMAL)

    def _mouse_cb(event: int, x: int, y: int,
                  flags: int, param: Any) -> None:
        roi_sel.mouse_callback(event, x, y, flags, param)
        sy_sel.mouse_callback(event, x, y, flags, param)

    cv2.setMouseCallback(WINDOW_MAIN, _mouse_cb)

    try:
        for frame in src:
            if frame is None:
                time.sleep(0.001)
                continue

            frame_num += 1

            # FPS
            now = time.perf_counter()
            dt = now - prev_t
            prev_t = now
            ifps = 1.0 / dt if dt > 0 else 0.0
            fps_ema = 0.1 * ifps + 0.9 * fps_ema

            # ROI selection mode
            if roi_sel.active:
                frozen = roi_sel.draw_frozen()
                if frozen is not None:
                    if DISPLAY_SCALE != 1.0:
                        dw = int(frozen.shape[1] * DISPLAY_SCALE)
                        dh = int(frozen.shape[0] * DISPLAY_SCALE)
                        frozen = cv2.resize(frozen, (dw, dh))
                    cv2.imshow(WINDOW_MAIN, frozen)
                key = cv2.waitKey(30) & 0xFF
                if key == 27:
                    roi_sel.cancel()
                elif not roi_sel.active and roi_sel.rect is not None:
                    search_roi = roi_sel.rect
                    detector.reset()
                continue

            # After ROI selection finishes
            if not roi_sel.active and roi_sel.rect is not None:
                search_roi = roi_sel.rect
                roi_sel.rect = None  # consumed

            # Strike-Y selection mode
            if sy_sel.active:
                frozen = sy_sel.draw_frozen()
                if frozen is not None:
                    if DISPLAY_SCALE != 1.0:
                        dw = int(frozen.shape[1] * DISPLAY_SCALE)
                        dh = int(frozen.shape[0] * DISPLAY_SCALE)
                        frozen = cv2.resize(frozen, (dw, dh))
                    cv2.imshow(WINDOW_MAIN, frozen)
                key = cv2.waitKey(30) & 0xFF
                if key == 27:
                    sy_sel.cancel()
                elif not sy_sel.active and sy_sel.target_y is not None:
                    predictor.set_target_y(sy_sel.target_y)
                continue

            if not sy_sel.active and sy_sel.target_y is not None:
                predictor.set_target_y(sy_sel.target_y)

            # Ball detection
            best = detector.detect(frame, search_roi)

            # Tracking
            # Pass ALL candidates so the tracker can pick the one closest
            # to its predicted position (much more robust than using only
            # the detector's top-scored pick).
            track = tracker.update(detector.candidates, best)

            # Use the tracker's selected candidate for recording
            tracked_ball = tracker.selected if tracker.selected else best

            # Prediction
            if track is not None and track.active:
                predictor.predict(track)

            # Recording
            if recorder.recording:
                recorder.add_frame(frame, tracked_ball)

            # Draw
            disp = vis.overlay(frame, detector, tracker, predictor,
                               search_roi, recorder.recording, fps_ema,
                               frame_num)
            if DISPLAY_SCALE != 1.0:
                dw = int(disp.shape[1] * DISPLAY_SCALE)
                dh = int(disp.shape[0] * DISPLAY_SCALE)
                disp = cv2.resize(disp, (dw, dh))
            cv2.imshow(WINDOW_MAIN, disp)

            if show_debug:
                dbg = vis.debug_panel(detector, frame, search_roi)
                cv2.imshow(WINDOW_DEBUG, dbg)

            # Keys
            # Use longer delay for folder playback so it's visible
            wait_ms = 33 if is_folder else 1
            key = cv2.waitKey(wait_ms) & 0xFF

            if key in (ord("q"), 27):
                break

            elif key == ord(" "):
                if recorder.recording:
                    recorder.stop()
                else:
                    recorder.start()
                    detector.reset()
                    tracker.reset()

            elif key == ord("d"):
                show_debug = not show_debug
                if not show_debug:
                    try:
                        cv2.destroyWindow(WINDOW_DEBUG)
                    except cv2.error:
                        pass
                print(f"[INFO] Debug {'ON' if show_debug else 'OFF'}")

            elif key == ord("c"):
                roi_sel.start(frame)

            elif key == ord("y"):
                sy_sel.start(frame)

            elif key == ord("s"):
                recorder.save()

            elif key == ord("r"):
                detector.reset()
                tracker.reset()
                print("[INFO] Tracker reset.")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")
    finally:
        cv2.destroyAllWindows()
        print("[INFO] Done.")


if __name__ == "__main__":
    main()
