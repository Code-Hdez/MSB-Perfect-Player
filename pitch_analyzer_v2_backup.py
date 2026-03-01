"""
Pitch / Ball Detection & Trajectory Analyzer  —  v2 (robust pipeline)

Detects the pitched baseball as it travels from pitcher to batter, tracks
its true position across frames, and follows where it goes.

Key improvements over v1
------------------------
* **Background model** — running-average background isolates foreground
  objects far more robustly than single-frame differencing.
* **Pitcher-body suppression** — large foreground blobs (pitcher arm /
  torso) are detected and their neighbourhood is penalised so the tracker
  never latches onto the windup animation.
* **Trajectory corridor** — an explicit zone (rectangle or polygon) that
  constrains where ball candidates are accepted.  Can be set manually,
  built from annotations, or (in the future) inferred automatically.
* **Kalman-filtered tracking** — constant-velocity Kalman filter keeps
  identity through brief occlusions and clutter.
* **Strict track lifecycle** — IDLE → TENTATIVE → CONFIRMED → LOST with
  layered confirmation rules: sustained positive-Y velocity, departure
  from the pitcher zone, minimum travel distance.

Controls (live mode)
--------------------
  SPACE  = Start / stop recording a pitch
  D      = Toggle debug panel
  C      = Click to define ball search ROI
  Y      = Click to set strike-zone Y level
  S      = Save current pitch recording to disk
  R      = Reset tracker
  Q/ESC  = Quit

Usage
-----
  python pitch_analyzer.py
  python pitch_analyzer.py --source folder --input ./pitches/20260227_205241
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

#  CONFIGURATION

SCREEN_ROI: Tuple[int, int, int, int] = (378, 127, 1542, 1019)
"""(left, top, right, bottom) pixel coords of the Dolphin game window."""

MONITOR_INDEX: int = 0

TARGET_FPS: int = 60

# Ball detection (HSV)

BALL_HSV_LOWER: np.ndarray = np.array([0, 0, 200])
BALL_HSV_UPPER: np.ndarray = np.array([180, 50, 255])
"""Very low saturation + very high value -> white / near-white.
The ball in flight can drop to V≈160 and miss this filter;
the rescue detection channel handles that case."""

TRAIL_HSV_LOWER: np.ndarray = np.array([0, 0, 140])
TRAIL_HSV_UPPER: np.ndarray = np.array([180, 70, 255])
"""Slightly dimmer range to catch motion-blur trail."""

# Contour filtering

BALL_MIN_AREA: int = 8
"""Minimum contour area (px^2) for a ball candidate."""

BALL_MAX_AREA: int = 300
"""Maximum contour area -- the ball is small even up close."""

BALL_FLIGHT_MAX_AREA: int = 160
"""Maximum area for a ball actively in flight (tighter, but allows
for the ball growing slightly as it approaches the camera)."""

BALL_MIN_CIRCULARITY: float = 0.20
"""Minimum circularity (4*pi*area / perimeter^2).  Low for motion blur."""

# Background model

BG_ALPHA: float = 0.02
"""Learning rate for the background model after warm-up.
Lower = more stable but slower to adapt to lighting."""

BG_WARMUP_FRAMES: int = 25
"""Frames to accumulate before background model is considered ready.
During warm-up, the learning rate is 1/n (plain running average)."""

BG_FG_THRESHOLD: int = 25
"""Grey-level difference vs background to be classified as foreground."""

# Frame differencing (fallback while bg warms up)

DIFF_THRESHOLD: int = 30
"""Grey-level difference for frame-to-frame differencing."""

DIFF_DILATE_ITER: int = 2
"""Dilation iterations on the motion mask."""

# Pitcher suppression

PITCHER_BODY_MIN_AREA: int = 200
"""Minimum foreground blob area (px^2) to be classified as pitcher body.
Blobs larger than this in the foreground are NOT the ball."""

ISOLATION_ZONE_SCALE: float = 0.6
"""Fraction of the large-blob bounding dimension used as suppression
radius.  Candidates inside this zone are penalised."""

ISOLATION_INNER: float = 0.5
"""Normalised distance <= this from a large blob -> full suppression."""

ISOLATION_OUTER: float = 2.0
"""Normalised distance >= this from a large blob -> no suppression."""

# Trajectory corridor

CORRIDOR_DEFAULT: Optional[Tuple[int, int, int, int]] = (380, 80, 820, 900)
"""Default rectangular corridor (x1, y1, x2, y2) in ROI coords.
Covers the central pitch lane generously.  Set to None to disable."""

CORRIDOR_MARGIN: int = 50
"""Extra margin when building corridor from annotations."""

CORRIDOR_PENALTY_DIST: float = 60.0
"""Candidates outside the corridor by more than this many pixels get
a hard 0 corridor score.  Between 0 and this distance, the score
linearly decreases from 1.0 to 0.0."""

# Ball tracker

TRACK_MAX_DIST_MIN: float = 50.0
"""Minimum search radius (px) for candidate-to-prediction matching.

In MSB the ball can move 15-25 px/frame and change direction
significantly at release, so a generous minimum is safer."""

TRACK_MAX_DIST_SPEED_K: float = 3.0
"""Multiplier on ball speed for the adaptive search radius."""

TRACK_GAP_EXPAND: float = 0.25
"""Per-missed-frame expansion factor on the search radius."""

TRACK_LOST_FRAMES: int = 5
"""Frames without a match before a confirmed track is declared lost."""

TRACK_TENTATIVE_LOST: int = 3
"""Frames without a match before a tentative track is killed.

Near the pitcher body the ball dims (V≈160, below the V=200 HSV
threshold) and cannot be detected for several frames.  A short
timeout lets the track die quickly so the tracker can re-acquire
the ball once it brightens below the pitcher zone."""

TRAJECTORY_HISTORY: int = 60
"""Past positions to keep for trajectory fitting."""

TRACK_MIN_CONFIRMATIONS: int = 8
"""Detections with forward motion needed to confirm.

Requires 8+ detections showing consistent downward motion
before accepting the track as real. This prevents premature
confirmation on pitcher arm oscillations (typically 3-5 frames)."""

TRACK_SIZE_RATIO: float = 4.0
"""Max area ratio between candidate and tracked average area."""

MIN_PITCH_VY: float = 5.0
"""Minimum average Y-velocity (px/frame) to accept as a real pitch."""

MAX_PITCHER_ZONE_FRAMES: int = 15
"""Maximum frames a tentative track may remain inside the pitcher zone
before being killed.  In MSB the ball can spend 5-8 frames in the
pitcher zone due to the release animation perspective."""

MIN_DEPARTURE_DIST: float = 40.0
"""Minimum Euclidean distance the ball must travel from its start
position before the track can be confirmed."""

MAX_VY_SIGN_CHANGES: int = 1
"""Maximum allowed sign changes in Y-velocity during tentative phase.
Real pitches move consistently forward (positive Y); the pitcher's
oscillating arm flips direction repeatedly."""

# Kalman filter tuning

KF_PROCESS_NOISE: float = 16.0
"""Process noise covariance diagonal.  Higher = more responsive to
acceleration (curves) but noisier position estimate.

Increased from 4.0 to handle the ball's significant direction
change at release (horizontal → downward perspective shift)."""

KF_MEASUREMENT_NOISE: float = 4.0
"""Measurement noise covariance diagonal.  Higher = trust measurements
less, smoother trajectory."""

# Pitcher zone

PITCHER_ZONE: Tuple[int, int, int, int] = (520, 80, 680, 260)
"""(x1, y1, x2, y2) -- region where the ball first appears after
the pitcher releases it.

Expanded from (540,100,660,220) because the pitcher release point
can vary and the ball may take several frames to clear this zone."""

REACQ_ZONE: Tuple[int, int, int, int] = (480, 260, 710, 600)
"""(x1, y1, x2, y2) -- re-acquisition zone below the pitcher.

When the ball dims near the pitcher body (V<200) and the tentative
track dies, the tracker searches this zone for re-acquiring the ball
once it brightens again further from the pitcher (V≈210+ by frame ~68).
Constrained to the central pitch lane, below the pitcher zone."""

REACQ_WINDOW: int = 30
"""Maximum frames after a track dies during which re-acquisition
in REACQ_ZONE is attempted.  Prevents spurious track starts when
no pitch is in progress."""

# Static element suppression

STATIC_CELL_SIZE: int = 12
STATIC_HIT_THRESHOLD: int = 3
"""How many times a cell must fire before it is suppressed as static.
Lowered to 3 so field markings are suppressed very quickly."""

# Pitch recording

RECORD_MAX_FRAMES: int = 120
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


def _in_rect(x: int, y: int,
             rect: Tuple[int, int, int, int]) -> bool:
    """Return True if (x, y) is inside the rectangle (x1, y1, x2, y2)."""
    return rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]


def _in_pitcher_zone(x: int, y: int) -> bool:
    return _in_rect(x, y, PITCHER_ZONE)


def _in_reacq_zone(x: int, y: int) -> bool:
    return _in_rect(x, y, REACQ_ZONE)


#  BALL CANDIDATE

class BallCandidate:
    """One detected blob that might be the ball."""

    __slots__ = ("center", "area", "circularity", "bbox", "contour",
                 "in_motion_mask", "isolation_score", "corridor_score",
                 "score")

    def __init__(self) -> None:
        self.center: Tuple[int, int] = (0, 0)
        self.area: float = 0.0
        self.circularity: float = 0.0
        self.bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)  # x,y,w,h
        self.contour: Optional[np.ndarray] = None
        self.in_motion_mask: bool = False
        self.isolation_score: float = 1.0   # 1=far from pitcher body
        self.corridor_score: float = 1.0    # 1=inside corridor
        self.score: float = 0.0

    def __repr__(self) -> str:
        return (f"Ball({self.center}, area={self.area:.0f}, "
                f"circ={self.circularity:.2f}, iso={self.isolation_score:.2f}, "
                f"corr={self.corridor_score:.2f}, score={self.score:.2f})")


#  BACKGROUND MODEL

class BackgroundModel:
    """Running-average background for foreground segmentation.

    During warm-up (first ``BG_WARMUP_FRAMES`` frames) the learning rate
    is 1/n so the background converges quickly.  After warm-up it drops
    to ``BG_ALPHA`` for slow, stable adaptation.

    Attributes
    ----------
    ready : bool
        True once warm-up is complete.
    fg_mask : np.ndarray | None
        Latest foreground binary mask (same size as input grey frame).
    """

    def __init__(self) -> None:
        self._bg: Optional[np.ndarray] = None          # float32 grey
        self._frame_count: int = 0
        self._warmup: bool = True
        self.ready: bool = False
        self.fg_mask: Optional[np.ndarray] = None
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # ------------------------------------------------------------------
    def update(self, grey: np.ndarray,
               learning: bool = True) -> np.ndarray:
        """Feed a new grey frame.  Returns the foreground binary mask."""
        if self._bg is None:
            self._bg = grey.astype(np.float32)
            self._frame_count = 1
            self.fg_mask = np.zeros_like(grey)
            return self.fg_mask

        self._frame_count += 1

        if learning:
            if self._warmup:
                alpha = 1.0 / self._frame_count
                if self._frame_count >= BG_WARMUP_FRAMES:
                    self._warmup = False
                    self.ready = True
            else:
                alpha = BG_ALPHA
            cv2.accumulateWeighted(grey, self._bg, alpha)

        # Foreground = pixels that differ significantly from the model
        bg_u8 = self._bg.astype(np.uint8)
        diff = cv2.absdiff(grey, bg_u8)
        _, fg = cv2.threshold(diff, BG_FG_THRESHOLD, 255, cv2.THRESH_BINARY)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self._kernel, iterations=1)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self._kernel, iterations=1)
        self.fg_mask = fg
        return fg

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self._bg = None
        self._frame_count = 0
        self._warmup = True
        self.ready = False
        self.fg_mask = None


#  PITCHER SUPPRESSOR

class PitcherSuppressor:
    """Identifies large foreground blobs (pitcher body / arm) and computes
    an isolation score for each ball candidate.

    Any small blob that is too close to a large foreground region is
    likely part of the pitcher animation, not the ball.
    """

    def __init__(self) -> None:
        self.large_blobs: List[Tuple[int, int, int, int, float]] = []
        # (cx, cy, w, h, area)
        self.suppression_zones: List[Tuple[int, int, int]] = []
        # (cx, cy, radius)

    # ------------------------------------------------------------------
    def analyze(self, fg_mask: np.ndarray,
                min_area: int = PITCHER_BODY_MIN_AREA) -> None:
        """Find large foreground blobs and set up suppression zones."""
        self.large_blobs.clear()
        self.suppression_zones.clear()

        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                bx, by, bw, bh = cv2.boundingRect(cnt)
                cx, cy = bx + bw // 2, by + bh // 2

            bx, by, bw, bh = cv2.boundingRect(cnt)
            self.large_blobs.append((cx, cy, bw, bh, area))
            radius = int(max(bw, bh) * ISOLATION_ZONE_SCALE)
            self.suppression_zones.append((cx, cy, radius))

    # ------------------------------------------------------------------
    def get_isolation_score(self, x: int, y: int) -> float:
        """Return 0.0 if *inside* a suppression zone, 1.0 if far enough.

        Linearly interpolated between ``ISOLATION_INNER`` and
        ``ISOLATION_OUTER`` normalised distances.
        """
        if not self.suppression_zones:
            return 1.0

        min_nd = float("inf")
        for sx, sy, r in self.suppression_zones:
            d = np.hypot(x - sx, y - sy)
            nd = d / max(r, 1)
            if nd < min_nd:
                min_nd = nd

        if min_nd <= ISOLATION_INNER:
            return 0.0
        if min_nd >= ISOLATION_OUTER:
            return 1.0
        return (min_nd - ISOLATION_INNER) / (ISOLATION_OUTER - ISOLATION_INNER)


#  TRAJECTORY CORRIDOR

class TrajectoryCorridor:
    """Defines the zone where the ball is expected to fly.

    Stage 1 (now)   — static rectangle or polygon set manually, from
                      config, or built from annotations.
    Stage 2 (later) — dynamic corridor that expands around the tracked
                      path in real time.
    """

    def __init__(self) -> None:
        self.rect: Optional[Tuple[int, int, int, int]] = None
        self.polygon: Optional[np.ndarray] = None   # Nx1x2 int32
        self.active: bool = False
        self._dynamic_points: List[Tuple[int, int]] = []

    # ------------------------------------------------------------------
    def set_rect(self, x1: int, y1: int, x2: int, y2: int) -> None:
        self.rect = (x1, y1, x2, y2)
        self.polygon = None
        self.active = True

    def set_polygon(self, points: List[Tuple[int, int]]) -> None:
        self.polygon = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
        self.rect = None
        self.active = True

    # ------------------------------------------------------------------
    def from_annotations(self, positions: List[Tuple[int, int]],
                         margin: int = CORRIDOR_MARGIN) -> None:
        """Build a bounding rectangle from labelled ball positions."""
        if not positions:
            return
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        self.set_rect(min(xs) - margin, min(ys) - margin,
                      max(xs) + margin, max(ys) + margin)

    # ------------------------------------------------------------------
    def contains(self, x: int, y: int) -> bool:
        if not self.active:
            return True
        if self.polygon is not None:
            return cv2.pointPolygonTest(
                self.polygon, (float(x), float(y)), False) >= 0
        if self.rect is not None:
            return _in_rect(x, y, self.rect)
        return True

    def distance_to_boundary(self, x: int, y: int) -> float:
        """Distance from point to corridor boundary.  Negative = inside."""
        if not self.active:
            return -1.0
        if self.polygon is not None:
            return -cv2.pointPolygonTest(
                self.polygon, (float(x), float(y)), True)
        if self.rect is not None:
            x1, y1, x2, y2 = self.rect
            dx = max(x1 - x, 0, x - x2)
            dy = max(y1 - y, 0, y - y2)
            inside = (x1 <= x <= x2) and (y1 <= y <= y2)
            d = np.hypot(dx, dy)
            return -d if inside else d
        return -1.0

    def get_corridor_score(self, x: int, y: int) -> float:
        """1.0 = inside corridor, linearly decreasing to 0.0 at
        ``CORRIDOR_PENALTY_DIST`` outside, then hard 0."""
        d = self.distance_to_boundary(x, y)
        if d <= 0:
            return 1.0
        if d >= CORRIDOR_PENALTY_DIST:
            return 0.0
        return 1.0 - d / CORRIDOR_PENALTY_DIST

    # ------------------------------------------------------------------
    def expand_dynamic(self, x: int, y: int,
                       margin: int = CORRIDOR_MARGIN) -> None:
        """Add a tracked position and widen the corridor to include it.
        Used in Stage-2 (automatic) mode."""
        self._dynamic_points.append((x, y))
        if len(self._dynamic_points) >= 3:
            self.from_annotations(self._dynamic_points, margin)

    # ------------------------------------------------------------------
    def save(self, path: Path) -> None:
        data: Dict[str, Any] = {"active": self.active}
        if self.rect:
            data["rect"] = list(self.rect)
        if self.polygon is not None:
            data["polygon"] = self.polygon.reshape(-1, 2).tolist()
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: Path) -> bool:
        if not path.exists():
            return False
        with open(path) as f:
            data = json.load(f)
        if "rect" in data:
            self.set_rect(*data["rect"])
        elif "polygon" in data:
            self.set_polygon(data["polygon"])
        self.active = data.get("active", True)
        return True

    def draw(self, img: np.ndarray,
             color: Tuple[int, ...] = COL_CYAN,
             thickness: int = 1) -> None:
        """Draw the corridor on *img*."""
        if not self.active:
            return
        if self.rect is not None:
            x1, y1, x2, y2 = self.rect
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            put_text(img, "CORRIDOR", (x1, y1 - 8), 0.30, color, 1)
        if self.polygon is not None:
            cv2.polylines(img, [self.polygon], True, color, thickness)
            pt = tuple(self.polygon[0, 0])
            put_text(img, "CORRIDOR", (pt[0], pt[1] - 8), 0.30, color, 1)


#  BALL DETECTOR

class BallDetector:
    """Detect the baseball in a game frame.

    Improved pipeline (v2)
    ----------------------
    1. **Background subtraction** — isolate foreground via running-average
       background model (falls back to frame-differencing during warm-up).
    2. **Pitcher-body suppression** — find large foreground blobs and
       penalise nearby small candidates.
    3. **HSV white threshold** — colour filter for the bright ball.
    4. **AND combine** — keep white blobs that overlap with foreground.
    5. **Contour filter** — area, circularity.
    6. **Static suppression** — grid-based HUD / field-marking filter.
    7. **Corridor scoring** — prefer candidates inside the corridor.
    8. **Composite score** — motion × circularity × area × isolation ×
       corridor.
    """

    def __init__(self) -> None:
        self._prev_grey: Optional[np.ndarray] = None
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self._kernel_small = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (3, 3))

        # Sub-components
        self.bg_model = BackgroundModel()
        self.suppressor = PitcherSuppressor()

        # Static element map
        self._static_map: Dict[Tuple[int, int], int] = {}

        # Cached masks for debug visualisation
        self.motion_mask: Optional[np.ndarray] = None
        self.bg_fg_mask: Optional[np.ndarray] = None
        self.white_mask: Optional[np.ndarray] = None
        self.combined_mask: Optional[np.ndarray] = None
        self.candidates: List[BallCandidate] = []
        self.best: Optional[BallCandidate] = None

        # Stored for rescue detection
        self._rescue_grey: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Call on scene change / new pitch."""
        self._prev_grey = None
        self.motion_mask = None
        self.bg_fg_mask = None
        self.white_mask = None
        self.combined_mask = None
        self.candidates.clear()
        self.best = None
        # Do NOT reset bg_model or _static_map — they persist

    def reset_full(self) -> None:
        """Full reset including background model."""
        self.reset()
        self.bg_model.reset()
        self._static_map.clear()

    # ------------------------------------------------------------------
    def _cell(self, x: int, y: int) -> Tuple[int, int]:
        return (x // STATIC_CELL_SIZE, y // STATIC_CELL_SIZE)

    def _is_static(self, x: int, y: int) -> bool:
        return self._static_map.get(self._cell(x, y), 0) >= STATIC_HIT_THRESHOLD

    def _record_static(self, x: int, y: int) -> None:
        cell = self._cell(x, y)
        self._static_map[cell] = self._static_map.get(cell, 0) + 1

    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray,
               search_roi: Optional[Tuple[int, int, int, int]] = None,
               corridor: Optional[TrajectoryCorridor] = None,
               track_active: bool = False
               ) -> Optional[BallCandidate]:
        """Detect ball candidates in *frame* (BGR).

        Parameters
        ----------
        search_roi : optional (x1, y1, x2, y2) to crop search area.
        corridor   : trajectory corridor for scoring.
        track_active : if True, the tracker has an active ball track
                       and the background model should NOT learn
                       (to prevent the moving ball from being absorbed).

        Returns the highest-scored BallCandidate, or None.
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
        self._rescue_grey = grey.copy()   # store for rescue_near()
        self._rescue_hsv = hsv.copy()     # store for rescue_near()

        # 1) Foreground mask
        # Background model produces best results; frame-diff is fallback.
        learning = not track_active  # freeze bg while tracking
        bg_fg = self.bg_model.update(grey, learning=learning)
        self.bg_fg_mask = bg_fg

        # Frame-to-frame differencing (always computed; used as secondary)
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

        # Choose primary foreground mask:
        #   - After bg warm-up: UNION of background-subtraction +
        #     frame-to-frame diff.  bg_fg catches static-scene departures;
        #     frame-diff catches rapidly moving objects even if the bg
        #     model has residual noise at those positions.
        #   - Before warm-up: frame-to-frame differencing only.
        if self.bg_model.ready:
            fg = cv2.bitwise_or(bg_fg, motion)
        else:
            fg = motion

        # 2) Pitcher-body suppression
        self.suppressor.analyze(fg, PITCHER_BODY_MIN_AREA)

        # 3) HSV white threshold
        white = cv2.inRange(hsv, BALL_HSV_LOWER, BALL_HSV_UPPER)
        white = cv2.morphologyEx(white, cv2.MORPH_OPEN,
                                 self._kernel_small, iterations=1)
        white = cv2.morphologyEx(white, cv2.MORPH_CLOSE,
                                 self._kernel_small, iterations=1)
        self.white_mask = white

        # 4) Combine: white AND foreground (primary candidates)
        combined = cv2.bitwise_and(white, fg)
        # NOTE: No dilation here! Dilation merges the tiny ball contour
        # with nearby pitcher-body pixels, creating oversized blobs that
        # get rejected by BALL_MAX_AREA.  The fg mask is already dilated.
        self.combined_mask = combined

        # 5) Extract candidates
        self.candidates.clear()
        self.best = None

        # Primary: combined (foreground + white)
        cands_motion = self._extract_candidates(
            combined, ox, oy, in_motion=True)
        # Secondary: white-only (fallback for first frames / slow ball)
        cands_white = self._extract_candidates(
            white, ox, oy, in_motion=False)

        # Merge, de-duplicate
        all_cands = cands_motion[:]
        for wc in cands_white:
            dup = False
            for mc in cands_motion:
                d = np.hypot(wc.center[0] - mc.center[0],
                             wc.center[1] - mc.center[1])
                if d < 20:
                    dup = True
                    break
            if not dup:
                all_cands.append(wc)

        if not all_cands:
            return None

        # 6) Static suppression
        filtered: List[BallCandidate] = []
        for c in all_cands:
            self._record_static(c.center[0], c.center[1])
            if self._is_static(c.center[0], c.center[1]):
                continue
            filtered.append(c)

        # No fallback: truly static blobs stay suppressed.
        # The ball moves fast enough (~12-20 px/frame) that it won't
        # saturate a static cell (size 12px) in 3 frames.
        if not filtered:
            self.candidates = []
            self.best = None
            return None

        # 7) Isolation + corridor scoring
        for c in filtered:
            c.isolation_score = self.suppressor.get_isolation_score(
                c.center[0] - ox, c.center[1] - oy)
            if corridor is not None:
                c.corridor_score = corridor.get_corridor_score(
                    c.center[0], c.center[1])
            else:
                c.corridor_score = 1.0

        # 8) Composite score
        for c in filtered:
            motion_w = 2.0 if c.in_motion_mask else 0.5
            area_norm = 1.0 - min(c.area / BALL_MAX_AREA, 1.0)
            circ_norm = min(c.circularity / 1.0, 1.0)
            # Isolation is a very soft factor (0.05 weight) because
            # the ball legitimately starts near the pitcher body.
            iso_w = max(c.isolation_score, 0.05)
            corr_w = max(c.corridor_score, 0.05)
            c.score = motion_w * (0.4 * circ_norm + 0.35 * area_norm
                                  + 0.05 * iso_w + 0.20 * corr_w)

        filtered.sort(key=lambda c: c.score, reverse=True)
        self.candidates = filtered
        self.best = filtered[0]
        return self.best

    # ------------------------------------------------------------------
    def _extract_candidates(self, mask: np.ndarray,
                            ox: int, oy: int,
                            in_motion: bool) -> List[BallCandidate]:
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
            cnt_full = cnt.copy()
            cnt_full[:, :, 0] += ox
            cnt_full[:, :, 1] += oy
            c.contour = cnt_full
            c.in_motion_mask = in_motion
            result.append(c)
        return result

    # ------------------------------------------------------------------
    def rescue_near(
        self, centre: Tuple[int, int], radius: int = 50
    ) -> Optional[BallCandidate]:
        """Emergency positional search near *centre* when normal
        candidate matching fails.

        Uses **local contrast** detection:  the ball is a small bright
        spot that stands out from its local neighbourhood (high positive
        deviation from the local mean), while the pitcher body is a
        uniformly-lit region with low per-pixel contrast.

        Steps:
          1. Compute per-pixel contrast = grey − blur(grey, 15×15).
          2. Threshold: contrast > 10 AND bg_fg > 0.
          3. Find the nearest lit pixel cluster to *centre*.

        Returns a BallCandidate, or None if nothing found.
        """
        if self.bg_fg_mask is None or self._rescue_grey is None:
            return None

        grey = self._rescue_grey
        bg_fg = self.bg_fg_mask
        cx, cy = centre
        h, w = grey.shape[:2]

        # Expand crop slightly beyond radius so the blur edge
        # doesn't create artefacts at the boundary
        pad = 15
        x1, y1 = max(0, cx - radius - pad), max(0, cy - radius - pad)
        x2, y2 = min(w, cx + radius + pad), min(h, cy + radius + pad)

        grey_roi = grey[y1:y2, x1:x2].astype(np.float32)
        fg_roi = bg_fg[y1:y2, x1:x2]
        if grey_roi.size == 0 or fg_roi.size == 0:
            return None

        # Local contrast: deviation from neighbourhood mean
        local_mean = cv2.blur(grey_roi, (15, 15))
        contrast = grey_roi - local_mean

        # Mask: high-contrast foreground pixels
        rescue_mask = np.zeros(grey_roi.shape, dtype=np.uint8)
        rescue_mask[(contrast > 10) & (fg_roi > 0)] = 255

        # Trim back to the actual search radius
        roi_h, roi_w = rescue_mask.shape[:2]
        local_cx = float(cx - x1)
        local_cy = float(cy - y1)

        # Get all lit pixels
        ys, xs = np.nonzero(rescue_mask)
        if len(xs) == 0:
            return None

        # Keep only pixels within the search radius
        dists = np.hypot(xs.astype(np.float64) - local_cx,
                         ys.astype(np.float64) - local_cy)
        within = dists <= radius
        if not np.any(within):
            return None

        xs = xs[within]
        ys = ys[within]
        dists = dists[within]

        nearest_idx = int(np.argmin(dists))

        # Compute centroid of lit pixels within ±4px of the nearest
        nx = int(xs[nearest_idx])
        ny = int(ys[nearest_idx])
        nearby = (np.abs(xs - nx) <= 4) & (np.abs(ys - ny) <= 4)
        cluster_xs = xs[nearby]
        cluster_ys = ys[nearby]

        bx = int(np.mean(cluster_xs)) + x1
        by = int(np.mean(cluster_ys)) + y1
        area = min(float(len(cluster_xs)), 15.0)

        c = BallCandidate()
        c.center = (bx, by)
        c.area = area
        c.circularity = 0.5
        c.in_motion_mask = True
        c.score = 0.5
        c.isolation_score = 0.0
        c.corridor_score = 1.0
        return c


#  TRACK STATE & BALL TRACK

class TrackState(Enum):
    IDLE = auto()
    TENTATIVE = auto()
    CONFIRMED = auto()
    LOST = auto()


class BallTrack:
    """Ball state tracked across frames with a Kalman filter.

    The Kalman filter uses a constant-velocity model:
      state   = [x, y, vx, vy]
      measure = [x, y]
    Process noise accommodates curve-ball acceleration.
    """

    def __init__(self, pos: Tuple[int, int], frame_idx: int,
                 area: float = 0.0) -> None:
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
            np.eye(4, dtype=np.float32) * KF_PROCESS_NOISE)
        self._kf.measurementNoiseCov = (
            np.eye(2, dtype=np.float32) * KF_MEASUREMENT_NOISE)
        self._kf.errorCovPost = np.eye(4, dtype=np.float32)
        self._kf.statePost = np.array(
            [[pos[0]], [pos[1]], [0], [0]], dtype=np.float32)

        # History
        self.positions: deque = deque(maxlen=TRAJECTORY_HISTORY)
        self.positions.append((pos[0], pos[1], frame_idx))
        self.frames_since_seen: int = 0
        self.total_frames: int = 1
        self.active: bool = True
        self.confirmed: bool = False

        # Track lifecycle helpers
        self._start_pos: Tuple[int, int] = pos
        self._start_frame: int = frame_idx
        self._frames_in_pitcher_zone: int = (
            1 if _in_pitcher_zone(pos[0], pos[1]) else 0)
        self._left_pitcher_zone: bool = (
            not _in_pitcher_zone(pos[0], pos[1]))
        self._vy_sign_changes: int = 0
        self._prev_vy_sign: int = 0   # -1, 0, +1
        self._positive_vy_count: int = 0

        # Area tracking
        self.avg_area: float = area if area > 0 else 20.0
        self._area_alpha: float = 0.3

    # Properties for backward compatibility

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
        """Run Kalman prediction step.  Returns predicted (x, y)."""
        pred = self._kf.predict()
        return (int(pred[0, 0]), int(pred[1, 0]))

    def kf_correct(self, x: int, y: int) -> None:
        meas = np.array([[np.float32(x)], [np.float32(y)]])
        self._kf.correct(meas)

    def kf_accept_prediction(self) -> None:
        """Accept the un-corrected prediction as the current state."""
        self._kf.statePost = self._kf.statePre.copy()
        self._kf.errorCovPost = self._kf.errorCovPre.copy()

    # High-level interface

    def update(self, pos: Tuple[int, int], frame_idx: int,
               area: float = 0.0) -> None:
        """Register a new detection and update the Kalman filter."""
        # Correct KF with measurement
        self.kf_correct(pos[0], pos[1])

        # Track vy sign changes (oscillation detection)
        vx, vy = self.velocity
        sign = 1 if vy > 1.0 else (-1 if vy < -1.0 else 0)
        if sign != 0 and self._prev_vy_sign != 0 and sign != self._prev_vy_sign:
            self._vy_sign_changes += 1
        if sign != 0:
            self._prev_vy_sign = sign
        if vy > 0:
            self._positive_vy_count += 1

        self.positions.append((pos[0], pos[1], frame_idx))
        self.frames_since_seen = 0
        self.total_frames += 1

        # Area EMA
        if area > 0:
            self.avg_area = (self._area_alpha * area
                             + (1 - self._area_alpha) * self.avg_area)

        # Pitcher zone accounting
        in_pz = _in_pitcher_zone(pos[0], pos[1])
        if in_pz:
            self._frames_in_pitcher_zone += 1
        else:
            self._left_pitcher_zone = True

        # Confirmation check
        self._check_confirmation()

    def _check_confirmation(self) -> None:
        if self.confirmed:
            return

        pts = list(self.positions)
        if len(pts) < TRACK_MIN_CONFIRMATIONS:
            return

        # 1) Enough positive-vy frames
        if self._positive_vy_count < TRACK_MIN_CONFIRMATIONS:
            return

        # 2) Must have either left the pitcher zone OR traveled enough
        cur = self.last_pos
        dist = np.hypot(cur[0] - self._start_pos[0],
                        cur[1] - self._start_pos[1])
        if not self._left_pitcher_zone and dist < MIN_DEPARTURE_DIST:
            return

        # 3) Average velocity check (overall trajectory must be forward)
        dy = pts[-1][1] - pts[0][1]
        df = pts[-1][2] - pts[0][2]
        if df <= 0:
            return
        avg_vy = dy / df
        if avg_vy < MIN_PITCH_VY:
            return

        # 4) Trajectory consistency: most consecutive pairs of detections
        #    should show increasing Y (ball moving toward batter).
        #    This rejects oscillating noise from pitcher arm or field.
        increasing_count = 0
        for i in range(1, len(pts)):
            if pts[i][1] > pts[i-1][1]:  # y increased
                increasing_count += 1
        fraction_increasing = increasing_count / max(len(pts) - 1, 1)
        if fraction_increasing < 0.4:  # at least 40% of steps must be forward
            return

        # 5) Not too many oscillations
        if self._vy_sign_changes > MAX_VY_SIGN_CHANGES:
            return

        self.confirmed = True

    def predict(self, n_frames: int = 1) -> Tuple[int, int]:
        """Simple linear extrapolation for n_frames."""
        vx, vy = self.velocity
        p = self.last_pos
        return (int(p[0] + vx * n_frames), int(p[1] + vy * n_frames))

    def mark_missed(self) -> None:
        """Called when no detection matched this frame."""
        self.frames_since_seen += 1
        # Accept the Kalman prediction
        self.kf_accept_prediction()
        max_gap = TRACK_LOST_FRAMES if self.confirmed else TRACK_TENTATIVE_LOST
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


#  BALL TRACKER

class BallTracker:
    """Track the pitched ball across frames.

    Lifecycle
    ---------
    ``IDLE`` → ``TENTATIVE`` → ``CONFIRMED`` → ``LOST``

    * **IDLE**: no track active — searching for a start in pitcher zone.
    * **TENTATIVE**: a track started but not yet confirmed.  Strict
      validation runs every frame:
        - kill if pitcher zone exceeded (``MAX_PITCHER_ZONE_FRAMES``),
        - kill if too many Y-velocity oscillations,
        - kill if stalled without confirming after N frames.
    * **CONFIRMED**: the ball is in flight.  Matching uses Kalman-
      predicted position with adaptive search radius.
    * **LOST**: track terminated (timeout or ball out of play).

    Pitcher-animation rejection
    ---------------------------
    1. Only start tracks from pitcher zone + in-motion + small area.
    2. Require departure from pitcher zone within ``MAX_PITCHER_ZONE_FRAMES``.
    3. Require sustained positive-Y velocity >= ``MIN_PITCH_VY``.
    4. Require minimum travel distance >= ``MIN_DEPARTURE_DIST``.
    5. Reject oscillating vy (> ``MAX_VY_SIGN_CHANGES`` sign flips).
    6. Penalise candidates near large foreground blobs (isolation score).
    """

    def __init__(self) -> None:
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

    # ------------------------------------------------------------------
    def _is_size_consistent(self, candidate: BallCandidate) -> bool:
        if self.track is None or self.track.avg_area <= 0:
            return True
        avg = self.track.avg_area
        ratio = max(candidate.area, avg) / max(min(candidate.area, avg), 1)
        return ratio <= TRACK_SIZE_RATIO

    # ------------------------------------------------------------------
    def _find_nearest_candidate(
        self, candidates: List[BallCandidate],
        pred: Tuple[int, int]
    ) -> Optional[BallCandidate]:
        """Match the closest candidate to the Kalman-predicted position."""
        if self.track is None:
            return None

        # Adaptive search radius — tighter for tentative tracks
        # to avoid matching distant static blobs when the rescue
        # channel would find the real ball nearby.
        speed = np.hypot(self.track.velocity[0], self.track.velocity[1])
        min_d = TRACK_MAX_DIST_MIN if self.track.confirmed else 30
        base_d = max(min_d, TRACK_MAX_DIST_SPEED_K * speed)
        max_d = base_d * (1.0 + TRACK_GAP_EXPAND *
                          self.track.frames_since_seen)

        best_dist = float("inf")
        chosen: Optional[BallCandidate] = None

        for c in candidates:
            if c.area > BALL_FLIGHT_MAX_AREA:
                continue
            if not self._is_size_consistent(c):
                continue

            d_pred = np.hypot(c.center[0] - pred[0],
                              c.center[1] - pred[1])

            # For tentative tracks, also consider last known position
            if self.track.confirmed:
                d = d_pred
            else:
                last = self.track.last_pos
                d_last = np.hypot(c.center[0] - last[0],
                                  c.center[1] - last[1])
                d = min(d_pred, d_last)

            if d > max_d:
                continue

            # Direction bonus: prefer candidates in the expected
            # direction of travel (forward = positive Y for pitches).
            # This prevents latching onto static blobs when the ball
            # is temporarily lost.
            dir_penalty = 1.0
            if self.track.confirmed and len(self.track.positions) >= 2:
                vx, vy = self.track.velocity
                if vy > 2.0:  # ball moving downward (toward batter)
                    # Candidate should be roughly in the forward direction
                    dy_cand = c.center[1] - self.track.last_pos[1]
                    if dy_cand < -5:  # candidate is BEHIND the expected direction
                        dir_penalty = 2.0  # double the effective distance

            effective_d = d * dir_penalty
            if effective_d < best_dist:
                best_dist = effective_d
                chosen = c

        return chosen

    # ------------------------------------------------------------------
    def _kill_tentative_if_bad(self) -> None:
        """Apply validation rules to kill bad tentative tracks."""
        t = self.track
        if t is None:
            return

        # --- Rules for TENTATIVE tracks ---
        if not t.confirmed:
            # Rule 1: stayed in pitcher zone too long
            if t._frames_in_pitcher_zone > MAX_PITCHER_ZONE_FRAMES:
                self.track = None
                self.state = TrackState.IDLE
                self.selected = None
                return

            # Rule 2: too many vy oscillations (pitcher arm swinging)
            if t._vy_sign_changes > MAX_VY_SIGN_CHANGES:
                self.track = None
                self.state = TrackState.IDLE
                self.selected = None
                return

            # Rule 3: stalled too long without confirming
            max_tentative = TRACK_MIN_CONFIRMATIONS + 10
            if t.total_frames > max_tentative:
                self.track = None
                self.state = TrackState.IDLE
                self.selected = None
                return

            # Rule 4: moving away from batter (negative avg vy)
            # Only check after enough frames to establish direction;
            # the ball may start with near-zero vy due to perspective.
            if t.total_frames >= 5 and t.avg_velocity_y() <= -2.0:
                self.track = None
                self.state = TrackState.IDLE
                self.selected = None
                return

            # Rule 5: track is "stalled" — hovering near the same spot
            # for many detections.  Real ball moves 10-20 px/frame; a
            # track stuck on a static field marking barely moves.
            if len(t.positions) >= 5:
                pts = list(t.positions)
                recent = pts[-5:]
                dx = abs(recent[-1][0] - recent[0][0])
                dy = abs(recent[-1][1] - recent[0][1])
                if dx < 15 and dy < 15:
                    # 5 detections with < 15px total movement → static
                    self.track = None
                    self.state = TrackState.IDLE
                    self.selected = None
                    return
        else:
            # --- Rules for CONFIRMED tracks ---
            # Kill confirmed track if vy reverses significantly
            # (means we were tracking the wrong thing)
            if t.total_frames >= 6 and t.avg_velocity_y() <= -5.0:
                self.track = None
                self.state = TrackState.LOST
                self.selected = None
                return

            # Kill confirmed track if stalled (hovering at same pos)
            if len(t.positions) >= 6:
                pts = list(t.positions)
                recent = pts[-6:]
                dx = abs(recent[-1][0] - recent[0][0])
                dy = abs(recent[-1][1] - recent[0][1])
                if dx < 20 and dy < 20:
                    self.track = None
                    self.state = TrackState.LOST
                    self.selected = None
                    return

    # ------------------------------------------------------------------
    def update(self, candidates: List[BallCandidate],
               best_candidate: Optional[BallCandidate] = None,
               detector: Optional['BallDetector'] = None,
               ) -> Optional[BallTrack]:
        """Run one tracker step with ALL detector candidates.

        Two-phase strategy
        ------------------
        Phase 1 : Ball detected in pitcher zone → tentative track.
        Blind   : Ball dims near pitcher (V<200) → track dies (3 frames).
        Phase 2 : Ball brightens below pitcher (V≥200) → re-acquisition
                  in ``REACQ_ZONE``, gated by ``REACQ_WINDOW``.
        """
        self._frame_idx += 1
        self.selected = None

        had_track = (self.track is not None and self.track.active)

        # Active track: predict then match
        if self.track is not None and self.track.active:
            pred = self.track.kf_predict()
            chosen = self._find_nearest_candidate(candidates, pred)

            if chosen is not None:
                self.track.update(chosen.center, self._frame_idx,
                                  area=chosen.area)
                self.selected = chosen

                if self.track.confirmed:
                    self.state = TrackState.CONFIRMED
                else:
                    self.state = TrackState.TENTATIVE

                # Validate tentative track
                self._kill_tentative_if_bad()
            else:
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

            # Priority 1: pitcher zone (normal pitch start)
            for c in candidates:
                if not c.in_motion_mask:
                    continue
                if c.area > BALL_FLIGHT_MAX_AREA:
                    continue
                if not _in_pitcher_zone(c.center[0], c.center[1]):
                    continue
                if c.score > best_start_score:
                    best_start_score = c.score
                    best_start = c

            # Priority 2: re-acquisition zone (ball re-emerged below
            # pitcher).  Only within REACQ_WINDOW of the last track
            # dying, to avoid spurious starts when no pitch is active.
            if best_start is None and (
                self._frame_idx - self._last_track_end_frame
            ) <= REACQ_WINDOW:
                best_rq: Optional[BallCandidate] = None
                best_rq_score = -1.0
                for c in candidates:
                    if not c.in_motion_mask:
                        continue
                    if c.area > BALL_FLIGHT_MAX_AREA:
                        continue
                    if not _in_reacq_zone(c.center[0], c.center[1]):
                        continue
                    if c.score > best_rq_score:
                        best_rq_score = c.score
                        best_rq = c
                best_start = best_rq

            if best_start is not None:
                self.track = BallTrack(
                    best_start.center, self._frame_idx,
                    area=best_start.area)
                self.state = TrackState.TENTATIVE
                self.selected = best_start

        return self.track


#  TRAJECTORY PREDICTOR

class TrajectoryPredictor:
    """Follow the ball trajectory and extrapolate the crossing point
    at a target Y-coordinate (strike zone line).

    Uses polynomial regression on the tracked positions.  This is
    short-horizon extrapolation for *following* the ball, NOT for
    predicting pitch outcome before the ball arrives.
    """

    def __init__(self, target_y: Optional[int] = None) -> None:
        self.target_y = target_y
        self.predicted_x: Optional[int] = None
        self.predicted_frame: Optional[int] = None
        self.fit_type: str = ""
        self.confidence: float = 0.0

    def set_target_y(self, y: int) -> None:
        self.target_y = y

    def predict(self, track: BallTrack) -> Optional[Tuple[int, int]]:
        self.predicted_x = None
        self.predicted_frame = None
        self.confidence = 0.0

        if self.target_y is None:
            return None

        positions = list(track.positions)
        n = len(positions)
        if n < 3:
            return None

        frames = np.array([p[2] for p in positions], dtype=np.float64)
        xs = np.array([p[0] for p in positions], dtype=np.float64)
        ys = np.array([p[1] for p in positions], dtype=np.float64)
        f0 = frames[0]
        frames_n = frames - f0

        # Check if ball is moving toward target
        if n >= 2:
            dy = ys[-1] - ys[0]
            target_dir = self.target_y - ys[-1]
            if dy * target_dir < 0:
                return None

        try:
            if n >= 5:
                cx = np.polyfit(frames_n, xs, 2)
                cy = np.polyfit(frames_n, ys, 2)
                self.fit_type = "quadratic"
            else:
                cx = np.polyfit(frames_n, xs, 1)
                cy = np.polyfit(frames_n, ys, 1)
                self.fit_type = "linear"
        except (np.linalg.LinAlgError, ValueError):
            return None

        poly_y = np.poly1d(cy)
        poly_y_shifted = poly_y - self.target_y
        roots = np.roots(poly_y_shifted)

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

        frames_to_target = best_f - current_f
        self.confidence = (min(1.0, n / 15.0)
                           * max(0.1, 1.0 - frames_to_target / 60.0))
        self.predicted_x = pred_x
        self.predicted_frame = pred_frame
        return (pred_x, self.target_y)

    def get_trajectory_points(self, track: BallTrack,
                              n_future: int = 30
                              ) -> List[Tuple[int, int]]:
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
            pts.append((int(poly_x(f)), int(poly_y(f))))
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
        self.frames.append(frame.copy())
        if ball is not None:
            self.detections.append({
                "center": list(ball.center),
                "area": ball.area,
                "circularity": ball.circularity,
                "bbox": list(ball.bbox),
                "in_motion": ball.in_motion_mask,
                "isolation": ball.isolation_score,
                "corridor": ball.corridor_score,
                "score": ball.score,
            })
        else:
            self.detections.append(None)
        if len(self.frames) >= RECORD_MAX_FRAMES:
            self.stop()
            return False
        return True

    def save(self) -> Optional[str]:
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
    """Draw overlays for ball detection, tracking, and corridor."""

    @staticmethod
    def overlay(frame: np.ndarray,
                detector: BallDetector,
                tracker: BallTracker,
                predictor: TrajectoryPredictor,
                corridor: Optional[TrajectoryCorridor],
                search_roi: Optional[Tuple[int, int, int, int]],
                recording: bool,
                fps: float,
                frame_num: int = 0) -> np.ndarray:
        vis = frame.copy()
        h, w = vis.shape[:2]

        # Frame number
        put_text(vis, f"Frame {frame_num}", (10, 20), 0.45, COL_WHITE, 1)

        # Pitcher zone
        pzx1, pzy1, pzx2, pzy2 = PITCHER_ZONE
        cv2.rectangle(vis, (pzx1, pzy1), (pzx2, pzy2), (128, 128, 0), 1)
        put_text(vis, "PITCHER ZONE", (pzx1, pzy1 - 6), 0.30,
                 (128, 128, 0), 1)

        # Corridor
        if corridor is not None:
            corridor.draw(vis, COL_CYAN, 1)

        # Search ROI
        if search_roi is not None:
            x1, y1, x2, y2 = search_roi
            cv2.rectangle(vis, (x1, y1), (x2, y2), (200, 200, 0), 1)
            put_text(vis, "SEARCH ROI", (x1, y1 - 8), 0.35,
                     (200, 200, 0), 1)

        # Suppression zones (pitcher body)
        for sx, sy, sr in detector.suppressor.suppression_zones:
            cv2.circle(vis, (sx, sy), sr, (0, 0, 128), 1)

        # All candidates
        for c in detector.candidates:
            col = COL_YELLOW if c.in_motion_mask else (128, 128, 128)
            cv2.circle(vis, c.center, 3, col, -1)

        # Tracker state label
        state_label = tracker.state.name
        state_col = {
            TrackState.IDLE: COL_WHITE,
            TrackState.TENTATIVE: COL_ORANGE,
            TrackState.CONFIRMED: COL_GREEN,
            TrackState.LOST: COL_RED,
        }.get(tracker.state, COL_WHITE)
        put_text(vis, f"STATE: {state_label}", (10, 40), 0.45,
                 state_col, 1)
        put_text(vis, f"BG: {'ready' if detector.bg_model.ready else 'warming'}",
                 (10, 58), 0.35, COL_CYAN, 1)

        # Tracked candidate
        sel = tracker.selected
        if sel is not None:
            cv2.circle(vis, sel.center, 10, COL_GREEN, 2)
            cv2.circle(vis, sel.center, 2, COL_RED, -1)
            bx, by, bw, bh = sel.bbox
            cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), COL_GREEN, 1)
            put_text(vis,
                     (f"TRACKED  a={sel.area:.0f} c={sel.circularity:.2f}"
                      f" iso={sel.isolation_score:.1f}"),
                     (bx, by - 8), 0.38, COL_GREEN, 1)
        elif detector.best is not None:
            b = detector.best
            cv2.circle(vis, b.center, 8, COL_ORANGE, 2)
            cv2.circle(vis, b.center, 2, COL_ORANGE, -1)
            bx, by, bw, bh = b.bbox
            cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), COL_ORANGE, 1)
            put_text(vis, f"CAND  a={b.area:.0f} c={b.circularity:.2f}",
                     (bx, by - 8), 0.38, COL_ORANGE, 1)

        # Trajectory trail
        track = tracker.track
        if track is not None and track.active and len(track.positions) >= 2:
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

            # Predicted trajectory
            traj_pts = predictor.get_trajectory_points(track, n_future=20)
            if len(traj_pts) >= 2:
                for i in range(len(traj_pts) - 1):
                    if i > n and i % 2 == 0:
                        cv2.line(vis, traj_pts[i], traj_pts[i + 1],
                                 COL_YELLOW, 1)

            # Prediction crosshair
            if (predictor.predicted_x is not None
                    and predictor.target_y is not None):
                px = predictor.predicted_x
                py = predictor.target_y
                cv2.drawMarker(vis, (px, py), COL_RED,
                               cv2.MARKER_CROSS, 20, 2)
                cv2.circle(vis, (px, py), 12, COL_RED, 2)
                put_text(vis,
                         f"CROSS ({px},{py})  {predictor.fit_type}  "
                         f"conf={predictor.confidence:.0%}",
                         (px + 16, py - 6), 0.38, COL_RED, 1)

            # Target Y line
            if predictor.target_y is not None:
                cv2.line(vis, (0, predictor.target_y),
                         (w, predictor.target_y), COL_RED, 1)
                put_text(vis, "STRIKE Y",
                         (w - 100, predictor.target_y - 6),
                         0.35, COL_RED, 1)

            # Track info bar
            status = "CONFIRMED" if track.confirmed else "tentative"
            put_text(vis,
                     f"[{status}]  frames={track.total_frames}  "
                     f"vel=({track.velocity[0]:.1f},{track.velocity[1]:.1f})  "
                     f"missed={track.frames_since_seen}  "
                     f"pz={track._frames_in_pitcher_zone}",
                     (10, h - 55), 0.40, COL_CYAN, 1)

        # Recording indicator
        if recording:
            cv2.circle(vis, (w - 30, 30), 12, COL_RED, -1)
            put_text(vis, "REC", (w - 65, 36), 0.50, COL_RED, 2)

        # FPS + controls hint
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

        # Background foreground mask
        panels.append(_tile(detector.bg_fg_mask,
                            "BG foreground", COL_CYAN))
        # Frame-diff motion mask
        panels.append(_tile(detector.motion_mask,
                            "Frame diff", (150, 150, 150)))
        # White mask
        panels.append(_tile(detector.white_mask,
                            "White (HSV)", COL_YELLOW))
        # Combined mask
        panels.append(_tile(detector.combined_mask,
                            "Combined (fg+white)", COL_GREEN))

        # Candidates on ROI
        if search_roi is not None:
            roi_vis = crop(frame, search_roi).copy()
        else:
            roi_vis = frame.copy()
        if roi_vis.size > 0:
            for c in detector.candidates:
                col = COL_GREEN if c.in_motion_mask else COL_YELLOW
                cx_off = c.center[0] - (search_roi[0] if search_roi else 0)
                cy_off = c.center[1] - (search_roi[1] if search_roi else 0)
                cv2.circle(roi_vis, (cx_off, cy_off), 5, col, 2)
                cv2.putText(roi_vis, f"{c.score:.1f}",
                            (cx_off + 6, cy_off - 4), FONT, 0.30, col, 1)
            panels.append(_tile(roi_vis,
                                f"Candidates ({len(detector.candidates)})",
                                COL_MAGENTA))
        else:
            panels.append(_tile(None, "No ROI", COL_RED))

        # Pitcher suppression overlay
        supp_vis = np.zeros_like(frame) if frame is not None else np.zeros((PH, PW, 3), np.uint8)
        if detector.bg_fg_mask is not None:
            fg3 = cv2.cvtColor(detector.bg_fg_mask, cv2.COLOR_GRAY2BGR)
            if fg3.shape[:2] == supp_vis.shape[:2]:
                supp_vis = fg3.copy()
        for sx, sy, sr in detector.suppressor.suppression_zones:
            cv2.circle(supp_vis, (sx, sy), sr, COL_RED, 2)
        for b in detector.suppressor.large_blobs:
            cx, cy, bw, bh, _ = b
            cv2.rectangle(supp_vis, (cx - bw // 2, cy - bh // 2),
                          (cx + bw // 2, cy + bh // 2), COL_ORANGE, 1)
        panels.append(_tile(supp_vis,
                            f"Suppression ({len(detector.suppressor.large_blobs)} blobs)",
                            COL_RED))

        # Arrange 3x2
        while len(panels) < 6:
            panels.append(np.zeros((PH, PW, 3), np.uint8))
        row1 = np.hstack(panels[:3])
        row2 = np.hstack(panels[3:6])
        return np.vstack([row1, row2])


#  ROI SELECTOR  (click-to-set search area)

class ROISelector:
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
        if event != cv2.EVENT_LBUTTONDOWN or not self.active:
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
        put_text(vis, "Click TOP-LEFT then BOTTOM-RIGHT",
                 (10, 30), 0.55, COL_YELLOW, 2)
        if self._pt1 is not None:
            cv2.drawMarker(vis, self._pt1, COL_CYAN,
                           cv2.MARKER_CROSS, 20, 2)
        if self.rect is not None:
            x1, y1, x2, y2 = self.rect
            cv2.rectangle(vis, (x1, y1), (x2, y2), COL_CYAN, 2)
        return vis


#  STRIKE-Y SELECTOR

class StrikeYSelector:
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
        if event != cv2.EVENT_LBUTTONDOWN or not self.active:
            return
        self.target_y = int(y / self.scale)
        self.active = False
        self._frozen = None
        print(f"[STRIKE-Y] Target Y = {self.target_y}")

    def draw_frozen(self) -> Optional[np.ndarray]:
        if self._frozen is None:
            return None
        vis = self._frozen.copy()
        overlay = np.zeros_like(vis)
        vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)
        put_text(vis, "Click the Y-level where the ball reaches the batter",
                 (10, 30), 0.55, COL_YELLOW, 2)
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
        description="MSB Pitch / Ball Detection & Trajectory Analyzer v2")
    ap.add_argument("--source", choices=["live", "folder"], default="live")
    ap.add_argument("--input", default=None,
                    help="Folder of pitch frames (for folder source)")
    ap.add_argument("--corridor", default=None,
                    help="Path to corridor JSON (optional)")
    args = ap.parse_args()

    # Components
    detector  = BallDetector()
    tracker   = BallTracker()
    predictor = TrajectoryPredictor()
    recorder  = PitchRecorder()
    vis       = PitchVisualiser()
    roi_sel   = ROISelector(display_scale=DISPLAY_SCALE)
    sy_sel    = StrikeYSelector(display_scale=DISPLAY_SCALE)

    # Trajectory corridor
    corridor = TrajectoryCorridor()
    if args.corridor and Path(args.corridor).exists():
        corridor.load(Path(args.corridor))
        print(f"[INFO] Loaded corridor from {args.corridor}")
    elif CORRIDOR_DEFAULT is not None:
        corridor.set_rect(*CORRIDOR_DEFAULT)
        print(f"[INFO] Using default corridor: {CORRIDOR_DEFAULT}")

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

    print("\n CONTROLS")
    print("  SPACE = Start / stop recording")
    print("  D     = Toggle debug panel")
    print("  C     = Click to define ball search ROI")
    print("  Y     = Click to set strike-zone Y level")
    print("  S     = Save recorded pitch")
    print("  R     = Reset tracker")
    print("  Q/ESC = Quit\n")

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

            if not roi_sel.active and roi_sel.rect is not None:
                search_roi = roi_sel.rect
                roi_sel.rect = None

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

            # Detection
            track_active = (tracker.track is not None
                            and tracker.track.active
                            and tracker.track.confirmed)
            best = detector.detect(frame, search_roi, corridor,
                                   track_active=track_active)

            # Tracking
            track = tracker.update(detector.candidates, best)

            tracked_ball = tracker.selected if tracker.selected else best

            # Following / extrapolation
            if track is not None and track.active:
                predictor.predict(track)
                # Dynamic corridor expansion (stage 2 prep)
                if track.confirmed:
                    corridor.expand_dynamic(track.last_pos[0],
                                            track.last_pos[1])

            # Recording
            if recorder.recording:
                recorder.add_frame(frame, tracked_ball)

            # Draw
            disp = vis.overlay(frame, detector, tracker, predictor,
                               corridor, search_roi,
                               recorder.recording, fps_ema, frame_num)
            if DISPLAY_SCALE != 1.0:
                dw = int(disp.shape[1] * DISPLAY_SCALE)
                dh = int(disp.shape[0] * DISPLAY_SCALE)
                disp = cv2.resize(disp, (dw, dh))
            cv2.imshow(WINDOW_MAIN, disp)

            if show_debug:
                dbg = vis.debug_panel(detector, frame, search_roi)
                cv2.imshow(WINDOW_DEBUG, dbg)

            # Keys
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
