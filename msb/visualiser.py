"""
Visualisation overlays and interactive selectors for the pitch analyser.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import cv2
import numpy as np

from msb.config import Config
from msb.detector import BallCandidate, BallDetector
from msb.tracker import BallTrack, BallTracker, TrackState
from msb.predictor import TrajectoryPredictor
from msb.corridor import TrajectoryCorridor
from msb.utils import (
    crop, put_text, in_rect,
    COL_GREEN, COL_RED, COL_YELLOW, COL_CYAN, COL_WHITE,
    COL_BLACK, COL_MAGENTA, COL_ORANGE, FONT,
)


#  PITCH VISUALISER

class PitchVisualiser:
    """Draw overlays for ball detection, tracking, and corridor."""

    @staticmethod
    def overlay(
        frame: np.ndarray,
        detector: BallDetector,
        tracker: BallTracker,
        predictor: TrajectoryPredictor,
        corridor: Optional[TrajectoryCorridor],
        search_roi: Optional[Tuple[int, int, int, int]],
        recording: bool,
        fps: float,
        frame_num: int = 0,
        cfg: Optional[Config] = None,
    ) -> np.ndarray:
        cfg = cfg or Config()
        vis = frame.copy()
        h, w = vis.shape[:2]

        put_text(vis, f"Frame {frame_num}", (10, 20), 0.45, COL_WHITE, 1)

        # Pitcher zone
        pz = cfg.pitcher_zone
        cv2.rectangle(vis, (pz[0], pz[1]), (pz[2], pz[3]),
                      (128, 128, 0), 1)
        put_text(vis, "PITCHER ZONE", (pz[0], pz[1] - 6), 0.30,
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

        # Suppression zones
        for sx, sy, sr in detector.suppressor.suppression_zones:
            cv2.circle(vis, (sx, sy), sr, (0, 0, 128), 1)

        # All candidates
        for c in detector.candidates:
            col = COL_YELLOW if c.in_motion_mask else (128, 128, 128)
            cv2.circle(vis, c.center, 3, col, -1)

        # Track state label
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

            last = track.last_pos
            vx, vy = track.velocity
            if abs(vx) + abs(vy) > 1:
                tip = (int(last[0] + vx * 3), int(last[1] + vy * 3))
                cv2.arrowedLine(vis, last, tip, COL_MAGENTA, 2,
                                tipLength=0.3)

            traj_pts = predictor.get_trajectory_points(track, n_future=20)
            if len(traj_pts) >= 2:
                for i in range(len(traj_pts) - 1):
                    if i > n and i % 2 == 0:
                        cv2.line(vis, traj_pts[i], traj_pts[i + 1],
                                 COL_YELLOW, 1)

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

            if predictor.target_y is not None:
                cv2.line(vis, (0, predictor.target_y),
                         (w, predictor.target_y), COL_RED, 1)
                put_text(vis, "STRIKE Y",
                         (w - 100, predictor.target_y - 6),
                         0.35, COL_RED, 1)

            status = "CONFIRMED" if track.confirmed else "tentative"
            put_text(vis,
                     f"[{status}]  frames={track.total_frames}  "
                     f"vel=({track.velocity[0]:.1f},{track.velocity[1]:.1f})  "
                     f"missed={track.frames_since_seen}  "
                     f"pz={track._frames_in_pitcher_zone}",
                     (10, h - 55), 0.40, COL_CYAN, 1)

        if recording:
            cv2.circle(vis, (w - 30, 30), 12, COL_RED, -1)
            put_text(vis, "REC", (w - 65, 36), 0.50, COL_RED, 2)

        put_text(vis, f"FPS {fps:.0f}", (w - 110, h - 8), 0.45,
                 COL_GREEN, 1)
        hints = ("[SPACE] Record  [D] Debug  [C] Set ROI  "
                 "[Y] Set strike-Y  [S] Save  [Q] Quit")
        put_text(vis, hints, (10, h - 8), 0.35, COL_WHITE, 1)

        return vis

    @staticmethod
    def debug_panel(
        detector: BallDetector,
        frame: np.ndarray,
        search_roi: Optional[Tuple[int, int, int, int]],
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

        panels.append(_tile(detector.bg_fg_mask,
                            "BG foreground", COL_CYAN))
        panels.append(_tile(detector.motion_mask,
                            "Frame diff", (150, 150, 150)))
        panels.append(_tile(detector.white_mask,
                            "White (HSV)", COL_YELLOW))
        panels.append(_tile(detector.combined_mask,
                            "Combined (fg+white)", COL_GREEN))

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

        supp_vis = (np.zeros_like(frame) if frame is not None
                    else np.zeros((PH, PW, 3), np.uint8))
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
        put_text(vis, "Click the Y-level where ball reaches batter",
                 (10, 30), 0.55, COL_YELLOW, 2)
        return vis
