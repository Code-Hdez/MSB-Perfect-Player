"""
Trajectory predictor — polynomial extrapolation for the ball crossing
at a target Y-coordinate (strike-zone line).

This is short-horizon *following*, not outcome prediction.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from msb.config import Config
from msb.tracker import BallTrack


class TrajectoryPredictor:
    """Fit a polynomial to tracked positions and extrapolate the
    crossing point at ``target_y``."""

    def __init__(self, target_y: Optional[int] = None,
                 cfg: Optional[Config] = None) -> None:
        self.cfg = cfg or Config()
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
