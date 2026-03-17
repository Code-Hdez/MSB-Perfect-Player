"""
Trajectory predictor — polynomial extrapolation for the ball crossing
at a target Y-coordinate (strike-zone line).

This is short-horizon *following*, not outcome prediction.

v2: Uses **wall-clock timestamps** (``time.perf_counter()``) as the
independent variable instead of frame indices.  This makes prediction
FPS-invariant — critical for a variable-rate pipeline.

The public interface is backward-compatible: ``predict()`` still returns
``(x, y)`` and populates ``predicted_x``.  New attributes expose time-
based outputs (``time_to_crossing_sec``).
"""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

import numpy as np

from msb.config import Config
from msb.tracker import BallTrack


class TrajectoryPredictor:
    """Fit a polynomial to tracked positions and extrapolate the
    crossing point at ``target_y``.

    The independent variable is wall-clock time (seconds), not frame
    index.  Track positions are expected to carry a timestamp as the
    4th element of each entry ``(x, y, frame_idx, timestamp)``.  If
    timestamps are absent, frame indices are used as a fallback
    (backward compatibility).
    """

    def __init__(self, target_y: Optional[int] = None,
                 cfg: Optional[Config] = None) -> None:
        self.cfg = cfg or Config()
        self.target_y = target_y

        # Legacy outputs (still populated for overlay / HUD compat)
        self.predicted_x: Optional[int] = None
        self.predicted_frame: Optional[int] = None
        self.fit_type: str = ""
        self.confidence: float = 0.0

        # New wall-clock outputs
        self.time_to_crossing_sec: Optional[float] = None
        self.crossing_timestamp: Optional[float] = None
        self.predicted_vx: float = 0.0   # px/sec at crossing

        # Prediction smoothing (temporal EMA)
        self._smooth_alpha: float = 0.35   # blend factor (0 = all history)
        self._prev_pred_x: Optional[float] = None
        self._prev_ttc: Optional[float] = None

    def set_target_y(self, y: int) -> None:
        self.target_y = y

    def reset(self) -> None:
        """Clear all prediction state for a clean next pitch cycle."""
        self.predicted_x = None
        self.predicted_frame = None
        self.fit_type = ""
        self.confidence = 0.0
        self.time_to_crossing_sec = None
        self.crossing_timestamp = None
        self.predicted_vx = 0.0
        self._prev_pred_x = None
        self._prev_ttc = None

    # Helpers

    @staticmethod
    def _extract_time(positions: list) -> np.ndarray:
        """Extract the time axis from position tuples.

        Returns wall-clock timestamps (seconds) if available (tuple
        length >= 4), otherwise falls back to frame indices.
        """
        if len(positions[0]) >= 4:
            return np.array([p[3] for p in positions], dtype=np.float64)
        return np.array([p[2] for p in positions], dtype=np.float64)

    # Main prediction

    def predict(self, track: BallTrack) -> Optional[Tuple[int, int]]:
        """Extrapolate crossing at ``target_y``.

        Returns ``(predicted_x, target_y)`` or ``None``.
        """
        self.predicted_x = None
        self.predicted_frame = None
        self.time_to_crossing_sec = None
        self.crossing_timestamp = None
        self.confidence = 0.0
        self.predicted_vx = 0.0

        if self.target_y is None:
            return None

        positions = list(track.positions)
        n = len(positions)
        if n < 2:
            return None

        ts = self._extract_time(positions)
        xs = np.array([p[0] for p in positions], dtype=np.float64)
        ys = np.array([p[1] for p in positions], dtype=np.float64)
        t0 = ts[0]
        ts_n = ts - t0   # normalised time (start at 0)

        dy = ys[-1] - ys[0]
        target_dir = self.target_y - ys[-1]
        near_target = abs(self.target_y - ys[-1]) < 250  # generous tolerance
        if dy * target_dir < 0 and not near_target:
            return None

        # Fit polynomials (time → x, time → y)
        try:
            if n >= 5:
                cx = np.polyfit(ts_n, xs, 2)
                cy = np.polyfit(ts_n, ys, 2)
                self.fit_type = "quadratic"
            else:
                cx = np.polyfit(ts_n, xs, 1)
                cy = np.polyfit(ts_n, ys, 1)
                self.fit_type = "linear"
        except (np.linalg.LinAlgError, ValueError):
            return None

        # Solve y(t) = target_y
        poly_y = np.poly1d(cy)
        roots = np.roots(poly_y - self.target_y)

        current_t = ts_n[-1]
        best_t: Optional[float] = None
        for r in roots:
            if np.isreal(r):
                rv = float(np.real(r))
                if rv > current_t - 0.5:
                    if best_t is None or abs(rv - current_t) < abs(best_t - current_t):
                        best_t = rv

        if best_t is None:
            return None

        poly_x = np.poly1d(cx)
        pred_x = int(poly_x(best_t))

        # Wall-clock outputs
        time_to_cross = best_t - current_t
        self.time_to_crossing_sec = max(0.001, time_to_cross)
        self.crossing_timestamp = t0 + best_t

        # Velocity at crossing (derivative of poly_x)
        dpoly_x = np.polyder(poly_x)
        self.predicted_vx = float(dpoly_x(best_t))

        # Legacy frame-based output (approximate)
        frames = np.array([p[2] for p in positions], dtype=np.float64)
        if len(frames) >= 2 and (ts[-1] - ts[0]) > 1e-6:
            avg_fps = (frames[-1] - frames[0]) / (ts[-1] - ts[0])
        else:
            avg_fps = 30.0
        pred_frame = int(frames[-1] + time_to_cross * avg_fps)
        self.predicted_frame = pred_frame

        # Confidence: more data + shorter horizon = higher confidence
        dt_span = ts_n[-1] - ts_n[0]
        self.confidence = (
            min(1.0, n / 15.0)
            * max(0.1, 1.0 - time_to_cross / max(dt_span * 3, 0.5))
        )

        # Temporal smoothing: blend with previous prediction to reduce jitter
        alpha = self._smooth_alpha
        if self._prev_pred_x is not None:
            pred_x = int(alpha * pred_x + (1.0 - alpha) * self._prev_pred_x)
        self._prev_pred_x = float(pred_x)
        if self._prev_ttc is not None and self.time_to_crossing_sec is not None:
            self.time_to_crossing_sec = (
                alpha * self.time_to_crossing_sec
                + (1.0 - alpha) * self._prev_ttc
            )
        self._prev_ttc = self.time_to_crossing_sec

        if pred_x < -50 or pred_x > 1250:
            self._prev_pred_x = None
            self._prev_ttc = None
            return None

        self.predicted_x = pred_x
        return (pred_x, self.target_y)

    def get_trajectory_points(self, track: BallTrack,
                              n_future: int = 30
                              ) -> List[Tuple[int, int]]:
        """Return a list of ``(x, y)`` points along the fitted curve."""
        positions = list(track.positions)
        n = len(positions)
        if n < 2:
            return [(p[0], p[1]) for p in positions]

        ts = self._extract_time(positions)
        xs = np.array([p[0] for p in positions], dtype=np.float64)
        ys = np.array([p[1] for p in positions], dtype=np.float64)
        t0 = ts[0]
        ts_n = ts - t0

        try:
            deg = 2 if n >= 5 else 1
            cx = np.polyfit(ts_n, xs, deg)
            cy = np.polyfit(ts_n, ys, deg)
        except (np.linalg.LinAlgError, ValueError):
            return [(p[0], p[1]) for p in positions]

        poly_x = np.poly1d(cx)
        poly_y = np.poly1d(cy)

        # Estimate future time range from average speed
        if len(ts_n) >= 2 and ts_n[-1] > ts_n[0]:
            avg_dt_per_frame = (ts_n[-1] - ts_n[0]) / (n - 1)
        else:
            avg_dt_per_frame = 1.0 / 30.0
        future_span = avg_dt_per_frame * n_future

        pts: List[Tuple[int, int]] = []
        t_start = ts_n[0]
        t_end = ts_n[-1] + future_span
        n_pts = max(int((t_end - t_start) / avg_dt_per_frame) + 1, 2)
        for t in np.linspace(t_start, t_end, n_pts):
            pts.append((int(poly_x(t)), int(poly_y(t))))
        return pts
