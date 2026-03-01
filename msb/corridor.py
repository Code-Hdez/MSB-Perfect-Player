"""
Trajectory corridor — defines the zone where the ball is expected to fly.

Stage 1 (now):  static rectangle or polygon, set manually, from config,
                or built from annotations.
Stage 2 (later): dynamic corridor that expands around the tracked path
                 in real time.  The ``expand_dynamic`` method and
                 ``_dynamic_points`` list are the integration hooks.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from msb.config import Config
from msb.utils import put_text, COL_CYAN, in_rect


class TrajectoryCorridor:
    """Constrains ball-candidate search to a spatial zone."""

    def __init__(self, cfg: Optional[Config] = None) -> None:
        self.cfg = cfg or Config()
        self.rect: Optional[Tuple[int, int, int, int]] = None
        self.polygon: Optional[np.ndarray] = None   # Nx1x2 int32
        self.active: bool = False
        self._dynamic_points: List[Tuple[int, int]] = []

    # Setters

    def set_rect(self, x1: int, y1: int, x2: int, y2: int) -> None:
        self.rect = (x1, y1, x2, y2)
        self.polygon = None
        self.active = True

    def set_polygon(self, points: List[Tuple[int, int]]) -> None:
        self.polygon = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
        self.rect = None
        self.active = True

    def from_annotations(self, positions: List[Tuple[int, int]],
                         margin: Optional[int] = None) -> None:
        """Build a bounding rectangle from labelled ball positions."""
        if not positions:
            return
        m = margin if margin is not None else self.cfg.corridor_margin
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        self.set_rect(min(xs) - m, min(ys) - m,
                      max(xs) + m, max(ys) + m)

    # Queries

    def contains(self, x: int, y: int) -> bool:
        if not self.active:
            return True
        if self.polygon is not None:
            return cv2.pointPolygonTest(
                self.polygon, (float(x), float(y)), False) >= 0
        if self.rect is not None:
            return in_rect(x, y, self.rect)
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
            d = float(np.hypot(dx, dy))
            return -d if inside else d
        return -1.0

    def get_corridor_score(self, x: int, y: int) -> float:
        """1.0 = inside corridor; linearly decays to 0.0 at penalty_dist."""
        d = self.distance_to_boundary(x, y)
        if d <= 0:
            return 1.0
        pen = self.cfg.corridor_penalty_dist
        if d >= pen:
            return 0.0
        return 1.0 - d / pen

    # Stage-2 hook

    def expand_dynamic(self, x: int, y: int,
                       margin: Optional[int] = None) -> None:
        """Add a tracked position and widen the corridor to include it."""
        self._dynamic_points.append((x, y))
        if len(self._dynamic_points) >= 3:
            self.from_annotations(self._dynamic_points, margin)

    # Persistence

    def save(self, path) -> None:
        data: Dict[str, Any] = {"active": self.active}
        if self.rect:
            data["rect"] = list(self.rect)
        if self.polygon is not None:
            data["polygon"] = self.polygon.reshape(-1, 2).tolist()
        with open(str(path), "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path) -> bool:
        import pathlib
        p = pathlib.Path(path)
        if not p.exists():
            return False
        with open(p) as f:
            data = json.load(f)
        if "rect" in data:
            self.set_rect(*data["rect"])
        elif "polygon" in data:
            self.set_polygon(data["polygon"])
        self.active = data.get("active", True)
        return True

    # Drawing

    def draw(self, img: np.ndarray,
             color: Tuple[int, ...] = COL_CYAN,
             thickness: int = 1) -> None:
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
