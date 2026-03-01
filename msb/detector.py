"""
Ball detection pipeline — background model, pitcher suppression,
HSV thresholding, contour filtering, and composite scoring.

All thresholds come from the :class:`~msb.config.Config` object passed
to the constructor.  No module-level mutable state.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from msb.config import Config
from msb.corridor import TrajectoryCorridor
from msb.utils import crop


#  BALL CANDIDATE

class BallCandidate:
    """One detected blob that might be the ball."""

    __slots__ = (
        "center", "area", "circularity", "bbox", "contour",
        "in_motion_mask", "isolation_score", "corridor_score",
        "brightness_score", "score",
    )

    def __init__(self) -> None:
        self.center: Tuple[int, int] = (0, 0)
        self.area: float = 0.0
        self.circularity: float = 0.0
        self.bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)  # x,y,w,h
        self.contour: Optional[np.ndarray] = None
        self.in_motion_mask: bool = False
        self.isolation_score: float = 1.0   # 1=far from pitcher body
        self.corridor_score: float = 1.0    # 1=inside corridor
        self.brightness_score: float = 0.5  # 0=dark, 1=bright
        self.score: float = 0.0

    def __repr__(self) -> str:
        return (f"Ball({self.center}, area={self.area:.0f}, "
                f"circ={self.circularity:.2f}, bri={self.brightness_score:.2f}, "
                f"corr={self.corridor_score:.2f}, score={self.score:.2f})")


#  BACKGROUND MODEL

class BackgroundModel:
    """Running-average background for foreground segmentation."""

    def __init__(self, cfg: Optional[Config] = None) -> None:
        self.cfg = cfg or Config()
        self._bg: Optional[np.ndarray] = None          # float32 grey
        self._frame_count: int = 0
        self._warmup: bool = True
        self.ready: bool = False
        self.fg_mask: Optional[np.ndarray] = None
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def update(self, grey: np.ndarray,
               learning: bool = True) -> np.ndarray:
        """Feed a new grey frame.  Returns the foreground binary mask."""
        cfg = self.cfg

        if self._bg is None or self._bg.shape != grey.shape:
            self._bg = grey.astype(np.float32)
            self._frame_count = 1
            self._warmup = True
            self.ready = False
            self.fg_mask = np.zeros_like(grey)
            return self.fg_mask

        self._frame_count += 1

        if learning:
            if self._warmup:
                alpha = 1.0 / self._frame_count
                if self._frame_count >= cfg.bg_warmup_frames:
                    self._warmup = False
                    self.ready = True
            else:
                alpha = cfg.bg_alpha
            cv2.accumulateWeighted(grey, self._bg, alpha)

        bg_u8 = self._bg.astype(np.uint8)
        diff = cv2.absdiff(grey, bg_u8)
        _, fg = cv2.threshold(diff, cfg.bg_fg_threshold, 255,
                              cv2.THRESH_BINARY)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self._kernel, iterations=1)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self._kernel, iterations=1)
        self.fg_mask = fg
        return fg

    def reset(self) -> None:
        self._bg = None
        self._frame_count = 0
        self._warmup = True
        self.ready = False
        self.fg_mask = None


#  PITCHER SUPPRESSOR

class PitcherSuppressor:
    """Identifies large foreground blobs (pitcher body / arm) and
    computes an isolation score for each ball candidate."""

    def __init__(self, cfg: Optional[Config] = None) -> None:
        self.cfg = cfg or Config()
        self.large_blobs: List[Tuple[int, int, int, int, float]] = []
        self.suppression_zones: List[Tuple[int, int, int]] = []

    def analyze(self, fg_mask: np.ndarray,
                min_area: Optional[int] = None) -> None:
        cfg = self.cfg
        min_a = min_area if min_area is not None else cfg.pitcher_body_min_area
        self.large_blobs.clear()
        self.suppression_zones.clear()

        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_a:
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
            radius = int(max(bw, bh) * cfg.isolation_zone_scale)
            self.suppression_zones.append((cx, cy, radius))

    def get_isolation_score(self, x: int, y: int) -> float:
        """0.0 = inside suppression zone, 1.0 = far enough away."""
        cfg = self.cfg
        if not self.suppression_zones:
            return 1.0

        min_nd = float("inf")
        for sx, sy, r in self.suppression_zones:
            d = np.hypot(x - sx, y - sy)
            nd = d / max(r, 1)
            if nd < min_nd:
                min_nd = nd

        if min_nd <= cfg.isolation_inner:
            return 0.0
        if min_nd >= cfg.isolation_outer:
            return 1.0
        return (min_nd - cfg.isolation_inner) / (
            cfg.isolation_outer - cfg.isolation_inner)


#  BALL DETECTOR

class BallDetector:
    """Detect the baseball in a game frame.

    Pipeline
    --------
    1. Background subtraction → foreground mask.
    2. Frame differencing → motion mask (secondary).
    3. Pitcher-body suppression.
    4. HSV white threshold → white mask.
    5. Trail HSV threshold → trail mask (dimmer ball).
    6. AND combine: white+fg (primary), trail+fg (secondary).
    7. Contour filter: area, circularity.
    8. Static suppression: grid-based HUD / field-marking filter.
    9. Corridor scoring.
    10. Composite score: motion × circularity × area × isolation × corridor.
    """

    def __init__(self, cfg: Optional[Config] = None) -> None:
        self.cfg = cfg or Config()
        self._prev_grey: Optional[np.ndarray] = None
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self._kernel_small = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (3, 3))

        self.bg_model = BackgroundModel(self.cfg)
        self.suppressor = PitcherSuppressor(self.cfg)

        self._static_map: Dict[Tuple[int, int], int] = {}

        # Cached masks for debug visualisation
        self.motion_mask: Optional[np.ndarray] = None
        self.bg_fg_mask: Optional[np.ndarray] = None
        self.white_mask: Optional[np.ndarray] = None
        self.trail_mask: Optional[np.ndarray] = None
        self.combined_mask: Optional[np.ndarray] = None
        self.candidates: List[BallCandidate] = []
        self.best: Optional[BallCandidate] = None

        # For rescue detection
        self._rescue_grey: Optional[np.ndarray] = None
        self._rescue_hsv: Optional[np.ndarray] = None

    # Resets

    def reset(self) -> None:
        """Soft reset (new pitch).  Preserves bg model + static map."""
        self._prev_grey = None
        self.motion_mask = None
        self.bg_fg_mask = None
        self.white_mask = None
        self.trail_mask = None
        self.combined_mask = None
        self.candidates.clear()
        self.best = None

    def reset_full(self) -> None:
        """Full reset including background model."""
        self.reset()
        self.bg_model.reset()
        self._static_map.clear()

    # Static-element grid

    def _cell(self, x: int, y: int) -> Tuple[int, int]:
        s = self.cfg.static_cell_size
        return (x // s, y // s)

    def _is_static(self, x: int, y: int) -> bool:
        return self._static_map.get(
            self._cell(x, y), 0) >= self.cfg.static_hit_threshold

    def _record_static(self, x: int, y: int) -> None:
        cell = self._cell(x, y)
        self._static_map[cell] = self._static_map.get(cell, 0) + 1

    # Main detection

    def detect(
        self,
        frame: np.ndarray,
        search_roi: Optional[Tuple[int, int, int, int]] = None,
        corridor: Optional[TrajectoryCorridor] = None,
        track_active: bool = False,
    ) -> Optional[BallCandidate]:
        """Detect ball candidates in *frame* (BGR).

        Returns the highest-scored :class:`BallCandidate`, or ``None``.
        """
        cfg = self.cfg

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
        self._rescue_grey = grey.copy()
        self._rescue_hsv = hsv.copy()

        # 1) Foreground mask
        learning = not track_active
        bg_fg = self.bg_model.update(grey, learning=learning)
        self.bg_fg_mask = bg_fg

        # 2) Frame-to-frame differencing
        if (self._prev_grey is not None
                and self._prev_grey.shape == grey.shape):
            diff = cv2.absdiff(grey, self._prev_grey)
            _, motion = cv2.threshold(diff, cfg.diff_threshold, 255,
                                      cv2.THRESH_BINARY)
            motion = cv2.dilate(motion, self._kernel,
                                iterations=cfg.diff_dilate_iter)
        else:
            motion = np.zeros_like(grey)
        self._prev_grey = grey.copy()
        self.motion_mask = motion

        # Choose primary foreground mask
        if self.bg_model.ready:
            fg = cv2.bitwise_or(bg_fg, motion)
        else:
            fg = motion

        # 3) Pitcher-body suppression
        self.suppressor.analyze(fg)

        # 4) HSV brightness mask (primary — widened)
        ball_lo = np.array(cfg.ball_hsv_lower)
        ball_hi = np.array(cfg.ball_hsv_upper)
        white = cv2.inRange(hsv, ball_lo, ball_hi)
        white = cv2.morphologyEx(white, cv2.MORPH_OPEN,
                                 self._kernel_small, iterations=1)
        white = cv2.morphologyEx(white, cv2.MORPH_CLOSE,
                                 self._kernel_small, iterations=1)
        self.white_mask = white

        # 5) Trail HSV threshold (dimmer ball in motion blur)
        trail_lo = np.array(cfg.trail_hsv_lower)
        trail_hi = np.array(cfg.trail_hsv_upper)
        trail = cv2.inRange(hsv, trail_lo, trail_hi)
        trail = cv2.morphologyEx(trail, cv2.MORPH_OPEN,
                                 self._kernel_small, iterations=1)
        self.trail_mask = trail

        # 6) Combine: (white OR trail) AND foreground
        brightness = cv2.bitwise_or(white, trail)
        combined = cv2.bitwise_and(brightness, fg)
        self.combined_mask = combined

        # 7) Extract candidates from multiple channels
        self.candidates.clear()
        self.best = None

        # Primary: combined (brightness + foreground)
        cands_combined = self._extract_candidates(
            combined, ox, oy, in_motion=True)
        # Secondary: brightness-only (no fg needed — for first frames)
        cands_bright = self._extract_candidates(
            white, ox, oy, in_motion=False)
        # Tertiary: fg-only with brightness verification at centroid
        cands_fg = self._extract_fg_bright_candidates(
            fg, grey, ox, oy)
        # Quaternary: peaks inside large fg blobs
        cands_peaks = self._extract_peak_candidates(
            fg, grey, ox, oy)

        # Merge + deduplicate
        all_cands = cands_combined[:]
        for secondary in (cands_bright, cands_fg, cands_peaks):
            for sc in secondary:
                dup = False
                for mc in all_cands:
                    d = np.hypot(sc.center[0] - mc.center[0],
                                 sc.center[1] - mc.center[1])
                    if d < 15:
                        dup = True
                        break
                if not dup:
                    all_cands.append(sc)

        if not all_cands:
            return None

        # 8) Static suppression
        filtered: List[BallCandidate] = []
        for c in all_cands:
            self._record_static(c.center[0], c.center[1])
            if self._is_static(c.center[0], c.center[1]):
                continue
            filtered.append(c)

        if not filtered:
            self.candidates = []
            self.best = None
            return None

        # 9) Isolation + corridor + brightness + contrast scoring
        # Pre-compute blurred grey for local contrast analysis
        grey_blur = cv2.blur(grey.astype(np.float32), (21, 21))

        for c in filtered:
            c.isolation_score = self.suppressor.get_isolation_score(
                c.center[0] - ox, c.center[1] - oy)
            if corridor is not None:
                c.corridor_score = corridor.get_corridor_score(
                    c.center[0], c.center[1])
            else:
                c.corridor_score = 1.0

            lx = c.center[0] - ox
            ly = c.center[1] - oy

            # Brightness score from greyscale at centroid
            if 0 <= ly < grey.shape[0] and 0 <= lx < grey.shape[1]:
                bval = float(grey[ly, lx])
                # Local contrast: how much brighter than neighbourhood
                local_mean = float(grey_blur[ly, lx])
                contrast = (bval - local_mean)
            else:
                bval = 100.0
                contrast = 0.0
            c.brightness_score = min(bval / 200.0, 1.0)

            # Contrast score: ball stands out, pitcher body doesnt
            contrast_score = max(0.0, min(contrast / 40.0, 1.0))

            # Store combined brightness (used in matching)
            c.brightness_score = (
                0.5 * min(bval / 200.0, 1.0) +
                0.5 * contrast_score
            )

        # 10) Composite score
        for c in filtered:
            motion_w = 2.0 if c.in_motion_mask else 0.5
            area_norm = 1.0 - min(c.area / cfg.ball_max_area, 1.0)
            circ_norm = min(c.circularity / 1.0, 1.0)
            bri_norm = c.brightness_score
            corr_w = max(c.corridor_score, 0.05)
            iso_w = max(c.isolation_score, 0.05)
            c.score = motion_w * (
                0.20 * circ_norm
                + 0.15 * area_norm
                + 0.30 * bri_norm
                + 0.30 * corr_w
                + 0.05 * iso_w
            )

        filtered.sort(key=lambda c: c.score, reverse=True)
        self.candidates = filtered
        self.best = filtered[0]
        return self.best

    # Candidate extraction

    def _extract_candidates(
        self, mask: np.ndarray, ox: int, oy: int, in_motion: bool,
    ) -> List[BallCandidate]:
        cfg = self.cfg
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result: List[BallCandidate] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (cfg.ball_min_area <= area <= cfg.ball_max_area):
                continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter < 1:
                continue
            circularity = 4.0 * np.pi * area / (perimeter * perimeter)
            if circularity < cfg.ball_min_circularity:
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

    # Fg-only candidate extraction with brightness check

    def _extract_fg_bright_candidates(
        self,
        fg: np.ndarray,
        grey: np.ndarray,
        ox: int,
        oy: int,
    ) -> List[BallCandidate]:
        """Extract candidates from a foreground mask that are bright
        enough at their centroid (no HSV colour requirement)."""
        cfg = self.cfg
        contours, _ = cv2.findContours(
            fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result: List[BallCandidate] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (cfg.ball_min_area <= area <= cfg.ball_max_area):
                continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter < 1:
                continue
            circularity = 4.0 * np.pi * area / (perimeter * perimeter)
            if circularity < 0.12:          # relaxed for fg-only path
                continue

            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx_local = int(M["m10"] / M["m00"])
                cy_local = int(M["m01"] / M["m00"])
            else:
                bx, by, bw, bh = cv2.boundingRect(cnt)
                cx_local, cy_local = bx + bw // 2, by + bh // 2

            # Brightness check at centroid — must be reasonably bright
            if 0 <= cy_local < grey.shape[0] and 0 <= cx_local < grey.shape[1]:
                if grey[cy_local, cx_local] < 110:
                    continue
            else:
                continue

            bx, by, bw, bh = cv2.boundingRect(cnt)
            c = BallCandidate()
            c.center = (cx_local + ox, cy_local + oy)
            c.area = area
            c.circularity = circularity
            c.bbox = (bx + ox, by + oy, bw, bh)
            cnt_full = cnt.copy()
            cnt_full[:, :, 0] += ox
            cnt_full[:, :, 1] += oy
            c.contour = cnt_full
            c.in_motion_mask = True         # fg = moving pixels
            result.append(c)
        return result

    # Peak extraction from large foreground blobs

    def _extract_peak_candidates(
        self,
        fg: np.ndarray,
        grey: np.ndarray,
        ox: int,
        oy: int,
    ) -> List[BallCandidate]:
        """Find bright peaks inside fg contours that are too large to
        pass the normal area filter (merged ball+body blobs)."""
        cfg = self.cfg
        contours, _ = cv2.findContours(
            fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result: List[BallCandidate] = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Only look at contours ABOVE the normal max
            if area <= cfg.ball_max_area or area > 8000:
                continue

            bx, by, bw, bh = cv2.boundingRect(cnt)
            if bw < 3 or bh < 3:
                continue

            # Crop grey and fg to bounding rect
            g_roi = grey[by:by + bh, bx:bx + bw].copy()
            f_roi = fg[by:by + bh, bx:bx + bw]

            # Mask to only count pixels inside the contour
            cnt_mask = np.zeros((bh, bw), dtype=np.uint8)
            cnt_shifted = cnt - np.array([bx, by])
            cv2.drawContours(cnt_mask, [cnt_shifted], 0, 255, -1)

            # Local contrast: pixel brightness minus neighbourhood mean
            blur = cv2.blur(g_roi.astype(np.float32), (15, 15))
            contrast = g_roi.astype(np.float32) - blur

            # Bright peak: high local contrast AND inside the contour
            peak_mask = np.zeros((bh, bw), dtype=np.uint8)
            peak_mask[(contrast > 12) & (cnt_mask > 0) & (g_roi > 100)] = 255

            if np.count_nonzero(peak_mask) == 0:
                continue

            # Find connected components in peak mask
            pk_contours, _ = cv2.findContours(
                peak_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for pk in pk_contours:
                pk_area = cv2.contourArea(pk)
                if pk_area < cfg.ball_min_area or pk_area > cfg.ball_max_area:
                    continue
                pk_perim = cv2.arcLength(pk, True)
                if pk_perim < 1:
                    continue
                pk_circ = 4.0 * np.pi * pk_area / (pk_perim * pk_perim)
                if pk_circ < 0.10:
                    continue

                Mpk = cv2.moments(pk)
                if Mpk["m00"] > 0:
                    pcx = int(Mpk["m10"] / Mpk["m00"])
                    pcy = int(Mpk["m01"] / Mpk["m00"])
                else:
                    pbx, pby, pbw, pbh = cv2.boundingRect(pk)
                    pcx, pcy = pbx + pbw // 2, pby + pbh // 2

                pbx, pby, pbw, pbh = cv2.boundingRect(pk)
                c = BallCandidate()
                c.center = (pcx + bx + ox, pcy + by + oy)
                c.area = pk_area
                c.circularity = pk_circ
                c.bbox = (pbx + bx + ox, pby + by + oy, pbw, pbh)
                c.in_motion_mask = True
                result.append(c)

        return result

    # Rescue detection

    def rescue_near(
        self, centre: Tuple[int, int], radius: int = 50,
    ) -> Optional[BallCandidate]:
        """Emergency positional search near *centre* using local
        contrast detection when normal candidate matching fails."""
        if self.bg_fg_mask is None or self._rescue_grey is None:
            return None

        grey = self._rescue_grey
        bg_fg = self.bg_fg_mask
        cx, cy = centre
        h, w = grey.shape[:2]

        pad = 15
        x1 = max(0, cx - radius - pad)
        y1 = max(0, cy - radius - pad)
        x2 = min(w, cx + radius + pad)
        y2 = min(h, cy + radius + pad)

        grey_roi = grey[y1:y2, x1:x2].astype(np.float32)
        fg_roi = bg_fg[y1:y2, x1:x2]
        if grey_roi.size == 0 or fg_roi.size == 0:
            return None

        local_mean = cv2.blur(grey_roi, (15, 15))
        contrast = grey_roi - local_mean

        rescue_mask = np.zeros(grey_roi.shape, dtype=np.uint8)
        rescue_mask[(contrast > 10) & (fg_roi > 0)] = 255

        local_cx = float(cx - x1)
        local_cy = float(cy - y1)
        ys_arr, xs_arr = np.nonzero(rescue_mask)
        if len(xs_arr) == 0:
            return None

        dists = np.hypot(xs_arr.astype(np.float64) - local_cx,
                         ys_arr.astype(np.float64) - local_cy)
        within = dists <= radius
        if not np.any(within):
            return None

        xs_arr = xs_arr[within]
        ys_arr = ys_arr[within]
        dists = dists[within]

        nearest_idx = int(np.argmin(dists))
        nx = int(xs_arr[nearest_idx])
        ny = int(ys_arr[nearest_idx])
        nearby = (np.abs(xs_arr - nx) <= 4) & (np.abs(ys_arr - ny) <= 4)
        cluster_xs = xs_arr[nearby]
        cluster_ys = ys_arr[nearby]

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
