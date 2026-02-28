"""
Hitbox Detection & Character Fingerprint System

Capture workflow (COLLECT mode)

  1. Press [C] to start a capture.
  2. The live feed FREEZES.  Instruction text appears.
  3. Click TOP-LEFT then BOTTOM-RIGHT of the **character body** region.
  4. Click TOP-LEFT then BOTTOM-RIGHT of the **strike-zone cursor** region.
  5. Sample is saved automatically.  Feed resumes.
  Press [ESC] at any point during clicking to cancel.

Recognition mode ([R] key — toggle COLLECT ↔ RECOGNIZE)

  Loads the fingerprint database built in COLLECT mode.  Each frame is
  matched against saved fingerprints.  Best character name + confidence is
  shown on-screen.  Press [R] again to return to COLLECT mode.

Tracking mode ([T] key — toggle tracking ON/OFF)

  Loads the movement-calibration data saved for the current character
  (set with [N]) and activates real-time batter tracking + strike-zone
  prediction.  Requires at least one calibration run to have been saved
  for the active character.  The BatterStateClassifier also loads so that
  non-normal animation frames (swings / idles) are automatically frozen
  and excluded from template-match updates.  Press [T] again to stop.

Set character name ([N] key)

  Prompts for a character name in the terminal.  Required before capturing
  samples ([C]) or starting tracking ([T]).

Debug panel ([D] key)

  Shows the user-defined strike-zone ROI with:
    • HSV colour mask  • Canny edges  • All detected contour outlines
    • Batter ROI + dominant colours + ORB key-points
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

#  CONFIGURATION

# Screen capture

SCREEN_ROI: Tuple[int, int, int, int] = (378, 127, 1542, 1019)
"""(left, top, right, bottom) pixel coords of the Dolphin game window."""

MONITOR_INDEX: int = 0
"""
DXGI output index of the monitor Dolphin is running on.
  0 = primary monitor
  1 = secondary monitor
Change this if Dolphin is on your second screen.
"""

TARGET_FPS: int = 60

# Hitbox colour detection (HSV) - for strike-zone outline detection

HITBOX_HSV_LOWER: np.ndarray = np.array([12, 70, 150])
HITBOX_HSV_UPPER: np.ndarray = np.array([38, 255, 255])
"""HSV range for the golden/amber hitbox pentagon."""

HITBOX_MIN_AREA: int = 400
HITBOX_MAX_AREA: int = 60_000
HITBOX_MIN_SOLIDITY: float = 0.45

# Fingerprint parameters

HIST_BINS: Tuple[int, int, int] = (12, 12, 12)
ORB_N_FEATURES: int = 500
DOMINANT_COLORS_K: int = 5

# Matching

HIST_MATCH_THRESHOLD: float = 0.35
RECOGNITION_TOP_K: int = 3

# Paths

DATASET_DIR: Path = Path("./dataset")

# Display

DISPLAY_SCALE: float = 0.85
"""Scale factor for the main window (< 1 shrinks on high-res monitors).
Click coordinates are automatically mapped back to original frame space."""

# Movement tracking & anti-flicker

BATTER_SEARCH_MARGIN: int = 80
"""Extra pixels around the calibrated batter area for template search."""

STRIKE_HOLD_FRAMES: int = 15
"""Frames to hold strike-zone position while it blinks off."""

STRIKE_CONFIDENCE_DECAY: float = 0.90
"""Per-frame confidence multiplier while the strike-zone is held."""

STRIKE_SEARCH_PADDING: int = 40
"""Extra pixels around predicted strike-zone for gold-hitbox detection."""

# Batter state classification

STATE_NORMAL_THRESHOLD: float = 0.55
"""Combined similarity score >= this to ENTER NORMAL state (from NON_NORMAL)."""

STATE_LEAVE_THRESHOLD: float = 0.42
"""Combined similarity must drop below this to LEAVE NORMAL state (hysteresis)."""

STATE_WINDOW_SIZE: int = 8
"""Rolling window size for majority-vote temporal smoothing of batter state."""

STATE_CLASSIFY_INTERVAL: int = 3
"""Run the state classifier every N frames (1 = every frame). Higher = less
CPU cost but slower reaction to state changes."""

TRACK_DOWNSCALE: int = 2
"""Downscale factor for template matching in _find_batter (2 = half res).
Higher = faster but less precise localisation."""

WINDOW_MAIN: str = "MSB Hitbox Detector"
WINDOW_DEBUG: str = "MSB Debug"

# Colours (BGR) & font

COL_GREEN  = (0, 255, 0)
COL_RED    = (0, 0, 255)
COL_YELLOW = (0, 255, 255)
COL_CYAN   = (255, 255, 0)
COL_WHITE  = (255, 255, 255)
COL_BLACK  = (0, 0, 0)
COL_MAGENTA = (255, 0, 255)
COL_ORANGE  = (0, 165, 255)
FONT       = cv2.FONT_HERSHEY_SIMPLEX


#  UTILITY HELPERS

def crop(frame: np.ndarray, rect: Tuple[int, int, int, int]) -> np.ndarray:
    """Crop *frame* to pixel rectangle (x1, y1, x2, y2). Returns a view."""
    x1, y1, x2, y2 = rect
    return frame[max(0, y1):y2, max(0, x1):x2]


def put_text(img: np.ndarray, text: str, org: Tuple[int, int],
             scale: float = 0.6, color: Tuple[int, ...] = COL_GREEN,
             thickness: int = 2) -> None:
    """Draw *text* with a dark shadow for readability."""
    x, y = org
    cv2.putText(img, text, (x + 1, y + 1), FONT, scale,
                COL_BLACK, thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), FONT, scale,
                color, thickness, cv2.LINE_AA)


def hsv_histogram(img_bgr: np.ndarray,
                  bins: Tuple[int, int, int] = HIST_BINS,
                  mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Compute a normalised 3-D HSV histogram."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], mask,
                        list(bins), [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist


def dominant_colours(img_bgr: np.ndarray,
                     k: int = DOMINANT_COLORS_K) -> np.ndarray:
    """K-means dominant colours -> shape (k, 3) BGR, sorted by frequency."""
    pixels = img_bgr.reshape(-1, 3).astype(np.float32)
    if len(pixels) < k:
        return pixels.astype(np.uint8)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centres = cv2.kmeans(
        pixels, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    order = np.argsort(-np.bincount(labels.flatten(), minlength=k))
    return centres[order].astype(np.uint8)


def orb_descriptors(img_bgr: np.ndarray,
                    n: int = ORB_N_FEATURES
                    ) -> Tuple[Optional[list], Optional[np.ndarray]]:
    """ORB key-points and descriptors on a grey-scale version."""
    grey = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=n)
    return orb.detectAndCompute(grey, None)


def draw_dashed_rect(img: np.ndarray,
                     pt1: Tuple[int, int], pt2: Tuple[int, int],
                     color: Tuple[int, ...], thickness: int = 1,
                     dash: int = 8, gap: int = 5) -> None:
    """Draw a dashed rectangle (OpenCV has no built-in dashed lines)."""
    x1, y1 = pt1
    x2, y2 = pt2
    step = dash + gap
    for x in range(x1, x2, step):
        cv2.line(img, (x, y1), (min(x + dash, x2), y1), color, thickness)
        cv2.line(img, (x, y2), (min(x + dash, x2), y2), color, thickness)
    for y in range(y1, y2, step):
        cv2.line(img, (x1, y), (x1, min(y + dash, y2)), color, thickness)
        cv2.line(img, (x2, y), (x2, min(y + dash, y2)), color, thickness)


#  CLICK-TO-SELECT ROI SYSTEM

class ClickPhase(Enum):
    """State machine phases for the 4-click capture flow."""
    IDLE            = auto()
    CHAR_TL         = auto()   # waiting for character top-left
    CHAR_BR         = auto()   # waiting for character bottom-right
    STRIKE_TL       = auto()   # waiting for strike-zone top-left
    STRIKE_BR       = auto()   # waiting for strike-zone bottom-right
    DONE            = auto()
    CANCELLED       = auto()


_PHASE_PROMPTS = {
    ClickPhase.CHAR_TL:   "Click TOP-LEFT of CHARACTER region",
    ClickPhase.CHAR_BR:   "Click BOTTOM-RIGHT of CHARACTER region",
    ClickPhase.STRIKE_TL: "Click TOP-LEFT of STRIKE-ZONE region",
    ClickPhase.STRIKE_BR: "Click BOTTOM-RIGHT of STRIKE-ZONE region",
}


class ClickCollector:
    """Collect 4 mouse clicks (2 rectangles) on a frozen frame.

    Coordinates are stored in *original frame space* (before display scaling).
    """

    def __init__(self, display_scale: float = 1.0) -> None:
        self.scale = display_scale
        self.phase = ClickPhase.IDLE
        self.char_rect: Optional[Tuple[int, int, int, int]] = None   # (x1,y1,x2,y2)
        self.strike_rect: Optional[Tuple[int, int, int, int]] = None
        self._points: List[Tuple[int, int]] = []

    def start(self) -> None:
        """Enter click-selection mode."""
        self.phase = ClickPhase.CHAR_TL
        self.char_rect = None
        self.strike_rect = None
        self._points.clear()

    def cancel(self) -> None:
        self.phase = ClickPhase.CANCELLED

    @property
    def active(self) -> bool:
        return self.phase not in (ClickPhase.IDLE, ClickPhase.DONE,
                                  ClickPhase.CANCELLED)

    @property
    def prompt(self) -> str:
        return _PHASE_PROMPTS.get(self.phase, "")

    def mouse_callback(self, event: int, x: int, y: int,
                       flags: int, param: Any) -> None:
        """OpenCV mouse callback — register left-clicks."""
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if not self.active:
            return

        # Map from display coords -> original frame coords
        ox = int(x / self.scale)
        oy = int(y / self.scale)
        self._points.append((ox, oy))

        if self.phase == ClickPhase.CHAR_TL:
            self.phase = ClickPhase.CHAR_BR

        elif self.phase == ClickPhase.CHAR_BR:
            tl = self._points[-2]
            br = self._points[-1]
            self.char_rect = (min(tl[0], br[0]), min(tl[1], br[1]),
                              max(tl[0], br[0]), max(tl[1], br[1]))
            self.phase = ClickPhase.STRIKE_TL

        elif self.phase == ClickPhase.STRIKE_TL:
            self.phase = ClickPhase.STRIKE_BR

        elif self.phase == ClickPhase.STRIKE_BR:
            tl = self._points[-2]
            br = self._points[-1]
            self.strike_rect = (min(tl[0], br[0]), min(tl[1], br[1]),
                                max(tl[0], br[0]), max(tl[1], br[1]))
            self.phase = ClickPhase.DONE

    def draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw click-mode UI on the frozen frame (in original coords)."""
        vis = frame.copy()
        h, w = vis.shape[:2]

        # Dim the frame slightly to indicate frozen state
        overlay = np.zeros_like(vis)
        vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)

        # Instruction banner
        put_text(vis, "CAPTURE MODE -- click to define ROIs  (ESC to cancel)",
                 (10, 30), 0.55, COL_YELLOW, 2)

        prompt = self.prompt
        if prompt:
            put_text(vis, prompt, (10, 62), 0.65, COL_WHITE, 2)

        # Draw completed character rect
        if self.char_rect is not None:
            x1, y1, x2, y2 = self.char_rect
            cv2.rectangle(vis, (x1, y1), (x2, y2), COL_CYAN, 2)
            put_text(vis, "CHARACTER", (x1, y1 - 8), 0.45, COL_CYAN, 1)

        # Draw in-progress first click of character
        elif (self.phase == ClickPhase.CHAR_BR
              and len(self._points) >= 1):
            px, py = self._points[-1]
            cv2.drawMarker(vis, (px, py), COL_CYAN, cv2.MARKER_CROSS, 20, 2)
            put_text(vis, "TL", (px + 5, py - 5), 0.40, COL_CYAN, 1)

        # Draw completed strike-zone rect
        if self.strike_rect is not None:
            x1, y1, x2, y2 = self.strike_rect
            cv2.rectangle(vis, (x1, y1), (x2, y2), COL_YELLOW, 2)
            put_text(vis, "STRIKE-ZONE", (x1, y1 - 8), 0.45, COL_YELLOW, 1)

        # Draw in-progress first click of strike-zone
        elif (self.phase == ClickPhase.STRIKE_BR
              and len(self._points) >= 3):
            px, py = self._points[-1]
            cv2.drawMarker(vis, (px, py), COL_YELLOW, cv2.MARKER_CROSS, 20, 2)
            put_text(vis, "TL", (px + 5, py - 5), 0.40, COL_YELLOW, 1)

        # Step indicator
        step_map = {
            ClickPhase.CHAR_TL:   "Step 1/4",
            ClickPhase.CHAR_BR:   "Step 2/4",
            ClickPhase.STRIKE_TL: "Step 3/4",
            ClickPhase.STRIKE_BR: "Step 4/4",
        }
        step = step_map.get(self.phase, "")
        if step:
            put_text(vis, step, (w - 160, 30), 0.50, COL_GREEN, 1)

        return vis


#  HITBOX DETECTOR  (runs inside a user-provided pixel ROI)

class HitboxResult:
    """Container for one frame's hitbox detection output."""
    __slots__ = ("found", "bbox", "contour", "contour_local", "mask",
                 "all_contours_local",
                 "confidence", "area", "centroid", "hu_moments",
                 "roi_rect")

    def __init__(self) -> None:
        self.found: bool = False
        self.bbox: Optional[Tuple[int, int, int, int]] = None
        self.contour: Optional[np.ndarray] = None          # full-frame coords
        self.contour_local: Optional[np.ndarray] = None    # local to ROI
        self.mask: Optional[np.ndarray] = None             # binary mask (ROI)
        self.all_contours_local: List[np.ndarray] = []     # all valid contours (local)
        self.confidence: float = 0.0
        self.area: float = 0.0
        self.centroid: Optional[Tuple[int, int]] = None
        self.hu_moments: Optional[np.ndarray] = None
        self.roi_rect: Optional[Tuple[int, int, int, int]] = None


class HitboxDetector:
    """Detect the golden hitbox / strike-zone cursor within a pixel ROI.

    Pipeline: crop -> HSV threshold -> morph close/open -> contours -> filter.
    """

    def __init__(self, *,
                 hsv_lower: np.ndarray = HITBOX_HSV_LOWER,
                 hsv_upper: np.ndarray = HITBOX_HSV_UPPER,
                 min_area: int = HITBOX_MIN_AREA,
                 max_area: int = HITBOX_MAX_AREA,
                 min_solidity: float = HITBOX_MIN_SOLIDITY):
        self.hsv_lo = hsv_lower
        self.hsv_hi = hsv_upper
        self.min_area = min_area
        self.max_area = max_area
        self.min_solidity = min_solidity
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    def detect(self, frame: np.ndarray,
               roi_rect: Optional[Tuple[int, int, int, int]] = None
               ) -> HitboxResult:
        """Run detection on a BGR game frame.

        Parameters
        ----------
        roi_rect : (x1, y1, x2, y2) pixel rectangle to search within.
                   If None, searches the whole frame.
        """
        result = HitboxResult()
        h, w = frame.shape[:2]

        if roi_rect is not None:
            rx, ry = roi_rect[0], roi_rect[1]
            roi_img = crop(frame, roi_rect)
            result.roi_rect = roi_rect
        else:
            rx, ry = 0, 0
            roi_img = frame

        if roi_img.size == 0:
            return result

        # Colour threshold
        hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lo, self.hsv_hi)

        # Morphological clean-up
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self._kernel, iterations=1)

        result.mask = mask

        # Contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        best_score = 0.0
        best_cnt: Optional[np.ndarray] = None
        best_solidity = 0.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (self.min_area <= area <= self.max_area):
                continue
            hull_area = cv2.contourArea(cv2.convexHull(cnt))
            if hull_area == 0:
                continue
            solidity = area / hull_area
            if solidity < self.min_solidity:
                continue

            # Store valid contour for debug display
            result.all_contours_local.append(cnt)

            score = area * solidity
            if score > best_score:
                best_score, best_cnt, best_solidity = score, cnt, solidity

        if best_cnt is None:
            return result

        # Bounding box
        bx, by, bw, bh = cv2.boundingRect(best_cnt)

        # Centroid
        M = cv2.moments(best_cnt)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"]) + rx
            cy = int(M["m01"] / M["m00"]) + ry
        else:
            cx, cy = bx + rx + bw // 2, by + ry + bh // 2

        # Hu moments
        hu = cv2.HuMoments(M).flatten()
        hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)

        # Full-frame contour
        cnt_full = best_cnt.copy()
        cnt_full[:, :, 0] += rx
        cnt_full[:, :, 1] += ry

        result.found          = True
        result.bbox           = (bx + rx, by + ry, bw, bh)
        result.contour        = cnt_full
        result.contour_local  = best_cnt
        result.confidence     = min(best_solidity * 1.2, 1.0)
        result.area           = cv2.contourArea(best_cnt)
        result.centroid       = (cx, cy)
        result.hu_moments     = hu_log
        return result


#  CHARACTER FINGERPRINT

class Fingerprint:
    """Visual fingerprint extracted from a single game frame."""

    __slots__ = (
        "character", "timestamp",
        "batter_roi", "strike_roi",
        "batter_hist", "batter_dominant", "batter_orb_desc", "batter_orb_n",
        "hitbox_found", "hitbox_area", "hitbox_aspect",
        "hitbox_hu", "hitbox_centroid_frac",
    )

    def __init__(self, character: str = "", ts: float = 0.0) -> None:
        self.character = character
        self.timestamp = ts or time.time()
        self.batter_roi: Optional[Tuple[int, int, int, int]] = None
        self.strike_roi: Optional[Tuple[int, int, int, int]] = None
        self.batter_hist: Optional[np.ndarray] = None
        self.batter_dominant: Optional[np.ndarray] = None
        self.batter_orb_desc: Optional[np.ndarray] = None
        self.batter_orb_n: int = 0
        self.hitbox_found: bool = False
        self.hitbox_area: float = 0.0
        self.hitbox_aspect: float = 0.0
        self.hitbox_hu: Optional[np.ndarray] = None
        self.hitbox_centroid_frac: Optional[Tuple[float, float]] = None

    def to_json(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "character":     self.character,
            "timestamp":     self.timestamp,
            "batter_orb_n":  self.batter_orb_n,
            "hitbox_found":  self.hitbox_found,
            "hitbox_area":   self.hitbox_area,
            "hitbox_aspect": self.hitbox_aspect,
        }
        if self.batter_roi is not None:
            d["batter_roi"] = list(self.batter_roi)
        if self.strike_roi is not None:
            d["strike_roi"] = list(self.strike_roi)
        if self.hitbox_centroid_frac is not None:
            d["hitbox_centroid_frac"] = list(self.hitbox_centroid_frac)
        if self.hitbox_hu is not None:
            d["hitbox_hu"] = self.hitbox_hu.tolist()
        if self.batter_dominant is not None:
            d["batter_dominant"] = self.batter_dominant.tolist()
        return d


class CharacterFingerprinter:
    """Extract a Fingerprint from a game frame using user-provided ROIs."""

    def __init__(self, *,
                 hist_bins: Tuple[int, ...] = HIST_BINS,
                 orb_n: int = ORB_N_FEATURES,
                 dom_k: int = DOMINANT_COLORS_K):
        self.hist_bins = hist_bins
        self.orb_n = orb_n
        self.dom_k = dom_k

    def extract(self, frame: np.ndarray, hbox: HitboxResult,
                batter_rect: Optional[Tuple[int, int, int, int]] = None,
                strike_rect: Optional[Tuple[int, int, int, int]] = None,
                character: str = "") -> Fingerprint:
        """Build a fingerprint using explicit pixel ROIs."""
        h, w = frame.shape[:2]
        fp = Fingerprint(character=character)
        fp.batter_roi = batter_rect
        fp.strike_roi = strike_rect

        # Batter region
        if batter_rect is not None:
            batter = crop(frame, batter_rect)
            if batter.size > 0:
                fp.batter_hist = hsv_histogram(batter, self.hist_bins)
                small = cv2.resize(batter, (80, 80))
                fp.batter_dominant = dominant_colours(small, self.dom_k)
                kps, desc = orb_descriptors(batter, self.orb_n)
                fp.batter_orb_desc = desc
                fp.batter_orb_n = len(kps) if kps else 0

        # Hitbox features
        if hbox.found:
            fp.hitbox_found = True
            fp.hitbox_area = hbox.area
            bx, by, bw, bh = hbox.bbox  # type: ignore[misc]
            fp.hitbox_aspect = bw / bh if bh else 0.0
            fp.hitbox_hu = hbox.hu_moments
            if hbox.centroid:
                fp.hitbox_centroid_frac = (hbox.centroid[0] / w,
                                           hbox.centroid[1] / h)
        return fp


#  FINGERPRINT DATABASE

class FingerprintDB:
    """Persist and retrieve per-character fingerprint samples.

    On-disk layout
      dataset/samples/<character>/<id>/
        frame.png, batter_roi.png, hitbox_roi.png,
        batter_hist.npy, batter_orb_desc.npy, metadata.json
    """

    def __init__(self, root: Path = DATASET_DIR) -> None:
        self.root = root
        self.samples = root / "samples"
        self.db_file = root / "database.json"
        self._index: Dict[str, List[Dict[str, Any]]] = {}
        self._hists:    Dict[str, List[np.ndarray]] = {}
        self._orb_desc: Dict[str, List[np.ndarray]] = {}

    def save_sample(self, fp: Fingerprint, frame: np.ndarray,
                    batter_img: Optional[np.ndarray] = None,
                    hitbox_img: Optional[np.ndarray] = None) -> str:
        slug = fp.character.lower().replace(" ", "_")
        char_dir = self.samples / slug
        sid = f"{slug}_{int(fp.timestamp * 1000)}"
        sdir = char_dir / sid
        sdir.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(sdir / "frame.png"), frame)
        if batter_img is not None and batter_img.size > 0:
            cv2.imwrite(str(sdir / "batter_roi.png"), batter_img)
        if hitbox_img is not None and hitbox_img.size > 0:
            cv2.imwrite(str(sdir / "hitbox_roi.png"), hitbox_img)

        if fp.batter_hist is not None:
            np.save(str(sdir / "batter_hist.npy"), fp.batter_hist)
        if fp.batter_orb_desc is not None:
            np.save(str(sdir / "batter_orb_desc.npy"), fp.batter_orb_desc)

        with open(sdir / "metadata.json", "w") as fh:
            json.dump(fp.to_json(), fh, indent=2)

        self._load_index()
        self._index.setdefault(slug, []).append({
            "sample_id": sid,
            "path": str(sdir.relative_to(self.root)),
            "timestamp": fp.timestamp,
        })
        self._save_index()
        print(f"[SAVED] {sid}  ->  {sdir}")
        return sid

    def load_all(self) -> Dict[str, List[Fingerprint]]:
        result: Dict[str, List[Fingerprint]] = {}
        self._hists.clear()
        self._orb_desc.clear()

        if not self.samples.exists():
            return result

        for char_dir in sorted(self.samples.iterdir()):
            if not char_dir.is_dir():
                continue
            name = char_dir.name
            result[name] = []
            self._hists[name] = []
            self._orb_desc[name] = []

            for sdir in sorted(char_dir.iterdir()):
                if not sdir.is_dir():
                    continue
                meta_path = sdir / "metadata.json"
                if not meta_path.exists():
                    continue
                with open(meta_path) as fh:
                    meta = json.load(fh)

                fp = Fingerprint(character=meta.get("character", name),
                                 ts=meta.get("timestamp", 0))
                fp.hitbox_found  = meta.get("hitbox_found", False)
                fp.hitbox_area   = meta.get("hitbox_area", 0)
                fp.hitbox_aspect = meta.get("hitbox_aspect", 0)
                fp.batter_orb_n  = meta.get("batter_orb_n", 0)
                if "batter_roi" in meta:
                    fp.batter_roi = tuple(meta["batter_roi"])
                if "strike_roi" in meta:
                    fp.strike_roi = tuple(meta["strike_roi"])
                if "hitbox_hu" in meta:
                    fp.hitbox_hu = np.array(meta["hitbox_hu"])
                if "hitbox_centroid_frac" in meta:
                    fp.hitbox_centroid_frac = tuple(meta["hitbox_centroid_frac"])
                if "batter_dominant" in meta:
                    fp.batter_dominant = np.array(meta["batter_dominant"],
                                                  dtype=np.uint8)

                hist_p = sdir / "batter_hist.npy"
                if hist_p.exists():
                    fp.batter_hist = np.load(str(hist_p))
                    self._hists[name].append(fp.batter_hist)

                orb_p = sdir / "batter_orb_desc.npy"
                if orb_p.exists():
                    fp.batter_orb_desc = np.load(str(orb_p))
                    self._orb_desc[name].append(fp.batter_orb_desc)

                result[name].append(fp)

        return result

    def avg_histograms(self) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for name, hists in self._hists.items():
            if hists:
                avg = np.mean(hists, axis=0).astype(np.float32)
                cv2.normalize(avg, avg)
                out[name] = avg
        return out

    def _load_index(self) -> None:
        if self.db_file.exists():
            with open(self.db_file) as fh:
                self._index = json.load(fh)
        else:
            self._index = {}

    def _save_index(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with open(self.db_file, "w") as fh:
            json.dump(self._index, fh, indent=2)


#  FINGERPRINT MATCHER

class MatchResult:
    __slots__ = ("character", "confidence", "hist_score", "orb_score")

    def __init__(self, character: str, confidence: float,
                 hist_score: float, orb_score: float) -> None:
        self.character  = character
        self.confidence = confidence
        self.hist_score = hist_score
        self.orb_score  = orb_score


class FingerprintMatcher:
    """Compare a query Fingerprint against the loaded database."""

    def __init__(self, db: FingerprintDB) -> None:
        self.db = db
        self.avg_hists: Dict[str, np.ndarray] = {}
        self.all_fps: Dict[str, List[Fingerprint]] = {}
        self._bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def load(self) -> None:
        self.all_fps = self.db.load_all()
        self.avg_hists = self.db.avg_histograms()
        n_char = len(self.all_fps)
        n_samp = sum(len(v) for v in self.all_fps.values())
        print(f"[DB] {n_samp} samples across {n_char} character(s)")

    def match(self, fp: Fingerprint,
              top_k: int = RECOGNITION_TOP_K) -> List[MatchResult]:
        if not self.avg_hists:
            return []

        results: List[MatchResult] = []
        for name, avg_h in self.avg_hists.items():
            h_score = 0.0
            if fp.batter_hist is not None:
                h_score = max(0.0, cv2.compareHist(
                    fp.batter_hist.astype(np.float32),
                    avg_h.astype(np.float32),
                    cv2.HISTCMP_CORREL))

            o_score = 0.0
            if (fp.batter_orb_desc is not None
                    and name in self.db._orb_desc):
                for stored in self.db._orb_desc[name]:
                    if stored is None or len(stored) < 2:
                        continue
                    try:
                        pairs = self._bf.knnMatch(fp.batter_orb_desc,
                                                  stored, k=2)
                        good = sum(1 for p in pairs
                                   if len(p) == 2
                                   and p[0].distance < 0.75 * p[1].distance)
                        total = sum(1 for p in pairs if len(p) == 2)
                        if total > 0:
                            o_score = max(o_score, good / total)
                    except cv2.error:
                        pass

            confidence = 0.70 * h_score + 0.30 * o_score
            results.append(MatchResult(name, confidence, h_score, o_score))

        results.sort(key=lambda r: r.confidence, reverse=True)
        return results[:top_k]


#  BATTER STATE CLASSIFIER  (NORMAL vs NON-NORMAL animation detection)

class BatterState(Enum):
    """Visual/animation state of the batter."""
    NORMAL     = auto()   # standard batting stance (waiting for pitch)
    NON_NORMAL = auto()   # charged swing or idle/silly animation


class BatterStateClassifier:
    """Lightweight real-time classifier: NORMAL vs NON_NORMAL batter state.

    Uses two complementary signals computed against the stored NORMAL
    reference images (from calibration captures):

      1. **Grey-scale template similarity** — resize reference to current
         crop size and compute normalised cross-correlation (NCC).  Directly
         measures pixel-level pose similarity.
      2. **HSV histogram correlation** — compares colour distribution in
         the batter region.  Robust to small spatial shifts.

    Temporal smoothing
    ------------------
      * **Hysteresis**: enter NORMAL at ``STATE_NORMAL_THRESHOLD``, leave
        only when score drops below ``STATE_LEAVE_THRESHOLD``.
      * **Majority vote**: rolling window of ``STATE_WINDOW_SIZE`` frames.
    """

    def __init__(self) -> None:
        self._ref_grey:  List[np.ndarray] = []   # grey batter crops
        self._ref_hists: List[np.ndarray] = []   # normalised HSV histograms
        self._window: deque = deque(maxlen=STATE_WINDOW_SIZE)
        self.current_state: BatterState = BatterState.NORMAL
        self.raw_score: float = 1.0              # combined similarity
        self.grey_score: float = 1.0             # best grey NCC
        self.hist_score: float = 1.0             # best histogram correl
        self.loaded: bool = False
        self._frame_counter: int = 0             # for interval-based skipping

    # Load / reset

    def load_references(self, character: str, db_root: Path) -> bool:
        """Load NORMAL-state reference images from calibration samples.

        Returns True if at least one reference was loaded.
        """
        slug = character.lower().replace(" ", "_")
        char_dir = db_root / "samples" / slug

        self._ref_grey.clear()
        self._ref_hists.clear()
        self._window.clear()
        self.current_state = BatterState.NORMAL
        self.raw_score = 1.0
        self._frame_counter = 0

        if not char_dir.exists():
            self.loaded = False
            return False

        for sdir in sorted(char_dir.iterdir()):
            if not sdir.is_dir():
                continue
            tmpl_path = sdir / "batter_roi.png"
            if not tmpl_path.exists():
                continue
            img = cv2.imread(str(tmpl_path))
            if img is None:
                continue
            grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            self._ref_grey.append(grey)
            self._ref_hists.append(hsv_histogram(img))

        self.loaded = len(self._ref_grey) >= 1
        n = len(self._ref_grey)
        print(f"[STATE] Loaded {n} NORMAL reference(s) for state classification")
        return self.loaded

    def reset(self) -> None:
        """Reset state to defaults (e.g., when tracking is turned off)."""
        self._window.clear()
        self.current_state = BatterState.NORMAL
        self.raw_score = 1.0
        self.grey_score = 1.0
        self.hist_score = 1.0

    # Per-frame classification

    def classify(self, frame: np.ndarray,
                 batter_roi_rect: Optional[Tuple[int, int, int, int]]
                 ) -> BatterState:
        """Classify the batter's current animation state.

        Parameters
        ----------
        frame : current game frame (BGR).
        batter_roi_rect : (x1, y1, x2, y2) region where the batter is
                          expected to be (typically the last NORMAL ROI).
        """
        if not self.loaded or batter_roi_rect is None:
            return BatterState.NORMAL  # no references ⇒ can't classify

        # Frame-skip: only run the expensive comparison every N frames
        self._frame_counter += 1
        if self._frame_counter % STATE_CLASSIFY_INTERVAL != 0:
            return self.current_state

        batter_crop = crop(frame, batter_roi_rect)
        if batter_crop.size == 0:
            return self.current_state

        # Downscale for speed (half resolution is fine for state detection)
        small = cv2.resize(batter_crop, (0, 0), fx=0.5, fy=0.5,
                           interpolation=cv2.INTER_AREA)
        batter_grey = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        batter_hist = hsv_histogram(small)
        bh, bw = batter_grey.shape[:2]

        best_grey = 0.0
        best_hist = 0.0

        for ref_grey, ref_hist in zip(self._ref_grey, self._ref_hists):
            # Grey NCC (resize reference to current crop size)
            ref_resized = cv2.resize(ref_grey, (bw, bh))
            res = cv2.matchTemplate(batter_grey, ref_resized,
                                    cv2.TM_CCOEFF_NORMED)
            best_grey = max(best_grey, float(res[0][0]))

            # Histogram correlation
            h_score = max(0.0, cv2.compareHist(
                batter_hist.astype(np.float32),
                ref_hist.astype(np.float32),
                cv2.HISTCMP_CORREL))
            best_hist = max(best_hist, h_score)

        self.grey_score = best_grey
        self.hist_score = best_hist

        # Weighted combination (grey is sharper discriminator)
        combined = 0.60 * best_grey + 0.40 * best_hist
        self.raw_score = combined

        # Hysteresis thresholds
        if self.current_state == BatterState.NORMAL:
            frame_is_normal = (combined >= STATE_LEAVE_THRESHOLD)
        else:
            frame_is_normal = (combined >= STATE_NORMAL_THRESHOLD)

        # Rolling-window majority vote
        self._window.append(frame_is_normal)

        if len(self._window) >= 3:
            normal_count = sum(self._window)
            if normal_count > len(self._window) / 2:
                self.current_state = BatterState.NORMAL
            else:
                self.current_state = BatterState.NON_NORMAL
        else:
            self.current_state = (
                BatterState.NORMAL if frame_is_normal
                else BatterState.NON_NORMAL)

        return self.current_state


#  MOVEMENT TRACKER  (batter tracking + strike-zone prediction & anti-flicker)

class MovementTracker:
    """Track the batter's position and predict / stabilise the strike zone.

    Workflow
    --------
    1. ``load_calibration()`` reads the saved calibration samples
       (batter_roi, strike_roi centres) and builds:

       - A set of grey-scale batter templates for template-matching.
       - An affine model  strike_centre = f(batter_centre)  via least-squares.
       - A wide search region covering every possible strike-zone position.

    2. ``update(frame, detector)`` is called every frame:

       a. Template-match the batter to find its current centre.
       b. Predict strike-zone centre & ROI from the affine model.
       c. Run the gold-hitbox detector inside the predicted ROI.
       d. **Anti-flicker**: if the hitbox blinks off, hold the last known
          position for up to ``STRIKE_HOLD_FRAMES`` frames, continuously
          updating the held centre from the batter prediction so it
          tracks movement even during blinks.
    """

    def __init__(self) -> None:
        # Calibration data
        self._cal_batter_centers: List[Tuple[int, int]] = []
        self._cal_strike_centers: List[Tuple[int, int]] = []
        self._cal_batter_rois: List[Tuple[int, int, int, int]] = []
        self._cal_strike_rois: List[Tuple[int, int, int, int]] = []
        self._cal_strike_sizes: List[Tuple[int, int]] = []
        self._templates: List[np.ndarray] = []          # grey-scale batter crops
        self._template_sizes: List[Tuple[int, int]] = []  # (w, h)
        self._templates_ds: List[np.ndarray] = []         # downscaled templates
        self._template_sizes_ds: List[Tuple[int, int]] = []

        # Batter search region (union of all calibration batter ROIs + margin)
        self._search_region: Optional[Tuple[int, int, int, int]] = None
        # Wide strike-zone search (union of all calibration strike ROIs + pad)
        self._wide_strike_search: Optional[Tuple[int, int, int, int]] = None

        # Affine model coefficients: strike = [bx, by, 1] @ coeff
        self._sx_coeff: Optional[np.ndarray] = None
        self._sy_coeff: Optional[np.ndarray] = None

        # Per-frame tracking state
        self.batter_center: Optional[Tuple[int, int]] = None
        self.batter_roi_pred: Optional[Tuple[int, int, int, int]] = None
        self.batter_match_score: float = 0.0

        self.strike_center_pred: Optional[Tuple[int, int]] = None
        self.strike_roi_pred: Optional[Tuple[int, int, int, int]] = None
        self.strike_roi_search: Optional[Tuple[int, int, int, int]] = None

        # Strike-zone stabiliser (anti-flicker)
        self.sz_detected: bool = False
        self.sz_held: bool = False
        self.sz_centroid: Optional[Tuple[int, int]] = None
        self.sz_contour: Optional[np.ndarray] = None
        self.sz_bbox: Optional[Tuple[int, int, int, int]] = None
        self.sz_confidence: float = 0.0
        self.sz_frames_since_detect: int = 999

        # Last stable NORMAL-state positions
        self._last_normal_center: Optional[Tuple[int, int]] = None
        self._last_normal_roi: Optional[Tuple[int, int, int, int]] = None
        self._last_normal_score: float = 0.0

        self.loaded: bool = False
        self.character: str = ""

    # Load calibration

    def load_calibration(self, character: str, db_root: Path) -> bool:
        """Load calibration from ``dataset/samples/<character>/``.

        Returns True if >= 2 valid samples (with both ROIs and a batter
        image) were found.
        """
        self.character = character
        slug = character.lower().replace(" ", "_")
        char_dir = db_root / "samples" / slug

        if not char_dir.exists():
            print(f"[TRACK] No samples directory for '{character}'")
            return False

        self._cal_batter_centers.clear()
        self._cal_strike_centers.clear()
        self._cal_batter_rois.clear()
        self._cal_strike_rois.clear()
        self._cal_strike_sizes.clear()
        self._templates.clear()
        self._template_sizes.clear()
        self._templates_ds.clear()
        self._template_sizes_ds.clear()

        for sdir in sorted(char_dir.iterdir()):
            if not sdir.is_dir():
                continue
            meta_path = sdir / "metadata.json"
            if not meta_path.exists():
                continue
            with open(meta_path) as fh:
                meta = json.load(fh)
            if "batter_roi" not in meta or "strike_roi" not in meta:
                continue

            br = tuple(meta["batter_roi"])
            sr = tuple(meta["strike_roi"])
            bc = ((br[0] + br[2]) // 2, (br[1] + br[3]) // 2)
            sc = ((sr[0] + sr[2]) // 2, (sr[1] + sr[3]) // 2)

            self._cal_batter_centers.append(bc)
            self._cal_strike_centers.append(sc)
            self._cal_batter_rois.append(br)
            self._cal_strike_rois.append(sr)
            self._cal_strike_sizes.append((sr[2] - sr[0], sr[3] - sr[1]))

            # Load batter template (grey-scale for fast matching)
            tmpl_path = sdir / "batter_roi.png"
            if tmpl_path.exists():
                img = cv2.imread(str(tmpl_path))
                if img is not None:
                    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    self._templates.append(grey)
                    self._template_sizes.append((grey.shape[1], grey.shape[0]))
                    # Pre-compute downscaled template
                    ds = TRACK_DOWNSCALE
                    grey_ds = cv2.resize(
                        grey, (grey.shape[1] // ds, grey.shape[0] // ds),
                        interpolation=cv2.INTER_AREA)
                    self._templates_ds.append(grey_ds)
                    self._template_sizes_ds.append(
                        (grey_ds.shape[1], grey_ds.shape[0]))

        n = len(self._cal_batter_centers)
        if n < 2:
            print(f"[TRACK] Need >= 2 calibration samples, found {n}")
            return False

        margin = BATTER_SEARCH_MARGIN
        self._search_region = (
            max(0, min(r[0] for r in self._cal_batter_rois) - margin),
            max(0, min(r[1] for r in self._cal_batter_rois) - margin),
            max(r[2] for r in self._cal_batter_rois) + margin,
            max(r[3] for r in self._cal_batter_rois) + margin,
        )

        pad = STRIKE_SEARCH_PADDING + 20
        self._wide_strike_search = (
            max(0, min(r[0] for r in self._cal_strike_rois) - pad),
            max(0, min(r[1] for r in self._cal_strike_rois) - pad),
            max(r[2] for r in self._cal_strike_rois) + pad,
            max(r[3] for r in self._cal_strike_rois) + pad,
        )

        self._fit_model()
        self.loaded = True

        # Initialise last-normal from the first calibration sample so
        # the state classifier has a starting ROI on the very first frame.
        self._last_normal_center = self._cal_batter_centers[0]
        self._last_normal_roi = self._cal_batter_rois[0]
        self._last_normal_score = 0.0

        print(f"[TRACK] Loaded {n} calibration points for '{character}'")
        print(f"[TRACK] Batter search : {self._search_region}")
        print(f"[TRACK] SZ wide search: {self._wide_strike_search}")
        return True

    def _fit_model(self) -> None:
        """Least-squares affine: strike_centre = f(batter_centre)."""
        A = np.array([[bc[0], bc[1], 1.0]
                       for bc in self._cal_batter_centers])
        sx = np.array([sc[0] for sc in self._cal_strike_centers],
                      dtype=np.float64)
        sy = np.array([sc[1] for sc in self._cal_strike_centers],
                      dtype=np.float64)

        self._sx_coeff, *_ = np.linalg.lstsq(A, sx, rcond=None)
        self._sy_coeff, *_ = np.linalg.lstsq(A, sy, rcond=None)

        pred_sx = A @ self._sx_coeff
        pred_sy = A @ self._sy_coeff
        err = np.sqrt((pred_sx - sx) ** 2 + (pred_sy - sy) ** 2)
        print(f"[TRACK] Affine model mean err: {err.mean():.1f} px, "
              f"max: {err.max():.1f} px")

    # Per-frame update

    def update(self, frame: np.ndarray,
               detector: HitboxDetector,
               batter_state: "BatterState | None" = None,
               ) -> HitboxResult:
        """Process one frame.  Returns the HitboxResult (may be from the
        predicted search region).  Check ``sz_*`` attributes for the
        stabilised strike-zone state.

        Parameters
        ----------
        batter_state : If provided and != NORMAL, the tracker will NOT
                       update the batter position from template matching.
                       Instead it freezes at the last known NORMAL position
                       (state-gated update).
        """
        if not self.loaded:
            return detector.detect(frame)

        h, w = frame.shape[:2]

        # 1) State-gated position update
        is_normal = (batter_state is None
                     or batter_state == BatterState.NORMAL)

        if is_normal:
            # Only run expensive template matching when NORMAL
            self._find_batter(frame)
            # Accept the template-match result; save as last-known NORMAL
            if self.batter_center is not None:
                self._last_normal_center = self.batter_center
                self._last_normal_roi = self.batter_roi_pred
                self._last_normal_score = self.batter_match_score
        else:
            # NON_NORMAL — skip template matching entirely, use frozen pos
            if self._last_normal_center is not None:
                self.batter_center = self._last_normal_center
                self.batter_roi_pred = self._last_normal_roi
                self.batter_match_score = self._last_normal_score

        # 3) Predict strike-zone from (gated) batter position
        if self.batter_center is not None:
            self._predict_strike(self.batter_center, w, h)

        # 4) Detect gold hitbox.  Use narrowed search if prediction is
        #    available, otherwise fall back to the wide calibrated region.
        search_roi = self.strike_roi_search
        if search_roi is None:
            search_roi = self._wide_strike_search
        hbox = detector.detect(frame, search_roi)

        # 5) Stabilise against blinks
        self._stabilise(hbox)

        return hbox

    # Template matching

    def _find_batter(self, frame: np.ndarray) -> None:
        """Multi-template match within the calibrated search region.

        Uses downscaled templates and search image (``TRACK_DOWNSCALE``) for
        speed, then maps the result back to full-resolution coordinates.
        """
        if not self._templates_ds or self._search_region is None:
            return

        ds = TRACK_DOWNSCALE
        search_img = crop(frame, self._search_region)
        if search_img.size == 0:
            return
        search_grey = cv2.cvtColor(search_img, cv2.COLOR_BGR2GRAY)
        # Downscale the search region
        search_ds = cv2.resize(
            search_grey,
            (search_grey.shape[1] // ds, search_grey.shape[0] // ds),
            interpolation=cv2.INTER_AREA)
        sh_ds, sw_ds = search_ds.shape[:2]
        ox, oy = self._search_region[0], self._search_region[1]

        best_score = -1.0
        best_center: Optional[Tuple[int, int]] = None
        best_roi: Optional[Tuple[int, int, int, int]] = None

        for tmpl_ds, (tw, th), (tw_full, th_full) in zip(
                self._templates_ds, self._template_sizes_ds,
                self._template_sizes):
            if th >= sh_ds or tw >= sw_ds:
                continue
            res = cv2.matchTemplate(search_ds, tmpl_ds,
                                    cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            if max_val > best_score:
                best_score = max_val
                # Map back to full-resolution coordinates
                fx = max_loc[0] * ds
                fy = max_loc[1] * ds
                cx = fx + tw_full // 2 + ox
                cy = fy + th_full // 2 + oy
                best_center = (cx, cy)
                best_roi = (fx + ox, fy + oy,
                            fx + tw_full + ox, fy + th_full + oy)

        self.batter_match_score = best_score
        if best_score >= 0.25:
            self.batter_center = best_center
            self.batter_roi_pred = best_roi

    # Strike-zone prediction

    def _predict_strike(self, bc: Tuple[int, int],
                        fw: int, fh: int) -> None:
        """Predict strike-zone centre & ROI from batter centre."""
        bx, by = bc
        inp = np.array([bx, by, 1.0])
        px = int(inp @ self._sx_coeff)
        py = int(inp @ self._sy_coeff)
        self.strike_center_pred = (px, py)

        # Interpolate strike-zone size via IDW
        sw, sh_ = self._idw_strike_size(bc)
        hx, hy = sw // 2, sh_ // 2
        self.strike_roi_pred = (
            max(0, px - hx), max(0, py - hy),
            min(fw, px + hx), min(fh, py + hy))

        pad = STRIKE_SEARCH_PADDING
        self.strike_roi_search = (
            max(0, px - hx - pad), max(0, py - hy - pad),
            min(fw, px + hx + pad), min(fh, py + hy + pad))

    def _idw_strike_size(self, bc: Tuple[int, int]) -> Tuple[int, int]:
        """Inverse-distance-weighted interpolation of strike-zone size."""
        eps = 1e-3
        weights: List[float] = []
        for cal_bc in self._cal_batter_centers:
            d = max(eps, np.hypot(bc[0] - cal_bc[0], bc[1] - cal_bc[1]))
            weights.append(1.0 / d ** 2)
        tw = sum(weights)
        sw = int(sum(w * s[0]
                     for w, s in zip(weights, self._cal_strike_sizes)) / tw)
        sh = int(sum(w * s[1]
                     for w, s in zip(weights, self._cal_strike_sizes)) / tw)
        return (sw, sh)

    # Anti-flicker stabiliser

    def _stabilise(self, hbox: HitboxResult) -> None:
        """Update the stabilised strike-zone state.

        When the gold hitbox is detected: store it as ground truth.
        When it blinks off: hold the last/predicted position for up to
        ``STRIKE_HOLD_FRAMES`` frames with decaying confidence, updating
        the held centroid from the batter-based prediction so it follows
        movement even during blinks.
        """
        if hbox.found:
            self.sz_detected = True
            self.sz_held = False
            self.sz_centroid = hbox.centroid
            self.sz_contour = hbox.contour
            self.sz_bbox = hbox.bbox
            self.sz_confidence = hbox.confidence
            self.sz_frames_since_detect = 0
        else:
            self.sz_detected = False
            self.sz_frames_since_detect += 1

            if (self.sz_centroid is not None
                    and self.sz_frames_since_detect <= STRIKE_HOLD_FRAMES):
                self.sz_held = True
                self.sz_confidence *= STRIKE_CONFIDENCE_DECAY
                # Update held centroid from prediction so it tracks movement
                if self.strike_center_pred is not None:
                    self.sz_centroid = self.strike_center_pred
            else:
                self.sz_held = False
                self.sz_confidence = 0.0
                self.sz_centroid = None
                self.sz_contour = None
                self.sz_bbox = None


#  VISUALISER

class Visualiser:
    """Overlay drawing and debug-panel building."""

    @staticmethod
    def overlay(frame: np.ndarray, hbox: HitboxResult,
                matches: List[MatchResult], mode: str,
                char_label: str, fps: float,
                user_char_roi: Optional[Tuple[int, int, int, int]] = None,
                user_strike_roi: Optional[Tuple[int, int, int, int]] = None,
                tracker: Optional[MovementTracker] = None,
                batter_state: "BatterState | None" = None,
                state_score: float = 0.0,
                ) -> np.ndarray:
        vis = frame.copy()
        h, w = vis.shape[:2]
        tracking = tracker is not None and tracker.loaded

        # Tracking overlay
        if tracking:
            # Batter tracking box (colour reflects state)
            is_normal = (batter_state is None
                         or batter_state == BatterState.NORMAL)
            if tracker.batter_roi_pred is not None:
                bx1, by1, bx2, by2 = tracker.batter_roi_pred
                box_col = COL_CYAN if is_normal else COL_ORANGE
                cv2.rectangle(vis, (bx1, by1), (bx2, by2), box_col, 2)
                state_lbl = "NORMAL" if is_normal else "NON-NORMAL (HELD)"
                put_text(vis,
                         f"BATTER  {state_lbl}  "
                         f"score={state_score:.2f}",
                         (bx1, by1 - 8), 0.42, box_col, 1)

            # Predicted strike-zone ROI (dashed yellow)
            if tracker.strike_roi_pred is not None:
                sx1, sy1, sx2, sy2 = tracker.strike_roi_pred
                draw_dashed_rect(vis, (sx1, sy1), (sx2, sy2),
                                 COL_YELLOW, 1, dash=8, gap=5)
                put_text(vis, "PREDICTED SZ",
                         (sx1, sy1 - 8), 0.38, COL_YELLOW, 1)

            # Strike zone — detected (green) or held (orange)
            if tracker.sz_detected and hbox.found:
                # Solid green — real detection
                cv2.drawContours(vis, [hbox.contour], -1, COL_GREEN, 2)
                bx, by, bw, bh = hbox.bbox
                cv2.rectangle(vis, (bx, by), (bx + bw, by + bh),
                              COL_GREEN, 2)
                if hbox.centroid:
                    cv2.circle(vis, hbox.centroid, 6, COL_RED, -1)
                    cv2.circle(vis, hbox.centroid, 10, COL_GREEN, 2)
                put_text(vis,
                         f"SZ DETECTED  {hbox.confidence:.0%}  "
                         f"area={hbox.area:.0f}",
                         (bx, by - 10), 0.48, COL_GREEN, 1)
            elif tracker.sz_held and tracker.sz_centroid is not None:
                # Orange — blink-off, holding last known / predicted
                cx, cy = tracker.sz_centroid
                cv2.drawMarker(vis, (cx, cy), COL_ORANGE,
                               cv2.MARKER_CROSS, 24, 2)
                cv2.circle(vis, (cx, cy), 10, COL_ORANGE, 2)
                put_text(vis,
                         f"SZ HELD  {tracker.sz_confidence:.0%}  "
                         f"blink-off {tracker.sz_frames_since_detect}f",
                         (cx + 14, cy - 6), 0.42, COL_ORANGE, 1)
            else:
                put_text(vis, "SZ: searching...",
                         (10, h - 55), 0.45, COL_RED)

            # Tracking status line
            st_col = COL_GREEN if tracker.sz_detected else COL_ORANGE
            status = "TRACKING"
            if not is_normal:
                status += "  |  BATTER: HELD"
            if tracker.sz_detected:
                status += "  |  SZ: DETECTED"
            elif tracker.sz_held:
                status += (f"  |  SZ: HELD "
                           f"({tracker.sz_frames_since_detect}f)")
            else:
                status += "  |  SZ: LOST"
            put_text(vis, status, (10, h - 32), 0.42, st_col, 1)

        else:
            # Static ROI display (non-tracking)
            if user_char_roi is not None:
                x1, y1, x2, y2 = user_char_roi
                cv2.rectangle(vis, (x1, y1), (x2, y2), COL_CYAN, 2)
                put_text(vis, "CHARACTER",
                         (x1, y1 - 8), 0.45, COL_CYAN, 1)

            if user_strike_roi is not None:
                x1, y1, x2, y2 = user_strike_roi
                cv2.rectangle(vis, (x1, y1), (x2, y2), COL_YELLOW, 2)
                put_text(vis, "STRIKE-ZONE",
                         (x1, y1 - 8), 0.45, COL_YELLOW, 1)

            # Hitbox detection result (within the strike-zone ROI)
            if hbox.found:
                cv2.drawContours(vis, [hbox.contour], -1, COL_GREEN, 2)
                bx, by, bw, bh = hbox.bbox  # type: ignore[misc]
                cv2.rectangle(vis, (bx, by), (bx + bw, by + bh),
                              COL_GREEN, 2)
                if hbox.centroid:
                    cv2.circle(vis, hbox.centroid, 5, COL_RED, -1)
                    cv2.circle(vis, hbox.centroid, 8, COL_GREEN, 2)
                put_text(vis, f"HITBOX  {hbox.confidence:.0%}  "
                              f"area={hbox.area:.0f}",
                         (bx, by - 10), 0.48, COL_GREEN, 1)
            else:
                if user_strike_roi is not None:
                    put_text(vis,
                             "HITBOX: not detected in strike-zone ROI",
                             (10, h - 50), 0.45, COL_RED)
                else:
                    put_text(vis,
                             "No strike-zone ROI set -- press [C] to define",
                             (10, h - 50), 0.45, COL_YELLOW)

        # Recognition results
        y = 28
        if mode == "recognize" and matches:
            put_text(vis, "RECOGNITION", (10, y), 0.55, COL_WHITE)
            y += 26
            for i, m in enumerate(matches):
                col = (COL_GREEN if i == 0
                       and m.confidence >= HIST_MATCH_THRESHOLD
                       else COL_YELLOW)
                txt = (f" {i+1}. {m.character}  "
                       f"{m.confidence:.1%}  "
                       f"(H:{m.hist_score:.2f} O:{m.orb_score:.2f})")
                put_text(vis, txt, (10, y), 0.48, col, 1)
                y += 22
            if matches[0].confidence < HIST_MATCH_THRESHOLD:
                put_text(vis, "  low confidence -- unknown?",
                         (10, y), 0.42, COL_RED, 1)
                y += 22
        elif mode == "collect":
            lbl = (f"Collecting: {char_label}" if char_label
                   else "COLLECT -- press [C] to capture, [N] to set name")
            put_text(vis, lbl, (10, y), 0.50, COL_CYAN)
            y += 26

        # Status bar
        put_text(vis, f"FPS {fps:.0f}", (w - 110, 26), 0.50, COL_GREEN, 1)
        mode_label = mode.upper()
        if tracking:
            mode_label += " + TRACK"
        put_text(vis, f"MODE {mode_label}", (w - 240, 50), 0.45, COL_WHITE, 1)
        hints = "[C]apture [N]ame [T]rack [R]ecog [D]ebug [Q]uit"
        put_text(vis, hints, (10, h - 8), 0.38, COL_WHITE, 1)
        return vis

    @staticmethod
    def debug_panel(frame: np.ndarray, hbox: HitboxResult,
                    fp: Optional[Fingerprint] = None,
                    user_char_roi: Optional[Tuple[int, int, int, int]] = None,
                    user_strike_roi: Optional[Tuple[int, int, int, int]] = None,
                    ) -> np.ndarray:
        """Build the debug panel.

        When a user-defined strike-zone ROI exists, shows:
          1. HSV mask within that ROI
          2. Strike-zone ROI with ALL detected contour outlines
          3. Canny edges within the strike-zone ROI
        Also shows batter ROI, dominant colours, and ORB key-points.
        """
        h, w = frame.shape[:2]
        PW, PH = 280, 220
        panels: List[np.ndarray] = []

        def _tile(img: np.ndarray, label: str,
                  col: Tuple[int, ...] = COL_GREEN) -> np.ndarray:
            if img.size == 0:
                t = np.zeros((PH, PW, 3), np.uint8)
            else:
                t = cv2.resize(img, (PW, PH))
            if t.ndim == 2:
                t = cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)
            put_text(t, label, (4, 18), 0.45, col, 1)
            return t

        # Strike-zone panels (only when ROI is set)
        if user_strike_roi is not None:
            sz_crop = crop(frame, user_strike_roi)
            if sz_crop.size > 0:
                # 1) HSV mask
                if hbox.mask is not None:
                    panels.append(_tile(hbox.mask, "SZ HSV mask", COL_YELLOW))
                else:
                    # Compute it ourselves
                    hsv = cv2.cvtColor(sz_crop, cv2.COLOR_BGR2HSV)
                    m = cv2.inRange(hsv, HITBOX_HSV_LOWER, HITBOX_HSV_UPPER)
                    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kern, iterations=2)
                    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  kern, iterations=1)
                    panels.append(_tile(m, "SZ HSV mask", COL_YELLOW))

                # 2) Strike-zone ROI with contour outlines
                sz_vis = sz_crop.copy()
                # Draw ALL valid contours (yellow) and best (green)
                if hbox.all_contours_local:
                    for cnt in hbox.all_contours_local:
                        cv2.drawContours(sz_vis, [cnt], -1, COL_YELLOW, 1)
                if hbox.found and hbox.contour_local is not None:
                    cv2.drawContours(sz_vis, [hbox.contour_local], -1,
                                     COL_GREEN, 2)
                    # Draw convex hull for shape reference
                    hull = cv2.convexHull(hbox.contour_local)
                    cv2.drawContours(sz_vis, [hull], -1, COL_MAGENTA, 1)
                n_cnt = len(hbox.all_contours_local) if hbox.all_contours_local else 0
                panels.append(_tile(sz_vis,
                                    f"SZ contours ({n_cnt})", COL_GREEN))

                # 3) Canny edges on strike-zone
                edges = cv2.Canny(cv2.cvtColor(sz_crop, cv2.COLOR_BGR2GRAY),
                                  50, 150)
                panels.append(_tile(edges, "SZ edges", COL_YELLOW))

        # Batter panels (only when ROI is set)
        bcrop = None
        if user_char_roi is not None:
            bcrop = crop(frame, user_char_roi)
            if bcrop.size > 0:
                panels.append(_tile(bcrop, "Batter ROI", COL_CYAN))

        # Dominant-colour swatch
        if fp and fp.batter_dominant is not None:
            swatch = np.zeros((PH, PW, 3), np.uint8)
            n = len(fp.batter_dominant)
            sh = PH // max(n, 1)
            for i, c in enumerate(fp.batter_dominant):
                y1 = i * sh
                y2 = (i + 1) * sh if i < n - 1 else PH
                swatch[y1:y2] = c
                hex_str = f"#{c[2]:02X}{c[1]:02X}{c[0]:02X}"
                tc = COL_WHITE if np.mean(c) < 128 else COL_BLACK
                cv2.putText(swatch, hex_str, (5, y1 + sh // 2 + 5),
                            FONT, 0.40, tc, 1, cv2.LINE_AA)
            put_text(swatch, "Dominant colours", (4, 16), 0.42, COL_WHITE, 1)
            panels.append(swatch)

        # Canny edges on batter
        if bcrop is not None and bcrop.size > 0:
            edges = cv2.Canny(cv2.cvtColor(bcrop, cv2.COLOR_BGR2GRAY),
                              50, 150)
            panels.append(_tile(edges, "Batter edges", COL_CYAN))

        # ORB key-points on batter
        if bcrop is not None and bcrop.size > 0:
            kps, _ = orb_descriptors(bcrop, 200)
            kp_vis = cv2.drawKeypoints(bcrop, kps, None,
                                       color=COL_GREEN,
                                       flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            panels.append(_tile(kp_vis,
                                f"ORB kps ({len(kps) if kps else 0})",
                                COL_CYAN))

        if not panels:
            blank = np.zeros((PH, PW * 2, 3), np.uint8)
            put_text(blank, "No ROIs defined -- press [C] first",
                     (10, PH // 2), 0.50, COL_YELLOW)
            return blank

        # Arrange into a grid (3 columns)
        cols = 3
        while len(panels) % cols != 0:
            panels.append(np.zeros((PH, PW, 3), np.uint8))
        rows = []
        for r in range(0, len(panels), cols):
            rows.append(np.hstack(panels[r:r + cols]))
        return np.vstack(rows)


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
        print("[ERROR] dxcam.create() returned None -- check GPU/drivers.")
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
    while True:
        for f in files:
            img = cv2.imread(str(f))
            if img is not None:
                yield img


def source_video(path: str):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open {path}")
        sys.exit(1)
    print(f"[INFO] Video: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))} frames "
          f"@ {cap.get(cv2.CAP_PROP_FPS):.1f} FPS")
    while True:
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
            if not ok:
                break
        yield frame
    cap.release()


#  MAIN

def main() -> None:
    ap = argparse.ArgumentParser(
        description="MSB Hitbox Detector & Character Fingerprint System")
    ap.add_argument("--mode", choices=["collect", "recognize"],
                    default="collect")
    ap.add_argument("--source", choices=["live", "folder", "video"],
                    default="live")
    ap.add_argument("--input", default=None,
                    help="Path to image folder / video file")
    ap.add_argument("--character", default="",
                    help="Pre-set character name for collect mode")
    ap.add_argument("--dataset", default=str(DATASET_DIR),
                    help="Dataset root directory")
    args = ap.parse_args()

    ds_root = Path(args.dataset)

    # Components
    detector      = HitboxDetector()
    fingerprinter = CharacterFingerprinter()
    db            = FingerprintDB(ds_root)
    vis           = Visualiser()
    clicker       = ClickCollector(display_scale=DISPLAY_SCALE)

    mode: str       = args.mode
    char_label: str = args.character
    show_debug: bool = False

    # Persistent user-defined ROIs (survive across frames, updated on capture)
    user_char_roi:   Optional[Tuple[int, int, int, int]] = None
    user_strike_roi: Optional[Tuple[int, int, int, int]] = None
    frozen_frame: Optional[np.ndarray] = None  # frame frozen during click mode

    # Movement tracker
    tracker = MovementTracker()
    tracking_active: bool = False

    # Batter state classifier
    state_classifier = BatterStateClassifier()
    batter_state: BatterState = BatterState.NORMAL

    # Matcher (recognition mode)
    matcher: Optional[FingerprintMatcher] = None
    if mode == "recognize":
        matcher = FingerprintMatcher(db)
        matcher.load()
        if not matcher.avg_hists:
            print("[WARN] Database empty -- switching to collect mode.")
            mode = "collect"

    # Frame source
    print(f"[INFO] mode={mode}  source={args.source}")
    if args.source == "live":
        src = source_live()
    elif args.source == "folder":
        if not args.input:
            print("[ERROR] --input required for folder source"); sys.exit(1)
        src = source_folder(args.input)
    else:
        if not args.input:
            print("[ERROR] --input required for video source"); sys.exit(1)
        src = source_video(args.input)

    # FPS tracking
    prev_t = time.perf_counter()
    fps_ema = float(TARGET_FPS)

    print()
    print(" CONTROLS")
    print("  C  = Capture sample (freeze + click ROIs)")
    print("  N  = Set / change character name")
    print("  T  = Toggle TRACKING (needs calibration samples)")
    print("  R  = Toggle recognise <-> collect")
    print("  D  = Toggle debug panel")
    print("  Q / ESC = Quit  (ESC also cancels click mode)")
    print()

    cv2.namedWindow(WINDOW_MAIN, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_MAIN, clicker.mouse_callback)

    try:
        for frame in src:
            if frame is None:
                time.sleep(0.001)
                continue

            # FPS
            now = time.perf_counter()
            dt = now - prev_t
            prev_t = now
            ifps = 1.0 / dt if dt > 0 else 0.0
            fps_ema = 0.1 * ifps + 0.9 * fps_ema

            # Click mode: frozen frame
            if clicker.active:
                if frozen_frame is not None:
                    click_disp = clicker.draw_overlay(frozen_frame)
                    if DISPLAY_SCALE != 1.0:
                        dw = int(click_disp.shape[1] * DISPLAY_SCALE)
                        dh = int(click_disp.shape[0] * DISPLAY_SCALE)
                        click_disp = cv2.resize(click_disp, (dw, dh))
                    cv2.imshow(WINDOW_MAIN, click_disp)

                key = cv2.waitKey(30) & 0xFF
                if key == 27:  # ESC cancels
                    clicker.cancel()
                    frozen_frame = None
                    print("[INFO] Capture cancelled.")
                continue

            # Click mode just finished
            if clicker.phase == ClickPhase.DONE:
                user_char_roi   = clicker.char_rect
                user_strike_roi = clicker.strike_rect
                clicker.phase   = ClickPhase.IDLE

                print(f"[ROI] Character:   {user_char_roi}")
                print(f"[ROI] Strike-zone: {user_strike_roi}")

                # Run detection on the frozen frame
                if frozen_frame is not None:
                    # State gate: reject capture if NOT normal
                    if (state_classifier.loaded
                            and user_char_roi is not None):
                        cap_state = state_classifier.classify(
                            frozen_frame, user_char_roi)
                        if cap_state != BatterState.NORMAL:
                            reject = frozen_frame.copy()
                            cv2.rectangle(
                                reject, (0, 0),
                                (reject.shape[1] - 1,
                                 reject.shape[0] - 1),
                                COL_RED, 10)
                            put_text(
                                reject,
                                "CAPTURE REJECTED",
                                (reject.shape[1] // 6,
                                 reject.shape[0] // 2 - 30),
                                1.0, COL_RED, 3)
                            put_text(
                                reject,
                                "Batter not in NORMAL stance",
                                (reject.shape[1] // 6,
                                 reject.shape[0] // 2 + 20),
                                0.7, COL_ORANGE, 2)
                            put_text(
                                reject,
                                f"State score: "
                                f"{state_classifier.raw_score:.2f}",
                                (reject.shape[1] // 6,
                                 reject.shape[0] // 2 + 55),
                                0.6, COL_WHITE, 1)
                            if DISPLAY_SCALE != 1.0:
                                dw = int(reject.shape[1]
                                         * DISPLAY_SCALE)
                                dh = int(reject.shape[0]
                                         * DISPLAY_SCALE)
                                reject = cv2.resize(
                                    reject, (dw, dh))
                            cv2.imshow(WINDOW_MAIN, reject)
                            cv2.waitKey(1500)
                            print(
                                "[WARN] Capture REJECTED: "
                                "batter not in NORMAL stance "
                                f"(score="
                                f"{state_classifier.raw_score:.2f})")
                            frozen_frame = None
                            continue

                    hbox = detector.detect(frozen_frame, user_strike_roi)
                    fp = fingerprinter.extract(
                        frozen_frame, hbox,
                        batter_rect=user_char_roi,
                        strike_rect=user_strike_roi,
                        character=char_label)

                    # Save
                    batter_img = (crop(frozen_frame, user_char_roi)
                                  if user_char_roi else None)
                    hitbox_img = (crop(frozen_frame, user_strike_roi)
                                  if user_strike_roi else None)
                    db.save_sample(fp, frozen_frame, batter_img, hitbox_img)

                    # Green flash feedback
                    flash = frozen_frame.copy()
                    if user_char_roi:
                        cv2.rectangle(flash,
                                      (user_char_roi[0], user_char_roi[1]),
                                      (user_char_roi[2], user_char_roi[3]),
                                      COL_CYAN, 3)
                    if user_strike_roi:
                        cv2.rectangle(flash,
                                      (user_strike_roi[0], user_strike_roi[1]),
                                      (user_strike_roi[2], user_strike_roi[3]),
                                      COL_YELLOW, 3)
                    cv2.rectangle(flash, (0, 0),
                                  (flash.shape[1] - 1, flash.shape[0] - 1),
                                  COL_GREEN, 10)
                    put_text(flash, f"SAVED: {char_label}",
                             (flash.shape[1] // 5, flash.shape[0] // 2),
                             1.0, COL_GREEN, 3)
                    if DISPLAY_SCALE != 1.0:
                        dw = int(flash.shape[1] * DISPLAY_SCALE)
                        dh = int(flash.shape[0] * DISPLAY_SCALE)
                        flash = cv2.resize(flash, (dw, dh))
                    cv2.imshow(WINDOW_MAIN, flash)
                    cv2.waitKey(400)

                    # Auto-reload calibration if tracking is active
                    if tracking_active:
                        tracker.load_calibration(char_label, ds_root)
                        state_classifier.load_references(
                            char_label, ds_root)

                frozen_frame = None
                continue

            if clicker.phase == ClickPhase.CANCELLED:
                clicker.phase = ClickPhase.IDLE
                frozen_frame = None

            # Normal pipeline
            if tracking_active and tracker.loaded:
                # Tracking pipeline
                # Classify batter state (uses last-normal ROI)
                classify_roi = (tracker._last_normal_roi
                                or tracker.batter_roi_pred
                                or tracker._search_region)
                if state_classifier.loaded and classify_roi:
                    batter_state = state_classifier.classify(
                        frame, classify_roi)
                else:
                    batter_state = BatterState.NORMAL
                hbox = tracker.update(frame, detector, batter_state)
                # For fingerprinting, use tracked ROIs
                fp = fingerprinter.extract(
                    frame, hbox,
                    batter_rect=tracker.batter_roi_pred,
                    strike_rect=(
                        tracker.strike_roi_pred
                        if tracker.strike_roi_pred
                        else user_strike_roi),
                    character=char_label)
            else:
                # Static ROI pipeline
                batter_state = BatterState.NORMAL
                hbox = detector.detect(frame, user_strike_roi)
                fp = fingerprinter.extract(
                    frame, hbox,
                    batter_rect=user_char_roi,
                    strike_rect=user_strike_roi,
                    character=char_label)

            matches: List[MatchResult] = []
            if mode == "recognize" and matcher and matcher.avg_hists:
                matches = matcher.match(fp)

            # Draw main overlay
            active_tracker = tracker if tracking_active else None
            disp = vis.overlay(frame, hbox, matches, mode,
                               char_label, fps_ema,
                               user_char_roi, user_strike_roi,
                               tracker=active_tracker,
                               batter_state=(
                                   batter_state
                                   if tracking_active else None),
                               state_score=(
                                   state_classifier.raw_score
                                   if tracking_active else 0.0))

            if DISPLAY_SCALE != 1.0:
                dw = int(disp.shape[1] * DISPLAY_SCALE)
                dh = int(disp.shape[0] * DISPLAY_SCALE)
                disp = cv2.resize(disp, (dw, dh))
            cv2.imshow(WINDOW_MAIN, disp)

            if show_debug:
                dbg_char = (tracker.batter_roi_pred
                            if tracking_active and tracker.loaded
                            else user_char_roi)
                dbg_strike = (tracker.strike_roi_search
                              if tracking_active and tracker.loaded
                              else user_strike_roi)
                dbg = vis.debug_panel(frame, hbox, fp,
                                      dbg_char, dbg_strike)
                cv2.imshow(WINDOW_DEBUG, dbg)

            # Keys
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break

            elif key == ord("d"):
                show_debug = not show_debug
                if not show_debug:
                    try:
                        cv2.destroyWindow(WINDOW_DEBUG)
                    except cv2.error:
                        pass
                print(f"[INFO] Debug panel {'ON' if show_debug else 'OFF'}")

            elif key == ord("r"):
                if mode == "collect":
                    mode = "recognize"
                    if matcher is None:
                        matcher = FingerprintMatcher(db)
                    matcher.load()
                    if not matcher.avg_hists:
                        print("[WARN] DB empty -- staying in collect mode.")
                        mode = "collect"
                    else:
                        print("[INFO] -> RECOGNIZE mode")
                else:
                    mode = "collect"
                    print("[INFO] -> COLLECT mode")

            elif key == ord("n"):
                cv2.setWindowTitle(WINDOW_MAIN,
                                   "Type character name in terminal...")
                char_label = input("[INPUT] Character name: ").strip()
                print(f"[INFO] Character = '{char_label}'")
                cv2.setWindowTitle(WINDOW_MAIN, WINDOW_MAIN)

            elif key == ord("t"):
                if not tracking_active:
                    if char_label:
                        ok = tracker.load_calibration(char_label, ds_root)
                        if ok:
                            state_classifier.load_references(
                                char_label, ds_root)
                            tracking_active = True
                            batter_state = BatterState.NORMAL
                            print("[INFO] -> TRACKING ON")
                        else:
                            print("[WARN] Could not load calibration "
                                  f"for '{char_label}'.")
                    else:
                        print("[WARN] Set character name first "
                              "(press [N]).")
                else:
                    tracking_active = False
                    batter_state = BatterState.NORMAL
                    state_classifier.reset()
                    print("[INFO] Tracking OFF")

            elif key == ord("c"):
                # Start capture: ensure we have a character name first
                if not char_label:
                    cv2.setWindowTitle(WINDOW_MAIN,
                                       "Type character name in terminal...")
                    char_label = input(
                        "[INPUT] Character name for this sample: ").strip()
                    cv2.setWindowTitle(WINDOW_MAIN, WINDOW_MAIN)
                if char_label:
                    frozen_frame = frame.copy()
                    clicker.start()
                    print("[INFO] Frame frozen. Click to define ROIs...")
                else:
                    print("[WARN] No name provided -- capture cancelled.")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")
    finally:
        cv2.destroyAllWindows()
        print("[INFO] Done.")


if __name__ == "__main__":
    main()
