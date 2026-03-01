"""
Frame Annotator — Human-in-the-loop ball annotation tool

Load recorded pitch frames, step through them, click to tag the ball
centre, mark frames where the ball is not visible, and save annotations
to JSON.  Optionally overlays the algorithm's detector/tracker output so
a human can quickly correct labels.

Usage
-----
  python frame_annotator.py pitches/20260227_205241
  python frame_annotator.py pitches/20260227_205241 --annotations annot.json
  python frame_annotator.py pitches/20260227_205241 --auto-detect

Controls
--------
  Right / D           Next frame
  Left  / A           Previous frame
  Space               Play / Pause (auto-advance)
  Home                Jump to first frame
  End                 Jump to last frame
  G                   Go to frame number (type in terminal)
  Left-click          Tag ball centre at cursor position
  V                   Mark ball as NOT visible in this frame
  Delete / Backspace  Remove annotation for this frame
  T                   Toggle algorithm detector overlay
  S                   Save annotations to JSON
  Q / ESC             Quit (prompts to save if unsaved changes)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Import detector/tracker from the main module for overlay mode
try:
    from pitch_analyzer import (
        BallDetector, BallTracker, TrajectoryCorridor, BallCandidate,
        CORRIDOR_DEFAULT, PITCHER_ZONE, COL_GREEN, COL_RED, COL_YELLOW,
        COL_CYAN, COL_WHITE, COL_ORANGE, COL_MAGENTA, COL_BLACK, FONT,
        put_text,
    )
    _HAS_DETECTOR = True
except ImportError:
    _HAS_DETECTOR = False

WINDOW_NAME = "MSB Frame Annotator"
DISPLAY_SCALE: float = 0.85

# zAnnotation schema
# {
#   "pitch_folder": "pitches/...",
#   "n_frames": 120,
#   "annotator": "human",
#   "created": "2026-02-28T...",
#   "annotations": {
#       "55": {"x": 595, "y": 170, "visible": true},
#       "89": {"visible": false},
#       ...
#   }
# }


class Annotator:
    """Interactive frame annotation tool with OpenCV GUI."""

    def __init__(self, folder: Path, annotation_path: Optional[Path],
                 auto_detect: bool = False) -> None:
        self.folder = folder
        self.annotation_path = annotation_path or folder / "annotations.json"
        self.auto_detect = auto_detect and _HAS_DETECTOR

        # Load frames
        exts = {".png", ".jpg", ".jpeg", ".bmp"}
        self.files = sorted(
            f for f in folder.iterdir() if f.suffix.lower() in exts)
        if not self.files:
            print(f"[ERROR] No image files in {folder}")
            sys.exit(1)
        self.n_frames = len(self.files)
        print(f"[INFO] Loaded {self.n_frames} frames from {folder}")

        # State
        self.idx: int = 0
        self.annotations: Dict[str, Dict[str, Any]] = {}
        self.unsaved: bool = False
        self.playing: bool = False
        self.show_detector: bool = self.auto_detect
        self._click_pos: Optional[Tuple[int, int]] = None
        self._frame_cache: Optional[np.ndarray] = None
        self._cache_idx: int = -1

        # Detector (for overlay)
        self.detector: Optional[Any] = None
        self.tracker: Optional[Any] = None
        self.corridor: Optional[Any] = None
        if self.auto_detect and _HAS_DETECTOR:
            self.detector = BallDetector()
            self.tracker = BallTracker()
            self.corridor = TrajectoryCorridor()
            if CORRIDOR_DEFAULT:
                self.corridor.set_rect(*CORRIDOR_DEFAULT)

        # Load existing annotations
        self._load_annotations()

    # I/O

    def _load_annotations(self) -> None:
        if self.annotation_path.exists():
            with open(self.annotation_path) as f:
                data = json.load(f)
            self.annotations = data.get("annotations", {})
            n = len(self.annotations)
            print(f"[INFO] Loaded {n} existing annotations "
                  f"from {self.annotation_path}")
        else:
            self.annotations = {}

    def save_annotations(self) -> None:
        data = {
            "pitch_folder": str(self.folder),
            "n_frames": self.n_frames,
            "annotator": "human",
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "annotations": self.annotations,
        }
        with open(self.annotation_path, "w") as f:
            json.dump(data, f, indent=2)
        self.unsaved = False
        print(f"[SAVE] Annotations saved to {self.annotation_path}")

    # Frame loading

    def load_frame(self, idx: int) -> np.ndarray:
        if idx == self._cache_idx and self._frame_cache is not None:
            return self._frame_cache
        frame = cv2.imread(str(self.files[idx]))
        if frame is None:
            frame = np.zeros((480, 640, 3), np.uint8)
        self._frame_cache = frame
        self._cache_idx = idx
        return frame

    # Detector overlay

    def _run_detector(self, frame: np.ndarray) -> None:
        """Run the algorithm on the current frame for overlay."""
        if self.detector is None:
            return
        best = self.detector.detect(frame, corridor=self.corridor)
        self.tracker.update(self.detector.candidates, best)

    def _run_detector_sequence(self) -> None:
        """Run the detector on all frames from 0 to current index
        to build proper background model and tracking state."""
        if self.detector is None:
            return
        self.detector.reset_full()
        self.tracker.reset()
        for i in range(self.idx + 1):
            frame = self.load_frame(i)
            best = self.detector.detect(frame, corridor=self.corridor)
            self.tracker.update(self.detector.candidates, best)

    # Drawing

    def draw(self) -> np.ndarray:
        frame = self.load_frame(self.idx)
        vis = frame.copy()
        h, w = vis.shape[:2]
        key = str(self.idx)

        # Detector overlay
        if self.show_detector and self.detector is not None:
            # Draw all candidates as small dots
            for c in self.detector.candidates:
                col = (0, 200, 200) if c.in_motion_mask else (100, 100, 100)
                cv2.circle(vis, c.center, 3, col, -1)

            # Draw tracker selection
            sel = self.tracker.selected if self.tracker else None
            if sel is not None:
                cv2.circle(vis, sel.center, 8, (0, 200, 0), 2)
                cv2.circle(vis, sel.center, 2, (0, 0, 200), -1)
                bx, by, bw, bh = sel.bbox
                cv2.rectangle(vis, (bx, by), (bx + bw, by + bh),
                              (0, 200, 0), 1)
                put_text(vis,
                         f"ALGO ({sel.center[0]},{sel.center[1]}) "
                         f"a={sel.area:.0f} iso={sel.isolation_score:.1f}",
                         (bx, by - 8), 0.35, (0, 200, 0), 1)

            # Show suppression zones
            if hasattr(self.detector, 'suppressor'):
                for sx, sy, sr in self.detector.suppressor.suppression_zones:
                    cv2.circle(vis, (sx, sy), sr, (0, 0, 128), 1)

        # Pitcher zone
        if _HAS_DETECTOR:
            pz = PITCHER_ZONE
            cv2.rectangle(vis, (pz[0], pz[1]), (pz[2], pz[3]),
                          (128, 128, 0), 1)

        # Human annotation
        if key in self.annotations:
            ann = self.annotations[key]
            if ann.get("visible", False):
                ax, ay = ann["x"], ann["y"]
                # Green crosshair for annotated position
                cv2.drawMarker(vis, (ax, ay), (0, 255, 0),
                               cv2.MARKER_CROSS, 20, 2)
                cv2.circle(vis, (ax, ay), 8, (0, 255, 0), 2)
                put_text(vis, f"ANN ({ax},{ay})",
                         (ax + 12, ay - 8), 0.38, (0, 255, 0), 1)
            else:
                put_text(vis, "NOT VISIBLE (annotated)",
                         (10, h - 40), 0.45, (0, 0, 255), 1)

        # Status bar
        bar_y = 22
        put_text(vis, f"Frame {self.idx}/{self.n_frames - 1}",
                 (10, bar_y), 0.50, (255, 255, 255), 1)

        status_parts = []
        if key in self.annotations:
            ann = self.annotations[key]
            if ann.get("visible"):
                status_parts.append(f"TAGGED ({ann['x']},{ann['y']})")
            else:
                status_parts.append("NOT VISIBLE")
        else:
            status_parts.append("NO ANNOTATION")

        annotated_count = len(self.annotations)
        status_parts.append(f"Total: {annotated_count}/{self.n_frames}")

        if self.unsaved:
            status_parts.append("*UNSAVED*")
        if self.playing:
            status_parts.append("PLAYING")
        if self.show_detector:
            status_parts.append("ALGO ON")

        put_text(vis, "  |  ".join(status_parts),
                 (10, bar_y + 22), 0.40, (200, 200, 200), 1)

        # Navigation minimap (thin bar showing annotated frames)
        bar_h = 8
        bar_w = w - 20
        bar_x = 10
        bar_top = h - 20
        cv2.rectangle(vis, (bar_x, bar_top),
                      (bar_x + bar_w, bar_top + bar_h),
                      (60, 60, 60), -1)
        # Mark annotated frames
        for fkey in self.annotations:
            fi = int(fkey)
            fx = bar_x + int(fi / max(self.n_frames - 1, 1) * bar_w)
            ann = self.annotations[fkey]
            col = (0, 200, 0) if ann.get("visible") else (0, 0, 200)
            cv2.line(vis, (fx, bar_top), (fx, bar_top + bar_h), col, 1)
        # Current position
        cx = bar_x + int(self.idx / max(self.n_frames - 1, 1) * bar_w)
        cv2.line(vis, (cx, bar_top - 2), (cx, bar_top + bar_h + 2),
                 (255, 255, 255), 2)

        # Controls hint
        hints = ("[</>] Navigate  [Space] Play  [Click] Tag  "
                 "[V] Not visible  [Del] Remove  [T] Algo  [S] Save  [Q] Quit")
        put_text(vis, hints, (10, h - 4), 0.30, (180, 180, 180), 1)

        return vis

    # Mouse callback

    def _on_mouse(self, event: int, x: int, y: int,
                  flags: int, param: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            # Convert display coords to frame coords
            ox = int(x / DISPLAY_SCALE)
            oy = int(y / DISPLAY_SCALE)
            self._click_pos = (ox, oy)

    # Main loop

    def run(self) -> None:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, self._on_mouse)

        # If auto-detect, run detector sequence up to current frame
        if self.auto_detect and self.detector is not None:
            print("[INFO] Running detector on all frames (building "
                  "background model)...")
            self.idx = self.n_frames - 1
            self._run_detector_sequence()
            self.idx = 0
            self.detector.reset_full()
            self.tracker.reset()
            # Re-run up to frame 0
            self._run_detector_sequence()
            print("[INFO] Detector ready.")

        last_play_time = time.time()

        while True:
            # Handle click
            if self._click_pos is not None:
                ox, oy = self._click_pos
                self._click_pos = None
                key = str(self.idx)
                self.annotations[key] = {
                    "x": ox, "y": oy, "visible": True
                }
                self.unsaved = True

            vis = self.draw()

            # Scale for display
            if DISPLAY_SCALE != 1.0:
                dw = int(vis.shape[1] * DISPLAY_SCALE)
                dh = int(vis.shape[0] * DISPLAY_SCALE)
                vis = cv2.resize(vis, (dw, dh))

            cv2.imshow(WINDOW_NAME, vis)

            # Auto-play
            if self.playing:
                now = time.time()
                if now - last_play_time > 1.0 / 15.0:  # 15 fps playback
                    last_play_time = now
                    if self.idx < self.n_frames - 1:
                        self.idx += 1
                        if self.show_detector and self.detector is not None:
                            frame = self.load_frame(self.idx)
                            self._run_detector(frame)
                    else:
                        self.playing = False

            k = cv2.waitKey(30) & 0xFF

            if k in (ord("q"), 27):  # Quit
                if self.unsaved:
                    print("[WARN] Unsaved annotations! Save? (y/n)")
                    resp = input("> ").strip().lower()
                    if resp in ("y", "yes", "s", "si"):
                        self.save_annotations()
                break

            elif k == ord("s"):
                self.save_annotations()

            elif k == ord(" "):
                self.playing = not self.playing
                last_play_time = time.time()

            elif k == 83 or k == ord("d"):  # Right arrow or D
                self.playing = False
                if self.idx < self.n_frames - 1:
                    self.idx += 1
                    if self.show_detector and self.detector is not None:
                        frame = self.load_frame(self.idx)
                        self._run_detector(frame)

            elif k == 81 or k == ord("a"):  # Left arrow or A
                self.playing = False
                if self.idx > 0:
                    self.idx -= 1
                    # Re-run detector from scratch to this frame
                    if self.show_detector and self.detector is not None:
                        self._run_detector_sequence()

            elif k == 80:  # Home
                self.idx = 0
                self.playing = False
                if self.show_detector and self.detector is not None:
                    self._run_detector_sequence()

            elif k == 87:  # End
                self.idx = self.n_frames - 1
                self.playing = False
                if self.show_detector and self.detector is not None:
                    self._run_detector_sequence()

            elif k == ord("g"):  # Go to frame
                self.playing = False
                try:
                    n = int(input(f"Go to frame [0-{self.n_frames - 1}]: "))
                    self.idx = max(0, min(n, self.n_frames - 1))
                    if self.show_detector and self.detector is not None:
                        self._run_detector_sequence()
                except ValueError:
                    print("[WARN] Invalid frame number")

            elif k == ord("v"):  # Mark not visible
                key = str(self.idx)
                self.annotations[key] = {"visible": False}
                self.unsaved = True

            elif k in (255, 8):  # Delete / Backspace
                key = str(self.idx)
                if key in self.annotations:
                    del self.annotations[key]
                    self.unsaved = True

            elif k == ord("t"):  # Toggle algorithm overlay
                if not _HAS_DETECTOR:
                    print("[WARN] pitch_analyzer not available for overlay")
                else:
                    self.show_detector = not self.show_detector
                    if self.show_detector:
                        if self.detector is None:
                            self.detector = BallDetector()
                            self.tracker = BallTracker()
                            self.corridor = TrajectoryCorridor()
                            if CORRIDOR_DEFAULT:
                                self.corridor.set_rect(*CORRIDOR_DEFAULT)
                        print("[INFO] Running detector sequence...")
                        self._run_detector_sequence()
                        print("[INFO] Algo overlay ON")
                    else:
                        print("[INFO] Algo overlay OFF")

        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(
        description="MSB Frame Annotator — tag ball positions in recorded "
                    "pitch frames")
    ap.add_argument("folder", help="Path to pitch recording folder "
                    "(e.g. pitches/20260227_205241)")
    ap.add_argument("--annotations", "-a", default=None,
                    help="Path to annotations JSON file "
                    "(default: <folder>/annotations.json)")
    ap.add_argument("--auto-detect", action="store_true",
                    help="Run the algorithm detector as overlay "
                    "for faster annotation")
    args = ap.parse_args()

    folder = Path(args.folder)
    if not folder.is_dir():
        print(f"[ERROR] Not a directory: {folder}")
        sys.exit(1)

    ann_path = Path(args.annotations) if args.annotations else None
    annotator = Annotator(folder, ann_path, auto_detect=args.auto_detect)
    annotator.run()


if __name__ == "__main__":
    main()
