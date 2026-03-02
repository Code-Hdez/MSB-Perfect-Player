"""
frame_annotator.py — Interactively annotate ball positions in pitch frames.

Click to mark the ball center in each frame. Press [N] to mark "not visible".
Saves annotations in a deterministic JSON schema compatible with
export_yolo.py and validate_tracking.py.

Usage
-----
  python frame_annotator.py pitches/20260301_030216
  python frame_annotator.py pitches/20260301_030216 -o custom_annotations.json

Controls
--------
  Left-click = Mark ball center at cursor position
  N          = Mark ball as NOT visible in this frame
  B / Left   = Go back one frame (undo last annotation for review)
  S          = Save annotations to disk (auto-saves on completion)
  Q / ESC    = Quit (prompts to save if unsaved changes)
  R          = Remove annotation for current frame and re-annotate

Schema (annotations.json)
-------------------------
See docs/annotations_schema.md for the full specification.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

WINDOW_NAME = "MSB Frame Annotator"
FONT = cv2.FONT_HERSHEY_SIMPLEX


class FrameAnnotator:
    """Interactive frame-by-frame ball position annotator."""

    def __init__(
        self,
        folder: Path,
        output_path: Optional[Path] = None,
        display_scale: float = 1.0,
    ) -> None:
        self.folder = folder
        self.output_path = output_path or (folder / "annotations.json")
        self.scale = display_scale

        # Load frames
        exts = {".png", ".jpg", ".jpeg", ".bmp"}
        self.files = sorted(
            f for f in folder.iterdir() if f.suffix.lower() in exts
        )
        if not self.files:
            print(f"[ERROR] No image files in {folder}")
            sys.exit(1)

        self.n_frames = len(self.files)
        self.current_idx = 0
        self.annotations: Dict[str, Dict[str, Any]] = {}
        self.click_pos: Optional[Tuple[int, int]] = None
        self.unsaved_changes = False

        # Read first frame to get image dimensions (for bounds clamping)
        first = cv2.imread(str(self.files[0]))
        if first is not None:
            self._img_h, self._img_w = first.shape[:2]
        else:
            self._img_h, self._img_w = 0, 0

        # Load existing annotations if present
        if self.output_path.exists():
            with open(self.output_path) as f:
                data = json.load(f)
            self.annotations = data.get("annotations", {})
            # Skip to first unannotated frame
            for i in range(self.n_frames):
                if str(i) not in self.annotations:
                    self.current_idx = i
                    break
            else:
                self.current_idx = self.n_frames - 1
            n_existing = len(self.annotations)
            print(f"[INFO] Loaded {n_existing} existing annotations. "
                  f"Resuming at frame {self.current_idx}.")

        print(f"[INFO] {self.n_frames} frames from {folder}")
        print(f"[INFO] Output: {self.output_path}")

    def _mouse_callback(
        self, event: int, x: int, y: int, flags: int, param: Any
    ) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            # Convert display coords back to frame coords
            real_x = int(x / self.scale) if self.scale != 1.0 else x
            real_y = int(y / self.scale) if self.scale != 1.0 else y
            # Clamp to image bounds (prevents out-of-range annotations)
            h, w = self._img_h, self._img_w
            real_x = max(0, min(real_x, w - 1))
            real_y = max(0, min(real_y, h - 1))
            self.click_pos = (real_x, real_y)

    def _draw_frame(self) -> np.ndarray:
        """Load and annotate the current frame for display."""
        frame = cv2.imread(str(self.files[self.current_idx]))
        if frame is None:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)

        disp = frame.copy()
        h, w = disp.shape[:2]
        idx_str = str(self.current_idx)

        # Draw existing annotation if any
        ann = self.annotations.get(idx_str)
        if ann is not None:
            if ann.get("visible"):
                cx, cy = ann["x"], ann["y"]
                cv2.drawMarker(disp, (cx, cy), (0, 255, 0),
                               cv2.MARKER_CROSS, 20, 2)
                cv2.circle(disp, (cx, cy), 8, (0, 255, 0), 2)
                cv2.putText(disp, f"({cx},{cy})", (cx + 12, cy - 12),
                            FONT, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            else:
                cv2.putText(disp, "NOT VISIBLE", (w // 2 - 80, h // 2),
                            FONT, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

        # Status bar
        n_done = len(self.annotations)
        pct = n_done / self.n_frames * 100 if self.n_frames > 0 else 0
        status = (f"Frame {self.current_idx}/{self.n_frames - 1}  "
                  f"| Annotated: {n_done}/{self.n_frames} ({pct:.0f}%)  "
                  f"| {'* UNSAVED' if self.unsaved_changes else 'Saved'}")
        cv2.rectangle(disp, (0, 0), (w, 30), (40, 40, 40), -1)
        cv2.putText(disp, status, (10, 20), FONT, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)

        # Controls help
        help_lines = [
            "Click=mark ball  N=not visible  B=back  R=redo  S=save  Q=quit"
        ]
        for j, line in enumerate(help_lines):
            cv2.putText(disp, line, (10, h - 10 - j * 18), FONT, 0.4,
                        (200, 200, 200), 1, cv2.LINE_AA)

        # Scale for display
        if self.scale != 1.0:
            dw = int(w * self.scale)
            dh = int(h * self.scale)
            disp = cv2.resize(disp, (dw, dh))

        return disp

    def save(self) -> None:
        """Save annotations to disk."""
        data = {
            "schema_version": "1.0",
            "folder": str(self.folder),
            "n_frames": self.n_frames,
            "image_dimensions": {
                "width": self._img_w,
                "height": self._img_h,
            },
            "frame_files": [f.name for f in self.files],
            "annotations": self.annotations,
        }
        with open(self.output_path, "w") as f:
            json.dump(data, f, indent=2)
        self.unsaved_changes = False
        print(f"[SAVED] {len(self.annotations)} annotations → "
              f"{self.output_path}")

    def run(self) -> None:
        """Main annotation loop."""
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, self._mouse_callback)

        while True:
            self.click_pos = None
            disp = self._draw_frame()
            cv2.imshow(WINDOW_NAME, disp)

            key = cv2.waitKey(30) & 0xFF

            # Check for click
            if self.click_pos is not None:
                cx, cy = self.click_pos
                self.annotations[str(self.current_idx)] = {
                    "x": cx, "y": cy, "visible": True,
                    "frame_file": self.files[self.current_idx].name,
                }
                self.unsaved_changes = True
                print(f"  F{self.current_idx:03d}: ball at ({cx}, {cy})")
                # Auto-advance
                if self.current_idx < self.n_frames - 1:
                    self.current_idx += 1
                self.click_pos = None
                continue

            if key == ord("n"):
                # Mark not visible
                self.annotations[str(self.current_idx)] = {
                    "x": None, "y": None, "visible": False,
                    "frame_file": self.files[self.current_idx].name,
                }
                self.unsaved_changes = True
                print(f"  F{self.current_idx:03d}: not visible")
                if self.current_idx < self.n_frames - 1:
                    self.current_idx += 1

            elif key == ord("b") or key == 81:  # B or Left arrow
                if self.current_idx > 0:
                    self.current_idx -= 1

            elif key == ord("r"):
                # Remove current annotation
                idx_str = str(self.current_idx)
                if idx_str in self.annotations:
                    del self.annotations[idx_str]
                    self.unsaved_changes = True
                    print(f"  F{self.current_idx:03d}: annotation removed")

            elif key == ord("s"):
                self.save()

            elif key in (ord("q"), 27):
                if self.unsaved_changes:
                    print("[WARN] Unsaved changes! Saving before exit...")
                    self.save()
                break

            # Auto-navigation with arrow keys
            elif key == 83:  # Right arrow
                if self.current_idx < self.n_frames - 1:
                    self.current_idx += 1
            elif key == 82:  # Up arrow — jump forward 10
                self.current_idx = min(self.n_frames - 1,
                                       self.current_idx + 10)
            elif key == 84:  # Down arrow — jump back 10
                self.current_idx = max(0, self.current_idx - 10)

        cv2.destroyAllWindows()

        # Final stats
        n_vis = sum(1 for v in self.annotations.values()
                    if v.get("visible"))
        n_not_vis = sum(1 for v in self.annotations.values()
                        if not v.get("visible"))
        print(f"\n[DONE] {len(self.annotations)} total annotations: "
              f"{n_vis} visible, {n_not_vis} not visible")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Annotate ball positions in pitch recording frames")
    ap.add_argument("folder",
                    help="Path to pitch recording folder with frame images")
    ap.add_argument("-o", "--output", default=None,
                    help="Output annotations JSON "
                         "(default: <folder>/annotations.json)")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="Display scale factor (default: 1.0)")
    args = ap.parse_args()

    folder = Path(args.folder)
    if not folder.is_dir():
        print(f"[ERROR] Not a directory: {folder}")
        sys.exit(1)

    annotator = FrameAnnotator(
        folder,
        output_path=Path(args.output) if args.output else None,
        display_scale=args.scale,
    )
    annotator.run()


if __name__ == "__main__":
    main()
