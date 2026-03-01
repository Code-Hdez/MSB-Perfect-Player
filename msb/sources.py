"""
Frame sources — generators that yield BGR frames from live capture
or from a folder of saved images.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional, Tuple

from msb.config import Config


def source_live(cfg: Optional[Config] = None):
    """Yield frames from the live screen via dxcam."""
    cfg = cfg or Config()
    try:
        import dxcam
    except ImportError:
        print("[ERROR] dxcam not installed.  pip install dxcam")
        sys.exit(1)
    cam = dxcam.create(output_idx=cfg.monitor_index, output_color="BGR")
    if cam is None:
        print("[ERROR] dxcam.create() returned None.")
        sys.exit(1)
    cam.start(target_fps=cfg.target_fps, region=cfg.screen_roi)
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
    """Yield frames from a folder of image files (sorted by name)."""
    import cv2
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
