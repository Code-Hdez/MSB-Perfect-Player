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
    """Yield frames from the live screen via dxcam (preferred) or mss (fallback)."""
    cfg = cfg or Config()

    # Try dxcam first (fastest, uses DXGI Desktop Duplication)
    cam = _try_dxcam(cfg)
    if cam is not None:
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
        return

    # Fallback: mss (works everywhere, slightly slower)
    print("[INFO] Falling back to mss screen capture")
    try:
        import mss
        import numpy as np
    except ImportError:
        print("[ERROR] Neither dxcam nor mss available. "
              "pip install dxcam mss")
        sys.exit(1)

    roi = cfg.screen_roi  # (left, top, right, bottom)
    monitor = {
        "left": roi[0], "top": roi[1],
        "width": roi[2] - roi[0], "height": roi[3] - roi[1],
    }
    target_dt = 1.0 / cfg.target_fps if cfg.target_fps > 0 else 0.0
    print(f"[INFO] mss capture region: {monitor}")

    with mss.mss() as sct:
        while True:
            t0 = time.perf_counter()
            shot = sct.grab(monitor)
            # mss returns BGRA; convert to BGR
            frame = np.asarray(shot)[:, :, :3].copy()
            yield frame
            # Throttle to target FPS
            elapsed = time.perf_counter() - t0
            if elapsed < target_dt:
                time.sleep(target_dt - elapsed)


def _try_dxcam(cfg: Config):
    """Attempt to create a dxcam camera. Returns None on failure."""
    try:
        import dxcam
    except ImportError:
        print("[WARN] dxcam not installed, will use mss fallback")
        return None

    import ctypes
    for kwargs in [
        dict(output_idx=cfg.monitor_index, output_color="BGR"),
        dict(device_idx=0, output_idx=cfg.monitor_index, output_color="BGR"),
    ]:
        try:
            cam = dxcam.create(**kwargs)
            if cam is None:
                continue
            cam.start(target_fps=cfg.target_fps, region=cfg.screen_roi)
            cam.stop()
            print(f"[INFO] dxcam OK: {kwargs}")
            return cam
        except (ctypes.COMError, IndexError, Exception) as e:
            print(f"[WARN] dxcam failed ({kwargs}): {e}")
    print("[WARN] All dxcam attempts failed")
    return None


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
