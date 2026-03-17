"""
Frame sources — generators that yield BGR frames from live capture
or from a folder of saved images.

Capture priority (fastest → slowest):
  1. **bettercam** — maintained fork of dxcam, handles laptop hybrid
     GPUs (Intel + NVIDIA) correctly.  ``pip install bettercam``
  2. **dxcam** — original DXGI Desktop Duplication wrapper.
     Known to fail on some laptop GPUs.  ``pip install dxcam``
  3. **Win32 BitBlt** — GDI-based capture via ctypes.  No external
     deps, works on all Windows GPUs, ~50-80 fps.
  4. **mss** — pure-Python fallback that works everywhere.
     Slower (~30–40 fps) but always reliable.  ``pip install mss``
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes as wintypes
import sys
import threading
import time
import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from msb.config import Config


# CAPTURE THREAD

class CaptureThread:
    """Runs screen capture in a background thread.

    The consumer always gets the *most recent* frame via ``latest()``,
    so the processing pipeline never waits for capture and never
    processes a stale frame.
    """

    def __init__(self, source_gen) -> None:
        self._gen = source_gen
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._ts: float = 0.0
        self._seq: int = 0          # monotonic capture counter
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop,
                                        daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def latest(self) -> Tuple[Optional[np.ndarray], float, int]:
        """Return the most recent (frame, timestamp, seq) triple."""
        with self._lock:
            return self._frame, self._ts, self._seq

    def _capture_loop(self) -> None:
        try:
            for item in self._gen:
                if not self._running:
                    break
                if isinstance(item, tuple):
                    frame, ts = item
                else:
                    frame, ts = item, time.perf_counter()
                with self._lock:
                    self._frame = frame
                    self._ts = ts
                    self._seq += 1
        except Exception as exc:
            print(f"[CaptureThread] error: {exc}")
        finally:
            self._running = False


# LIVE CAPTURE

def source_live(cfg: Optional[Config] = None):
    """Yield BGR frames from the live screen at *cfg.target_fps*.

    Tries bettercam → dxcam → Win32 BitBlt → mss, in that order.
    """
    cfg = cfg or Config()

    warnings.filterwarnings("ignore", category=ResourceWarning)
    import io, contextlib

    # 1) bettercam
    with contextlib.redirect_stderr(io.StringIO()):
        cam = _try_bettercam(cfg)
    if cam is not None:
        yield from _run_dxgi_cam(cam, cfg, lib_name="bettercam")
        return

    # 2) dxcam
    with contextlib.redirect_stderr(io.StringIO()):
        cam = _try_dxcam(cfg)
    if cam is not None:
        yield from _run_dxgi_cam(cam, cfg, lib_name="dxcam")
        return

    # 3) Win32 BitBlt
    yield from _source_bitblt(cfg)


def _run_dxgi_cam(cam, cfg: Config, lib_name: str = "dxcam"):
    """Shared generator for any dxcam-compatible camera object."""
    cam.start(target_fps=cfg.target_fps, region=cfg.screen_roi)
    time.sleep(0.25)
    print(f"[INFO] {lib_name} capture running @ target {cfg.target_fps} fps")
    null_count = 0
    try:
        while True:
            frame = cam.get_latest_frame()
            if frame is None:
                null_count += 1
                if null_count > 120:
                    print(f"[WARN] {lib_name}: 120 consecutive null frames, "
                          "falling back to mss")
                    break
                time.sleep(0.001)
                continue
            null_count = 0
            yield (frame, time.perf_counter())
    finally:
        try:
            cam.stop()
        except Exception:
            pass


# bettercam

def _try_bettercam(cfg: Config):
    """Try bettercam (maintained dxcam fork with hybrid-GPU fixes)."""
    try:
        import bettercam
    except ImportError:
        print("[INFO] bettercam not installed (pip install bettercam)")
        return None

    attempts = [
        dict(output_idx=cfg.monitor_index, output_color="BGR"),
        dict(device_idx=0, output_idx=cfg.monitor_index, output_color="BGR"),
    ]
    try:
        n_devices = len(bettercam.device_info()) if hasattr(bettercam, "device_info") else 1
    except Exception:
        n_devices = 1
    if n_devices > 1:
        attempts.append(
            dict(device_idx=1, output_idx=cfg.monitor_index,
                 output_color="BGR"))

    for kwargs in attempts:
        try:
            cam = bettercam.create(**kwargs)
            if cam is None:
                continue
            cam.start(target_fps=cfg.target_fps, region=cfg.screen_roi)
            time.sleep(0.1)
            test = cam.get_latest_frame()
            cam.stop()
            if test is not None:
                print(f"[INFO] bettercam OK: {kwargs}")
                return bettercam.create(**kwargs)  # fresh instance
            print(f"[WARN] bettercam returned null frame ({kwargs})")
        except (ctypes.COMError, IndexError, OSError, Exception) as exc:
            print(f"[WARN] bettercam failed ({kwargs}): {exc}")
            try:
                cam.stop()
            except Exception:
                pass

    print("[WARN] All bettercam attempts failed")
    return None


# dxcam

def _try_dxcam(cfg: Config):
    """Attempt to create a dxcam camera. Returns None on failure."""
    try:
        import dxcam
    except ImportError:
        print("[INFO] dxcam not installed")
        return None

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
            return dxcam.create(**kwargs)
        except (ctypes.COMError, IndexError, Exception) as exc:
            print(f"[WARN] dxcam failed ({kwargs}): {exc}")
    print("[WARN] All dxcam attempts failed")
    return None


# Win32 BitBlt

# Win32 constants
_SRCCOPY       = 0x00CC0020
_DIB_RGB_COLORS = 0
_BI_RGB         = 0


class _BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize",          wintypes.DWORD),
        ("biWidth",         wintypes.LONG),
        ("biHeight",        wintypes.LONG),
        ("biPlanes",        wintypes.WORD),
        ("biBitCount",      wintypes.WORD),
        ("biCompression",   wintypes.DWORD),
        ("biSizeImage",     wintypes.DWORD),
        ("biXPelsPerMeter", wintypes.LONG),
        ("biYPelsPerMeter", wintypes.LONG),
        ("biClrUsed",       wintypes.DWORD),
        ("biClrImportant",  wintypes.DWORD),
    ]


def _source_bitblt(cfg: Config):
    """Fast screen capture using Win32 GDI BitBlt with reused resources.

    No external dependencies — only ctypes + numpy.
    Persistent DC / bitmap means minimal per-frame overhead.
    Typically 50–80+ fps for a 1164×892 region on modern hardware.
    """
    user32 = ctypes.windll.user32
    gdi32  = ctypes.windll.gdi32

    roi = cfg.screen_roi  # (left, top, right, bottom)
    width  = roi[2] - roi[0]
    height = roi[3] - roi[1]

    hdc_screen = user32.GetDC(0)  # desktop DC
    hdc_mem    = gdi32.CreateCompatibleDC(hdc_screen)
    hbmp       = gdi32.CreateCompatibleBitmap(hdc_screen, width, height)
    old_bmp    = gdi32.SelectObject(hdc_mem, hbmp)

    bmi = _BITMAPINFOHEADER()
    bmi.biSize        = ctypes.sizeof(_BITMAPINFOHEADER)
    bmi.biWidth       = width
    bmi.biHeight      = -height  # top-down
    bmi.biPlanes      = 1
    bmi.biBitCount    = 32       # BGRA
    bmi.biCompression = _BI_RGB

    buf_size = width * height * 4
    buf = ctypes.create_string_buffer(buf_size)

    target_dt = 1.0 / cfg.target_fps if cfg.target_fps > 0 else 0.0

    print(f"[INFO] Win32 BitBlt capture: {width}x{height} "
          f"@ target {cfg.target_fps} fps")

    try:
        while True:
            t0 = time.perf_counter()

            # Blit screen region → memory DC
            gdi32.BitBlt(
                hdc_mem, 0, 0, width, height,
                hdc_screen, roi[0], roi[1],
                _SRCCOPY,
            )

            # Read pixels into our buffer
            gdi32.GetDIBits(
                hdc_mem, hbmp, 0, height,
                buf, ctypes.byref(bmi), _DIB_RGB_COLORS,
            )

            # BGRA → BGR (drop alpha channel)
            bgra = np.frombuffer(buf, dtype=np.uint8).reshape(
                height, width, 4)
            frame = np.ascontiguousarray(bgra[:, :, :3])

            yield (frame, time.perf_counter())

            # Throttle to target FPS
            elapsed = time.perf_counter() - t0
            if elapsed < target_dt:
                time.sleep(target_dt - elapsed)
    finally:
        # Release GDI resources
        gdi32.SelectObject(hdc_mem, old_bmp)
        gdi32.DeleteObject(hbmp)
        gdi32.DeleteDC(hdc_mem)
        user32.ReleaseDC(0, hdc_screen)
        print("[INFO] BitBlt capture resources released")


# mss fallback

def _source_mss(cfg: Config):
    """Yield frames via mss (pure-Python, ~30–40 fps)."""
    print("[INFO] Using mss screen capture (fallback)")
    try:
        import mss
    except ImportError:
        print("[ERROR] No capture backend available. Install one:\n"
              "  pip install bettercam   (recommended)\n"
              "  pip install dxcam\n"
              "  pip install mss         (slowest fallback)")
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
            yield (frame, time.perf_counter())
            elapsed = time.perf_counter() - t0
            if elapsed < target_dt:
                time.sleep(target_dt - elapsed)


# FOLDER SOURCE

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
