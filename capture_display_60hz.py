"""
capture_display_60hz.py

Real-time screen capture at 60 FPS using dxcam (DXGI Desktop Duplication)
with OpenCV display and on-screen diagnostics.

Notes:
    - Dolphin must run in Borderless Windowed or Windowed mode.
      Fullscreen exclusive may block DXGI Desktop Duplication.
    - Adjust ROI to capture only the game area for better performance.
"""

import sys
import time

import cv2
import dxcam
import numpy as np

# Configuration — tweak these values as needed

TARGET_FPS: int = 60
"""Target capture/display framerate."""

# ROI: tuple[int, int, int, int] | None = None
ROI = (378, 127, 1542, 1019)
"""
Screen region to capture as (left, top, right, bottom) in pixel coordinates.
Set to None to capture the entire primary monitor.
Example for a 1280×720 window starting at (100, 200):
    ROI = (100, 200, 1380, 920)
Using a tight ROI significantly reduces capture cost and processing load.
"""

DISPLAY_SCALE: float = 1.0
"""
Scale factor for the display window (0.5 = half size, 1.0 = original).
Lowering this reduces cv2.imshow overhead on high-res captures.
"""

WINDOW_NAME: str = "MSB Capture — 60Hz"
"""OpenCV window title."""

# Colors for overlay text (BGR)
_GREEN = (0, 255, 0)
_WHITE = (255, 255, 255)
_FONT = cv2.FONT_HERSHEY_SIMPLEX

# Helper functions

def draw_overlay(frame: np.ndarray, fps: float, frame_ms: float) -> np.ndarray:
    """Draw diagnostic text (FPS, frame time, resolution) on the frame."""
    h, w = frame.shape[:2]
    overlay_lines = [
        f"FPS: {fps:.1f}",
        f"Frame: {frame_ms:.1f} ms",
        f"Res: {w}x{h}",
    ]
    y_offset = 30
    for i, line in enumerate(overlay_lines):
        y = y_offset + i * 28
        # Shadow for readability
        cv2.putText(frame, line, (11, y + 1), _FONT, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, line, (10, y), _FONT, 0.7, _GREEN, 2, cv2.LINE_AA)
    return frame


def create_camera(target_fps: int, roi: tuple | None) -> dxcam.DXCamera:
    """
    Create and start a dxcam capture instance.
    dxcam internally runs a capture thread with a small ring buffer.
    We use output_color="BGR" to avoid a per-frame BGRA→BGR conversion.
    """
    cam = dxcam.create(output_color="BGR")
    if cam is None:
        print("[ERROR] dxcam.create() returned None. Check GPU/driver compatibility.")
        sys.exit(1)

    # Start the internal capture loop.
    # region= crops at the DXGI level (GPU-side), which is the most efficient ROI.
    if roi is not None:
        cam.start(target_fps=target_fps, region=roi)
    else:
        cam.start(target_fps=target_fps)

    return cam

# Main loop

def main() -> None:
    print(f"[INFO] Starting capture — target {TARGET_FPS} FPS")
    if ROI is not None:
        print(f"[INFO] ROI: {ROI}")
    if DISPLAY_SCALE != 1.0:
        print(f"[INFO] Display scale: {DISPLAY_SCALE}")

    cam = create_camera(TARGET_FPS, ROI)

    # FPS measurement using exponential moving average for smooth readout
    fps_ema: float = float(TARGET_FPS)
    alpha: float = 0.1  # EMA smoothing factor (lower = smoother)
    prev_time: float = time.perf_counter()

    # Allow a short warm-up for dxcam's capture thread
    time.sleep(0.1)

    null_streak: int = 0  # consecutive None frames counter

    try:
        while True:
            # Always grab the most recent frame (no backlog accumulation).
            frame = cam.get_latest_frame()

            if frame is None:
                null_streak += 1
                if null_streak > TARGET_FPS * 2:
                    # ~2 seconds of no frames → likely a real problem
                    print("[WARN] No frames received for ~2 s. Retrying capture…")
                    cam.stop()
                    time.sleep(0.5)
                    cam = create_camera(TARGET_FPS, ROI)
                    null_streak = 0
                    prev_time = time.perf_counter()
                # Yield CPU briefly on None to avoid busy-spin
                time.sleep(0.001)
                continue

            null_streak = 0

            # Timing
            now = time.perf_counter()
            dt = now - prev_time
            prev_time = now
            instant_fps = 1.0 / dt if dt > 0 else 0.0
            fps_ema = alpha * instant_fps + (1.0 - alpha) * fps_ema
            frame_ms = dt * 1000.0

            # Optional downscale for display
            display = frame
            if DISPLAY_SCALE != 1.0:
                new_w = int(frame.shape[1] * DISPLAY_SCALE)
                new_h = int(frame.shape[0] * DISPLAY_SCALE)
                # INTER_NEAREST is the cheapest interpolation — fine for preview
                display = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

            # Draw overlay
            display = draw_overlay(display, fps_ema, frame_ms)

            # Show
            cv2.imshow(WINDOW_NAME, display)

            # Input
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # q or ESC
                print("[INFO] Exit requested.")
                break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")

    finally:
        # Clean shutdown: stop capture thread + destroy GUI
        cam.stop()
        cv2.destroyAllWindows()
        print("[INFO] Capture stopped. Resources released.")


if __name__ == "__main__":
    main()
