"""Diagnose why ball detection fails after frame 57.

Checks each mask (white, bg_fg, motion, combined) at expected ball positions.
"""
import sys
sys.path.insert(0, ".")
import cv2
import numpy as np
from pathlib import Path
from pitch_analyzer import (
    BallDetector, BALL_FLIGHT_MAX_AREA, BALL_HSV_LOWER, BALL_HSV_UPPER,
    BALL_MIN_AREA, BALL_MAX_AREA
)

folder = Path("pitches/20260227_205241")
files = sorted(f for f in folder.iterdir() if f.suffix.lower() == ".png")
print(f"Loaded {len(files)} frames\n")

detector = BallDetector()

# Expected ball trajectory (extrapolated from frames 54-57)
# Frame 54: (602, 165), 55: (602, 192), 56: (605, 211), 57: (613, 214)
# Avg vy ~ 16 px/frame, vx ~ 3 px/frame
expected = {}
base_x, base_y = 613, 214
for f in range(58, 90):
    dt = f - 57
    ex = int(base_x + 3 * dt)
    ey = int(base_y + 16 * dt)
    expected[f] = (ex, ey)

R = 25  # search radius around expected position

for i, fpath in enumerate(files):
    frame = cv2.imread(str(fpath))
    if frame is None:
        continue

    # Always run detection to keep bg model + frame diff in sync
    detector.detect(frame)

    if i < 54 or i > 85:
        continue

    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Get masks from detector (stored after detect())
    white_mask = detector.white_mask
    bg_fg_mask = detector.bg_fg_mask
    motion_mask = detector.motion_mask
    combined_mask = detector.combined_mask

    if i in expected:
        ex, ey = expected[i]
    elif i <= 57:
        # Use known positions
        known = {54: (602, 165), 55: (602, 192), 56: (605, 211), 57: (613, 214)}
        ex, ey = known.get(i, (600, 190))
    else:
        continue

    # Check pixel values in a box around expected position
    h, w = grey.shape[:2]
    x1 = max(0, ex - R)
    x2 = min(w, ex + R)
    y1 = max(0, ey - R)
    y2 = min(h, ey + R)

    white_region = white_mask[y1:y2, x1:x2] if white_mask is not None else None
    bg_fg_region = bg_fg_mask[y1:y2, x1:x2] if bg_fg_mask is not None else None
    motion_region = motion_mask[y1:y2, x1:x2] if motion_mask is not None else None
    combined_region = combined_mask[y1:y2, x1:x2] if combined_mask is not None else None

    white_px = np.count_nonzero(white_region) if white_region is not None else -1
    bg_fg_px = np.count_nonzero(bg_fg_region) if bg_fg_region is not None else -1
    motion_px = np.count_nonzero(motion_region) if motion_region is not None else -1
    comb_px = np.count_nonzero(combined_region) if combined_region is not None else -1

    # Check what's actually at the expected position in the raw frame
    if 0 <= ey < h and 0 <= ex < w:
        bgr_val = frame[ey, ex]
        hsv_val = hsv[ey, ex]
        grey_val = grey[ey, ex]
    else:
        bgr_val = (-1, -1, -1)
        hsv_val = (-1, -1, -1)
        grey_val = -1

    # Get candidates near expected position
    near_cands = []
    for c in detector.candidates:
        d = np.hypot(c.center[0] - ex, c.center[1] - ey)
        if d < 50:
            near_cands.append((d, c))

    # Print diagnostics
    print(f"Frame {i:3d}: expected=({ex:4d},{ey:4d})")
    print(f"  BGR={bgr_val}  HSV={hsv_val}  Grey={grey_val}")
    print(f"  Mask px in {R}px box: white={white_px:3d}  bg_fg={bg_fg_px:3d}  motion={motion_px:3d}  combined={comb_px:3d}")
    print(f"  Total candidates: {len(detector.candidates)}")
    if near_cands:
        near_cands.sort()
        print(f"  Nearby candidates:")
        for d, c in near_cands[:5]:
            flag = "MOTION" if c.in_motion_mask else "static"
            print(f"    d={d:5.1f} ({c.center[0]:4d},{c.center[1]:4d}) area={c.area:5.0f} circ={c.circularity:.2f} {flag}")
    else:
        print(f"  NO nearby candidates within 50px")
    print()
