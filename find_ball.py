"""Find ALL white blobs in frames 55-88 to locate the real ball trajectory."""
import sys
sys.path.insert(0, ".")
import cv2
import numpy as np
from pathlib import Path
from pitch_analyzer import BALL_HSV_LOWER, BALL_HSV_UPPER, BALL_MIN_AREA, BALL_MAX_AREA

folder = Path("pitches/20260227_205241")
files = sorted(f for f in folder.iterdir() if f.suffix.lower() == ".png")
print(f"Loaded {len(files)} frames\n")

prev_grey = None

for i, fpath in enumerate(files):
    frame = cv2.imread(str(fpath))
    if frame is None:
        continue

    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # White mask
    white = cv2.inRange(hsv, BALL_HSV_LOWER, BALL_HSV_UPPER)
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)

    # Frame diff
    if prev_grey is not None:
        diff = cv2.absdiff(grey, prev_grey)
        _, motion = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
        motion = cv2.dilate(motion, np.ones((5, 5), np.uint8), iterations=2)
    else:
        motion = np.zeros_like(grey)
    prev_grey = grey.copy()

    if i < 54 or i > 88:
        continue

    # Find all white blobs
    contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    white_blobs = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 4:  # very small threshold to catch everything
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        perimeter = cv2.arcLength(cnt, True)
        circ = 4.0 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        # Check if this blob overlaps with motion
        blob_mask = np.zeros_like(grey)
        cv2.drawContours(blob_mask, [cnt], -1, 255, -1)
        overlap = cv2.bitwise_and(motion, blob_mask)
        has_motion = np.count_nonzero(overlap) > 0
        white_blobs.append((cx, cy, area, circ, has_motion))

    # Also find motion-only blobs (non-white ball detection)
    motion_only = cv2.bitwise_and(motion, cv2.bitwise_not(white))
    m_contours, _ = cv2.findContours(motion_only, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    motion_blobs = []
    for cnt in m_contours:
        area = cv2.contourArea(cnt)
        if not (BALL_MIN_AREA <= area <= 200):
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        perimeter = cv2.arcLength(cnt, True)
        circ = 4.0 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        if circ > 0.3:  # somewhat circular
            motion_blobs.append((cx, cy, area, circ))

    # Print
    ball_sized = [(cx, cy, a, c, m) for cx, cy, a, c, m in white_blobs
                  if BALL_MIN_AREA <= a <= BALL_MAX_AREA]
    small_white = [(cx, cy, a, c, m) for cx, cy, a, c, m in white_blobs if a < 200]

    print(f"Frame {i:3d}: {len(white_blobs)} white blobs, {len(ball_sized)} ball-sized, {len(motion_blobs)} motion-only")
    for cx, cy, a, c, m in small_white:
        flag = "+MOT" if m else "    "
        in_corridor = "CORR" if 380 <= cx <= 820 else "    "
        print(f"  white ({cx:4d},{cy:4d}) area={a:5.0f} circ={c:.2f} {flag} {in_corridor}")
    if motion_blobs:
        for cx, cy, a, c in motion_blobs[:10]:
            in_corridor = "CORR" if 380 <= cx <= 820 else "    "
            print(f"  motio ({cx:4d},{cy:4d}) area={a:5.0f} circ={c:.2f}      {in_corridor}")
    print()
