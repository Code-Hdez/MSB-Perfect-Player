"""Diagnostic: Dump all contours + candidate details for frames 60-90.
Shows exactly what the detector sees at each frame near the ball."""
import sys
sys.path.insert(0, ".")
import cv2, numpy as np
from pathlib import Path
from pitch_analyzer import (
    BallDetector, TrajectoryCorridor, CORRIDOR_DEFAULT,
    BALL_HSV_LOWER, BALL_HSV_UPPER, BALL_MIN_AREA, BALL_MAX_AREA,
    BALL_MIN_CIRCULARITY, BG_FG_THRESHOLD,
)

folder = Path(sys.argv[1] if len(sys.argv) > 1 else "pitches/20260227_205241")
files = sorted(f for f in folder.iterdir() if f.suffix.lower() in {".png", ".jpg"})

detector = BallDetector()
corridor = TrajectoryCorridor()
if CORRIDOR_DEFAULT:
    corridor.set_rect(*CORRIDOR_DEFAULT)

# GT positions for reference
GT = {60: (570, 217), 70: (573, 342), 80: (573, 543)}

DIAG_FRAMES = set(range(60, 91))

for i, f in enumerate(files):
    frame = cv2.imread(str(f))
    if frame is None:
        continue

    # Run detection (to build BG model, etc.)
    best = detector.detect(frame, corridor=corridor, track_active=False)

    if i not in DIAG_FRAMES:
        continue

    gt = GT.get(i, None)
    gt_str = f"  GT=({gt[0]},{gt[1]})" if gt else ""
    n_cands = len(detector.candidates)
    print(f"\n=== Frame {i}{gt_str} ===  candidates={n_cands}")

    # Show candidates sorted by score
    for j, c in enumerate(detector.candidates[:10]):
        dist_gt = ""
        if gt:
            d = np.hypot(c.center[0] - gt[0], c.center[1] - gt[1])
            dist_gt = f"  dist_gt={d:.0f}"
        print(f"  [{j}] ({c.center[0]:4d},{c.center[1]:4d}) a={c.area:5.0f} "
              f"circ={c.circularity:.2f} motion={c.in_motion_mask} "
              f"score={c.score:.3f}{dist_gt}")

    # Check specific GT pixel
    if gt:
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gx, gy = gt
        if 0 <= gy < grey.shape[0] and 0 <= gx < grey.shape[1]:
            g = grey[gy, gx]
            h, s, v = hsv[gy, gx]
            bg_fg_val = detector.bg_fg_mask[gy, gx] if detector.bg_fg_mask is not None else -1
            motion_val = detector.motion_mask[gy, gx] if detector.motion_mask is not None else -1
            white_val = detector.white_mask[gy, gx] if detector.white_mask is not None else -1
            combined_val = detector.combined_mask[gy, gx] if detector.combined_mask is not None else -1
            print(f"  GT pixel: grey={g} HSV=({h},{s},{v}) bg_fg={bg_fg_val} "
                  f"motion={motion_val} white={white_val} combined={combined_val}")

    # Dump raw combined mask contours near y=300-600
    if detector.combined_mask is not None:
        cnts, _ = cv2.findContours(detector.combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        interesting = []
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cy = int(M["m01"] / M["m00"])
                cx = int(M["m10"] / M["m00"])
            else:
                bx, by, bw, bh = cv2.boundingRect(cnt)
                cx, cy = bx + bw // 2, by + bh // 2
            if 200 < cy < 700 and 450 < cx < 750:
                per = cv2.arcLength(cnt, True)
                circ = 4 * np.pi * area / (per * per) if per > 0 else 0
                interesting.append((cx, cy, area, circ))
        if interesting:
            interesting.sort(key=lambda t: t[2], reverse=True)
            print(f"  Raw contours (y=200-700, x=450-750): {len(interesting)}")
            for cx, cy, a, circ in interesting[:8]:
                gt_d = ""
                if gt:
                    d = np.hypot(cx - gt[0], cy - gt[1])
                    gt_d = f" dist_gt={d:.0f}"
                print(f"    ({cx:4d},{cy:4d}) area={a:6.1f} circ={circ:.2f}{gt_d}")
