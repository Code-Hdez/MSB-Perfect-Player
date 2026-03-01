"""Show ALL candidates for frames 50-90 to find the real ball trajectory."""
import sys
sys.path.insert(0, ".")
import cv2
import numpy as np
from pathlib import Path
from pitch_analyzer import BallDetector, BALL_FLIGHT_MAX_AREA

folder = Path("pitches/20260227_205241")
files = sorted(f for f in folder.iterdir() if f.suffix.lower() == ".png")
print(f"Loaded {len(files)} frames\n")

detector = BallDetector()

for i, f in enumerate(files):
    if i < 50 or i > 92:
        # Still process to keep frame differencing correct
        frame = cv2.imread(str(f))
        if frame is not None:
            detector.detect(frame)
        continue

    frame = cv2.imread(str(f))
    if frame is None:
        continue

    detector.detect(frame)
    cands = detector.candidates

    # Filter to small candidates (ball-sized) that are in motion
    small = [c for c in cands if c.area <= BALL_FLIGHT_MAX_AREA and c.in_motion_mask]
    all_small = [c for c in cands if c.area <= BALL_FLIGHT_MAX_AREA]

    print(f"Frame {i:3d}: {len(cands)} total cands, {len(small)} small+motion, {len(all_small)} small+any")
    for j, c in enumerate(all_small):
        flag = " MOTION" if c.in_motion_mask else " static"
        print(f"    [{j}] ({c.center[0]:4d},{c.center[1]:4d}) area={c.area:5.0f} circ={c.circularity:.2f} score={c.score:.2f}{flag}")
    print()
