"""Quick test: run ball detection + tracking on the saved pitch frames
and print what the tracker actually selects per frame."""
import sys
sys.path.insert(0, ".")
import cv2
import numpy as np
from pathlib import Path
from pitch_analyzer import (
    BallDetector, BallTracker, TrajectoryPredictor,
    TRACK_MAX_DIST_MIN, TRACK_LOST_FRAMES, PITCHER_ZONE
)

folder = Path(sys.argv[1] if len(sys.argv) > 1 else "pitches/20260227_205241")
exts = {".png", ".jpg"}
files = sorted(f for f in folder.iterdir() if f.suffix.lower() in exts)
print(f"Loaded {len(files)} frames")
print(f"Pitcher zone: {PITCHER_ZONE}")
print()

detector = BallDetector()
tracker = BallTracker()

for i, f in enumerate(files):
    frame = cv2.imread(str(f))
    if frame is None:
        print(f"Frame {i:3d}: COULD NOT READ")
        continue

    best = detector.detect(frame)
    track = tracker.update(detector.candidates, best)
    sel = tracker.selected

    if sel is not None:
        c = sel.center
        trk_info = ""
        if track and track.active:
            conf = "CONFIRMED" if track.confirmed else "tentative"
            trk_info = (f"  vel=({track.velocity[0]:+.1f},{track.velocity[1]:+.1f})"
                        f"  pts={len(track.positions)}  [{conf}]"
                        f"  missed={track.frames_since_seen}")
        print(f"Frame {i:3d}: TRACKED ({c[0]:4d},{c[1]:4d})  a={sel.area:5.0f} circ={sel.circularity:.2f} motion={sel.in_motion_mask}{trk_info}")
    elif best is not None:
        c = best.center
        print(f"Frame {i:3d}: notrack  best=({c[0]:4d},{c[1]:4d})  a={best.area:5.0f}")
    else:
        print(f"Frame {i:3d}: NONE")

    # Print track state changes
    if track and not track.active:
        print(f"          ^ TRACK LOST after {track.total_frames} frames")
    elif track and track.active and track.frames_since_seen > 0:
        pred = track.predicted_next
        print(f"          ^ searching... predicted=({pred[0]},{pred[1]}) missed={track.frames_since_seen}")
