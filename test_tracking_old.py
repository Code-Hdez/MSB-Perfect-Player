"""Quick test: run ball detection + tracking on the saved pitch frames
and print what the tracker actually selects per frame.

Updated for v2 pipeline (background model, Kalman filter, pitcher
suppression, trajectory corridor).
"""
import sys
sys.path.insert(0, ".")
import cv2
import numpy as np
from pathlib import Path
from pitch_analyzer import (
    BallDetector, BallTracker, TrajectoryPredictor, TrajectoryCorridor,
    TrackState, TRACK_MAX_DIST_MIN, TRACK_LOST_FRAMES, PITCHER_ZONE,
    CORRIDOR_DEFAULT, REACQ_ZONE, _in_reacq_zone, _in_pitcher_zone,
)

folder = Path(sys.argv[1] if len(sys.argv) > 1 else "pitches/20260227_205241")
exts = {".png", ".jpg"}
files = sorted(f for f in folder.iterdir() if f.suffix.lower() in exts)
print(f"Loaded {len(files)} frames")
print(f"Pitcher zone: {PITCHER_ZONE}")
print(f"Reacq zone:   {REACQ_ZONE}")
print(f"Corridor default: {CORRIDOR_DEFAULT}")
print()

detector = BallDetector()
tracker = BallTracker()
corridor = TrajectoryCorridor()
if CORRIDOR_DEFAULT:
    corridor.set_rect(*CORRIDOR_DEFAULT)

for i, f in enumerate(files):
    frame = cv2.imread(str(f))
    if frame is None:
        print(f"Frame {i:3d}: COULD NOT READ")
        continue

    track_active = (tracker.track is not None
                    and tracker.track.active
                    and tracker.track.confirmed)
    best = detector.detect(frame, corridor=corridor,
                           track_active=track_active)
    track = tracker.update(detector.candidates, best, detector=detector)
    sel = tracker.selected

    state = tracker.state.name

    if sel is not None:
        c = sel.center
        trk_info = ""
        rescue_tag = ""
        if track and len(track.positions) == 1 and _in_reacq_zone(c[0], c[1]):
            rescue_tag = "  [REACQ]"
        if track and track.active:
            conf = "CONFIRMED" if track.confirmed else "tentative"
            trk_info = (f"  vel=({track.velocity[0]:+.1f},{track.velocity[1]:+.1f})"
                        f"  pts={len(track.positions)}  [{conf}]"
                        f"  missed={track.frames_since_seen}"
                        f"  pz={track._frames_in_pitcher_zone}"
                        f"  iso={sel.isolation_score:.2f}")
        print(f"Frame {i:3d}: TRACKED ({c[0]:4d},{c[1]:4d})  a={sel.area:5.0f} circ={sel.circularity:.2f} [{state}]{trk_info}{rescue_tag}")
    elif best is not None:
        c = best.center
        print(f"Frame {i:3d}: notrack  best=({c[0]:4d},{c[1]:4d})  a={best.area:5.0f} iso={best.isolation_score:.2f} [{state}]")
    else:
        print(f"Frame {i:3d}: NONE [{state}]")

    # Print track state changes
    if track and not track.active:
        print(f"          ^ TRACK LOST after {track.total_frames} frames")
    elif track and track.active and track.frames_since_seen > 0:
        pred = track.predicted_next
        print(f"          ^ searching... predicted=({pred[0]},{pred[1]}) missed={track.frames_since_seen}")

# Summary
print(f"\n--- BG model ready: {detector.bg_model.ready} "
      f"(frames processed: {detector.bg_model._frame_count}) ---")
