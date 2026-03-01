"""Quick test: run ball detection + tracking on saved pitch frames
and print what the tracker selects per frame.

Updated for the ``msb`` package pipeline.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

from msb import Config, BallDetector, BallTracker, TrajectoryCorridor, TrackState
from msb.utils import in_rect


def main() -> None:
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else "pitches/20260227_205241")

    # Load config (auto-detect config.toml in cwd)
    cfg_path = Path("config.toml")
    cfg = Config.load(str(cfg_path)) if cfg_path.exists() else Config()

    exts = {".png", ".jpg"}
    files = sorted(f for f in folder.iterdir() if f.suffix.lower() in exts)
    print(f"Loaded {len(files)} frames")
    print(f"Pitcher zone: {cfg.pitcher_zone}")
    print(f"Reacq zone:   {cfg.reacq_zone}")
    print(f"Corridor default: {cfg.corridor_default}")
    print()

    detector = BallDetector(cfg)
    tracker = BallTracker(cfg)
    corridor = TrajectoryCorridor(cfg)
    if cfg.corridor_default:
        corridor.set_rect(*cfg.corridor_default)

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
            rescue_tag = ""
            trk_info = ""
            if track and len(track.positions) == 1 and in_rect(c[0], c[1], cfg.reacq_zone):
                rescue_tag = "  [REACQ]"
            if track and track.active:
                conf = "CONFIRMED" if track.confirmed else "tentative"
                trk_info = (
                    f"  vel=({track.velocity[0]:+.1f},{track.velocity[1]:+.1f})"
                    f"  pts={len(track.positions)}  [{conf}]"
                    f"  missed={track.frames_since_seen}"
                    f"  pz={track._frames_in_pitcher_zone}"
                    f"  iso={sel.isolation_score:.2f}"
                )
            print(
                f"Frame {i:3d}: TRACKED ({c[0]:4d},{c[1]:4d})"
                f"  a={sel.area:5.0f} circ={sel.circularity:.2f}"
                f" [{state}]{trk_info}{rescue_tag}"
            )
        elif best is not None:
            c = best.center
            print(
                f"Frame {i:3d}: notrack  best=({c[0]:4d},{c[1]:4d})"
                f"  a={best.area:5.0f} iso={best.isolation_score:.2f}"
                f" [{state}]"
            )
        else:
            print(f"Frame {i:3d}: NONE [{state}]")

        # Track state changes
        if track and not track.active:
            print(f"          ^ TRACK LOST after {track.total_frames} frames")
        elif track and track.active and track.frames_since_seen > 0:
            pred = track.predicted_next
            print(
                f"          ^ searching... predicted=({pred[0]},{pred[1]})"
                f" missed={track.frames_since_seen}"
            )

    # Summary
    print(
        f"\n--- BG model ready: {detector.bg_model.ready} "
        f"(frames processed: {detector.bg_model._frame_count}) ---"
    )


if __name__ == "__main__":
    main()
