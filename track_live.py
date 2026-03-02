"""
track_live.py — Live ball tracking from screen capture (dxcam).

CLI tool: no need to edit library code for experiments.

Usage examples
--------------
  python track_live.py
  python track_live.py -c config.toml
  python track_live.py --corridor-rect 380 80 820 900
  python track_live.py --debug

Controls
--------
  SPACE  = Start / stop recording a pitch
  D      = Toggle debug panel
  C      = Click to define ball search ROI
  Y      = Click to set strike-zone Y level
  S      = Save current pitch recording to disk
  R      = Reset tracker
  Q/ESC  = Quit
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from msb.config import Config
from msb.detector import BallDetector
from msb.tracker import BallTracker
from msb.predictor import TrajectoryPredictor
from msb.corridor import TrajectoryCorridor
from msb.recorder import PitchRecorder
from msb.visualiser import PitchVisualiser, ROISelector, StrikeYSelector
from msb.sources import source_live
from msb.detector_ml import MLBallDetector
from msb.tracker_ml import MLBallTracker

WINDOW_MAIN = "MSB Pitch Analyzer"
WINDOW_DEBUG = "MSB Pitch Debug"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Live ball tracking from screen capture")
    ap.add_argument("-c", "--config", default=None,
                    help="Path to config file (TOML or JSON)")
    ap.add_argument("--corridor", default=None,
                    help="Path to corridor JSON")
    ap.add_argument("--corridor-rect", nargs=4, type=int, default=None,
                    metavar=("X1", "Y1", "X2", "Y2"),
                    help="Override corridor rectangle")
    ap.add_argument("--pitcher-zone", nargs=4, type=int, default=None,
                    metavar=("X1", "Y1", "X2", "Y2"),
                    help="Override pitcher zone")
    ap.add_argument("--model", "-m", default=None,
                    help="Path to YOLO .pt or .onnx model for ML detection")
    ap.add_argument("--conf", type=float, default=0.25,
                    help="ML detector confidence threshold (default: 0.25)")
    ap.add_argument("--imgsz", type=int, default=960,
                    help="ML detector inference resolution (default: 960)")
    ap.add_argument("--debug", action="store_true",
                    help="Start with debug panel enabled")
    args = ap.parse_args()

    # Load config
    if args.config:
        cfg = Config.load(args.config)
    else:
        cfg_path = Path("config.toml")
        if cfg_path.exists():
            cfg = Config.load(str(cfg_path))
        else:
            cfg = Config()

    overrides: Dict[str, Any] = {}
    if args.corridor_rect:
        overrides["corridor_default"] = tuple(args.corridor_rect)
    if args.pitcher_zone:
        overrides["pitcher_zone"] = tuple(args.pitcher_zone)
    cfg.apply_overrides(overrides)

    # Components
    if args.model:
        detector = MLBallDetector(args.model, conf=args.conf,
                                  imgsz=args.imgsz, cfg=cfg)
        tracker = MLBallTracker(cfg=cfg)
        print(f"[INFO] Using ML detector: {args.model} "
              f"(conf={args.conf}, imgsz={args.imgsz})")
    else:
        detector = BallDetector(cfg)
        tracker = BallTracker(cfg)
        print("[INFO] Using classical detector")
    predictor = TrajectoryPredictor(cfg=cfg)
    recorder  = PitchRecorder(cfg)
    vis       = PitchVisualiser()
    roi_sel   = ROISelector(display_scale=cfg.display_scale)
    sy_sel    = StrikeYSelector(display_scale=cfg.display_scale)

    corridor = TrajectoryCorridor(cfg)
    if args.corridor and Path(args.corridor).exists():
        corridor.load(Path(args.corridor))
        print(f"[INFO] Loaded corridor from {args.corridor}")
    elif cfg.corridor_default is not None:
        corridor.set_rect(*cfg.corridor_default)
        print(f"[INFO] Using default corridor: {cfg.corridor_default}")

    show_debug: bool = args.debug
    search_roi: Optional[Tuple[int, int, int, int]] = None
    frame_num: int = 0

    print(f"[INFO] source=live")
    src = source_live(cfg)

    prev_t = time.perf_counter()
    fps_ema = float(cfg.target_fps)

    print("\n CONTROLS")
    print("  SPACE = Start / stop recording")
    print("  D     = Toggle debug panel")
    print("  C     = Click to define ball search ROI")
    print("  Y     = Click to set strike-zone Y level")
    print("  S     = Save recorded pitch")
    print("  R     = Reset tracker")
    print("  Q/ESC = Quit\n")

    cv2.namedWindow(WINDOW_MAIN, cv2.WINDOW_NORMAL)

    def _mouse_cb(event: int, x: int, y: int,
                  flags: int, param: Any) -> None:
        roi_sel.mouse_callback(event, x, y, flags, param)
        sy_sel.mouse_callback(event, x, y, flags, param)

    cv2.setMouseCallback(WINDOW_MAIN, _mouse_cb)
    scale = cfg.display_scale

    try:
        for frame in src:
            if frame is None:
                time.sleep(0.001)
                continue

            frame_num += 1
            now = time.perf_counter()
            dt = now - prev_t
            prev_t = now
            ifps = 1.0 / dt if dt > 0 else 0.0
            fps_ema = 0.1 * ifps + 0.9 * fps_ema

            # ROI / Strike-Y selection modes
            if roi_sel.active:
                frozen = roi_sel.draw_frozen()
                if frozen is not None:
                    if scale != 1.0:
                        dw = int(frozen.shape[1] * scale)
                        dh = int(frozen.shape[0] * scale)
                        frozen = cv2.resize(frozen, (dw, dh))
                    cv2.imshow(WINDOW_MAIN, frozen)
                key = cv2.waitKey(30) & 0xFF
                if key == 27:
                    roi_sel.cancel()
                elif not roi_sel.active and roi_sel.rect is not None:
                    search_roi = roi_sel.rect
                    detector.reset()
                continue

            if not roi_sel.active and roi_sel.rect is not None:
                search_roi = roi_sel.rect
                roi_sel.rect = None

            if sy_sel.active:
                frozen = sy_sel.draw_frozen()
                if frozen is not None:
                    if scale != 1.0:
                        dw = int(frozen.shape[1] * scale)
                        dh = int(frozen.shape[0] * scale)
                        frozen = cv2.resize(frozen, (dw, dh))
                    cv2.imshow(WINDOW_MAIN, frozen)
                key = cv2.waitKey(30) & 0xFF
                if key == 27:
                    sy_sel.cancel()
                elif not sy_sel.active and sy_sel.target_y is not None:
                    predictor.set_target_y(sy_sel.target_y)
                continue

            if not sy_sel.active and sy_sel.target_y is not None:
                predictor.set_target_y(sy_sel.target_y)

            # Detection
            track_active = (tracker.track is not None
                            and tracker.track.active
                            and tracker.track.confirmed)
            best = detector.detect(frame, search_roi, corridor,
                                   track_active=track_active)

            # Tracking
            track = tracker.update(detector.candidates, best)
            tracked_ball = tracker.selected if tracker.selected else best

            # Following / extrapolation
            if track is not None and track.active:
                predictor.predict(track)
                if track.confirmed:
                    corridor.expand_dynamic(track.last_pos[0],
                                            track.last_pos[1])

            # Recording
            if recorder.recording:
                recorder.add_frame(frame, tracked_ball)

            # Draw
            disp = vis.overlay(frame, detector, tracker, predictor,
                               corridor, search_roi,
                               recorder.recording, fps_ema, frame_num,
                               cfg=cfg)
            if scale != 1.0:
                dw = int(disp.shape[1] * scale)
                dh = int(disp.shape[0] * scale)
                disp = cv2.resize(disp, (dw, dh))
            cv2.imshow(WINDOW_MAIN, disp)

            if show_debug:
                dbg = vis.debug_panel(detector, frame, search_roi)
                cv2.imshow(WINDOW_DEBUG, dbg)

            # Keys
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord(" "):
                if recorder.recording:
                    recorder.stop()
                else:
                    recorder.start()
                    detector.reset()
                    tracker.reset()
            elif key == ord("d"):
                show_debug = not show_debug
                if not show_debug:
                    try:
                        cv2.destroyWindow(WINDOW_DEBUG)
                    except cv2.error:
                        pass
                print(f"[INFO] Debug {'ON' if show_debug else 'OFF'}")
            elif key == ord("c"):
                roi_sel.start(frame)
            elif key == ord("y"):
                sy_sel.start(frame)
            elif key == ord("s"):
                recorder.save()
            elif key == ord("r"):
                detector.reset()
                tracker.reset()
                print("[INFO] Tracker reset.")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")
    finally:
        cv2.destroyAllWindows()
        print("[INFO] Done.")


if __name__ == "__main__":
    main()
