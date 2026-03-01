"""
Pitch / Ball Detection & Trajectory Analyzer — backward-compatibility wrapper.

This module re-exports all classes and constants from the ``msb`` package
so that existing scripts (diagnostic tools, test runners) continue to work
without modification.

**New code should import from ``msb.*`` directly.**

Usage unchanged::

    python pitch_analyzer.py
    python pitch_analyzer.py --source folder --input ./pitches/20260227_205241
"""

from __future__ import annotations

# Re-export everything from msb package

from msb.config import Config
from msb.utils import (
    crop, put_text, in_rect,
    COL_GREEN, COL_RED, COL_YELLOW, COL_CYAN, COL_WHITE,
    COL_BLACK, COL_MAGENTA, COL_ORANGE, FONT,
)
from msb.detector import (
    BallCandidate, BackgroundModel, PitcherSuppressor, BallDetector,
)
from msb.tracker import TrackState, BallTrack, BallTracker, TrackedBall
from msb.corridor import TrajectoryCorridor
from msb.predictor import TrajectoryPredictor
from msb.recorder import PitchRecorder
from msb.visualiser import PitchVisualiser, ROISelector, StrikeYSelector
from msb.sources import source_live, source_folder

# Default config → backward-compatible module-level constants

_cfg = Config()

SCREEN_ROI = _cfg.screen_roi
MONITOR_INDEX = _cfg.monitor_index
TARGET_FPS = _cfg.target_fps

BALL_HSV_LOWER = __import__("numpy").array(_cfg.ball_hsv_lower)
BALL_HSV_UPPER = __import__("numpy").array(_cfg.ball_hsv_upper)
TRAIL_HSV_LOWER = __import__("numpy").array(_cfg.trail_hsv_lower)
TRAIL_HSV_UPPER = __import__("numpy").array(_cfg.trail_hsv_upper)

BALL_MIN_AREA = _cfg.ball_min_area
BALL_MAX_AREA = _cfg.ball_max_area
BALL_FLIGHT_MAX_AREA = _cfg.ball_flight_max_area
BALL_MIN_CIRCULARITY = _cfg.ball_min_circularity

BG_ALPHA = _cfg.bg_alpha
BG_WARMUP_FRAMES = _cfg.bg_warmup_frames
BG_FG_THRESHOLD = _cfg.bg_fg_threshold

DIFF_THRESHOLD = _cfg.diff_threshold
DIFF_DILATE_ITER = _cfg.diff_dilate_iter

PITCHER_BODY_MIN_AREA = _cfg.pitcher_body_min_area
ISOLATION_ZONE_SCALE = _cfg.isolation_zone_scale
ISOLATION_INNER = _cfg.isolation_inner
ISOLATION_OUTER = _cfg.isolation_outer

CORRIDOR_DEFAULT = _cfg.corridor_default
CORRIDOR_MARGIN = _cfg.corridor_margin
CORRIDOR_PENALTY_DIST = _cfg.corridor_penalty_dist

TRACK_MAX_DIST_MIN = _cfg.track_max_dist_min
TRACK_MAX_DIST_SPEED_K = _cfg.track_max_dist_speed_k
TRACK_GAP_EXPAND = _cfg.track_gap_expand
TRACK_LOST_FRAMES = _cfg.track_lost_frames
TRACK_TENTATIVE_LOST = _cfg.track_tentative_lost
TRAJECTORY_HISTORY = _cfg.trajectory_history
TRACK_MIN_CONFIRMATIONS = _cfg.track_min_confirmations
TRACK_SIZE_RATIO = _cfg.track_size_ratio
MIN_PITCH_VY = _cfg.min_pitch_vy
MAX_PITCHER_ZONE_FRAMES = _cfg.max_pitcher_zone_frames
MIN_DEPARTURE_DIST = _cfg.min_departure_dist
MAX_VY_SIGN_CHANGES = _cfg.max_vy_sign_changes

KF_PROCESS_NOISE = _cfg.kf_process_noise
KF_MEASUREMENT_NOISE = _cfg.kf_measurement_noise

PITCHER_ZONE = _cfg.pitcher_zone
REACQ_ZONE = _cfg.reacq_zone
REACQ_WINDOW = _cfg.reacq_window

STATIC_CELL_SIZE = _cfg.static_cell_size
STATIC_HIT_THRESHOLD = _cfg.static_hit_threshold

RECORD_MAX_FRAMES = _cfg.record_max_frames
PITCHES_DIR = __import__("pathlib").Path(_cfg.pitches_dir)

DISPLAY_SCALE = _cfg.display_scale

WINDOW_MAIN = "MSB Pitch Analyzer"
WINDOW_DEBUG = "MSB Pitch Debug"


# Backward-compatible helper functions

def _in_rect(x: int, y: int, rect) -> bool:
    return in_rect(x, y, rect)


def _in_pitcher_zone(x: int, y: int) -> bool:
    return in_rect(x, y, PITCHER_ZONE)


def _in_reacq_zone(x: int, y: int) -> bool:
    return in_rect(x, y, REACQ_ZONE)


# Entry point (same as before)

def main() -> None:
    """Run the live or folder pitch analyser (legacy entry point)."""
    import argparse
    import time

    import cv2

    ap = argparse.ArgumentParser(
        description="MSB Pitch / Ball Detection & Trajectory Analyzer")
    ap.add_argument("--source", choices=["live", "folder"], default="live")
    ap.add_argument("--input", default=None,
                    help="Folder of pitch frames (for folder source)")
    ap.add_argument("--corridor", default=None,
                    help="Path to corridor JSON")
    ap.add_argument("-c", "--config", default=None,
                    help="Path to config file (TOML or JSON)")
    args = ap.parse_args()

    from pathlib import Path

    if args.config:
        cfg = Config.load(args.config)
    else:
        cfg_path = Path("config.toml")
        cfg = Config.load(str(cfg_path)) if cfg_path.exists() else Config()

    detector  = BallDetector(cfg)
    tracker   = BallTracker(cfg)
    predictor = TrajectoryPredictor(cfg=cfg)
    recorder  = PitchRecorder(cfg)
    vis       = PitchVisualiser()
    roi_sel   = ROISelector(display_scale=cfg.display_scale)
    sy_sel    = StrikeYSelector(display_scale=cfg.display_scale)

    corridor = TrajectoryCorridor(cfg)
    if args.corridor and Path(args.corridor).exists():
        corridor.load(Path(args.corridor))
    elif cfg.corridor_default is not None:
        corridor.set_rect(*cfg.corridor_default)

    show_debug = False
    search_roi = None
    frame_num = 0
    is_folder = (args.source == "folder")

    if args.source == "live":
        src = source_live(cfg)
    else:
        if not args.input:
            print("[ERROR] --input required for folder source")
            import sys; sys.exit(1)
        src = source_folder(args.input)

    prev_t = time.perf_counter()
    fps_ema = float(cfg.target_fps)

    print("\n CONTROLS")
    print("  SPACE = Record  |  D = Debug  |  C = ROI  |  Y = Strike-Y")
    print("  S = Save  |  R = Reset  |  Q/ESC = Quit\n")

    cv2.namedWindow(WINDOW_MAIN, cv2.WINDOW_NORMAL)

    def _mouse_cb(event, x, y, flags, param):
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

            track_active = (tracker.track is not None
                            and tracker.track.active
                            and tracker.track.confirmed)
            best = detector.detect(frame, search_roi, corridor,
                                   track_active=track_active)
            track = tracker.update(detector.candidates, best)
            tracked_ball = tracker.selected if tracker.selected else best

            if track is not None and track.active:
                predictor.predict(track)
                if track.confirmed:
                    corridor.expand_dynamic(track.last_pos[0],
                                            track.last_pos[1])

            if recorder.recording:
                recorder.add_frame(frame, tracked_ball)

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

            wait_ms = 33 if is_folder else 1
            key = cv2.waitKey(wait_ms) & 0xFF
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
            elif key == ord("c"):
                roi_sel.start(frame)
            elif key == ord("y"):
                sy_sel.start(frame)
            elif key == ord("s"):
                recorder.save()
            elif key == ord("r"):
                detector.reset()
                tracker.reset()

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")
    finally:
        cv2.destroyAllWindows()
        print("[INFO] Done.")


if __name__ == "__main__":
    main()
