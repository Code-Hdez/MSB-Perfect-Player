"""bat_diag.py - Automated batter with full batter tracking.

Detect ball (YOLO) → Track → Predict trajectory → Locate batter/strike
zone (character calibration) → Move batter to intercept → Swing.

Usage:
  python bat_diag.py -m weights/ball_best.pt --character papihongo
  python bat_diag.py -m weights/ball_best.pt --target-y 682   # no character, manual target
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes as _wt
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from msb.config import Config
from msb.detector_ml import MLBallDetector
from msb.tracker_ml import MLBallTracker, MLTrackState
from msb.predictor import TrajectoryPredictor
from msb.corridor import TrajectoryCorridor
from msb.dolphin_input import DolphinGamepadInput
from msb.sources import source_live
from msb.utils import put_text, COL_GREEN, COL_RED, COL_YELLOW, COL_CYAN, COL_WHITE, COL_MAGENTA

sys.path.insert(0, str(Path(__file__).resolve().parent / "features" / "batter_hitbox"))
try:
    from msb_hitbox_detector import (
        HitboxDetector,
        MovementTracker,
        BatterStateClassifier,
        BatterState,
    )
    _HAS_HITBOX = True
except ImportError as _ie:
    print(f"[WARN] Hitbox subsystem unavailable ({_ie}). "
          "Character tracking disabled — use --target-y instead.")
    _HAS_HITBOX = False

WINDOW = "DIAG"
FONT = cv2.FONT_HERSHEY_SIMPLEX

def _pt(*a) -> Tuple[int, ...]:
    return tuple(int(float(x)) for x in a)


# Window-focus helpers (Windows-only)

_user32 = ctypes.windll.user32

def _find_dolphin_hwnd() -> int:
    """Find the main Dolphin game-render window (HWND), or 0."""
    EnumWindows = _user32.EnumWindows
    WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, _wt.HWND, _wt.LPARAM)
    result = [0]

    def cb(hwnd, _lp):
        length = _user32.GetWindowTextLengthW(hwnd) + 1
        if length <= 1:
            return True
        buf = ctypes.create_unicode_buffer(length)
        _user32.GetWindowTextW(hwnd, buf, length)
        t = buf.value.lower()
        if _user32.IsWindowVisible(hwnd):
            if ('jit64' in t or 'jit arm' in t or 'interpreter' in t
                    or 'dolphin' in t and ('|' in t or 'gc' in t)):
                result[0] = hwnd
                return False
        return True

    EnumWindows(WNDENUMPROC(cb), 0)
    return result[0]


def _set_cv_window_no_activate(window_name: str) -> None:
    """Add WS_EX_NOACTIVATE to an OpenCV window so it never steals focus."""
    try:
        hwnd = _user32.FindWindowW(None, window_name)
        if hwnd:
            GWL_EXSTYLE = -20
            WS_EX_NOACTIVATE = 0x08000000
            WS_EX_APPWINDOW = 0x00040000
            exstyle = _user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            exstyle = (exstyle | WS_EX_NOACTIVATE) & ~WS_EX_APPWINDOW
            _user32.SetWindowLongW(hwnd, GWL_EXSTYLE, exstyle)
    except Exception:
        pass


def _focus_dolphin(hwnd: int) -> None:
    """Give keyboard/input focus back to Dolphin's game window."""
    if hwnd:
        try:
            SWP_NOMOVE = 0x0002
            SWP_NOSIZE = 0x0001
            SWP_SHOWWINDOW = 0x0040
            _user32.SetWindowPos(hwnd, 0, 0, 0, 0, 0,
                                 SWP_NOMOVE | SWP_NOSIZE | SWP_SHOWWINDOW)
            _user32.SetForegroundWindow(hwnd)
        except Exception:
            pass


# Simple logger
_log_lines: list[str] = []
_LOG_MAX = 200

def LOG(tag: str, msg: str) -> None:
    """Print AND buffer a timestamped log line."""
    ts = time.perf_counter()
    line = f"[{ts:12.3f}] [{tag:>8}] {msg}"
    print(line)
    _log_lines.append(line)
    if len(_log_lines) > _LOG_MAX:
        _log_lines.pop(0)


def main() -> None:
    ap = argparse.ArgumentParser(description="MSB Perfect Batter")
    ap.add_argument("-m", "--model", required=True)
    ap.add_argument("--character", type=str, default=None,
                    help="Character slug (e.g. 'papihongo'). Auto-sets target-y from calibration.")
    ap.add_argument("--dataset-dir", type=str, default="dataset",
                    help="Path to dataset folder with samples/")
    ap.add_argument("--target-y", type=int, default=None,
                    help="Override strike zone Y (auto-derived from --character if omitted)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--max-reach", type=float, default=100.0,
                    help="Pixels ±from aim center for full stick deflection")
    ap.add_argument("--deadzone", type=float, default=8.0)
    ap.add_argument("--anticipation-ms", type=int, default=200)
    # Aim-tuning args
    ap.add_argument("--aim-x", type=int, default=None,
                    help="Fixed horizontal aim centre (overrides strike-zone X). "
                         "Auto-derived from calibration if omitted.")
    ap.add_argument("--contact-offset-x", type=float, default=0.0,
                    help="Pixel offset added to filtered prediction before "
                         "computing stick error (tune ±5–20)")
    ap.add_argument("--freeze-frames", type=int, default=5,
                    help="Freeze horizontal target this many frames before impact")
    ap.add_argument("--smooth-alpha", type=float, default=0.30,
                    help="EMA alpha for prediction smoothing (0=ignore new, 1=no smooth)")
    ap.add_argument("--max-target-step", type=float, default=25.0,
                    help="Max px/frame change on filtered target (rate limiter)")
    ap.add_argument("--min-track-frames", type=int, default=4,
                    help="Min confirmed track frames before aiming")
    ap.add_argument("--min-vy", type=float, default=3.0,
                    help="Min downward velocity (px/frame) to consider a live pitch")
    ap.add_argument("--gain", type=float, default=0.5,
                    help="Stick gain: dx/max_reach * gain maps to ±0.5 stick")
    ap.add_argument("-c", "--config", default=None)
    ap.add_argument("--corridor-rect", nargs=4, type=int, default=None,
                    metavar=("X1", "Y1", "X2", "Y2"))
    args = ap.parse_args()

    # Config
    if args.config:
        cfg = Config.load(args.config)
    else:
        cfg_path = Path("config.toml")
        cfg = Config.load(str(cfg_path)) if cfg_path.exists() else Config()
    if args.corridor_rect:
        cfg.apply_overrides({"corridor_default": tuple(args.corridor_rect)})

    # Pipeline components
    detector = MLBallDetector(args.model, conf=args.conf, imgsz=args.imgsz, cfg=cfg)
    tracker  = MLBallTracker(cfg=cfg)
    predictor = TrajectoryPredictor(cfg=cfg)
    # corridor disabled — ball is not always inside the corridor
    # corridor = TrajectoryCorridor(cfg)
    # if cfg.corridor_default is not None:
    #     corridor.set_rect(*cfg.corridor_default)
    corridor = None

    # Batter tracking
    mv_tracker = None
    bt_classifier = None
    hbox_detector = None
    batter_tracking_active = False

    if _HAS_HITBOX and args.character:
        db_root = Path(args.dataset_dir)
        hbox_detector = HitboxDetector()
        mv_tracker = MovementTracker()
        # bt_classifier disabled — always treat batter as ready
        # bt_classifier = BatterStateClassifier()

        mv_ok = mv_tracker.load_calibration(args.character, db_root)
        bt_ok = True  # no classifier needed

        if mv_ok and bt_ok:
            batter_tracking_active = True
            LOG("INIT", f"Batter tracking active for '{args.character}'")

            # Auto-derive target_y from calibration strike-zone centres
            if args.target_y is None and mv_tracker._cal_strike_centers:
                avg_y = int(round(
                    sum(c[1] for c in mv_tracker._cal_strike_centers)
                    / len(mv_tracker._cal_strike_centers)
                ))
                args.target_y = avg_y
                LOG("INIT", f"target_y auto-set from calibration: {avg_y}")
        else:
            LOG("INIT", f"Batter calibration failed for '{args.character}' — check dataset/samples/")
    elif args.character and not _HAS_HITBOX:
        LOG("INIT", "--character given but hitbox module not importable")

    # Fallback target_y
    if args.target_y is None:
        args.target_y = 682
        LOG("INIT", f"No character / no calibration — using default target_y={args.target_y}")

    predictor.set_target_y(args.target_y)

    # Resolve aim_center_x
    _aim_x: Optional[int] = args.aim_x
    if _aim_x is None and batter_tracking_active and mv_tracker._cal_strike_centers:
        _aim_x = int(round(
            sum(c[0] for c in mv_tracker._cal_strike_centers)
            / len(mv_tracker._cal_strike_centers)
        ))
        LOG("INIT", f"aim_x auto-set from calibration strike centres: {_aim_x}")

    LOG("INIT", f"target_y={args.target_y}  aim_x={_aim_x}  conf={args.conf}  imgsz={args.imgsz}")
    LOG("INIT", f"max_reach={args.max_reach}  deadzone={args.deadzone}  gain={args.gain}")
    LOG("INIT", f"smooth_alpha={args.smooth_alpha}  max_step={args.max_target_step}  freeze={args.freeze_frames}")
    LOG("INIT", f"contact_offset_x={args.contact_offset_x}  min_track={args.min_track_frames}  min_vy={args.min_vy}")
    LOG("INIT", f"corridor=DISABLED")
    LOG("INIT", f"batter_tracking={'ON' if batter_tracking_active else 'OFF'}")

    # Gamepad
    dolphin = DolphinGamepadInput()
    if not dolphin.connect():
        LOG("INIT", "FATAL: cannot create virtual gamepad!")
        return
    LOG("INIT", "Gamepad connected OK")

    hp = dolphin.get_health()
    LOG("INIT", f"Gamepad health: {hp}")

    LOG("INIT", "Sending 60 frames of LEFT stick (0.0, 0.5)...")
    for i in range(60):
        dolphin.apply(0.0, 0.5, False)
        time.sleep(1/60)
    LOG("INIT", "Sending 60 frames of RIGHT stick (1.0, 0.5)...")
    for i in range(60):
        dolphin.apply(1.0, 0.5, False)
        time.sleep(1/60)
    LOG("INIT", "Sending neutral...")
    dolphin.apply(0.5, 0.5, False)
    time.sleep(0.1)
    hp = dolphin.get_health()
    LOG("INIT", f"Post-test health: {hp}")
    LOG("INIT", ">>> If character moved L then R, gamepad is WORKING <<<")
    LOG("INIT", ">>> Starting main loop in 2 seconds... <<<")
    time.sleep(2.0)

    # Screen source
    src = source_live(cfg)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    # Focus management
    _dolphin_hwnd = _find_dolphin_hwnd()
    if _dolphin_hwnd:
        LOG("INIT", f"Dolphin window found (hwnd={_dolphin_hwnd})")
    else:
        LOG("INIT", "Dolphin window NOT found — is the game running?")
        LOG("INIT", "*** Enable 'Background Input' in Dolphin controller config ***")
    _set_cv_window_no_activate(WINDOW)
    if _dolphin_hwnd:
        _focus_dolphin(_dolphin_hwnd)
        LOG("INIT", "Dolphin set as foreground window")

    frame_num = 0
    prev_t = time.perf_counter()
    fps_ema = 30.0
    _focus_check_ts = 0.0

    # Simple state
    _last_health_log = 0.0
    _swing_cooldown = 0
    _has_swung = False
    _last_stick_x = 0.5
    _last_stick_y = 0.5
    _target_x: Optional[float] = None
    _cached_batter_ready = True
    _sz_center: Optional[Tuple[int, int]] = None

    # Aim pipeline state
    _pred_x_raw: Optional[float] = None      # raw crossing X from predictor
    _pred_x_filtered: Optional[float] = None # EMA-smoothed prediction
    _pred_x_frozen: Optional[float] = None   # locked target near impact
    _target_frozen = False                    # True once frozen for this pitch

    _prev_brightness = 0.0
    _scene_stable_count = 0
    _BRIGHTNESS_MIN = 40
    _BRIGHTNESS_DELTA_MAX = 20
    _SCENE_SETTLE_FRAMES = 6
    _scene_ok = False

    _anticipation_frames = max(1, round(args.anticipation_ms / 1000.0 * 30))

    # PERF: preallocate detector masks once (avoids 5MB/frame alloc)
    _empty_mask: Optional[np.ndarray] = None

    # PERF: skip-frame detection (run YOLO every N frames when idle)
    _det_skip_interval = 2     # detect every 2nd frame when no active track
    _last_best = None          # cached last detection result
    _last_n_cands = 0

    # PERF: cache gamepad health (avoid repeated dict creation)
    _cached_hp: dict = {"alive": True, "hz": 0.0, "reports": 0,
                        "errors": 0, "consec_err": 0, "slot": -1, "gen": 0}

    LOG("LOOP", "=== ENTERING MAIN LOOP ===")

    try:
        for item in src:
            if isinstance(item, tuple):
                frame, _capture_ts = item
            else:
                frame = item
            if frame is None:
                time.sleep(0.001)
                continue

            frame_num += 1
            now = time.perf_counter()
            dt = now - prev_t
            prev_t = now
            fps_ema = 0.1 * (1/max(dt, 1e-9)) + 0.9 * fps_ema

            # Update anticipation from fps
            _anticipation_frames = max(1, round(args.anticipation_ms / 1000.0 * max(fps_ema, 1.0)))

            # 1) DETECT
            track_active = (tracker.track is not None
                            and tracker.track.active
                            and tracker.track.confirmed)

            if _empty_mask is None or _empty_mask.shape[:2] != frame.shape[:2]:
                _empty_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                detector.bg_fg_mask = _empty_mask
                detector.motion_mask = _empty_mask
                detector.white_mask = _empty_mask
                detector.trail_mask = _empty_mask
                detector.combined_mask = _empty_mask

            run_detect = track_active or (frame_num % _det_skip_interval == 0)
            if run_detect:
                best = detector.detect(frame,
                                       track_active=track_active)
                n_cands = len(detector.candidates)
                _last_best = best
                _last_n_cands = n_cands
            else:
                best = _last_best
                n_cands = _last_n_cands

            # 2) TRACK
            track = tracker.update(detector.candidates, best)

            # 3) BATTER TRACKING (every 6th frame)
            brightness = float(np.mean(frame[::8, ::8]))
            bright_delta = abs(brightness - _prev_brightness)
            _prev_brightness = brightness

            if brightness < _BRIGHTNESS_MIN or bright_delta > _BRIGHTNESS_DELTA_MAX:
                _scene_stable_count = 0
                _scene_ok = False
            else:
                _scene_stable_count += 1
                if _scene_stable_count >= _SCENE_SETTLE_FRAMES:
                    _scene_ok = True

            if (batter_tracking_active
                    and _scene_ok
                    and frame_num % 6 == 0):
                mv_tracker.update(frame, hbox_detector,
                                  batter_state=BatterState.NORMAL)

            if batter_tracking_active:
                _sz_center = mv_tracker.sz_centroid
                if _sz_center is None:
                    _sz_center = mv_tracker.strike_center_pred

            # Fallback: frame center + target_y
            if _sz_center is None:
                _cx = frame.shape[1] // 2
                _sz_center = (_cx, args.target_y)

            # 4) PREDICT + SMOOTH + FREEZE
            crossing = None
            fti_phys = 999.0
            _pred_x_raw = None

            if _aim_x is None:
                _aim_x = frame.shape[1] // 2
                LOG("INIT", f"aim_x fallback to frame center: {_aim_x}")

            if track is not None and track.active:
                crossing = predictor.predict(track)
                ball_y = track.last_pos[1]
                vy = track.velocity[1]
                if vy > args.min_vy and ball_y < args.target_y:
                    fti_phys = (args.target_y - ball_y) / vy
                elif vy > 1.0 and abs(ball_y - args.target_y) < 40:
                    fti_phys = 0.0

            # Build raw target X
            if (crossing is not None
                    and fti_phys < 200
                    and track is not None
                    and track.total_frames >= args.min_track_frames):
                _pred_x_raw = float(crossing[0])
            elif (track is not None and track.active
                  and track.total_frames >= args.min_track_frames):
                bx, by = track.last_pos[:2]
                vx, vy = track.velocity[:2]
                if vy > args.min_vy and by < args.target_y:
                    fti_est = (args.target_y - by) / vy
                    _pred_x_raw = bx + vx * fti_est

            # Smooth (EMA + rate limiter)
            if _pred_x_raw is not None:
                if _pred_x_filtered is None:
                    _pred_x_filtered = _pred_x_raw
                else:
                    # EMA
                    new_f = (1.0 - args.smooth_alpha) * _pred_x_filtered + args.smooth_alpha * _pred_x_raw
                    # Rate limiter
                    delta = new_f - _pred_x_filtered
                    if abs(delta) > args.max_target_step:
                        new_f = _pred_x_filtered + args.max_target_step * (1.0 if delta > 0 else -1.0)
                    _pred_x_filtered = new_f

            # Freeze near impact
            if (not _target_frozen
                    and _pred_x_filtered is not None
                    and fti_phys <= args.freeze_frames):
                _pred_x_frozen = _pred_x_filtered
                _target_frozen = True
                LOG("AIM", f"TARGET FROZEN at x={_pred_x_frozen:.0f} fti={fti_phys:.1f}")

            # Final target X
            if _target_frozen and _pred_x_frozen is not None:
                _target_x = _pred_x_frozen + args.contact_offset_x
            elif _pred_x_filtered is not None:
                _target_x = _pred_x_filtered + args.contact_offset_x
            else:
                _target_x = None

            if crossing is not None and _pred_x_raw is not None and frame_num % 10 == 0:
                filt_s = f"{_pred_x_filtered:.0f}" if _pred_x_filtered is not None else "?"
                LOG("PREDICT", f"f={frame_num} raw={_pred_x_raw:.0f} filt={filt_s} "
                               f"{'FRZ' if _target_frozen else 'live'} "
                               f"fti={fti_phys:.1f} vy={track.velocity[1]:.1f}")

            # 5) DECIDE STICK + SWING
            stick_x = 0.5
            stick_y = 0.5
            do_swing = False

            home_x = _aim_x

            if _swing_cooldown > 0:
                _swing_cooldown -= 1
                stick_x = _last_stick_x
                stick_y = _last_stick_y
                if _swing_cooldown == 0:
                    _has_swung = False
                    _last_stick_x = 0.5
                    _last_stick_y = 0.5
                    _pred_x_filtered = None
                    _pred_x_frozen = None
                    _target_frozen = False
                    _target_x = None
                    LOG("SWING", "Cooldown expired — reset for next pitch")
                    tracker.reset()
                    predictor.predicted_x = None
                    predictor.predicted_frame = None

            elif not _has_swung:
                # MAP TARGET X → STICK
                if _target_x is not None:
                    dx = _target_x - home_x
                    if abs(dx) > args.deadzone:
                        raw_stick = 0.5 + (dx / args.max_reach) * args.gain
                    else:
                        raw_stick = 0.5
                    stick_x = max(0.0, min(1.0, raw_stick))

                _last_stick_x = stick_x
                _last_stick_y = stick_y

                # Log alignment periodically
                if _target_x is not None and frame_num % 5 == 0:
                    LOG("ALIGN", f"tgt={_target_x:.0f} aim={home_x} "
                                 f"dx={_target_x - home_x:.0f} stk={stick_x:.3f} "
                                 f"{'FRZ' if _target_frozen else 'live'}")

                # SWING DECISION
                if fti_phys <= _anticipation_frames and fti_phys >= -2:
                    do_swing = True
                    _has_swung = True
                    _swing_cooldown = 10
                    tgt_s = f"{_target_x:.0f}" if _target_x is not None else "?"
                    LOG("SWING", f">>> SWING! fti={fti_phys:.1f} "
                                f"tgt={tgt_s} aim={home_x} "
                                f"stk={stick_x:.3f} "
                                f"ball_y={track.last_pos[1]:.0f} vy={track.velocity[1]:.1f} "
                                f"raw={_pred_x_raw} filt={_pred_x_filtered} frz={_target_frozen}")

            # 6) SEND TO GAMEPAD
            dolphin.apply(stick_x, stick_y, do_swing)

            if do_swing or (frame_num % 30 == 0 and (stick_x != 0.5 or stick_y != 0.5)):
                LOG("APPLY", f"f={frame_num} stk=({stick_x:.3f},{stick_y:.3f}) "
                             f"sw={do_swing} cd={_swing_cooldown}")

            if now - _last_health_log > 2.0:
                _last_health_log = now
                hp = dolphin.get_health()
                alive_str = "ALIVE" if hp['alive'] else ">>> DEAD <<<"
                LOG("HEALTH", f"{alive_str} hz={hp['hz']:.0f} "
                              f"reports={hp['reports']} errors={hp['errors']} "
                              f"slot={hp['slot']} gen={hp['gen']} "
                              f"consec_err={hp['consec_err']}")

            if now - _focus_check_ts > 5.0:
                _focus_check_ts = now
                if _dolphin_hwnd:
                    fg = _user32.GetForegroundWindow()
                    if fg != _dolphin_hwnd:
                        _focus_dolphin(_dolphin_hwnd)

            # 7) DRAW
            h_f, w_f = frame.shape[:2]

            if batter_tracking_active and mv_tracker.sz_centroid is not None:
                cv2.circle(frame, _pt(mv_tracker.sz_centroid[0],
                                      mv_tracker.sz_centroid[1]),
                           6, COL_CYAN, -1)

            cv2.line(frame, (home_x, 0), (home_x, h_f), (120, 120, 120), 1)

            if _target_x is not None:
                tx_i = max(0, min(w_f - 1, int(_target_x)))
                cv2.line(frame, (tx_i, 0), (tx_i, h_f), COL_GREEN, 2)

            # Tracked ball
            if track is not None and track.active:
                pt = _pt(track.last_pos[0], track.last_pos[1])
                cv2.circle(frame, pt, 12, COL_GREEN, 2)
                cv2.circle(frame, pt, 3, COL_RED, -1)

            # Prediction cross
            if crossing is not None:
                px, py = _pt(crossing[0], crossing[1])
                px = max(-9999, min(9999, px))
                py = max(-9999, min(9999, py))
                if -w_f < px < 2 * w_f and -h_f < py < 2 * h_f:
                    cv2.drawMarker(frame, (px, py), COL_RED,
                                   cv2.MARKER_CROSS, 20, 2)

            # Stick indicator (top-right)
            sx_px = int(w_f - 60 + (stick_x - 0.5) * 40)
            sy_px = int(60 + (stick_y - 0.5) * 40)
            cv2.circle(frame, (w_f - 60, 60), 25, COL_WHITE, 1)
            cv2.circle(frame, (sx_px, sy_px), 6,
                       COL_GREEN if not do_swing else COL_RED, -1)

            # 3-line HUD
            put_text(frame, f"FPS {fps_ema:.0f}  f{frame_num}  aim={home_x}",
                     (10, 22), 0.45, COL_WHITE, 1)
            fti_s = f"FTI={fti_phys:.0f}" if fti_phys < 200 else ""
            frz_s = "FRZ" if _target_frozen else ""
            tgt_s = f"tgt={_target_x:.0f}" if _target_x is not None else ""
            put_text(frame, f"Stk {stick_x:.2f} {tgt_s} {fti_s} {frz_s} CD={_swing_cooldown}",
                     (10, 44), 0.45, COL_CYAN, 1)
            raw_s = f"R={_pred_x_raw:.0f}" if _pred_x_raw is not None else ""
            filt_s = f"F={_pred_x_filtered:.0f}" if _pred_x_filtered is not None else ""
            put_text(frame, f"{raw_s} {filt_s}",
                     (10, 64), 0.40, COL_YELLOW, 1)

            # Flash overlays
            if _swing_cooldown > 0:
                put_text(frame, "SWING",
                         (w_f // 2 - 50, h_f // 2), 0.7, (0, 128, 255), 2)

            cv2.imshow(WINDOW, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("r"):
                tracker.reset()
                predictor.predicted_x = None
                predictor.predicted_frame = None
                dolphin.safe_neutral()
                _swing_cooldown = 0
                _has_swung = False
                _last_stick_x = 0.5
                _last_stick_y = 0.5
                _target_x = None
                _pred_x_filtered = None
                _pred_x_frozen = None
                _target_frozen = False
                LOG("INPUT", "RESET triggered by user")

    except KeyboardInterrupt:
        LOG("EXIT", "Interrupted")
    finally:
        dolphin.safe_neutral()
        dolphin.close()
        cv2.destroyAllWindows()
        LOG("EXIT", "Done")


if __name__ == "__main__":
    main()
