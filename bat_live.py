"""
bat_live.py — Automated batting from live screen capture.

Integrates the full pipeline:
  Screen capture (dxcam @ 60 fps, mss fallback)
  → YOLO ball detection
  → Kalmanfiltered tracking
  → Polynomial trajectory prediction
  → Batterposition / strikezone localisation (MovementTracker)
  → Batterstate gating (BatterStateClassifier: NORMAL / NON_NORMAL)
  → BatterController (statemachine: IDLE → ALIGNING → SWINGING → COOLDOWN)
  → DolphinGamepadInput (analog stick + A button via virtual Xbox 360 gamepad)

Usage

  python bat_live.py --model weights/ball_best.pt --character Mario

  # With explicit targetY and tuning:
  python bat_live.py -m weights/ball_best.pt \\
     --character Mario --target-y 700 --anticipation-ms 200 --deadzone 18

  # Dry-run (no virtual gamepad — just shows overlays):
  python bat_live.py -m weights/ball_best.pt --character Mario --dry-run

Controls

  Q / ESC  = Quit
  R        = Reset tracker + controller, centre stick
  D        = Toggle debug HUD
  Y        = Click to set strikezone Y level (same as track_live.py)
  +/      = Adjust swing anticipation on the fly (+/ 25 ms)
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes as _wt
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from msb.config import Config
from msb.detector_ml import MLBallDetector
from msb.tracker_ml import MLBallTracker, MLTrackState
from msb.predictor import TrajectoryPredictor
from msb.corridor import TrajectoryCorridor
from msb.batter_controller import BatterController, ControlState
from msb.dolphin_input import DolphinGamepadInput
from msb.sources import source_live, CaptureThread
from msb.utils import put_text, COL_GREEN, COL_RED, COL_YELLOW, COL_CYAN, COL_WHITE, COL_MAGENTA, COL_ORANGE
from msb.game_phase import GamePhase, GamePhaseManager
from msb.swing_controller import SwingState, SwingController
from msb.metrics import FrameMetrics, SessionMetrics

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
          "Batter tracking / state gating disabled.")
    _HAS_HITBOX = False


WINDOW = "MSB Auto Batter"
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _pt(*args) -> Tuple[int, ...]:
    """Force pure-Python ints for OpenCV drawing (numpy scalars crash 4.13)."""
    return tuple(int(float(a)) for a in args)


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


# OVERLAY DRAWING

def draw_ball_overlays(
    vis: np.ndarray,
    detector,
    tracker: MLBallTracker,
    predictor: TrajectoryPredictor,
    corridor=None,
) -> None:
    """Draw ball detection, tracking and prediction overlays (in-place).

    These are drawn INDEPENDENTLY of the controller — if YOLO sees
    the ball, it shows up here regardless of batter state.
    """
    h, w = vis.shape[:2]

    # All YOLO detections (small yellow dots)
    for c in detector.candidates:
        pt = _pt(c.center[0], c.center[1])
        cv2.circle(vis, pt, 5, COL_YELLOW, -1)
        cv2.circle(vis, pt, 5, (0, 180, 180), 1)

    # Tracked ball (larger green ring + red centre + bbox + label)
    sel = tracker.selected
    if sel is not None:
        pt = _pt(sel.center[0], sel.center[1])
        cv2.circle(vis, pt, 12, COL_GREEN, 2)
        cv2.circle(vis, pt, 3, COL_RED, -1)
        # Bounding box
        bx, by, bw, bh = sel.bbox
        cv2.rectangle(vis, _pt(bx, by), _pt(bx + bw, by + bh), COL_GREEN, 1)
        put_text(vis, f"BALL conf={sel.score:.0%}",
                 _pt(bx, by - 6), 0.35, COL_GREEN, 1)

    # Trajectory trail (green→red gradient)
    track = tracker.track
    if track is not None and track.active and len(track.positions) >= 2:
        positions = list(track.positions)
        n = len(positions)
        for i in range(1, n):
            alpha = i / n
            col = (0, int(255 * alpha), int(255 * (1 - alpha)))
            pt1 = _pt(positions[i - 1][0], positions[i - 1][1])
            pt2 = _pt(positions[i][0], positions[i][1])
            cv2.line(vis, pt1, pt2, col, 2)

        # Velocity arrow
        last = track.last_pos
        vx, vy = track.velocity
        if abs(vx) + abs(vy) > 1:
            tip = _pt(last[0] + vx * 3, last[1] + vy * 3)
            cv2.arrowedLine(vis, _pt(last[0], last[1]), tip,
                            COL_MAGENTA, 2, tipLength=0.3)

        # Future trajectory (dashed yellow)
        traj_pts = predictor.get_trajectory_points(track, n_future=20)
        if len(traj_pts) >= 2:
            for i in range(max(0, n - 1), len(traj_pts) - 1):
                if i % 2 == 0:
                    p1 = _pt(traj_pts[i][0], traj_pts[i][1])
                    p2 = _pt(traj_pts[i + 1][0], traj_pts[i + 1][1])
                    cv2.line(vis, p1, p2, COL_YELLOW, 1)

        # Track info bar
        status = "CONFIRMED" if track.confirmed else "tentative"
        put_text(vis,
                 f"[{status}] frames={track.total_frames} "
                 f"vel=({vx:.1f},{vy:.1f}) "
                 f"missed={track.frames_since_seen}",
                 (10, h - 26), 0.38, COL_CYAN, 1)

    # Prediction cross (drawn DIRECTLY from predictor, not controller)
    # Only show a live crossing when the tracker still has a plausible
    # active ball track. This avoids stale crosses hanging around on the
    # batter after the pitch has already ended.
    if (predictor.predicted_x is not None
            and predictor.target_y is not None
            and tracker.track is not None
            and tracker.track.active
            and tracker.track.total_frames >= 3
            and predictor.confidence >= 0.15):
        px, py = _pt(predictor.predicted_x, predictor.target_y)
        # Only draw if prediction is on-screen
        if 0 <= px <= w and 0 <= py <= h:
            cv2.drawMarker(vis, (px, py), COL_RED,
                            cv2.MARKER_CROSS, 24, 2)
            cv2.circle(vis, (px, py), 14, COL_RED, 2)
            put_text(vis,
                     f"CROSS ({px},{py}) {predictor.fit_type} "
                     f"conf={predictor.confidence:.0%}",
                     (px + 18, py - 8), 0.38, COL_RED, 1)

    # Target Y line
    if predictor.target_y is not None:
        ty = _pt(predictor.target_y)[0]
        cv2.line(vis, (0, ty), (w, ty), COL_RED, 1)
        put_text(vis, f"STRIKE Y = {ty}", (w - 140, ty - 6),
                 0.35, COL_RED, 1)


def draw_hud(
    vis: np.ndarray,
    ctrl: BatterController,
    tracker: MLBallTracker,
    detector,
    sz_center: Optional[Tuple[int, int]],
    batter_ready: bool,
    fps: float,
    frame_num: int,
    dry_run: bool,
    mv_tracker=None,
) -> None:
    """Draw controller / batter HUD overlays on *vis* (in-place)."""
    h, w = vis.shape[:2]

    # Top bar
    put_text(vis, f"Frame {frame_num}", (10, 20), 0.45, COL_WHITE, 1)
    put_text(vis, f"FPS {fps:.0f}", (w - 110, 20), 0.45, COL_GREEN, 1)
    put_text(vis, f"BALLS: {len(detector.candidates)}",
             (w - 220, 20), 0.45, COL_YELLOW, 1)

    # Tracker state
    state_col = {
        MLTrackState.IDLE: COL_WHITE,
        MLTrackState.ACTIVE: COL_GREEN,
        MLTrackState.LOST: COL_RED,
    }.get(tracker.state, COL_WHITE)
    put_text(vis, f"TRACK: {tracker.state.name}", (10, 40), 0.45,
             state_col, 1)

    # Controller state
    cs_colors = {
        ControlState.IDLE: COL_WHITE,
        ControlState.TRACKING: COL_CYAN,
        ControlState.ALIGNING: COL_YELLOW,
        ControlState.SWINGING: COL_RED,
        ControlState.COOLDOWN: COL_MAGENTA,
    }
    cs_col = cs_colors.get(ctrl.state, COL_WHITE)
    put_text(vis, f"CTRL: {ctrl.state.name}", (10, 76), 0.45,
             cs_col, 1)

    # Frames to impact
    if ctrl.frames_to_impact is not None:
        ftxt = f"Impact in {ctrl.frames_to_impact} frames"
        col = COL_RED if ctrl.frames_to_impact <= ctrl.swing_anticipation_frames else COL_YELLOW
        put_text(vis, ftxt, (10, 94), 0.45, col, 1)

    # Stick values
    put_text(vis, (f"Stick: ({ctrl.stick_x:.2f}, {ctrl.stick_y:.2f})"
                   f"  Swing: {'YES' if ctrl.swing else 'no'}"),
             (10, 112), 0.40, COL_CYAN, 1)

    # Anticipation reminder
    put_text(vis, f"Antic: {ctrl.swing_anticipation_frames}f  "
                  f"DZ: {ctrl.position_threshold_px:.0f}px",
             (10, 130), 0.35, COL_WHITE, 1)

    if dry_run:
        put_text(vis, "DRY RUN (no pipe)", (w // 2 - 100, 20), 0.55,
                 COL_ORANGE, 2)

    # Strikezone centre marker
    if sz_center is not None:
        sz_pt = _pt(sz_center[0], sz_center[1])
        cv2.drawMarker(vis, sz_pt, COL_CYAN,
                        cv2.MARKER_TILTED_CROSS, 18, 2)
        put_text(vis, "SZ", (sz_pt[0] + 12, sz_pt[1] - 4),
                 0.35, COL_CYAN, 1)

    # Error vector (crossing → strike zone)
    if sz_center is not None and ctrl.last_crossing is not None:
        cx, cy = int(float(ctrl.last_crossing[0])), int(float(ctrl.last_crossing[1]))
        # Only draw if the crossing is reasonably on-screen
        if -w < cx < 2 * w and -h < cy < 2 * h:
            sz_pt = _pt(sz_center[0], sz_center[1])
            cr_pt = (cx, cy)
            cv2.arrowedLine(vis, sz_pt, cr_pt,
                            COL_MAGENTA, 2, tipLength=0.15)

    # Swing flash
    if ctrl.state == ControlState.SWINGING:
        put_text(vis, ">>> SWING <<<",
                 (w // 2 - 90, h // 2), 0.9, COL_RED, 3)

    # Key hints
    hints = "[Q] Quit  [R] Reset  [D] Debug  [Y] SetY  [+/-] Antic"
    put_text(vis, hints, (10, h - 8), 0.33, COL_WHITE, 1)


# MAIN

def main()-> None:
    ap = argparse.ArgumentParser(
        description="MSB Auto Batter — live screen capture → Dolphin input")

    # Required
    ap.add_argument("--model", "-m", required=True,
                    help="Path to YOLO .pt or .onnx model")

    # Dolphin input (virtual gamepad)
    ap.add_argument("--dry-run", action="store_true",
                    help="Run without sending Dolphin inputs (overlay only)")
    ap.add_argument("--no-gamepad", action="store_true",
                    help="Alias for --dry-run")

    # Hitbox / batter tracking
    ap.add_argument("--character", default=None,
                    help="Character name for batter-tracking calibration "
                         "(e.g. Mario). If omitted, batter tracking is "
                         "disabled and strike-zone centre is not used.")
    ap.add_argument("--dataset-dir", default="./dataset",
                    help="Root directory with hitbox calibration samples")

    # Tuning (all timing in MILLISECONDS — auto-adapted to current FPS)
    ap.add_argument("--target-y", type=int, default=None,
                    help="Strike-zone Y (pixels, ROI-relative). "
                         "Required for prediction. Can also set with [Y].")
    ap.add_argument("--anticipation-ms", type=int, default=200,
                    help="Swing anticipation in ms (default: 200). "
                         "How long before predicted impact to press A.")
    ap.add_argument("--deadzone", type=float, default=15.0,
                    help="Position deadzone in pixels (default: 15)")
    ap.add_argument("--gain", type=float, default=0.008,
                    help="Proportional gain for stick (default: 0.008)")
    ap.add_argument("--swing-hold-ms", type=int, default=100,
                    help="How long to hold A in ms (default: 100)")
    ap.add_argument("--cooldown-ms", type=int, default=500,
                    help="Cooldown after swing in ms (default: 500)")
    ap.add_argument("--lockout-ms", type=int, default=500,
                    help="Post-swing lockout in ms (default: 500). "
                         "How long to pause auto-batting after a swing "
                         "before looking for the next pitch.")
    ap.add_argument("--no-state-gate", action="store_true",
                    help="Ignore batter-state classifier (always treat "
                         "as NORMAL). Useful when classifier is unreliable.")

    # Detector
    ap.add_argument("--conf", type=float, default=0.25,
                    help="YOLO confidence threshold (default: 0.25)")
    ap.add_argument("--imgsz", type=int, default=640,
                    help="YOLO inference resolution (default: 640)")

    # Pipeline config
    ap.add_argument("-c", "--config", default=None,
                    help="Path to config.toml / config.json")
    ap.add_argument("--corridor-rect", nargs=4, type=int, default=None,
                    metavar=("X1", "Y1", "X2", "Y2"))

    args = ap.parse_args()
    dry_run: bool = args.dry_run or args.no_gamepad
    no_state_gate: bool = args.no_state_gate

    # Config
    if args.config:
        cfg = Config.load(args.config)
    else:
        cfg_path = Path("config.toml")
        cfg = Config.load(str(cfg_path)) if cfg_path.exists() else Config()
    if args.corridor_rect:
        cfg.apply_overrides({"corridor_default": tuple(args.corridor_rect)})

    # Ball detection + tracking
    detector = MLBallDetector(args.model, conf=args.conf,
                              imgsz=args.imgsz, cfg=cfg)
    tracker = MLBallTracker(cfg=cfg)
    predictor = TrajectoryPredictor(cfg=cfg)
    if args.target_y is not None:
        predictor.set_target_y(args.target_y)

    # corridor disabled — ball is not always inside the corridor
    # corridor = TrajectoryCorridor(cfg)
    # if cfg.corridor_default is not None:
    #     corridor.set_rect(*cfg.corridor_default)
    corridor = None

    # Batter tracking (optional)
    mv_tracker: Optional[Any] = None
    bt_classifier: Optional[Any] = None
    hbox_detector: Optional[Any] = None
    batter_tracking_active: bool = False
    stable_aim_center: Optional[Tuple[int, int]] = None
    home_target_x: Optional[int] = None

    if _HAS_HITBOX and args.character:
        db_root = Path(args.dataset_dir)
        hbox_detector = HitboxDetector()
        mv_tracker = MovementTracker()
        # bt_classifier disabled — always treat batter as ready
        # bt_classifier = BatterStateClassifier()

        mv_ok = mv_tracker.load_calibration(args.character, db_root)
        bt_ok = True  # no classifier needed

        mv_tracker.contact_offset = (cfg.contact_offset_x,
                                     cfg.contact_offset_y)

        if mv_ok and bt_ok:
            batter_tracking_active = True
            print(f"[INFO] Batter tracking active for '{args.character}'")

            if mv_tracker._cal_strike_centers:
                stable_aim_center = (
                    int(round(sum(c[0] for c in mv_tracker._cal_strike_centers)
                              / len(mv_tracker._cal_strike_centers))),
                    int(round(sum(c[1] for c in mv_tracker._cal_strike_centers)
                              / len(mv_tracker._cal_strike_centers))),
                )
                home_target_x = stable_aim_center[0] - 60
                print(f"[INFO] aim center auto-set from calibration: "
                      f"{stable_aim_center}")
                print(f"[INFO] home target auto-set left of contact: "
                      f"x={home_target_x}")

            if args.target_y is None and mv_tracker._cal_strike_centers:
                avg_y = int(round(
                    sum(c[1] for c in mv_tracker._cal_strike_centers)
                    / len(mv_tracker._cal_strike_centers)
                ))
                predictor.set_target_y(avg_y)
                print(f"[INFO] target_y auto-set from calibration: {avg_y}")
        else:
            if not mv_ok:
                print(f"[WARN] MovementTracker calibration failed for "
                      f"'{args.character}' in {db_root}")
            if not bt_ok:
                print(f"[WARN] BatterStateClassifier refs failed for "
                      f"'{args.character}' in {db_root}")
            batter_tracking_active = False
    elif args.character and not _HAS_HITBOX:
        print("[WARN] --character given but hitbox module not importable")

    # Controller
    controller = BatterController(
        position_threshold_px=args.deadzone,
        swing_anticipation_frames=3,   # placeholder, overwritten per frame
        proportional_gain=args.gain,
        swing_hold_frames=2,           # placeholder, overwritten per frame
        cooldown_frames=8,             # placeholder, overwritten per frame
    )
    _anticipation_ms = args.anticipation_ms
    _swing_hold_ms = args.swing_hold_ms
    _cooldown_ms = args.cooldown_ms
    _lockout_seconds = args.lockout_ms / 1000.0

    # Game-phase FSM
    phase_cfg = {
        "ssim_batter_thresh": cfg.phase_ssim_batter_thresh,
        "ssim_deviation_thresh": cfg.phase_ssim_deviation_thresh,
        "hysteresis_enter_bip": cfg.phase_hysteresis_enter,
        "hysteresis_exit_bip": cfg.phase_hysteresis_exit,
        "ref_thumb_size": (cfg.phase_ref_thumb_w, cfg.phase_ref_thumb_h),
        "ref_refresh_interval": cfg.phase_ref_refresh_interval,
        "post_swing_lockout_sec": cfg.phase_post_swing_lockout_sec,
        "reset_timeout_sec": cfg.phase_reset_timeout_sec,
        "min_pitch_vy_trigger": cfg.phase_min_pitch_vy_trigger,
    }
    phase_mgr = GamePhaseManager(phase_cfg)

    swing_ctrl = SwingController(
        Kp=args.gain,
        Kd=cfg.ctrl_kd,
        Kff=cfg.ctrl_kff,
        pipeline_latency_sec=cfg.ctrl_pipeline_latency_sec,
        swing_startup_sec=max(
            0.01,
            args.anticipation_ms / 1000.0 - cfg.ctrl_pipeline_latency_sec,
        ),
        swing_hold_sec=cfg.ctrl_swing_hold_sec,
        swing_window_sec=cfg.ctrl_swing_window_sec,
        cooldown_sec=cfg.ctrl_cooldown_sec,
        deadzone_px=args.deadzone,
    )

    # Session-level metrics
    session_metrics: Optional[SessionMetrics] = None
    if cfg.metrics_enabled:
        session_metrics = SessionMetrics(fps_window=cfg.metrics_fps_window)

    # Dolphin input
    dolphin: Optional[DolphinGamepadInput] = None
    if not dry_run:
        dolphin = DolphinGamepadInput()
        if not dolphin.connect():
            print("[ERROR] Cannot create virtual gamepad. "
                  "Install vgamepad: pip install vgamepad\n"
                  "Running in dry-run mode instead.")
            dolphin = None
            dry_run = True
        else:
            print("[INFO] Virtual gamepad active — configure Dolphin "
                  "Port 1 → XInput/0/Gamepad")
    if dry_run:
        print("[INFO] Dry-run mode — no inputs sent to Dolphin")

    # Screen source
    _cap = CaptureThread(source_live(cfg))
    _cap.start()

    # Window
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    show_debug = False
    scale = cfg.display_scale

    # Focus management
    _dolphin_hwnd = _find_dolphin_hwnd()
    if _dolphin_hwnd:
        print(f"[INFO] Dolphin window found (hwnd={_dolphin_hwnd})")
    else:
        print("[WARN] Dolphin window NOT found — is the game running?")
        print("       Start the game in Dolphin BEFORE running bat_live.py")

    _set_cv_window_no_activate(WINDOW)

    if _dolphin_hwnd:
        _focus_dolphin(_dolphin_hwnd)
        print("[INFO] Dolphin set as foreground window")
    else:
        print("[WARN] *** CRITICAL: Enable 'Background Input' in ***")
        print("       *** Dolphin controller config, or the      ***")
        print("       *** controller WILL stop working when our   ***")
        print("       *** window gets focus!                      ***")

    # StrikeY selector
    from msb.visualiser import StrikeYSelector
    sy_sel = StrikeYSelector(display_scale=scale)

    def _mouse_cb(event: int, x: int, y: int,
                  flags: int, param: Any)-> None:
        sy_sel.mouse_callback(event, x, y, flags, param)

    cv2.setMouseCallback(WINDOW, _mouse_cb)

    frame_num: int = 0
    prev_t = time.perf_counter()
    fps_ema = float(cfg.target_fps)
    _cached_batter_ready: bool = True
    _lockout_until: float = 0.0
    _rearm_frame: int = 0
    _focus_check_ts: float = 0.0
    _last_cap_seq: int = -1

    print()
    print("═" * 56)
    print("  MSB AUTO BATTER")
    print("═" * 56)
    print(f"  Model       : {args.model}")
    print(f"  Gamepad     : {'ACTIVE' if dolphin is not None else '(dry-run)'}")
    print(f"  Character   : {args.character or '(none)'}")
    _effective_target_y = predictor.target_y
    print(f"  Target Y    : {_effective_target_y or '(not set — press [Y])'}")
    print(f"  Anticipation: {_anticipation_ms} ms")
    print(f"  Swing hold  : {_swing_hold_ms} ms")
    print(f"  Cooldown    : {_cooldown_ms} ms")
    print(f"  Lockout     : {args.lockout_ms} ms")
    print(f"  Deadzone    : {args.deadzone} px")
    print(f"  Gain        : {args.gain}")
    print("═" * 56)
    print()

    try:
        while True:
            frame, capture_ts, cap_seq = _cap.latest()
            if frame is None or cap_seq == _last_cap_seq:
                time.sleep(0.001)
                continue
            _last_cap_seq = cap_seq

            frame_num += 1
            now = time.perf_counter()
            dt = now - prev_t
            prev_t = now
            fps_ema = 0.1 * (1.0 / max(dt, 1e-9)) + 0.9 * fps_ema

            # Per-frame metrics
            fm: Optional[FrameMetrics] = None
            if session_metrics is not None:
                fm = FrameMetrics()
                fm.mark("capture")

            # Game-phase FSM update
            ball_vy_px_per_sec = 0.0
            _ball_tracked = False
            _ball_y = 0.0
            if (tracker.track is not None and tracker.track.active
                    and tracker.track.total_frames >= 2):
                _ball_y = tracker.track.last_pos[1]
                ball_vy_px_per_sec = tracker.track.velocity_y_px_per_sec()
                _ball_tracked = True
                if (predictor.target_y is not None
                        and _ball_y > predictor.target_y + 40):
                    _ball_tracked = False
                    ball_vy_px_per_sec = 0.0
            batter_visible_for_phase = batter_tracking_active
            prev_phase = phase_mgr.phase
            phase_mgr.update(
                frame,
                ball_tracked=_ball_tracked,
                ball_vy=ball_vy_px_per_sec,
                ball_y=_ball_y,
                target_y=predictor.target_y,
                swung=swing_ctrl.swing,
                swing_committed=(swing_ctrl.state in (
                    SwingState.COMMITTED, SwingState.SWINGING)),
                batter_visible=batter_visible_for_phase,
            )
            if phase_mgr.phase != prev_phase:
                print(f"[PHASE] {prev_phase.name} → {phase_mgr.phase.name}")
                if session_metrics is not None:
                    session_metrics.record_phase_transition(
                        prev_phase.name, phase_mgr.phase.name)

                # Phase-gated resets
                if phase_mgr.phase == GamePhase.BALL_IN_PLAY:
                    tracker.reset()
                    predictor.reset()
                    print("[PHASE] BALL_IN_PLAY — tracker/predictor reset")
                elif phase_mgr.phase == GamePhase.RESET_TO_BATTER:
                    tracker.reset()
                    predictor.reset()
                    swing_ctrl.reset()
                    controller.reset()
                    if dolphin is not None:
                        dolphin.safe_neutral()
                    print("[PHASE] RESET_TO_BATTER — tracker/predictor/controller reset")

            # StrikeY selector overlay
            if sy_sel.active:
                frozen = sy_sel.draw_frozen()
                if frozen is not None:
                    if scale != 1.0:
                        dw = int(frozen.shape[1] * scale)
                        dh = int(frozen.shape[0] * scale)
                        frozen = cv2.resize(frozen, (dw, dh))
                    cv2.imshow(WINDOW, frozen)
                key = cv2.waitKey(30) & 0xFF
                if key == 27:
                    sy_sel.cancel()
                elif not sy_sel.active and sy_sel.target_y is not None:
                    predictor.set_target_y(sy_sel.target_y)
                    print(f"[INFO] target_y set to {sy_sel.target_y}")
                if dolphin is not None:
                    dolphin.apply(0.5, 0.5, False)
                continue
            if not sy_sel.active and sy_sel.target_y is not None:
                predictor.set_target_y(sy_sel.target_y)

            #
            # 1) DETECT ball
            #
            track_active = (tracker.track is not None
                            and tracker.track.active
                            and tracker.track.confirmed)
            if phase_mgr.is_batter_view:
                _track_pos = None
                if (tracker.track is not None and tracker.track.active):
                    _track_pos = tracker.track.last_pos[:2]
                best = detector.detect(frame,
                                       track_active=track_active,
                                       track_pos=_track_pos)
            else:
                best = None
                detector.candidates = []

            if fm is not None:
                fm.mark("inference")

            n_raw = len(detector.candidates)

            feed_candidates = detector.candidates
            feed_best = best

            #
            # 2) TRACK ball
            #
            track = tracker.update(feed_candidates, feed_best,
                                   timestamp=capture_ts)
            if fm is not None:
                fm.mark("tracking")

            fps_now = max(fps_ema, 1.0)
            stale_frames = max(8, int(0.6 * fps_now))   # ~0.6 s
            if track is not None and track.active and track.total_frames > stale_frames:
                avg_vy = track.avg_velocity_y()
                cur_y = track.last_pos[1]
                kill = False
                if abs(avg_vy) < 2.0:
                    kill = True
                if (predictor.target_y is not None
                        and cur_y > predictor.target_y + 60
                        and avg_vy >= 0):
                    kill = True
                if kill:
                    tracker.kill_track()
                    track = None
                    predictor.reset()

            #
            # 3) PREDICT crossing
            #
            crossing: Optional[Tuple[int, int]] = None
            if (track is not None and track.active
                    and phase_mgr.should_track_ball):
                crossing = predictor.predict(track)
                pass
            if fm is not None:
                fm.mark("prediction")

            # Sanity-check prediction: reject garbage values
            h_frame, w_frame = frame.shape[:2]
            if crossing is not None and predictor.predicted_frame is not None:
                fti = predictor.predicted_frame - frame_num
                px = crossing[0]
                fps_now = max(fps_ema, 1.0)
                max_impact_frames = int(3.0 * fps_now)  # max 3 seconds
                if (fti > max_impact_frames
                        or px < -200
                        or px > w_frame + 200):
                    crossing = None
                    predictor.reset()

            if crossing is not None and time.perf_counter() < _lockout_until:
                crossing = None

            if crossing is not None and track is not None:
                _vy = (track.avg_velocity_y()
                       if track.total_frames >= 2 else 0.0)
                _skip = False
                _reason = ""
                if track.total_frames < 2:
                    _skip = True
                    _reason = f"frames={track.total_frames}<2"
                elif _vy < 1.0:
                    _skip = True
                    _reason = f"vy={_vy:.1f}<1.0"
                if _skip:
                    if frame_num % 30 == 0:
                        print(f"[GATE] Blocked: {_reason}  "
                              f"pos={track.last_pos}  vy={_vy:.1f}")
                    crossing = None
                else:
                    print(f"[GATE] PASS  pos={track.last_pos}  "
                          f"vy={_vy:.1f}  frames={track.total_frames}  "
                          f"cross={crossing}  "
                          f"ttc={predictor.time_to_crossing_sec:.3f}s")

            #
            # 4) BATTER tracking
            #
            sz_center: Optional[Tuple[int, int]] = None
            batter_ready: bool = True  # default: assume ready

            if batter_tracking_active:
                if frame_num % 3 == 0:
                    mv_tracker.update(frame, hbox_detector,
                                      batter_state=BatterState.NORMAL)

                batter_ready = True

                # Best estimate of where the strike zone is right now
                sz_center = mv_tracker.sz_centroid
                if sz_center is None:
                    sz_center = mv_tracker.strike_center_pred

            if sz_center is None and predictor.target_y is not None:
                cx = frame.shape[1] // 2
                sz_center = (cx, predictor.target_y)

            control_center = sz_center
            if control_center is None and stable_aim_center is not None and predictor.target_y is not None:
                control_center = (stable_aim_center[0], predictor.target_y)

            if frame_num % 30 == 0:
                fps_now = max(fps_ema, 1.0)
                diag_parts = [
                    f"frame={frame_num}",
                    f"fps={fps_ema:.0f}",
                    f"phase={phase_mgr.phase.name}",
                    f"cands={n_raw}",
                    f"track={tracker.state.name}",
                    f"ctrl={controller.state.name}",
                    f"swing={swing_ctrl.state.name}",
                ]
                if tracker.track is not None and tracker.track.active:
                    t = tracker.track
                    diag_parts.append(f"pos={t.last_pos}")
                    diag_parts.append(f"vel=({t.velocity[0]:.1f},{t.velocity[1]:.1f})")
                    diag_parts.append(
                        f"vyps={t.velocity_y_px_per_sec():.0f}")
                    diag_parts.append(f"frames={t.total_frames}")
                if control_center is not None:
                    diag_parts.append(f"sz={control_center}")
                if stable_aim_center is not None:
                    diag_parts.append(f"aim={stable_aim_center}")
                if home_target_x is not None:
                    diag_parts.append(f"home_x={home_target_x}")
                diag_parts.append(f"ready={'Y' if batter_ready else 'N'}")
                if predictor.predicted_x is not None:
                    diag_parts.append(
                        f"cross=({predictor.predicted_x},{predictor.target_y})")
                if swing_ctrl.time_to_impact_sec is not None:
                    sec = swing_ctrl.time_to_impact_sec
                    diag_parts.append(
                        f"impact={sec:.2f}s")
                if swing_ctrl.swing:
                    diag_parts.append("SWING!")
                if swing_ctrl.stick_x != 0.5 or swing_ctrl.stick_y != 0.5:
                    diag_parts.append(
                        f"stick=({swing_ctrl.stick_x:.2f},{swing_ctrl.stick_y:.2f})")
                diag_parts.append(f"antic={_anticipation_ms}ms")
                if dolphin is not None:
                    hp = dolphin.get_health()
                    diag_parts.append(
                        f"gp={'ALIVE' if hp['alive'] else 'STALL'}"
                        f"/{hp['hz']:.0f}Hz"
                        f"/xi{hp['slot']}"
                        f"/gen{hp['gen']}")
                    if hp['consec_err'] > 0:
                        diag_parts.append(f"errs={hp['consec_err']}")
                _lockout_remain = _lockout_until - time.perf_counter()
                if _lockout_remain > 0:
                    diag_parts.append(f"LOCKOUT={_lockout_remain:.1f}s")
                print(f"[DIAG] {' | '.join(diag_parts)}")

            #
            # 5) CONTROLLER decision
            #
            fps_now = max(fps_ema, 1.0)

            swing_ctrl.Kp = args.gain
            swing_ctrl.deadzone_px = args.deadzone
            swing_ctrl.swing_startup_sec = max(
                0.01,
                _anticipation_ms / 1000.0 - swing_ctrl.pipeline_latency_sec,
            )

            # SwingController
            ttc = getattr(predictor, "time_to_crossing_sec", None)
            pvx = getattr(predictor, "predicted_vx", None)
            ttc_phys: Optional[float] = None
            if (track is not None and track.active
                    and predictor.target_y is not None):
                vyps = track.velocity_y_px_per_sec()
                dy_to_target = predictor.target_y - track.last_pos[1]
                if vyps > 25.0 and dy_to_target > 0:
                    ttc_phys = dy_to_target / vyps
                elif abs(dy_to_target) <= 40 and vyps > 1.0:
                    ttc_phys = 0.0
            if ttc_phys is not None:
                ttc = ttc_phys if ttc is None else min(ttc, ttc_phys)
            prev_swing_state = swing_ctrl.state
            if phase_mgr.should_send_control and batter_ready:
                swing_ctrl.update(
                    predicted_crossing=crossing,
                    strike_zone_center=control_center,
                    time_to_crossing_sec=ttc,
                    predicted_vx=pvx,
                    batter_ready=batter_ready,
                )
            else:
                # Outside controllable phases → idle the swing controller
                swing_ctrl.update(
                    predicted_crossing=None,
                    strike_zone_center=control_center,
                    time_to_crossing_sec=None,
                    predicted_vx=None,
                    batter_ready=False,
                )
            if swing_ctrl.state != prev_swing_state:
                print(f"[SWING] {prev_swing_state.name} → "
                      f"{swing_ctrl.state.name}")

            if fm is not None:
                fm.mark("control")

            if (prev_swing_state == SwingState.COOLDOWN
                    and swing_ctrl.state == SwingState.IDLE):
                tracker.reset()
                predictor.reset()
                controller.reset()
                _lockout_until = time.perf_counter() + _lockout_seconds
                phase_mgr.notify_swing_complete()
                print(f"[INFO] Pitch cycle complete — tracker reset, "
                      f"lockout {_lockout_seconds:.1f}s")
                if session_metrics is not None:
                    _err = swing_ctrl.error_px
                    _err_scalar = float(np.hypot(_err[0], _err[1])) \
                        if isinstance(_err, (tuple, list)) else float(_err)
                    session_metrics.record_swing(
                        error_px=_err_scalar,
                        ttc_sec=ttc if ttc is not None else 0.0,
                        hit=False,
                        phase=phase_mgr.phase.name,
                    )

            #
            # 6) SEND to Dolphin
            #
            if dolphin is not None:
                now_t = time.perf_counter()
                in_lockout = now_t < _lockout_until

                if in_lockout:
                    # During lockout: hold B for base-running, stick neutral
                    dolphin.apply(0.5, 0.5, False, press_b=True)
                    _rearm_frame = 0  # reset rearm
                else:
                    if _rearm_frame == 0 and _lockout_until > 0:
                        _rearm_frame = frame_num
                        print("[INFO] Lockout expired — rearm pulse")

                    in_rearm = (0 < _rearm_frame
                                and frame_num - _rearm_frame < 10)

                    if in_rearm:
                        nudge = 0.55 if (frame_num & 1) else 0.45
                        dolphin.apply(nudge, 0.5, False, press_b=False)
                    elif (phase_mgr.phase in (GamePhase.PRE_PITCH,
                                              GamePhase.RESET_TO_BATTER)
                          and home_target_x is not None
                          and sz_center is not None):
                        dx_home = home_target_x - sz_center[0]
                        if abs(dx_home) > max(5.0, args.deadzone * 0.5):
                            home_sx = max(0.0, min(1.0,
                                0.5 + dx_home * max(args.gain, 0.01)))
                        else:
                            home_sx = 0.5
                        dolphin.apply(home_sx, 0.5, False, press_b=False)
                    else:
                        dolphin.apply(swing_ctrl.stick_x,
                                      swing_ctrl.stick_y,
                                      swing_ctrl.swing,
                                      press_b=False)

                if now_t - _focus_check_ts > 5.0:
                    _focus_check_ts = now_t
                    if _dolphin_hwnd:
                        fg = _user32.GetForegroundWindow()
                        if fg != _dolphin_hwnd:
                            _focus_dolphin(_dolphin_hwnd)

            #
            # 7) VISUALISE
            #
            disp = frame

            draw_ball_overlays(disp, detector, tracker, predictor)

            # Controller / batter HUD
            draw_hud(disp, controller, tracker, detector,
                     sz_center, batter_ready, fps_ema, frame_num,
                     dry_run, mv_tracker=mv_tracker)

            # Phase + SwingController overlay
            h_d, w_d = disp.shape[:2]
            phase_col = COL_GREEN if phase_mgr.is_batter_view else COL_ORANGE
            put_text(disp,
                     f"PHASE: {phase_mgr.phase.name}  "
                     f"SWING: {swing_ctrl.state.name}",
                     (10, h_d - 44), 0.38, phase_col, 1)

            # Status banners (drawn after HUD)
            _lr = _lockout_until - time.perf_counter()
            h_d, w_d = disp.shape[:2]
            if _lr > 0:
                put_text(disp,
                         f"LOCKOUT {_lr:.1f}s  (B held)",
                         (w_d // 2 - 110, 50), 0.65, COL_ORANGE, 2)
            elif (controller.state == ControlState.IDLE
                    and not (0 < _rearm_frame
                             and frame_num - _rearm_frame < 10)):
                put_text(disp,
                         "WAITING FOR PITCH",
                         (w_d // 2 - 100, 50), 0.65, COL_WHITE, 2)

            # Gamepad health bar (always visible when gamepad active)
            if dolphin is not None:
                h_d, w_d = disp.shape[:2]
                hp = dolphin.get_health()
                hp_col = COL_GREEN if hp['alive'] else COL_RED
                hp_txt = (f"GP: {'OK' if hp['alive'] else 'STALL'}  "
                          f"{hp['hz']:.0f}Hz  "
                          f"xi={hp['slot']}  "
                          f"gen={hp['gen']}  "
                          f"rpt={hp['reports']}")
                put_text(disp, hp_txt,
                         (w_d - 380, h_d - 24), 0.33, hp_col, 1)

            if scale != 1.0:
                dw = int(disp.shape[1] * scale)
                dh = int(disp.shape[0] * scale)
                disp = cv2.resize(disp, (dw, dh))
            cv2.imshow(WINDOW, disp)

            # Record per-frame metrics
            if fm is not None and session_metrics is not None:
                fm.mark("display")
                session_metrics.record(fm, extras={
                    "phase": phase_mgr.phase.name,
                    "swing_state": swing_ctrl.state.name,
                    "ctrl_state": controller.state.name,
                    "batter_ready": batter_ready,
                })

            #
            # 8) KEY HANDLING
            #
            cv2.waitKey(1)

            def _key_pressed(vk: int) -> bool:
                """True once when key transitions to pressed (edge)."""
                return bool(_user32.GetAsyncKeyState(vk) & 0x0001)

            VK_Q, VK_R, VK_D, VK_Y = 0x51, 0x52, 0x44, 0x59
            VK_ESCAPE = 0x1B
            VK_OEM_PLUS, VK_OEM_MINUS = 0xBB, 0xBD

            if _key_pressed(VK_Q) or _key_pressed(VK_ESCAPE):
                break

            elif _key_pressed(VK_R):
                tracker.reset()
                controller.reset()
                swing_ctrl.reset()
                phase_mgr.reset()
                if dolphin is not None:
                    dolphin.safe_neutral()
                print("[INFO] Reset.")

            elif _key_pressed(VK_D):
                show_debug = not show_debug
                print(f"[INFO] Debug HUD {'ON' if show_debug else 'OFF'}")

            elif _key_pressed(VK_Y):
                sy_sel.start(frame)

            elif _key_pressed(VK_OEM_PLUS):
                _anticipation_ms += 25
                print(f"[INFO] Anticipation → {_anticipation_ms} ms")

            elif _key_pressed(VK_OEM_MINUS):
                _anticipation_ms = max(25, _anticipation_ms - 25)
                print(f"[INFO] Anticipation → {_anticipation_ms} ms")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")
    finally:
        # Ensure safe state
        _cap.stop()
        if dolphin is not None:
            dolphin.safe_neutral()
            dolphin.close()
        cv2.destroyAllWindows()

        # Session summary
        if session_metrics is not None:
            session_metrics.print_summary()
            if cfg.metrics_csv_path:
                session_metrics.export_csv(cfg.metrics_csv_path)

        print("[INFO] Exited cleanly.")


if __name__ == "__main__":
    main()
