"""Diagnostic: dump ALL candidates + raw contours at frames 55-70.
Answers why the ball at ~(570,217) vanishes from candidates."""
import sys, cv2, numpy as np
from pathlib import Path

sys.path.insert(0, ".")
from pitch_analyzer import (
    BallDetector, BallTracker, TrajectoryCorridor,
    CORRIDOR_DEFAULT, BALL_MIN_AREA, BALL_MAX_AREA,
    BALL_MIN_CIRCULARITY, BALL_FLIGHT_MAX_AREA,
    STATIC_HIT_THRESHOLD, STATIC_CELL_SIZE,
)

folder = Path(sys.argv[1] if len(sys.argv) > 1 else "pitches/20260227_205241")
files = sorted(f for f in folder.iterdir() if f.suffix.lower() in {".png", ".jpg"})
print(f"Loaded {len(files)} frames\n")

# User-confirmed ball positions (frame: (x, y))
BALL_GT = {60: (570, 217), 70: (573, 342), 80: (573, 543)}

detector = BallDetector()
tracker = BallTracker()
corridor = TrajectoryCorridor()
if CORRIDOR_DEFAULT:
    corridor.set_rect(*CORRIDOR_DEFAULT)

DIAG_START, DIAG_END = 55, 72

for i, f in enumerate(files):
    frame = cv2.imread(str(f))
    if frame is None:
        continue

    track_active = (tracker.track is not None
                    and tracker.track.active
                    and tracker.track.confirmed)
    best = detector.detect(frame, corridor=corridor, track_active=track_active)
    track = tracker.update(detector.candidates, best)

    if not (DIAG_START <= i <= DIAG_END):
        continue

    print(f"=== Frame {i} ===")

    # 1) Raw contours from combined mask BEFORE any filtering
    combined = detector.combined_mask
    if combined is not None:
        raw_contours, _ = cv2.findContours(
            combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Show contours near the ball region (y=150-350, x=500-650)
        near_ball = []
        for cnt in raw_contours:
            area = cv2.contourArea(cnt)
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                bx, by, bw, bh = cv2.boundingRect(cnt)
                cx, cy = bx + bw // 2, by + bh // 2
            # Show ALL contours in the region of interest
            if 450 <= cx <= 700 and 100 <= cy <= 400:
                peri = cv2.arcLength(cnt, True)
                circ = 4 * np.pi * area / (peri * peri) if peri > 1 else 0
                near_ball.append((cx, cy, area, circ))
        near_ball.sort(key=lambda x: (x[1], x[0]))  # sort by y then x
        print(f"  Raw combined-mask contours (x:450-700, y:100-400): {len(near_ball)}")
        for cx, cy, a, c in near_ball:
            tag = ""
            if a < BALL_MIN_AREA:
                tag += " [TOO_SMALL]"
            elif a > BALL_MAX_AREA:
                tag += " [TOO_BIG]"
            if c < BALL_MIN_CIRCULARITY:
                tag += " [LOW_CIRC]"
            print(f"    ({cx:4d},{cy:4d}) area={a:6.0f} circ={c:.3f}{tag}")

    # 2) Also check white-only mask contours
    white = detector.white_mask
    if white is not None:
        w_contours, _ = cv2.findContours(
            white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        near_ball_w = []
        for cnt in w_contours:
            area = cv2.contourArea(cnt)
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                bx, by, bw, bh = cv2.boundingRect(cnt)
                cx, cy = bx + bw // 2, by + bh // 2
            if 450 <= cx <= 700 and 100 <= cy <= 400:
                peri = cv2.arcLength(cnt, True)
                circ = 4 * np.pi * area / (peri * peri) if peri > 1 else 0
                near_ball_w.append((cx, cy, area, circ))
        near_ball_w.sort(key=lambda x: (x[1], x[0]))
        # Only show the small blobs near ball position
        small_w = [(cx, cy, a, c) for cx, cy, a, c in near_ball_w
                   if 500 <= cx <= 620 and 150 <= cy <= 350 and a <= 200]
        if small_w:
            print(f"  White-only small blobs near ball (x:500-620, y:150-350, a<=200): {len(small_w)}")
            for cx, cy, a, c in small_w:
                print(f"    ({cx:4d},{cy:4d}) area={a:6.0f} circ={c:.3f}")

    # 3) Check pixel values at GT ball position
    if i in BALL_GT:
        bx, by = BALL_GT[i]
        if combined is not None:
            val_comb = combined[by, bx] if 0 <= by < combined.shape[0] and 0 <= bx < combined.shape[1] else -1
        else:
            val_comb = -1
        if white is not None:
            val_white = white[by, bx] if 0 <= by < white.shape[0] and 0 <= bx < white.shape[1] else -1
        else:
            val_white = -1
        bg_fg = detector.bg_fg_mask
        if bg_fg is not None:
            val_bg = bg_fg[by, bx] if 0 <= by < bg_fg.shape[0] and 0 <= bx < bg_fg.shape[1] else -1
        else:
            val_bg = -1
        motion = detector.motion_mask
        if motion is not None:
            val_mot = motion[by, bx] if 0 <= by < motion.shape[0] and 0 <= bx < motion.shape[1] else -1
        else:
            val_mot = -1
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        val_grey = grey[by, bx]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        val_hsv = tuple(hsv[by, bx])
        print(f"  ** GT ball at ({bx},{by}): grey={val_grey} HSV={val_hsv}"
              f" white={val_white} bg_fg={val_bg} motion={val_mot} combined={val_comb}")
        # Check 5x5 neighbourhood
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                ny, nx = by + dy, bx + dx
                if 0 <= ny < combined.shape[0] and 0 <= nx < combined.shape[1]:
                    if combined[ny, nx] > 0:
                        print(f"    combined[{ny},{nx}]=255 (offset {dx},{dy})")
                        break
            else:
                continue
            break
        else:
            print(f"    NO combined-mask pixel within ±3px of GT!")

    # 4) Detector candidates (after all filtering)
    print(f"  Detector candidates (total): {len(detector.candidates)}")
    for c in detector.candidates[:15]:  # top 15 by score
        sx, sy = c.center
        static_cell = (sx // STATIC_CELL_SIZE, sy // STATIC_CELL_SIZE)
        static_count = detector._static_map.get(static_cell, 0)
        tag = f"  static_hits={static_count}"
        if static_count >= STATIC_HIT_THRESHOLD:
            tag += " [SUPPRESSED]"
        print(f"    ({sx:4d},{sy:4d}) area={c.area:5.0f} circ={c.circularity:.2f}"
              f" score={c.score:.3f} motion={c.in_motion_mask}{tag}")

    # 5) Tracker state
    sel = tracker.selected
    if sel:
        print(f"  Tracker selected: ({sel.center[0]},{sel.center[1]}) area={sel.area}")
    else:
        print(f"  Tracker selected: NONE")
    if tracker.track and tracker.track.active:
        t = tracker.track
        print(f"  Track: vel=({t.velocity[0]:+.1f},{t.velocity[1]:+.1f})"
              f" pts={len(t.positions)} confirmed={t.confirmed}"
              f" missed={t.frames_since_seen}")
    print()
