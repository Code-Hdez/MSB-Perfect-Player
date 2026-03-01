"""Generate visual debug frames showing ALL detection layers.

Output: pitches/20260227_205241/debug/ folder with annotated PNGs.
Each frame shows:
  - White mask overlay (blue tint)
  - Foreground/motion mask (green tint)
  - All detected bright+moving blobs (yellow circles)
  - Combined (white AND fg) blobs (red circles) 
  - Tracker output (magenta cross)
  - Trajectory A candidate (cyan circle) — the manually-identified fast trajectory
"""
import sys
sys.path.insert(0, ".")
import cv2
import numpy as np
from pathlib import Path
from pitch_analyzer import (
    BallDetector, BallTracker, TrajectoryCorridor, CORRIDOR_DEFAULT,
    PITCHER_ZONE, BALL_HSV_LOWER, BALL_HSV_UPPER
)

folder = Path("pitches/20260227_205241")
debug_dir = folder / "debug"
debug_dir.mkdir(exist_ok=True)

files = sorted(f for f in folder.iterdir() if f.suffix.lower() == ".png")
print(f"Loaded {len(files)} frames")

detector = BallDetector()
tracker = BallTracker()
corridor = TrajectoryCorridor()
corridor.set_rect(*CORRIDOR_DEFAULT)

prev_grey = None

for i, fpath in enumerate(files):
    frame = cv2.imread(str(fpath))
    if frame is None:
        continue

    if i < 50 or i > 90:
        # Process but don't save debug frames before/after pitch
        detector.detect(frame, corridor=corridor,
                        track_active=(tracker.track is not None and tracker.track.active))
        cands = detector.candidates
        tracker.update(cands, best_candidate=detector.best)
        prev_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        continue

    # Run detection
    track_active = tracker.track is not None and tracker.track.active
    detector.detect(frame, corridor=corridor, track_active=track_active)
    cands = detector.candidates
    track = tracker.update(cands, best_candidate=detector.best)

    # Build visualization
    vis = frame.copy()
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # White mask overlay (semi-transparent blue)
    white_mask = detector.white_mask
    if white_mask is not None:
        blue_overlay = np.zeros_like(vis)
        blue_overlay[:, :, 0] = 200  # Blue channel
        vis[white_mask > 0] = cv2.addWeighted(
            vis[white_mask > 0], 0.6,
            blue_overlay[white_mask > 0], 0.4, 0).astype(np.uint8) if np.any(white_mask > 0) else vis[white_mask > 0]

    # Motion mask overlay (semi-transparent green, only in corridor)
    motion_mask = detector.motion_mask
    if motion_mask is not None:
        green_layer = vis.copy()
        green_layer[motion_mask > 0, 1] = np.clip(
            green_layer[motion_mask > 0, 1].astype(int) + 60, 0, 255).astype(np.uint8)
        vis = cv2.addWeighted(vis, 0.7, green_layer, 0.3, 0)

    # Combined mask contours (red outline)
    combined = detector.combined_mask
    if combined is not None:
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, (0, 0, 255), 1)

    # Draw corridor rectangle
    cx1, cy1, cx2, cy2 = CORRIDOR_DEFAULT
    cv2.rectangle(vis, (cx1, cy1), (cx2, cy2), (100, 100, 100), 1)

    # Draw pitcher zone
    px1, py1, px2, py2 = PITCHER_ZONE
    cv2.rectangle(vis, (px1, py1), (px2, py2), (0, 255, 255), 1)

    # ALL candidates: yellow dots
    for c in cands:
        color = (0, 255, 255) if c.in_motion_mask else (200, 200, 0)
        cv2.circle(vis, c.center, 4, color, -1)
        cv2.putText(vis, f"{c.area:.0f}", (c.center[0]+6, c.center[1]-3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

    # Bright+moving detection (our custom analysis)
    if prev_grey is not None:
        diff = cv2.absdiff(grey, prev_grey)
        bright_now = (grey > 180).astype(np.uint8) * 255
        changed = (diff > 30).astype(np.uint8) * 255
        bright_appeared = cv2.bitwise_and(bright_now, changed)
        ba_contours, _ = cv2.findContours(bright_appeared, cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)
        for cnt in ba_contours:
            a = cv2.contourArea(cnt)
            if a < 4 or a > 200:
                continue
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            if cy > 100 and 380 <= cx <= 820:
                cv2.circle(vis, (cx, cy), 8, (255, 255, 0), 1)  # cyan outline

    # Tracker state
    if tracker.selected is not None:
        sel = tracker.selected.center
        cv2.drawMarker(vis, sel, (255, 0, 255), cv2.MARKER_CROSS, 20, 2)
    if tracker.track is not None and tracker.track.active:
        pred = tracker.track.last_pos
        cv2.circle(vis, (int(pred[0]), int(pred[1])), 12, (255, 0, 255), 2)
        # Draw trail
        pts = list(tracker.track.positions)
        for j in range(1, len(pts)):
            p1 = (pts[j-1][0], pts[j-1][1])
            p2 = (pts[j][0], pts[j][1])
            cv2.line(vis, p1, p2, (255, 0, 255), 1)

    # Label
    state_name = tracker.state.name if hasattr(tracker.state, 'name') else str(tracker.state)
    cv2.putText(vis, f"Frame {i} [{state_name}]",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Save
    out_path = debug_dir / f"debug_{i:04d}.png"
    cv2.imwrite(str(out_path), vis)
    prev_grey = grey.copy()

print(f"\nDone. Debug frames saved to: {debug_dir}")
print(f"Open them in an image viewer to see detection layers.")
