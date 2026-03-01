"""Deep diagnostic: for each annotated frame, show what the detector sees
near the ground-truth ball position.  Also dumps ALL candidates so we can
understand why the tracker picks the wrong one."""
import sys, cv2, json
import numpy as np
from pathlib import Path
from msb import Config, BallDetector, TrajectoryCorridor

folder = Path(sys.argv[1] if len(sys.argv) > 1 else "pitches/20260301_030216")
ann_path = folder / "annotations.json"
with open(ann_path) as f:
    data = json.load(f)
annotations = data["annotations"]

cfg = Config()
detector = BallDetector(cfg)
corridor = TrajectoryCorridor(cfg)
if cfg.corridor_default:
    corridor.set_rect(*cfg.corridor_default)

files = sorted(f for f in folder.iterdir() if f.suffix.lower() in {".png", ".jpg"})

for i, fpath in enumerate(files):
    frame = cv2.imread(str(fpath))
    if frame is None:
        continue

    best = detector.detect(frame, corridor=corridor, track_active=False)
    cands = detector.candidates

    key = str(i)
    ann = annotations.get(key)
    gt_visible = ann is not None and ann.get("visible", False)
    if not gt_visible:
        continue

    gx, gy = ann["x"], ann["y"]

    # Examine pixel at ground truth
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h_px, s_px, v_px = hsv[gy, gx]
    g_px = grey[gy, gx]

    # Check masks at GT position
    bg_fg_val = detector.bg_fg_mask[gy, gx] if detector.bg_fg_mask is not None else -1
    motion_val = detector.motion_mask[gy, gx] if detector.motion_mask is not None else -1
    white_val = detector.white_mask[gy, gx] if detector.white_mask is not None else -1
    combined_val = detector.combined_mask[gy, gx] if detector.combined_mask is not None else -1

    # Find nearest candidate to GT
    nearest_dist = 9999
    nearest_cand = None
    for c in cands:
        d = np.hypot(c.center[0] - gx, c.center[1] - gy)
        if d < nearest_dist:
            nearest_dist = d
            nearest_cand = c

    # Best candidate info
    best_str = "NONE"
    best_dist = -1
    if best is not None:
        best_dist = np.hypot(best.center[0] - gx, best.center[1] - gy)
        best_str = (f"({best.center[0]:3d},{best.center[1]:3d}) "
                    f"a={best.area:5.0f} circ={best.circularity:.2f} "
                    f"iso={best.isolation_score:.2f} score={best.score:.2f} "
                    f"dist_gt={best_dist:.0f}")

    nearest_str = "NONE"
    if nearest_cand is not None:
        nearest_str = (f"({nearest_cand.center[0]:3d},{nearest_cand.center[1]:3d}) "
                       f"a={nearest_cand.area:5.0f} circ={nearest_cand.circularity:.2f} "
                       f"iso={nearest_cand.isolation_score:.2f} score={nearest_cand.score:.2f} "
                       f"dist_gt={nearest_dist:.0f}")

    print(f"F{i:3d} GT=({gx},{gy})  grey={g_px:3d} HSV=({h_px},{s_px},{v_px})"
          f"  masks: bg={bg_fg_val:3d} mot={motion_val:3d} whi={white_val:3d} comb={combined_val:3d}"
          f"  #cands={len(cands):2d}")
    print(f"      best  = {best_str}")
    if nearest_cand is not best and nearest_cand is not None:
        print(f"      near  = {nearest_str}")
    if len(cands) <= 8:
        for j, c in enumerate(cands):
            d = np.hypot(c.center[0] - gx, c.center[1] - gy)
            tag = " <-- BEST" if c is best else ""
            tag += " <-- NEAR" if c is nearest_cand and c is not best else ""
            print(f"        [{j}] ({c.center[0]:3d},{c.center[1]:3d}) a={c.area:5.0f} "
                  f"circ={c.circularity:.2f} iso={c.isolation_score:.2f} "
                  f"mot={c.in_motion_mask} score={c.score:.2f} d_gt={d:.0f}{tag}")
    print()
