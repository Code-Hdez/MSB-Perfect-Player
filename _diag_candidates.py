"""Diagnostic: show all candidates at frames 49-68, compare with GT."""
import json, sys
from pathlib import Path
import cv2, numpy as np
from msb import Config, BallDetector, TrajectoryCorridor

folder = Path(sys.argv[1] if len(sys.argv) > 1 else "pitches/20260301_030216")
cfg = Config.load("config.toml") if Path("config.toml").exists() else Config()

with open(folder / "annotations.json") as f:
    ann = json.load(f)["annotations"]

files = sorted(f for f in folder.iterdir() if f.suffix.lower() in {".png", ".jpg"})
det = BallDetector(cfg)
corridor = TrajectoryCorridor(cfg)
if cfg.corridor_default:
    corridor.set_rect(*cfg.corridor_default)

for i, fp in enumerate(files):
    if i < 46 or i > 70:
        continue
    frame = cv2.imread(str(fp))
    if frame is None:
        continue
    det.detect(frame, corridor=corridor, track_active=False)
    
    a = ann.get(str(i), {})
    gt_vis = a.get("visible", False)
    gt_x = a.get("x", -1)
    gt_y = a.get("y", -1)
    
    cands = det.candidates
    best = det.best
    
    # Find nearest candidate to GT
    nearest_d = 9999
    nearest_c = None
    for c in cands:
        d = np.hypot(c.center[0] - gt_x, c.center[1] - gt_y)
        if d < nearest_d:
            nearest_d = d
            nearest_c = c
    
    print(f"\nFrame {i:3d}  GT={'VIS' if gt_vis else 'hid':3s} ({gt_x:4d},{gt_y:4d})  #cands={len(cands)}")
    if best:
        bd = np.hypot(best.center[0] - gt_x, best.center[1] - gt_y)
        print(f"  BEST: ({best.center[0]:4d},{best.center[1]:4d}) a={best.area:5.0f} "
              f"circ={best.circularity:.2f} bri={best.brightness_score:.2f} "
              f"corr={best.corridor_score:.2f} score={best.score:.2f}  d_gt={bd:.0f}")
    
    if nearest_c and nearest_c is not best:
        print(f"  NEAR: ({nearest_c.center[0]:4d},{nearest_c.center[1]:4d}) a={nearest_c.area:5.0f} "
              f"circ={nearest_c.circularity:.2f} bri={nearest_c.brightness_score:.2f} "
              f"corr={nearest_c.corridor_score:.2f} score={nearest_c.score:.2f}  d_gt={nearest_d:.0f}")
    elif nearest_c is best and nearest_d < 9999:
        print(f"  (best IS nearest, d_gt={nearest_d:.0f})")
    
    # Show top 5 candidates for context
    for j, c in enumerate(cands[:5]):
        d_gt = np.hypot(c.center[0] - gt_x, c.center[1] - gt_y)
        tag = " <-- BEST" if c is best else ""
        print(f"  [{j}] ({c.center[0]:4d},{c.center[1]:4d}) a={c.area:5.0f} "
              f"circ={c.circularity:.2f} bri={c.brightness_score:.2f} "
              f"mot={'Y' if c.in_motion_mask else 'n'} corr={c.corridor_score:.2f} "
              f"score={c.score:.2f}  d_gt={d_gt:.0f}{tag}")
