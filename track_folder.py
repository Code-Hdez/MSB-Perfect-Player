"""
track_folder.py — Run ball tracking on a folder of saved pitch frames.

CLI tool: no need to edit library code for experiments.

Usage examples
--------------
  # Classical detector (legacy):
  python track_folder.py pitches/20260227_205241

  # ML detector (recommended):
  python track_folder.py pitches/20260301_030216 --model runs/ball_detect/train/weights/best.pt

  python track_folder.py pitches/20260301_030216 --model best.pt --debug-video debug.mp4
  python track_folder.py pitches/20260301_030216 --model best.pt -o results.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from msb.config import Config
from msb.detector import BallDetector
from msb.tracker import BallTracker, TrackState
from msb.predictor import TrajectoryPredictor
from msb.corridor import TrajectoryCorridor
from msb.visualiser import PitchVisualiser
from msb.detector_ml import MLBallDetector
from msb.tracker_ml import MLBallTracker, MLTrackState
from msb.utils import (
    put_text, in_rect,
    COL_GREEN, COL_RED, COL_YELLOW, COL_CYAN, COL_WHITE,
    COL_MAGENTA, COL_ORANGE,
)


def load_frames(folder: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    return sorted(f for f in folder.iterdir() if f.suffix.lower() in exts)


def run_tracking(
    folder: Path,
    cfg: Config,
    model_path: Optional[str] = None,
    conf: float = 0.25,
    imgsz: int = 960,
    corridor_path: Optional[Path] = None,
    output_json: Optional[Path] = None,
    debug_video: Optional[Path] = None,
    debug_frames_dir: Optional[Path] = None,
    annotations_path: Optional[Path] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run the full detector + tracker pipeline on recorded frames.

    Returns a results dict (per-frame detections, track events, metrics).
    """
    files = load_frames(folder)
    if not files:
        print(f"[ERROR] No image files in {folder}")
        return {}

    n_frames = len(files)
    if verbose:
        print(f"[INFO] {n_frames} frames from {folder}")

    # Pipeline components
    use_ml = model_path is not None
    if use_ml:
        detector = MLBallDetector(model_path, conf=conf, imgsz=imgsz, cfg=cfg)
        tracker = MLBallTracker(cfg=cfg)
        if verbose:
            print(f"[INFO] Using ML detector: {model_path} "
                  f"(conf={conf}, imgsz={imgsz})")
    else:
        detector = BallDetector(cfg)
        tracker = BallTracker(cfg)
        if verbose:
            print("[INFO] Using classical detector")

    predictor = TrajectoryPredictor(cfg=cfg)
    vis = PitchVisualiser()

    corridor = TrajectoryCorridor(cfg)
    if corridor_path and corridor_path.exists():
        corridor.load(corridor_path)
        if verbose:
            print(f"[INFO] Loaded corridor from {corridor_path}")
    elif cfg.corridor_default is not None:
        corridor.set_rect(*cfg.corridor_default)

    # Annotations (optional, for comparison)
    annotations: Dict[str, Dict[str, Any]] = {}
    if annotations_path and annotations_path.exists():
        with open(annotations_path) as f:
            data = json.load(f)
        annotations = data.get("annotations", {})
        if verbose:
            print(f"[INFO] Loaded {len(annotations)} annotations")

    # Debug video / frames
    writer: Optional[cv2.VideoWriter] = None
    if debug_frames_dir:
        debug_frames_dir.mkdir(parents=True, exist_ok=True)

    # Per-frame output
    per_frame: List[Dict[str, Any]] = []
    track_events: List[Dict[str, Any]] = []
    prev_state = TrackState.IDLE if not use_ml else MLTrackState.IDLE

    for i, fpath in enumerate(files):
        frame = cv2.imread(str(fpath))
        if frame is None:
            per_frame.append({"frame": i, "error": "unreadable"})
            continue

        # Run pipeline
        track_active = (tracker.track is not None
                        and tracker.track.active
                        and tracker.track.confirmed)
        best = detector.detect(frame, corridor=corridor,
                               track_active=track_active)
        track = tracker.update(detector.candidates, best, detector=detector)

        sel = tracker.selected
        entry: Dict[str, Any] = {
            "frame": i,
            "state": tracker.state.name,
            "n_candidates": len(detector.candidates),
        }

        if sel is not None:
            entry["tracked"] = {
                "x": sel.center[0], "y": sel.center[1],
                "area": sel.area,
                "circularity": sel.circularity,
                "isolation": sel.isolation_score,
                "corridor": sel.corridor_score,
                "score": sel.score,
                "in_motion": sel.in_motion_mask,
            }
        elif best is not None:
            entry["best_untracked"] = {
                "x": best.center[0], "y": best.center[1],
                "area": best.area, "score": best.score,
            }

        if track and track.active:
            entry["track_info"] = {
                "confirmed": track.confirmed,
                "total_frames": track.total_frames,
                "velocity": list(track.velocity),
                "missed": track.frames_since_seen,
                "pitcher_zone_frames": track._frames_in_pitcher_zone,
            }

        # Annotation comparison
        ann = annotations.get(str(i))
        if ann and ann.get("visible") and "tracked" in entry:
            gt_x, gt_y = ann["x"], ann["y"]
            tx, ty = entry["tracked"]["x"], entry["tracked"]["y"]
            err = float(np.hypot(tx - gt_x, ty - gt_y))
            entry["annotation_error_px"] = err

        per_frame.append(entry)

        # Track state transitions
        cur_state = tracker.state
        if cur_state != prev_state:
            track_events.append({
                "frame": i,
                "from": prev_state.name,
                "to": cur_state.name,
            })
        prev_state = cur_state

        # Console output
        if verbose:
            if sel is not None:
                c = sel.center
                tag = "CONFIRMED" if (track and track.confirmed) else "tentative"
                vel_str = ""
                if track and track.active:
                    vel_str = (f"  vel=({track.velocity[0]:+.1f},"
                               f"{track.velocity[1]:+.1f})  "
                               f"pts={len(track.positions)}")
                print(f"Frame {i:3d}: TRACKED ({c[0]:4d},{c[1]:4d}) "
                      f"a={sel.area:5.0f} [{tag}]{vel_str}")
            elif best is not None:
                c = best.center
                print(f"Frame {i:3d}: notrack best=({c[0]:4d},{c[1]:4d}) "
                      f"[{cur_state.name}]")
            else:
                print(f"Frame {i:3d}: NONE [{cur_state.name}]")

        # Debug visualisation
        if debug_video is not None or debug_frames_dir is not None:
            dvis = vis.overlay(frame, detector, tracker, predictor,
                               corridor, None, False, 0.0, i, cfg=cfg)

            # Draw annotation ground truth if available
            if ann and ann.get("visible"):
                gt = (ann["x"], ann["y"])
                cv2.drawMarker(dvis, gt, COL_RED, cv2.MARKER_CROSS, 18, 2)
                cv2.circle(dvis, gt, 6, COL_RED, 1)
                if sel is not None:
                    cv2.line(dvis, sel.center, gt, COL_MAGENTA, 1)
                    err_val = entry.get("annotation_error_px")
                    if err_val is not None:
                        mid = ((sel.center[0] + gt[0]) // 2,
                               (sel.center[1] + gt[1]) // 2)
                        put_text(dvis, f"{err_val:.1f}px",
                                 (mid[0] + 5, mid[1] - 5), 0.35,
                                 COL_MAGENTA, 1)

            h, w = dvis.shape[:2]
            if debug_video is not None:
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(
                        str(debug_video), fourcc, 30, (w, h))
                writer.write(dvis)

            if debug_frames_dir is not None:
                out_path = debug_frames_dir / f"debug_{i:04d}.png"
                cv2.imwrite(str(out_path), dvis)

    if writer is not None:
        writer.release()
        if verbose:
            print(f"[INFO] Debug video saved: {debug_video}")
    if debug_frames_dir is not None and verbose:
        print(f"[INFO] Debug frames saved: {debug_frames_dir}")

    # Build results
    results: Dict[str, Any] = {
        "folder": str(folder),
        "n_frames": n_frames,
        "config": cfg.to_dict(),
        "track_events": track_events,
        "per_frame": per_frame,
    }

    # Summary statistics
    tracked_frames = [e for e in per_frame if "tracked" in e]
    results["summary"] = {
        "total_frames": n_frames,
        "tracked_frames": len(tracked_frames),
        "tracked_pct": len(tracked_frames) / n_frames * 100 if n_frames else 0,
        "track_transitions": len(track_events),
    }
    if annotations:
        errors = [e["annotation_error_px"] for e in per_frame
                  if "annotation_error_px" in e]
        if errors:
            arr = np.array(errors)
            results["summary"]["annotation_comparison"] = {
                "n_compared": len(errors),
                "mean_error_px": float(np.mean(arr)),
                "median_error_px": float(np.median(arr)),
                "max_error_px": float(np.max(arr)),
                "within_10px": float(np.mean(arr <= 10) * 100),
                "within_20px": float(np.mean(arr <= 20) * 100),
            }

    if verbose:
        print(f"\n--- Summary ---")
        print(f"  Tracked: {results['summary']['tracked_frames']}"
              f"/{n_frames} frames "
              f"({results['summary']['tracked_pct']:.1f}%)")
        print(f"  Track transitions: {len(track_events)}")
        if "annotation_comparison" in results.get("summary", {}):
            ac = results["summary"]["annotation_comparison"]
            print(f"  Annotation errors: mean={ac['mean_error_px']:.1f}px "
                  f"median={ac['median_error_px']:.1f}px "
                  f"max={ac['max_error_px']:.1f}px")

    # Save output
    if output_json:
        # Strip per_frame from serialisation if too large
        with open(output_json, "w") as f:
            json.dump(results, f, indent=2, default=str)
        if verbose:
            print(f"[INFO] Results saved to {output_json}")

    return results


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run ball tracking on a folder of saved pitch frames")
    ap.add_argument("folder", help="Path to pitch recording folder")
    ap.add_argument("-c", "--config", default=None,
                    help="Path to config file (TOML or JSON)")
    ap.add_argument("-o", "--output", default=None,
                    help="Output JSON path for tracking results")
    ap.add_argument("--debug-video", default=None,
                    help="Output path for debug video (e.g. debug.mp4)")
    ap.add_argument("--debug-frames", default=None,
                    help="Output folder for per-frame debug images")
    ap.add_argument("-a", "--annotations", default=None,
                    help="Annotations JSON for error comparison")
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
    ap.add_argument("-q", "--quiet", action="store_true",
                    help="Suppress per-frame console output")
    args = ap.parse_args()

    folder = Path(args.folder)
    if not folder.is_dir():
        print(f"[ERROR] Not a directory: {folder}")
        sys.exit(1)

    # Load config
    if args.config:
        cfg = Config.load(args.config)
    else:
        cfg_path = Path("config.toml")
        if cfg_path.exists():
            cfg = Config.load(str(cfg_path))
        else:
            cfg = Config()

    # CLI overrides
    overrides: Dict[str, Any] = {}
    if args.corridor_rect:
        overrides["corridor_default"] = tuple(args.corridor_rect)
    if args.pitcher_zone:
        overrides["pitcher_zone"] = tuple(args.pitcher_zone)
    cfg.apply_overrides(overrides)

    # Annotations
    ann_path = None
    if args.annotations:
        ann_path = Path(args.annotations)
    else:
        default_ann = folder / "annotations.json"
        if default_ann.exists():
            ann_path = default_ann

    run_tracking(
        folder, cfg,
        model_path=args.model,
        conf=args.conf,
        imgsz=args.imgsz,
        corridor_path=Path(args.corridor) if args.corridor else None,
        output_json=Path(args.output) if args.output else None,
        debug_video=Path(args.debug_video) if args.debug_video else None,
        debug_frames_dir=Path(args.debug_frames) if args.debug_frames else None,
        annotations_path=ann_path,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
