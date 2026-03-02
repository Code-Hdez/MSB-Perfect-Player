"""
validate_tracking.py — Evaluate ball tracking against ground-truth annotations.

Compares tracker predictions against manually annotated ball positions
and reports detection accuracy metrics.

Usage examples
--------------
  # Classical detector:
  python validate_tracking.py pitches/20260301_030216

  # ML detector:
  python validate_tracking.py pitches/20260301_030216 --model weights/ball_best.pt

  # Custom annotations file:
  python validate_tracking.py pitches/20260301_030216 -a my_annotations.json

  # Generate debug video with GT overlay:
  python validate_tracking.py pitches/20260301_030216 --model weights/ball_best.pt --debug-video val_debug.mp4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from msb.config import Config
from msb.detector import BallDetector, BallCandidate
from msb.tracker import BallTracker, TrackState
from msb.predictor import TrajectoryPredictor
from msb.corridor import TrajectoryCorridor
from msb.visualiser import PitchVisualiser
from msb.detector_ml import MLBallDetector
from msb.tracker_ml import MLBallTracker, MLTrackState
from msb.utils import put_text, COL_GREEN, COL_RED, COL_YELLOW, COL_MAGENTA


# Metrics

def compute_metrics(
    predictions: Dict[int, Tuple[int, int]],
    ground_truth: Dict[int, Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute tracking quality metrics.

    Parameters
    ----------
    predictions : dict
        {frame_idx: (x, y)} for frames where the tracker reported a position.
    ground_truth : dict
        {frame_idx: {"x": int, "y": int, "visible": bool}} from annotations.

    Returns
    -------
    dict with metric fields.
    """
    gt_visible = {
        int(k): v for k, v in ground_truth.items()
        if v.get("visible", False)
    }
    gt_not_visible = {
        int(k): v for k, v in ground_truth.items()
        if not v.get("visible", False)
    }

    # True positives: predicted AND visible in GT
    errors: List[float] = []
    true_positives = 0
    for fidx, gt in gt_visible.items():
        if fidx in predictions:
            px, py = predictions[fidx]
            err = float(np.hypot(px - gt["x"], py - gt["y"]))
            errors.append(err)
            true_positives += 1

    # Missed: visible in GT but not predicted
    missed = len(gt_visible) - true_positives

    # False positives: predicted but NOT visible in GT
    false_positives = 0
    for fidx in predictions:
        if fidx in gt_not_visible:
            false_positives += 1

    # False starts: predicted before ball is visible in GT
    gt_first_visible = min(gt_visible.keys()) if gt_visible else None
    false_starts = 0
    if gt_first_visible is not None:
        for fidx in predictions:
            if fidx < gt_first_visible:
                false_starts += 1

    # Dropouts: gaps in continuous tracking during visible window
    dropouts = 0
    if gt_visible:
        vis_frames = sorted(gt_visible.keys())
        tracking = False
        for fidx in range(vis_frames[0], vis_frames[-1] + 1):
            if fidx in predictions:
                tracking = True
            elif tracking and fidx in gt_visible:
                dropouts += 1
                tracking = False

    # Continuity: longest consecutive tracked run within visible window
    max_run = 0
    current_run = 0
    if gt_visible:
        vis_frames = sorted(gt_visible.keys())
        for fidx in range(vis_frames[0], vis_frames[-1] + 1):
            if fidx in predictions:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0

    err_arr = np.array(errors) if errors else np.array([])

    detection_rate = (true_positives / len(gt_visible) * 100
                      if gt_visible else 0.0)

    metrics: Dict[str, Any] = {
        "total_gt_visible": len(gt_visible),
        "total_gt_not_visible": len(gt_not_visible),
        "total_predicted": len(predictions),
        "true_positives": true_positives,
        "missed": missed,
        "false_positives": false_positives,
        "false_starts": false_starts,
        "dropouts": dropouts,
        "detection_rate_pct": round(detection_rate, 2),
        "longest_continuous_run": max_run,
    }

    if len(err_arr) > 0:
        metrics.update({
            "mean_error_px": round(float(np.mean(err_arr)), 2),
            "median_error_px": round(float(np.median(err_arr)), 2),
            "max_error_px": round(float(np.max(err_arr)), 2),
            "std_error_px": round(float(np.std(err_arr)), 2),
            "within_5px_pct": round(float(np.mean(err_arr <= 5) * 100), 2),
            "within_10px_pct": round(float(np.mean(err_arr <= 10) * 100), 2),
            "within_20px_pct": round(float(np.mean(err_arr <= 20) * 100), 2),
        })
    else:
        metrics.update({
            "mean_error_px": None,
            "median_error_px": None,
            "max_error_px": None,
            "std_error_px": None,
            "within_5px_pct": None,
            "within_10px_pct": None,
            "within_20px_pct": None,
        })

    return metrics


def print_metrics(metrics: Dict[str, Any]) -> None:
    """Pretty-print validation metrics to stdout."""
    print("\n" + "=" * 55)
    print("  VALIDATION METRICS")
    print("=" * 55)
    print(f"  GT visible frames:     {metrics['total_gt_visible']}")
    print(f"  GT not-visible frames: {metrics['total_gt_not_visible']}")
    print(f"  Predicted frames:      {metrics['total_predicted']}")
    print(f"  True positives:        {metrics['true_positives']}")
    print(f"  Missed:                {metrics['missed']}")
    print(f"  False positives:       {metrics['false_positives']}")
    print(f"  False starts:          {metrics['false_starts']}")
    print(f"  Dropouts:              {metrics['dropouts']}")
    print(f"  Detection rate:        {metrics['detection_rate_pct']:.1f}%")
    print(f"  Longest run:           {metrics['longest_continuous_run']} frames")
    print("-" * 55)
    if metrics["mean_error_px"] is not None:
        print(f"  Mean error:            {metrics['mean_error_px']:.2f} px")
        print(f"  Median error:          {metrics['median_error_px']:.2f} px")
        print(f"  Max error:             {metrics['max_error_px']:.2f} px")
        print(f"  Std error:             {metrics['std_error_px']:.2f} px")
        print(f"  Within  5px:           {metrics['within_5px_pct']:.1f}%")
        print(f"  Within 10px:           {metrics['within_10px_pct']:.1f}%")
        print(f"  Within 20px:           {metrics['within_20px_pct']:.1f}%")
    else:
        print("  (no overlapping detections to compute error)")
    print("=" * 55)

    # Quality assessment
    dr = metrics["detection_rate_pct"]
    mean_e = metrics["mean_error_px"]
    print("\n  QUALITY ASSESSMENT:")
    if dr >= 90 and mean_e is not None and mean_e <= 10:
        print("  ** EXCELLENT — ready for real-time use")
    elif dr >= 75 and mean_e is not None and mean_e <= 20:
        print("  * GOOD — usable, minor improvements possible")
    elif dr >= 50:
        print("  ~ FAIR — needs more training data or tuning")
    else:
        print("  ! POOR — significant issues, review pipeline")
    print()


# Tracking run

def run_validation(
    folder: Path,
    cfg: Config,
    model_path: Optional[str] = None,
    conf: float = 0.25,
    imgsz: int = 960,
    annotations_path: Optional[Path] = None,
    debug_video: Optional[Path] = None,
    debug_frames_dir: Optional[Path] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run tracker on frames and compare to annotations."""

    # Load frames
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    files = sorted(f for f in folder.iterdir() if f.suffix.lower() in exts)
    if not files:
        print(f"[ERROR] No image files in {folder}")
        return {}

    # Load annotations
    ann_path = annotations_path or (folder / "annotations.json")
    if not ann_path.exists():
        print(f"[ERROR] Annotations not found: {ann_path}")
        print("  Run frame_annotator.py first to create annotations.")
        return {}

    with open(ann_path) as f:
        ann_data = json.load(f)
    ground_truth = ann_data.get("annotations", {})
    if verbose:
        n_vis = sum(1 for v in ground_truth.values() if v.get("visible"))
        print(f"[INFO] {len(files)} frames, {len(ground_truth)} annotations "
              f"({n_vis} visible)")

    # Setup pipeline
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

    corridor = TrajectoryCorridor(cfg)
    if cfg.corridor_default is not None:
        corridor.set_rect(*cfg.corridor_default)

    vis = PitchVisualiser()

    # Debug output
    writer: Optional[cv2.VideoWriter] = None
    if debug_frames_dir:
        debug_frames_dir.mkdir(parents=True, exist_ok=True)

    # Run tracking
    predictions: Dict[int, Tuple[int, int]] = {}

    for i, fpath in enumerate(files):
        frame = cv2.imread(str(fpath))
        if frame is None:
            continue

        track_active = (tracker.track is not None
                        and tracker.track.active
                        and tracker.track.confirmed)
        best = detector.detect(frame, corridor=corridor,
                               track_active=track_active)
        track = tracker.update(detector.candidates, best, detector=detector)

        sel = tracker.selected
        if sel is not None:
            predictions[i] = sel.center
        elif best is not None and not use_ml:
            # For classical, include untracked best as fallback
            pass

        # Debug output
        if debug_video is not None or debug_frames_dir is not None:
            dvis = frame.copy()

            # Draw tracker overlay
            dvis = vis.overlay(frame, detector, tracker,
                               TrajectoryPredictor(cfg=cfg),
                               corridor, None, False, 0.0, i, cfg=cfg)

            # Draw ground truth
            gt = ground_truth.get(str(i))
            if gt and gt.get("visible"):
                gx, gy = gt["x"], gt["y"]
                cv2.drawMarker(dvis, (gx, gy), COL_RED,
                               cv2.MARKER_CROSS, 20, 2)
                cv2.circle(dvis, (gx, gy), 8, COL_RED, 1)
                put_text(dvis, "GT", (gx + 10, gy - 10), 0.4, COL_RED, 1)

                if sel is not None:
                    cv2.line(dvis, sel.center, (gx, gy), COL_MAGENTA, 1)
                    err = np.hypot(sel.center[0] - gx, sel.center[1] - gy)
                    mid = ((sel.center[0] + gx) // 2,
                           (sel.center[1] + gy) // 2)
                    put_text(dvis, f"{err:.1f}px",
                             (mid[0] + 5, mid[1] - 5), 0.35,
                             COL_MAGENTA, 1)

            # State label
            state_name = tracker.state.name
            put_text(dvis, f"F{i:03d} [{state_name}]",
                     (10, 25), 0.5, COL_YELLOW, 1)

            h, w = dvis.shape[:2]
            if debug_video is not None and writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(debug_video), fourcc, 30, (w, h))
            if writer is not None:
                writer.write(dvis)
            if debug_frames_dir is not None:
                cv2.imwrite(str(debug_frames_dir / f"val_{i:04d}.png"), dvis)

        if verbose:
            gt = ground_truth.get(str(i))
            gt_str = ""
            if gt and gt.get("visible"):
                gt_str = f" GT=({gt['x']},{gt['y']})"
            if sel is not None:
                print(f"  F{i:03d}: tracked ({sel.center[0]:4d},"
                      f"{sel.center[1]:4d}){gt_str}")
            elif gt and gt.get("visible"):
                print(f"  F{i:03d}: MISSED{gt_str}")

    if writer is not None:
        writer.release()
        if verbose:
            print(f"[INFO] Debug video saved: {debug_video}")

    # Compute metrics
    metrics = compute_metrics(predictions, ground_truth)

    return {
        "folder": str(folder),
        "annotations_file": str(ann_path),
        "model": model_path or "classical",
        "metrics": metrics,
        "predictions": {str(k): list(v) for k, v in predictions.items()},
    }


# CLI

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Validate ball tracking against ground-truth annotations")
    ap.add_argument("folder",
                    help="Path to pitch recording folder")
    ap.add_argument("-a", "--annotations", default=None,
                    help="Annotations JSON (default: <folder>/annotations.json)")
    ap.add_argument("-c", "--config", default=None,
                    help="Config file (TOML or JSON)")
    ap.add_argument("--model", "-m", default=None,
                    help="YOLO .pt or .onnx model for ML detection")
    ap.add_argument("--conf", type=float, default=0.25,
                    help="ML detector confidence threshold (default: 0.25)")
    ap.add_argument("--imgsz", type=int, default=960,
                    help="ML detector inference resolution (default: 960)")
    ap.add_argument("--debug-video", default=None,
                    help="Output debug video path (e.g. val_debug.mp4)")
    ap.add_argument("--debug-frames", default=None,
                    help="Output folder for per-frame debug images")
    ap.add_argument("-o", "--output", default=None,
                    help="Save results JSON to this path")
    ap.add_argument("-q", "--quiet", action="store_true",
                    help="Suppress per-frame output")
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
        cfg = Config.load(str(cfg_path)) if cfg_path.exists() else Config()

    results = run_validation(
        folder, cfg,
        model_path=args.model,
        conf=args.conf,
        imgsz=args.imgsz,
        annotations_path=Path(args.annotations) if args.annotations else None,
        debug_video=Path(args.debug_video) if args.debug_video else None,
        debug_frames_dir=Path(args.debug_frames) if args.debug_frames else None,
        verbose=not args.quiet,
    )

    if not results:
        sys.exit(1)

    print_metrics(results["metrics"])

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"[INFO] Results saved to {args.output}")


if __name__ == "__main__":
    main()
