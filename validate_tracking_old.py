"""
Validate Tracking — Compare tracker output against manual annotations.

Loads manual annotations (from frame_annotator.py) and runs the detection
+ tracking pipeline on the same recorded frames.  Computes per-frame
pixel error and reports summary statistics.

Optionally renders a debug video or image sequence showing:
  - All candidates (yellow/grey dots)
  - Chosen track point (green circle)
  - Annotation ground truth (red crosshair)
  - Error line connecting them
  - Track state label
  - Suppression zones

Usage
-----
  python validate_tracking.py pitches/20260227_205241
  python validate_tracking.py pitches/20260227_205241 -a annotations.json
  python validate_tracking.py pitches/20260227_205241 --debug-video debug.mp4
  python validate_tracking.py pitches/20260227_205241 --debug-frames debug_out/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from pitch_analyzer import (
    BallDetector, BallTracker, TrajectoryPredictor, TrajectoryCorridor,
    CORRIDOR_DEFAULT, PITCHER_ZONE, TrackState,
    COL_GREEN, COL_RED, COL_YELLOW, COL_CYAN, COL_WHITE,
    COL_ORANGE, COL_MAGENTA, COL_BLACK, FONT, put_text,
)


def load_annotations(path: Path) -> Dict[str, Dict[str, Any]]:
    """Load annotations JSON and return the annotations dict."""
    with open(path) as f:
        data = json.load(f)
    return data.get("annotations", {})


def load_frames(folder: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    files = sorted(f for f in folder.iterdir() if f.suffix.lower() in exts)
    return files


def run_validation(folder: Path,
                   annotations: Dict[str, Dict[str, Any]],
                   corridor_path: Optional[Path] = None,
                   debug_video: Optional[Path] = None,
                   debug_frames_dir: Optional[Path] = None,
                   ) -> Dict[str, Any]:
    """Run the full tracker on recorded frames, compare with annotations.

    Returns a dict of metrics.
    """
    files = load_frames(folder)
    if not files:
        print(f"[ERROR] No frames in {folder}")
        return {}

    n_frames = len(files)
    print(f"[INFO] {n_frames} frames, {len(annotations)} annotations")

    # Set up pipeline
    detector = BallDetector()
    tracker = BallTracker()
    corridor = TrajectoryCorridor()
    if corridor_path and corridor_path.exists():
        corridor.load(corridor_path)
    elif CORRIDOR_DEFAULT:
        corridor.set_rect(*CORRIDOR_DEFAULT)

    # Video writer
    writer: Optional[cv2.VideoWriter] = None
    if debug_frames_dir:
        debug_frames_dir.mkdir(parents=True, exist_ok=True)

    # Per-frame results
    errors: List[Optional[float]] = []
    tracker_detected: List[bool] = []
    ann_visible: List[bool] = []
    false_starts: int = 0
    track_swaps: int = 0
    dropouts: int = 0

    prev_track_state = TrackState.IDLE
    was_tracking = False

    for i, fpath in enumerate(files):
        frame = cv2.imread(str(fpath))
        if frame is None:
            errors.append(None)
            tracker_detected.append(False)
            ann_visible.append(False)
            continue

        key = str(i)
        ann = annotations.get(key)
        gt_visible = ann is not None and ann.get("visible", False)
        gt_pos: Optional[Tuple[int, int]] = None
        if gt_visible:
            gt_pos = (ann["x"], ann["y"])

        ann_visible.append(gt_visible)

        # Run pipeline
        track_active = (tracker.track is not None
                        and tracker.track.active
                        and tracker.track.confirmed)
        best = detector.detect(frame, corridor=corridor,
                               track_active=track_active)
        track = tracker.update(detector.candidates, best)

        sel = tracker.selected
        tracker_pos: Optional[Tuple[int, int]] = None
        if sel is not None:
            tracker_pos = sel.center
            tracker_detected.append(True)
        else:
            tracker_detected.append(False)

        # Compute error
        if gt_visible and tracker_pos is not None:
            err = np.hypot(tracker_pos[0] - gt_pos[0],
                           tracker_pos[1] - gt_pos[1])
            errors.append(float(err))
        elif gt_visible and tracker_pos is None:
            errors.append(None)  # missed detection
        else:
            errors.append(None)

        # Track lifecycle events
        cur_state = tracker.state
        if (prev_track_state == TrackState.IDLE
                and cur_state == TrackState.TENTATIVE):
            # New track started
            if not gt_visible:
                false_starts += 1  # started when ball not visible
        if (prev_track_state in (TrackState.CONFIRMED, TrackState.TENTATIVE)
                and cur_state == TrackState.LOST):
            if gt_visible:
                dropouts += 1
        prev_track_state = cur_state

        # Debug visualisation
        if debug_video is not None or debug_frames_dir is not None:
            vis = frame.copy()
            h, w = vis.shape[:2]

            # Pitcher zone
            pz = PITCHER_ZONE
            cv2.rectangle(vis, (pz[0], pz[1]), (pz[2], pz[3]),
                          (128, 128, 0), 1)

            # Corridor
            corridor.draw(vis, COL_CYAN, 1)

            # Suppression zones
            for sx, sy, sr in detector.suppressor.suppression_zones:
                cv2.circle(vis, (sx, sy), sr, (0, 0, 128), 1)

            # All candidates
            for c in detector.candidates:
                col = COL_YELLOW if c.in_motion_mask else (100, 100, 100)
                cv2.circle(vis, c.center, 3, col, -1)

            # Tracker pick
            if tracker_pos is not None:
                cv2.circle(vis, tracker_pos, 10, COL_GREEN, 2)
                cv2.circle(vis, tracker_pos, 2, COL_RED, -1)

            # Ground truth
            if gt_pos is not None:
                cv2.drawMarker(vis, gt_pos, COL_RED,
                               cv2.MARKER_CROSS, 18, 2)
                cv2.circle(vis, gt_pos, 6, COL_RED, 1)

            # Error line
            if gt_pos is not None and tracker_pos is not None:
                cv2.line(vis, tracker_pos, gt_pos, COL_MAGENTA, 1)
                err_val = errors[-1]
                if err_val is not None:
                    mid = ((tracker_pos[0] + gt_pos[0]) // 2,
                           (tracker_pos[1] + gt_pos[1]) // 2)
                    put_text(vis, f"{err_val:.1f}px",
                             (mid[0] + 5, mid[1] - 5), 0.35,
                             COL_MAGENTA, 1)

            # State / frame info
            state_label = tracker.state.name
            put_text(vis, f"Frame {i}  [{state_label}]",
                     (10, 22), 0.45, COL_WHITE, 1)

            if gt_visible and tracker_pos is None:
                put_text(vis, "MISSED", (10, 44), 0.50, COL_RED, 2)

            # Write
            if debug_video is not None:
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(
                        str(debug_video), fourcc, 30, (w, h))
                writer.write(vis)

            if debug_frames_dir is not None:
                out_path = debug_frames_dir / f"debug_{i:04d}.png"
                cv2.imwrite(str(out_path), vis)

    if writer is not None:
        writer.release()
        print(f"[INFO] Debug video saved: {debug_video}")
    if debug_frames_dir is not None:
        print(f"[INFO] Debug frames saved: {debug_frames_dir}")

    # Compute metrics
    # Only over frames where annotation says ball is visible
    gt_visible_indices = [i for i in range(n_frames) if ann_visible[i]]
    n_visible = len(gt_visible_indices)

    valid_errors = [errors[i] for i in gt_visible_indices
                    if errors[i] is not None]
    missed_count = sum(1 for i in gt_visible_indices
                       if errors[i] is None)
    detected_count = len(valid_errors)

    metrics: Dict[str, Any] = {
        "total_frames": n_frames,
        "annotated_visible": n_visible,
        "detected_when_visible": detected_count,
        "missed_when_visible": missed_count,
        "detection_rate": (detected_count / n_visible * 100
                           if n_visible > 0 else 0.0),
        "false_starts": false_starts,
        "dropouts": dropouts,
    }

    if valid_errors:
        arr = np.array(valid_errors)
        metrics["mean_error_px"] = float(np.mean(arr))
        metrics["median_error_px"] = float(np.median(arr))
        metrics["max_error_px"] = float(np.max(arr))
        metrics["std_error_px"] = float(np.std(arr))
        metrics["p90_error_px"] = float(np.percentile(arr, 90))
        metrics["within_5px"] = float(np.mean(arr <= 5) * 100)
        metrics["within_10px"] = float(np.mean(arr <= 10) * 100)
        metrics["within_20px"] = float(np.mean(arr <= 20) * 100)
    else:
        metrics["mean_error_px"] = None
        metrics["median_error_px"] = None
        metrics["max_error_px"] = None

    # Per-frame detail
    metrics["per_frame"] = []
    for i in range(n_frames):
        entry: Dict[str, Any] = {"frame": i}
        if ann_visible[i]:
            entry["gt_visible"] = True
            entry["tracker_detected"] = tracker_detected[i]
            entry["error_px"] = errors[i]
        else:
            entry["gt_visible"] = False
        metrics["per_frame"].append(entry)

    return metrics


def print_report(metrics: Dict[str, Any]) -> None:
    """Print a human-readable summary of validation results."""
    print("\n" + "=" * 60)
    print("  TRACKING VALIDATION REPORT")
    print("=" * 60)

    print(f"\n  Total frames:           {metrics['total_frames']}")
    print(f"  Annotated (visible):    {metrics['annotated_visible']}")
    print(f"  Detected when visible:  {metrics['detected_when_visible']}")
    print(f"  Missed when visible:    {metrics['missed_when_visible']}")
    print(f"  Detection rate:         {metrics['detection_rate']:.1f}%")
    print(f"  False starts:           {metrics['false_starts']}")
    print(f"  Track dropouts:         {metrics['dropouts']}")

    if metrics.get("mean_error_px") is not None:
        print(f"\n  Mean error:             {metrics['mean_error_px']:.1f} px")
        print(f"  Median error:           {metrics['median_error_px']:.1f} px")
        print(f"  Max error:              {metrics['max_error_px']:.1f} px")
        print(f"  Std deviation:          {metrics['std_error_px']:.1f} px")
        print(f"  90th percentile:        {metrics['p90_error_px']:.1f} px")
        print(f"  Within  5 px:           {metrics['within_5px']:.1f}%")
        print(f"  Within 10 px:           {metrics['within_10px']:.1f}%")
        print(f"  Within 20 px:           {metrics['within_20px']:.1f}%")
    else:
        print("\n  [No valid error measurements — "
              "no frames with both annotation and detection]")

    print("\n" + "=" * 60)

    # Quality assessment
    rate = metrics["detection_rate"]
    mean_err = metrics.get("mean_error_px")
    fs = metrics["false_starts"]

    print("\n  ASSESSMENT:")
    issues = []
    if rate < 80:
        issues.append(f"  - Detection rate ({rate:.1f}%) is below 80% target")
    if mean_err is not None and mean_err > 15:
        issues.append(f"  - Mean error ({mean_err:.1f}px) exceeds 15px target")
    if fs > 0:
        issues.append(f"  - {fs} false start(s) detected (should be 0)")
    if metrics["dropouts"] > 1:
        issues.append(f"  - {metrics['dropouts']} track dropout(s)")

    if not issues:
        print("  PASS — Tracking meets quality targets")
    else:
        print("  NEEDS IMPROVEMENT:")
        for issue in issues:
            print(issue)

    print()


def main():
    ap = argparse.ArgumentParser(
        description="Validate tracking pipeline against manual annotations")
    ap.add_argument("folder",
                    help="Path to pitch recording folder")
    ap.add_argument("-a", "--annotations", default=None,
                    help="Path to annotations JSON "
                    "(default: <folder>/annotations.json)")
    ap.add_argument("--corridor", default=None,
                    help="Path to corridor JSON")
    ap.add_argument("--debug-video", default=None,
                    help="Output path for debug video (e.g. debug.mp4)")
    ap.add_argument("--debug-frames", default=None,
                    help="Output folder for per-frame debug images")
    ap.add_argument("--save-metrics", default=None,
                    help="Save metrics JSON to this path")
    args = ap.parse_args()

    folder = Path(args.folder)
    if not folder.is_dir():
        print(f"[ERROR] Not a directory: {folder}")
        sys.exit(1)

    ann_path = Path(args.annotations) if args.annotations else folder / "annotations.json"
    if not ann_path.exists():
        print(f"[ERROR] Annotations not found: {ann_path}")
        print("  Run frame_annotator.py first to create annotations.")
        sys.exit(1)

    annotations = load_annotations(ann_path)
    if not annotations:
        print("[ERROR] No annotations in file")
        sys.exit(1)

    corridor_path = Path(args.corridor) if args.corridor else None
    debug_video = Path(args.debug_video) if args.debug_video else None
    debug_frames_dir = Path(args.debug_frames) if args.debug_frames else None

    metrics = run_validation(
        folder, annotations,
        corridor_path=corridor_path,
        debug_video=debug_video,
        debug_frames_dir=debug_frames_dir,
    )

    print_report(metrics)

    if args.save_metrics:
        # Remove per_frame detail for the saved version (too large)
        save_data = {k: v for k, v in metrics.items() if k != "per_frame"}
        with open(args.save_metrics, "w") as f:
            json.dump(save_data, f, indent=2)
        print(f"[INFO] Metrics saved to {args.save_metrics}")


if __name__ == "__main__":
    main()
