"""
export_yolo.py — Convert annotated pitch recordings to YOLO format dataset.

Reads all pitches/<id>/annotations.json files and generates a YOLO-format
dataset ready for training with Ultralytics YOLOv8.

Output structure
----------------
  yolo_dataset/
    dataset.yaml
    split_manifest.json
    images/
      train/  val/  test/
    labels/
      train/  val/  test/

Usage examples
--------------
  # Export all annotated pitches:
  python tools/export_yolo.py --all

  # Include negatives (frames where ball is not visible → empty labels):
  python tools/export_yolo.py --all --include-negatives

  # Custom bbox size and split:
  python tools/export_yolo.py --all --bbox-size 24 --train-ratio 0.8 --val-ratio 0.15

  # Clean rebuild:
  python tools/export_yolo.py --all --clean

  # Export specific pitch folders:
  python tools/export_yolo.py pitches/20260301_030216 pitches/20260302_140000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def find_annotated_pitches(pitches_dir: Path) -> List[Path]:
    """Find all pitch folders that have an annotations.json file."""
    if not pitches_dir.is_dir():
        return []
    results = []
    for d in sorted(pitches_dir.iterdir()):
        if d.is_dir() and (d / "annotations.json").exists():
            results.append(d)
    return results


def load_annotations(ann_path: Path) -> Dict[str, Any]:
    """Load and validate an annotations.json file."""
    with open(ann_path) as f:
        data = json.load(f)
    if "annotations" not in data:
        raise ValueError(f"Missing 'annotations' key in {ann_path}")
    return data


def compute_yolo_label(
    cx: int,
    cy: int,
    img_w: int,
    img_h: int,
    bbox_size: int = 20,
) -> str:
    """Create a YOLO label line for a ball detection.

    YOLO format: <class_id> <cx_norm> <cy_norm> <w_norm> <h_norm>
    Class 0 = baseball.

    Parameters
    ----------
    cx, cy : int
        Ball center in pixel coordinates.
    img_w, img_h : int
        Image dimensions.
    bbox_size : int
        Square bounding box side length in pixels.
    """
    half = bbox_size / 2.0
    # Clamp bbox to image boundaries
    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(img_w, cx + half)
    y2 = min(img_h, cy + half)

    bw = x2 - x1
    bh = y2 - y1
    bcx = (x1 + bw / 2) / img_w
    bcy = (y1 + bh / 2) / img_h
    bw_norm = bw / img_w
    bh_norm = bh / img_h

    return f"0 {bcx:.6f} {bcy:.6f} {bw_norm:.6f} {bh_norm:.6f}"


def export_dataset(
    pitch_folders: List[Path],
    output_dir: Path,
    bbox_size: int = 20,
    include_negatives: bool = False,
    train_ratio: float = 0.75,
    val_ratio: float = 0.20,
    seed: int = 42,
    clean: bool = False,
) -> Dict[str, Any]:
    """Export annotated pitches to YOLO dataset format.

    Splitting is done at the **pitch level** — all frames from a single
    pitch recording go to the same split.  This prevents data leakage
    (consecutive frames from the same pitch share background/context).

    Returns
    -------
    dict with export statistics and split manifest.
    """
    import cv2

    if clean and output_dir.exists():
        print(f"[CLEAN] Removing existing dataset: {output_dir}")
        shutil.rmtree(output_dir)

    # Create directory structure
    for split in ("train", "val", "test"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Collect samples grouped by pitch
    pitch_samples: Dict[str, List[Dict[str, Any]]] = {}
    stats = {
        "total_pitches": len(pitch_folders),
        "total_frames": 0,
        "visible_frames": 0,
        "not_visible_frames": 0,
        "skipped_frames": 0,
    }

    for pitch_dir in pitch_folders:
        ann_path = pitch_dir / "annotations.json"
        try:
            data = load_annotations(ann_path)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"[WARN] Skipping {pitch_dir}: {e}")
            continue

        annotations = data["annotations"]
        frame_files = data.get("frame_files", [])
        pitch_id = pitch_dir.name
        pitch_list: List[Dict[str, Any]] = []

        for frame_key, ann in annotations.items():
            frame_idx = int(frame_key)

            # Determine image path
            if frame_files and frame_idx < len(frame_files):
                img_name = frame_files[frame_idx]
            else:
                # Try common naming patterns
                candidates = [
                    f"frame_{frame_idx:06d}.png",
                    f"frame_{frame_idx:04d}.png",
                    f"{frame_idx:06d}.png",
                    f"{frame_idx:04d}.png",
                ]
                img_name = None
                for c in candidates:
                    if (pitch_dir / c).exists():
                        img_name = c
                        break
                # Also check frames/ subdirectory
                if img_name is None:
                    for c in candidates:
                        if (pitch_dir / "frames" / c).exists():
                            img_name = f"frames/{c}"
                            break

            if img_name is None:
                stats["skipped_frames"] += 1
                continue

            img_path = pitch_dir / img_name
            if not img_path.exists():
                stats["skipped_frames"] += 1
                continue

            visible = ann.get("visible", False)

            if visible:
                stats["visible_frames"] += 1
            else:
                stats["not_visible_frames"] += 1
                if not include_negatives:
                    continue

            pitch_list.append({
                "pitch_id": pitch_id,
                "frame_idx": frame_idx,
                "img_path": str(img_path),
                "visible": visible,
                "x": ann.get("x"),
                "y": ann.get("y"),
            })

        if pitch_list:
            pitch_samples[pitch_id] = pitch_list

    total_samples = sum(len(v) for v in pitch_samples.values())
    stats["total_frames"] = total_samples
    if total_samples == 0:
        print("[ERROR] No valid samples found.")
        return {"error": "no samples", **stats}

    print(f"[INFO] {total_samples} samples from "
          f"{len(pitch_samples)} pitches")
    print(f"  Visible: {stats['visible_frames']}, "
          f"Not visible: {stats['not_visible_frames']}, "
          f"Skipped: {stats['skipped_frames']}")

    # Pitch-level deterministic split
    # All frames from a single pitch go to the SAME split to prevent
    # data leakage (consecutive frames share background/context).
    random.seed(seed)
    pitch_ids = sorted(pitch_samples.keys())
    random.shuffle(pitch_ids)

    n_pitches = len(pitch_ids)
    n_train_p = max(1, round(n_pitches * train_ratio))
    n_val_p = max(1, round(n_pitches * val_ratio)) if n_pitches > 1 else 0
    # If only 1 pitch, put it all in train
    if n_pitches == 1:
        n_train_p, n_val_p = 1, 0
    elif n_pitches == 2:
        n_train_p, n_val_p = 1, 1
    # Remaining pitches go to test
    train_pitches = set(pitch_ids[:n_train_p])
    val_pitches = set(pitch_ids[n_train_p:n_train_p + n_val_p])
    test_pitches = set(pitch_ids[n_train_p + n_val_p:])

    print(f"[INFO] Pitch-level split (seed={seed}): "
          f"train={len(train_pitches)} pitches, "
          f"val={len(val_pitches)}, test={len(test_pitches)}")

    manifest: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
    pitch_split_map: Dict[str, str] = {}

    # Read one image to get baseline dimensions
    first_samples = next(iter(pitch_samples.values()))
    sample_img = cv2.imread(first_samples[0]["img_path"])
    if sample_img is None:
        print(f"[ERROR] Cannot read sample image: "
              f"{first_samples[0]['img_path']}")
        return {"error": "unreadable image", **stats}
    img_h, img_w = sample_img.shape[:2]
    print(f"[INFO] Image dimensions: {img_w}x{img_h}")

    label_warnings: List[str] = []

    # Export each sample
    for pid in pitch_ids:
        if pid in train_pitches:
            split = "train"
        elif pid in val_pitches:
            split = "val"
        else:
            split = "test"
        pitch_split_map[pid] = split

        for sample in pitch_samples[pid]:
            # Unique filename: pitch_id + zero-padded frame index
            unique_name = (f"{sample['pitch_id']}_"
                           f"{sample['frame_idx']:06d}")
            img_ext = Path(sample["img_path"]).suffix

            # Copy image
            dst_img = (output_dir / "images" / split
                       / f"{unique_name}{img_ext}")
            shutil.copy2(sample["img_path"], dst_img)

            # Create label
            dst_lbl = (output_dir / "labels" / split
                       / f"{unique_name}.txt")
            if sample["visible"] and sample["x"] is not None:
                # Read actual image dimensions (avoids assumption
                # that all frames are the same size)
                img = cv2.imread(sample["img_path"])
                if img is not None:
                    ih, iw = img.shape[:2]
                else:
                    ih, iw = img_h, img_w

                # Validate coordinates are within image bounds
                sx = max(0, min(sample["x"], iw - 1))
                sy = max(0, min(sample["y"], ih - 1))
                if sx != sample["x"] or sy != sample["y"]:
                    label_warnings.append(
                        f"  {unique_name}: clamped "
                        f"({sample['x']},{sample['y']}) → ({sx},{sy})"
                    )

                label_line = compute_yolo_label(sx, sy, iw, ih, bbox_size)

                # Validate normalised values are in [0,1]
                parts = label_line.split()
                for v in parts[1:]:
                    fv = float(v)
                    if fv < 0.0 or fv > 1.0:
                        label_warnings.append(
                            f"  {unique_name}: normalised value "
                            f"out of range: {label_line}"
                        )
                        break

                dst_lbl.write_text(label_line + "\n")
            else:
                # Empty label = negative sample (no objects)
                dst_lbl.write_text("")

            manifest[split].append(unique_name)

    if label_warnings:
        print(f"[WARN] {len(label_warnings)} label warnings:")
        for w in label_warnings[:10]:
            print(w)
        if len(label_warnings) > 10:
            print(f"  ... and {len(label_warnings) - 10} more")

    # Write dataset.yaml (relative path for portability)
    yaml_content = f"""# MSB Baseball Detection Dataset
# Auto-generated by tools/export_yolo.py
# Seed: {seed}, bbox_size: {bbox_size}px
# Split: pitch-level (no leakage between train/val/test)

path: .
train: images/train
val: images/val
test: images/test

nc: 1
names:
  0: baseball
"""
    (output_dir / "dataset.yaml").write_text(yaml_content)

    # Write split manifest
    manifest_data = {
        "seed": seed,
        "bbox_size": bbox_size,
        "include_negatives": include_negatives,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "split_strategy": "pitch-level",
        "image_dimensions": {"width": img_w, "height": img_h},
        "pitch_split_map": pitch_split_map,
        "split_counts": {
            "train": len(manifest["train"]),
            "val": len(manifest["val"]),
            "test": len(manifest["test"]),
        },
        "splits": manifest,
    }
    with open(output_dir / "split_manifest.json", "w") as f:
        json.dump(manifest_data, f, indent=2)

    # Summary
    print(f"\n[EXPORT COMPLETE]")
    print(f"  Output:  {output_dir}")
    print(f"  Train:   {len(manifest['train'])} samples "
          f"({len(train_pitches)} pitches)")
    print(f"  Val:     {len(manifest['val'])} samples "
          f"({len(val_pitches)} pitches)")
    print(f"  Test:    {len(manifest['test'])} samples "
          f"({len(test_pitches)} pitches)")
    print(f"  Config:  {output_dir / 'dataset.yaml'}")

    return {
        "output_dir": str(output_dir),
        "stats": stats,
        "split_counts": manifest_data["split_counts"],
        "dataset_yaml": str(output_dir / "dataset.yaml"),
    }


def integrity_check(output_dir: Path) -> bool:
    """Verify dataset integrity: matching images/labels and valid content."""
    ok = True
    for split in ("train", "val", "test"):
        img_dir = output_dir / "images" / split
        lbl_dir = output_dir / "labels" / split

        if not img_dir.exists() or not lbl_dir.exists():
            print(f"[CHECK FAIL] Missing directory for split '{split}'")
            ok = False
            continue

        img_stems = {f.stem for f in img_dir.iterdir() if f.is_file()}
        lbl_stems = {f.stem for f in lbl_dir.iterdir() if f.is_file()}

        missing_labels = img_stems - lbl_stems
        orphan_labels = lbl_stems - img_stems

        if missing_labels:
            print(f"[CHECK WARN] {split}: {len(missing_labels)} images "
                  f"without labels")
        if orphan_labels:
            print(f"[CHECK WARN] {split}: {len(orphan_labels)} labels "
                  f"without images")
            ok = False

        # Validate label file content
        bad_labels = 0
        for lbl_path in lbl_dir.iterdir():
            if not lbl_path.is_file() or lbl_path.suffix != ".txt":
                continue
            text = lbl_path.read_text().strip()
            if not text:
                continue  # empty = negative, valid
            for line_no, line in enumerate(text.splitlines(), 1):
                parts = line.split()
                if len(parts) != 5:
                    print(f"[CHECK FAIL] {lbl_path.name}:{line_no}: "
                          f"expected 5 fields, got {len(parts)}")
                    bad_labels += 1
                    ok = False
                    continue
                try:
                    cls_id = int(parts[0])
                    vals = [float(p) for p in parts[1:]]
                except ValueError:
                    print(f"[CHECK FAIL] {lbl_path.name}:{line_no}: "
                          f"non-numeric values")
                    bad_labels += 1
                    ok = False
                    continue
                if cls_id != 0:
                    print(f"[CHECK WARN] {lbl_path.name}:{line_no}: "
                          f"class_id={cls_id}, expected 0")
                for v in vals:
                    if v < 0.0 or v > 1.0:
                        print(f"[CHECK FAIL] {lbl_path.name}:{line_no}: "
                              f"value {v} outside [0,1]")
                        bad_labels += 1
                        ok = False
                        break
        if bad_labels:
            print(f"[CHECK] {split}: {bad_labels} invalid label(s)")

    if ok:
        print("[CHECK] Dataset integrity OK")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Export annotated pitch recordings to YOLO dataset format")
    ap.add_argument("folders", nargs="*", default=[],
                    help="Specific pitch folders to export")
    ap.add_argument("--all", action="store_true",
                    help="Export all annotated pitches from pitches/ directory")
    ap.add_argument("--pitches-dir", default="pitches",
                    help="Root pitches directory (default: pitches/)")
    ap.add_argument("-o", "--output", default="yolo_dataset",
                    help="Output dataset directory (default: yolo_dataset/)")
    ap.add_argument("--bbox-size", type=int, default=24,
                    help="Bounding box side length in pixels (default: 24). "
                         "Ball is ~10-15px; 24px captures ball + motion blur.")
    ap.add_argument("--include-negatives", action="store_true",
                    help="Include frames where ball is not visible as "
                         "negative samples")
    ap.add_argument("--train-ratio", type=float, default=0.75,
                    help="Training set ratio (default: 0.75)")
    ap.add_argument("--val-ratio", type=float, default=0.20,
                    help="Validation set ratio (default: 0.20)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for reproducible splits (default: 42)")
    ap.add_argument("--clean", action="store_true",
                    help="Remove existing dataset before export")
    ap.add_argument("--check", action="store_true",
                    help="Only run integrity check on existing dataset")
    args = ap.parse_args()

    output_dir = Path(args.output)

    if args.check:
        integrity_check(output_dir)
        return

    # Collect pitch folders
    pitch_folders: List[Path] = []
    if args.all:
        pitches_dir = Path(args.pitches_dir)
        pitch_folders = find_annotated_pitches(pitches_dir)
        if not pitch_folders:
            print(f"[ERROR] No annotated pitches found in {pitches_dir}/")
            print("  Run frame_annotator.py on pitch recordings first.")
            sys.exit(1)
    elif args.folders:
        for f in args.folders:
            p = Path(f)
            if p.is_dir() and (p / "annotations.json").exists():
                pitch_folders.append(p)
            else:
                print(f"[WARN] Skipping {f}: not a directory or no "
                      f"annotations.json")
    else:
        print("[ERROR] Specify --all or provide pitch folder paths.")
        ap.print_help()
        sys.exit(1)

    print(f"[INFO] Exporting {len(pitch_folders)} pitch folder(s)")

    result = export_dataset(
        pitch_folders,
        output_dir,
        bbox_size=args.bbox_size,
        include_negatives=args.include_negatives,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        clean=args.clean,
    )

    if "error" not in result:
        integrity_check(output_dir)


if __name__ == "__main__":
    main()
