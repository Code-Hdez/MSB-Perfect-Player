"""
colab_package_and_pull.py — Dataset packaging for Colab and artifact retrieval.

Handles the local side of the Colab training workflow:
  1. Zip the YOLO dataset for upload to Google Drive / Colab
  2. After training, copy artifacts back into the repo

Usage examples
--------------
  # Step 1 — Zip dataset for Colab:
  python tools/colab_package_and_pull.py --zip-dataset

  # Step 2 — After Colab training, pull artifacts:
  python tools/colab_package_and_pull.py --pull-artifacts path/to/colab_results/

  # Combine: specify custom paths:
  python tools/colab_package_and_pull.py --zip-dataset --dataset-dir yolo_dataset --zip-output yolo_dataset.zip
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def zip_dataset(
    dataset_dir: Path,
    output_zip: Path,
) -> None:
    """Zip the YOLO dataset directory for upload to Google Drive / Colab."""
    if not dataset_dir.is_dir():
        print(f"[ERROR] Dataset directory not found: {dataset_dir}")
        print("  Run: python tools/export_yolo.py --all --clean")
        sys.exit(1)

    # Verify dataset.yaml exists
    if not (dataset_dir / "dataset.yaml").exists():
        print(f"[ERROR] No dataset.yaml in {dataset_dir}")
        sys.exit(1)

    print(f"[INFO] Zipping {dataset_dir} → {output_zip}")

    # Count contents
    n_images = sum(
        1 for f in dataset_dir.rglob("*")
        if f.is_file() and f.suffix in {".png", ".jpg", ".jpeg", ".bmp"}
    )
    n_labels = sum(
        1 for f in dataset_dir.rglob("*.txt")
    )
    print(f"  Images: {n_images}, Labels: {n_labels}")

    # Create zip (without the parent directory in paths)
    archive_base = str(output_zip).replace(".zip", "")
    shutil.make_archive(archive_base, "zip", dataset_dir.parent,
                        dataset_dir.name)

    final_zip = Path(archive_base + ".zip")
    if final_zip != output_zip:
        shutil.move(str(final_zip), str(output_zip))

    size_mb = output_zip.stat().st_size / (1024 * 1024)
    print(f"[DONE] Created {output_zip} ({size_mb:.1f} MB)")
    print()
    print("  NEXT STEPS:")
    print("  1. Copy to Google Drive:  MyDrive/msb/yolo_dataset.zip")
    print("  2. Open notebooks/train_ball_detector_colab.ipynb in Colab")
    print("  3. Set runtime to GPU → Run All")


def pull_artifacts(
    source_dir: Path,
    weights_dir: Path,
) -> None:
    """Copy Colab training artifacts into the repo weights/ directory.

    Expects source_dir to contain at least best.pt.
    Optionally: last.pt, results.csv, args.yaml, confusion_matrix.png, etc.
    """
    if not source_dir.is_dir():
        print(f"[ERROR] Source directory not found: {source_dir}")
        sys.exit(1)

    # Find best.pt
    best_pt = None
    for pattern in ["best.pt", "weights/best.pt", "**/best.pt"]:
        matches = list(source_dir.glob(pattern))
        if matches:
            best_pt = matches[0]
            break

    if best_pt is None:
        print(f"[ERROR] best.pt not found in {source_dir}")
        sys.exit(1)

    # Create run directory with timestamp
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = weights_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Copy best.pt to canonical location
    dst_best = weights_dir / "ball_best.pt"
    shutil.copy2(str(best_pt), str(dst_best))
    print(f"[COPY] {best_pt} → {dst_best}")

    # Also copy to run directory
    shutil.copy2(str(best_pt), str(run_dir / "best.pt"))

    # Copy optional artifacts
    optional_files = [
        "last.pt", "weights/last.pt",
        "results.csv",
        "args.yaml",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "P_curve.png", "R_curve.png", "PR_curve.png", "F1_curve.png",
        "results.png",
    ]

    copied = []
    for pattern in optional_files:
        matches = list(source_dir.glob(pattern))
        if not matches:
            matches = list(source_dir.glob(f"**/{pattern}"))
        for m in matches:
            dst = run_dir / m.name
            if not dst.exists():
                shutil.copy2(str(m), str(dst))
                copied.append(m.name)

    # Copy ONNX if present
    for onnx in source_dir.glob("**/*.onnx"):
        dst = weights_dir / "ball_best.onnx"
        shutil.copy2(str(onnx), str(dst))
        shutil.copy2(str(onnx), str(run_dir / onnx.name))
        copied.append(onnx.name)

    print(f"[COPY] Additional artifacts: {', '.join(copied) if copied else 'none'}")
    print(f"[INFO] Run directory: {run_dir}")

    # Update weights/README.md
    readme_path = weights_dir / "README.md"
    readme_content = f"""# Model Weights

## Current best model
- **File**: `ball_best.pt`
- **Run**: `{run_id}`
- **Created**: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Usage
```bash
python track_live.py --model weights/ball_best.pt
python track_folder.py <pitch_folder> --model weights/ball_best.pt
python validate_tracking.py <pitch_folder> --model weights/ball_best.pt
```

## Run history
Check `runs/` for per-run artifacts (metrics, configs, plots).

## IMPORTANT
- Do NOT commit .pt / .onnx files to git (they are gitignored)
- Keep them in Google Drive or local storage
- Only metrics/logs in runs/ may be committed selectively
"""
    readme_path.write_text(readme_content)
    print(f"[INFO] Updated {readme_path}")
    print()
    print(f"[DONE] Artifacts pulled successfully.")
    print(f"  Use model: python track_live.py --model weights/ball_best.pt")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Package dataset for Colab training and pull artifacts back")
    ap.add_argument("--zip-dataset", action="store_true",
                    help="Zip YOLO dataset for upload to Google Drive / Colab")
    ap.add_argument("--dataset-dir", default="yolo_dataset",
                    help="YOLO dataset directory (default: yolo_dataset)")
    ap.add_argument("--zip-output", default="yolo_dataset.zip",
                    help="Output zip path (default: yolo_dataset.zip)")
    ap.add_argument("--pull-artifacts", default=None, metavar="DIR",
                    help="Pull training artifacts from Colab output directory")
    ap.add_argument("--weights-dir", default="weights",
                    help="Weights directory (default: weights)")
    args = ap.parse_args()

    if not args.zip_dataset and not args.pull_artifacts:
        print("[ERROR] Specify --zip-dataset and/or --pull-artifacts <dir>")
        ap.print_help()
        sys.exit(1)

    if args.zip_dataset:
        zip_dataset(
            Path(args.dataset_dir),
            Path(args.zip_output),
        )

    if args.pull_artifacts:
        pull_artifacts(
            Path(args.pull_artifacts),
            Path(args.weights_dir),
        )


if __name__ == "__main__":
    main()
