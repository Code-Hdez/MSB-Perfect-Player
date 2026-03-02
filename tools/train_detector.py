"""
train_detector.py — Train a YOLOv8 single-class baseball detector locally.

Wraps Ultralytics training with reproducibility metadata.

Usage examples
--------------
  python tools/train_detector.py --data yolo_dataset/dataset.yaml
  python tools/train_detector.py --data yolo_dataset/dataset.yaml --model yolov8s.pt --imgsz 960 --epochs 150
  python tools/train_detector.py --data yolo_dataset/dataset.yaml --device cpu

Notes
-----
  - For GPU training, install: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
  - Requires: pip install ultralytics (see requirements-ml.txt)
  - Outputs go to runs/<project>/<name>/
  - A repro.json is written alongside results for reproducibility
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def get_git_commit() -> Optional[str]:
    """Return current git commit hash, or None."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def hash_file(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def write_repro_json(
    run_dir: Path,
    args: argparse.Namespace,
    dataset_yaml_path: Path,
    duration_sec: float,
) -> None:
    """Write reproducibility metadata alongside training outputs."""
    repro: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "git_commit": get_git_commit(),
        "duration_sec": round(duration_sec, 1),
        "args": vars(args),
        "dataset_yaml_hash": hash_file(dataset_yaml_path),
    }

    # Add split manifest hash if present
    manifest_path = dataset_yaml_path.parent / "split_manifest.json"
    if manifest_path.exists():
        repro["split_manifest_hash"] = hash_file(manifest_path)

    # Ultralytics version
    try:
        import ultralytics
        repro["ultralytics_version"] = ultralytics.__version__
    except ImportError:
        repro["ultralytics_version"] = "not installed"

    # Torch version
    try:
        import torch
        repro["torch_version"] = torch.__version__
        repro["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            repro["cuda_device"] = torch.cuda.get_device_name(0)
    except ImportError:
        repro["torch_version"] = "not installed"

    out_path = run_dir / "repro.json"
    with open(out_path, "w") as f:
        json.dump(repro, f, indent=2)
    print(f"[INFO] Reproducibility metadata: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Train YOLOv8 baseball detector (single-class)")
    ap.add_argument("--data", required=True,
                    help="Path to dataset.yaml")
    ap.add_argument("--model", default="yolov8n.pt",
                    help="Base model (default: yolov8n.pt)")
    ap.add_argument("--imgsz", type=int, default=960,
                    help="Training image size (default: 960 for small objects)")
    ap.add_argument("--epochs", type=int, default=100,
                    help="Training epochs (default: 100)")
    ap.add_argument("--batch", type=int, default=-1,
                    help="Batch size (-1=auto, default: -1)")
    ap.add_argument("--device", default="0",
                    help="Device: '0' for GPU, 'cpu' for CPU (default: '0')")
    ap.add_argument("--project", default="runs",
                    help="Output project directory (default: runs)")
    ap.add_argument("--name", default="ball_detect",
                    help="Run name (default: ball_detect)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed (default: 42)")
    ap.add_argument("--workers", type=int, default=4,
                    help="Data loader workers (default: 4)")
    ap.add_argument("--patience", type=int, default=30,
                    help="Early stopping patience (default: 30)")
    ap.add_argument("--resume", action="store_true",
                    help="Resume from last checkpoint")
    args = ap.parse_args()

    dataset_yaml = Path(args.data)
    if not dataset_yaml.exists():
        print(f"[ERROR] Dataset config not found: {dataset_yaml}")
        print("  Run: python tools/export_yolo.py --all --clean")
        sys.exit(1)

    # Validate ultralytics is installed
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics not installed.")
        print("  pip install -r requirements-ml.txt")
        sys.exit(1)

    # Print training configuration
    print("=" * 55)
    print("  MSB YOLO TRAINING")
    print("=" * 55)
    print(f"  Dataset:  {dataset_yaml}")
    print(f"  Model:    {args.model}")
    print(f"  ImgSz:    {args.imgsz}")
    print(f"  Epochs:   {args.epochs}")
    print(f"  Batch:    {args.batch}")
    print(f"  Device:   {args.device}")
    print(f"  Seed:     {args.seed}")
    print(f"  Output:   {args.project}/{args.name}")
    print("=" * 55)

    # Log the equivalent CLI command for reference
    cmd_str = (
        f"yolo detect train "
        f"data={dataset_yaml} model={args.model} "
        f"imgsz={args.imgsz} epochs={args.epochs} batch={args.batch} "
        f"device={args.device} project={args.project} name={args.name} "
        f"seed={args.seed} workers={args.workers} patience={args.patience}"
    )
    print(f"\n[CMD] {cmd_str}\n")

    # Resolve dataset.yaml so the relative path: inside it is correct.
    # If dataset.yaml contains "path: ." the working dir must be its
    # parent, OR we rewrite path to be absolute before passing to YOLO.
    import re
    yaml_text = dataset_yaml.read_text()
    if re.search(r"^path:\s*\.", yaml_text, re.MULTILINE):
        # Rewrite the relative path to an absolute one so Ultralytics
        # can find the images regardless of the current working dir.
        abs_path = dataset_yaml.resolve().parent.as_posix()
        yaml_text = re.sub(
            r"^path:.*$",
            f"path: {abs_path}",
            yaml_text,
            flags=re.MULTILINE,
        )
        dataset_yaml.write_text(yaml_text)
        print(f"[INFO] Resolved dataset.yaml path → {abs_path}")

    # Train
    model = YOLO(args.model)
    t0 = time.time()

    results = model.train(
        data=str(dataset_yaml.resolve()),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        seed=args.seed,
        workers=args.workers,
        patience=args.patience,
        resume=args.resume,
        exist_ok=True,
        # Augmentation tuned for small-object detection (10–15px ball)
        mosaic=0.5,     # Reduced — full mosaic shrinks ball to ~7px
        mixup=0.1,
        scale=0.5,
        translate=0.2,
        fliplr=0.0,     # No horizontal flip — ball trajectory direction matters
        flipud=0.0,
        hsv_h=0.015,
        hsv_s=0.3,
        hsv_v=0.3,
        close_mosaic=15, # Disable mosaic for last 15 epochs (convergence)
        verbose=True,
    )

    duration = time.time() - t0

    # Find the actual output directory
    run_dir = Path(args.project) / args.name
    if not run_dir.exists():
        # Ultralytics may append a number
        candidates = sorted(Path(args.project).glob(f"{args.name}*"))
        if candidates:
            run_dir = candidates[-1]

    # Write reproducibility metadata
    if run_dir.exists():
        write_repro_json(run_dir, args, dataset_yaml, duration)
        best_pt = run_dir / "weights" / "best.pt"
        if best_pt.exists():
            print(f"\n[DONE] Training complete in {duration:.0f}s")
            print(f"  Best weights: {best_pt}")
            print(f"\n  To use in tracking:")
            print(f"    python track_live.py --model {best_pt}")
            print(f"    python track_folder.py <folder> --model {best_pt}")
        else:
            print(f"\n[WARN] best.pt not found in {run_dir / 'weights'}")
    else:
        print(f"\n[WARN] Run directory not found: {run_dir}")

    print(f"\n[INFO] Full results at: {run_dir}")


if __name__ == "__main__":
    main()
