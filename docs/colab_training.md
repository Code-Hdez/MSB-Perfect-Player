# Colab Training Workflow

Step-by-step guide for training the YOLOv8 baseball detector using Google Colab.

## Prerequisites

- Annotated pitch recordings in `pitches/*/annotations.json`
- Python environment with `opencv-python` installed locally

---

## Step 1 — Export YOLO Dataset (Local)

```bash
python tools/export_yolo.py --all --include-negatives --clean
```

This reads all `pitches/*/annotations.json` files and creates:
```
yolo_dataset/
  dataset.yaml
  split_manifest.json
  images/{train,val,test}/
  labels/{train,val,test}/
```

Options:
- `--bbox-size 24` — Adjust bounding box size (default: 20px)
- `--seed 42` — Deterministic split
- `--train-ratio 0.75 --val-ratio 0.20` — Adjust split ratios

## Step 2 — Zip Dataset (Local)

```bash
python tools/colab_package_and_pull.py --zip-dataset
```

Creates `yolo_dataset.zip` ready for upload.

## Step 3 — Upload to Google Drive

Copy `yolo_dataset.zip` to:
```
Google Drive → MyDrive/msb/yolo_dataset.zip
```

You can also use Colab's file upload as a fallback (see notebook).

## Step 4 — Open Notebook and Train

1. Open `notebooks/train_ball_detector_colab.ipynb` in Google Colab
2. Go to **Runtime → Change runtime type → GPU** (T4 or better)
3. **Run All** cells

The notebook will:
- Check GPU availability
- Install `ultralytics`
- Mount Drive and extract dataset
- Train YOLOv8 with optimized augmentation
- Save results back to Drive

## Step 5 — Retrieve Artifacts from Colab

After training completes, the notebook saves to Drive at:
```
MyDrive/msb/training_runs/<timestamp>/
MyDrive/msb/training_runs/ball_best.pt
```

Download these files to your local machine.

### Required artifacts to download:
| File | Required | Description |
|---|---|---|
| `best.pt` | **Yes** | Best model weights |
| `last.pt` | Recommended | Last epoch weights |
| `results.csv` | **Yes** | Training metrics per epoch |
| `args.yaml` | **Yes** | Training configuration |
| `results.png` | Recommended | Training curves plot |
| `confusion_matrix.png` | Recommended | Confusion matrix |
| `PR_curve.png` | Recommended | Precision-Recall curve |
| `best.onnx` | Optional | ONNX export for lighter inference |

## Step 6 — Pull Artifacts into Repo (Local)

```bash
python tools/colab_package_and_pull.py --pull-artifacts path/to/downloaded_run/
```

This copies:
- `best.pt` → `weights/ball_best.pt` (canonical location)
- All artifacts → `weights/runs/<timestamp>/`
- Updates `weights/README.md`

## Step 7 — Test Inference (Local)

```bash
# On recorded frames:
python track_folder.py pitches/20260301_030216 --model weights/ball_best.pt

# Live:
python track_live.py --model weights/ball_best.pt

# Validate against ground truth:
python validate_tracking.py pitches/20260301_030216 --model weights/ball_best.pt
```

## Local Training (Alternative)

If you have a local GPU (e.g., RTX 4050):

```bash
pip install -r requirements-ml.txt
python tools/train_detector.py --data yolo_dataset/dataset.yaml --epochs 150 --imgsz 960
```

Colab is preferred for longer training since it provides consistent GPU access.
