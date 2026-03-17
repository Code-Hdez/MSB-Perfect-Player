# MSB-Perfect-Player

Real-time baseball tracking system for *Mario Superstar Baseball* (GameCube / Dolphin emulator).  
Uses computer vision (OpenCV) and YOLOv8 to detect and track pitched baseballs.

## Project Structure

```
MSB-Perfect-Player/
├── msb/                        # Core runtime library
│   ├── config.py               #   Configuration management (TOML/JSON)
│   ├── detector.py             #   Classical ball detector (HSV + BG subtraction)
│   ├── detector_ml.py          #   YOLO-based ball detector (drop-in replacement)
│   ├── tracker.py              #   Kalman-filtered ball tracker
│   ├── tracker_ml.py           #   Simplified ML tracker
│   ├── corridor.py             #   Trajectory corridor constraints
│   ├── predictor.py            #   Polynomial trajectory extrapolation
│   ├── recorder.py             #   Pitch frame recorder
│   ├── sources.py              #   Frame sources (live capture / folder)
│   ├── visualiser.py           #   Overlay rendering and ROI selectors
│   └── utils.py                #   Shared utilities and colours
│
├── track_live.py               # Live tracking from screen capture
├── track_folder.py             # Offline tracking on recorded frames
├── validate_tracking.py        # Compare tracker vs ground-truth annotations
├── frame_annotator.py          # Interactive ball position annotator
├── find_roi.py                 # Mouse-click ROI coordinate finder
│
├── tools/                      # Offline pipeline tools
│   ├── export_yolo.py          #   Convert annotations → YOLO dataset
│   ├── train_detector.py       #   Local YOLO training wrapper
│   └── colab_package_and_pull.py  # Dataset zip + artifact retrieval
│
├── notebooks/                  # Colab notebooks
│   └── train_ball_detector_colab.ipynb
│
├── docs/                       # Documentation
│   ├── annotations_schema.md   #   Annotations JSON schema
│   └── colab_training.md       #   Step-by-step Colab training guide
│
├── features/                   # Feature subsystems
│   └── batter_hitbox/          #   Batter identification + hitbox detection
│       └── msb_hitbox_detector.py
│
├── weights/                    # Trained model weights (gitignored)
│   ├── README.md
│   └── runs/                   #   Per-run metrics and configs
│
├── archive/                    # Historical notes and chat logs
│   └── notes/
│
├── config.toml                 # Main pipeline configuration
├── requirements.txt            # Core dependencies
├── requirements-ml.txt         # ML/training dependencies
└── .gitignore
```

## Quick Start

### 1. Install dependencies

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. Configure screen capture

Edit `config.toml` — set `screen_roi` to your Dolphin window coordinates:
```bash
python find_roi.py            # Click top-left then bottom-right of game area
```

### 3. Live tracking (classical detector)

```bash
python track_live.py
```

### 4. Live tracking (ML detector)

```bash
python track_live.py --model weights/ball_best.pt
```

## Training Pipeline

Full end-to-end workflow for training the YOLO baseball detector:

### Step 1 — Record pitch frames
```bash
python track_live.py          # Press SPACE to record, S to save
```

### Step 2 — Annotate ball positions
```bash
python frame_annotator.py pitches/<recording_id>
```

### Step 3 — Export YOLO dataset
```bash
python tools/export_yolo.py --all --include-negatives --clean
```

### Step 4 — Train (Colab — recommended)
```bash
python tools/colab_package_and_pull.py --zip-dataset
# Upload yolo_dataset.zip to Google Drive → MyDrive/msb/
# Open notebooks/train_ball_detector_colab.ipynb in Colab
# Set runtime to GPU → Run All
```

### Step 4 (alt) — Train locally
```bash
pip install -r requirements-ml.txt
python tools/train_detector.py --data yolo_dataset/dataset.yaml
```

### Step 5 — Pull trained weights
```bash
python tools/colab_package_and_pull.py --pull-artifacts <downloaded_run_folder>
```

### Step 6 — Validate
```bash
python validate_tracking.py pitches/<recording_id> --model weights/ball_best.pt
```

See [docs/colab_training.md](docs/colab_training.md) for the full step-by-step guide.

## Controls (Live Mode)

| Key | Action |
|---|---|
| SPACE | Start / stop recording a pitch |
| D | Toggle debug panel |
| C | Click to define ball search ROI |
| Y | Click to set strike-zone Y level |
| S | Save recorded pitch to disk |
| R | Reset tracker |
| Q / ESC | Quit |

## Quality Targets

| Metric | Target |
|---|---|
| Detection rate | ≥ 90% of visible frames |
| Mean pixel error | ≤ 10 px |
| False starts | 0 |
| Dropouts | ≤ 2 per pitch |

## License

W.I.P.
