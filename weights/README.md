# Model Weights

This directory stores trained model weights for ball detection.

## Setup

After training (Colab or local), pull artifacts:

```bash
python tools/colab_package_and_pull.py --pull-artifacts <run_folder>
```

## Usage

```bash
python track_live.py --model weights/ball_best.pt
python track_folder.py <pitch_folder> --model weights/ball_best.pt
python validate_tracking.py <pitch_folder> --model weights/ball_best.pt
```

## IMPORTANT

- `.pt` and `.onnx` files are **gitignored** (too large for git)
- Store them in Google Drive or local storage
- Run metadata in `runs/` can be committed selectively
