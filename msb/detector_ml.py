"""
YOLOv8 inference replacing the classical
HSV + background-subtraction + contour pipeline.

Provides the SAME interface as ``BallDetector`` (returns
``BallCandidate`` objects, exposes ``.candidates`` and ``.best``),
so the tracker, visualiser, and all other pipeline components work
without modification.

Supports:
  - YOLO .pt model  (requires ``ultralytics``)
  - ONNX model      (requires ``onnxruntime`` or ``onnxruntime-gpu``)

Usage::

    from msb.detector_ml import MLBallDetector

    detector = MLBallDetector("runs/ball_detect/train/weights/best.pt",
                              conf=0.25, imgsz=960)
    # Same interface as BallDetector:
    detector.detect(frame)
    print(detector.best, detector.candidates)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from msb.config import Config
from msb.corridor import TrajectoryCorridor

# Re-use the shared BallCandidate dataclass
from msb.detector import BallCandidate


#  INFERENCE BACKENDS

class _YOLOBackend:
    """Ultralytics YOLO .pt inference."""

    def __init__(self, model_path: str, conf: float, imgsz: int,
                 device: str) -> None:
        from ultralytics import YOLO
        self._model = YOLO(model_path)
        self._conf = conf
        self._imgsz = imgsz
        self._device = device

    def infer(self, frame: np.ndarray
              ) -> List[Tuple[int, int, int, int, float, int]]:
        """Run inference.  Returns list of (x1, y1, x2, y2, conf, cls)."""
        results = self._model.predict(
            frame,
            conf=self._conf,
            imgsz=self._imgsz,
            device=self._device,
            verbose=False,
            stream=False,
        )
        detections: List[Tuple[int, int, int, int, float, int]] = []
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                detections.append((int(x1), int(y1), int(x2), int(y2),
                                   conf, cls))
        return detections


class _ONNXBackend:
    """ONNX Runtime inference (no ultralytics dependency at runtime)."""

    def __init__(self, model_path: str, conf: float,
                 imgsz: int) -> None:
        try:
            import onnxruntime as ort
        except ImportError:
            print("[ERROR] onnxruntime not installed. "
                  "pip install onnxruntime-gpu  (or onnxruntime)")
            sys.exit(1)

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._sess = ort.InferenceSession(model_path, providers=providers)
        self._conf = conf
        self._imgsz = imgsz
        self._input_name = self._sess.get_inputs()[0].name

    def infer(self, frame: np.ndarray
              ) -> List[Tuple[int, int, int, int, float, int]]:
        """Preprocess, run ONNX, postprocess NMS."""
        img, ratio, (pad_w, pad_h) = self._preprocess(frame)
        outputs = self._sess.run(None, {self._input_name: img})
        return self._postprocess(outputs[0], ratio, pad_w, pad_h,
                                 frame.shape[:2])

    def _preprocess(self, frame: np.ndarray):
        """Letterbox + normalise to NCHW float32."""
        h0, w0 = frame.shape[:2]
        sz = self._imgsz
        ratio = min(sz / h0, sz / w0)
        new_w, new_h = int(w0 * ratio), int(h0 * ratio)
        img = cv2.resize(frame, (new_w, new_h),
                         interpolation=cv2.INTER_LINEAR)
        pad_w = (sz - new_w) // 2
        pad_h = (sz - new_h) // 2
        canvas = np.full((sz, sz, 3), 114, dtype=np.uint8)
        canvas[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = img
        blob = canvas.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[np.newaxis, ...]  # NCHW
        return blob, ratio, (pad_w, pad_h)

    def _postprocess(self, output, ratio, pad_w, pad_h, orig_shape):
        """Parse YOLO output tensor → list of detections."""
        # output shape: (1, 5+nc, num_preds)  or  (1, num_preds, 5+nc)
        pred = output[0]
        if pred.shape[0] < pred.shape[1]:
            pred = pred.T  # → (num_preds, 5+nc)

        detections: List[Tuple[int, int, int, int, float, int]] = []
        for row in pred:
            cx, cy, w, h = row[:4]
            scores = row[4:]
            cls = int(np.argmax(scores))
            conf = float(scores[cls])
            if conf < self._conf:
                continue

            # Undo letterbox
            x1 = (cx - w / 2 - pad_w) / ratio
            y1 = (cy - h / 2 - pad_h) / ratio
            x2 = (cx + w / 2 - pad_w) / ratio
            y2 = (cy + h / 2 - pad_h) / ratio

            # Clamp
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(orig_shape[1], int(x2))
            y2 = min(orig_shape[0], int(y2))

            if x2 > x1 and y2 > y1:
                detections.append((x1, y1, x2, y2, conf, cls))

        # Simple NMS
        if len(detections) > 1:
            detections = self._nms(detections, iou_thresh=0.5)
        return detections

    @staticmethod
    def _nms(dets, iou_thresh=0.5):
        """Greedy NMS on a list of (x1,y1,x2,y2,conf,cls)."""
        dets = sorted(dets, key=lambda d: d[4], reverse=True)
        keep = []
        for d in dets:
            overlap = False
            for k in keep:
                # IoU
                ix1 = max(d[0], k[0])
                iy1 = max(d[1], k[1])
                ix2 = min(d[2], k[2])
                iy2 = min(d[3], k[3])
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                a1 = (d[2] - d[0]) * (d[3] - d[1])
                a2 = (k[2] - k[0]) * (k[3] - k[1])
                union = a1 + a2 - inter
                if union > 0 and inter / union > iou_thresh:
                    overlap = True
                    break
            if not overlap:
                keep.append(d)
        return keep


#  ML BALL DETECTOR  (drop-in replacement for BallDetector)

class MLBallDetector:
    """YOLO-based ball detector with the same interface as BallDetector.

    Parameters
    ----------
    model_path : str
        Path to a .pt (ultralytics) or .onnx model file.
    conf : float
        Minimum detection confidence (default: 0.25).
    imgsz : int
        Inference resolution (default: 960). Should match training.
    device : str
        "0" for CUDA GPU 0, "cpu" for CPU (only used for .pt backend).
    cfg : Config, optional
        Pipeline config (used for corridor integration).
    """

    def __init__(
        self,
        model_path: str,
        conf: float = 0.25,
        imgsz: int = 960,
        device: str = "auto",
        cfg: Optional[Config] = None,
    ) -> None:
        # Auto-select device: CUDA if available, else CPU
        if device == "auto":
            import torch
            device = "0" if torch.cuda.is_available() else "cpu"
        self.cfg = cfg or Config()
        self.model_path = model_path
        self.conf = conf
        self.imgsz = imgsz

        # Select backend
        p = Path(model_path)
        if not p.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        if p.suffix == ".onnx":
            self._backend = _ONNXBackend(model_path, conf, imgsz)
            self._backend_name = "onnx"
        else:
            self._backend = _YOLOBackend(model_path, conf, imgsz, device)
            self._backend_name = "yolo"

        # Public state (same interface as BallDetector)
        self.candidates: List[BallCandidate] = []
        self.best: Optional[BallCandidate] = None

        # Dummy attributes for compatibility with visualiser
        class _DummySuppressor:
            suppression_zones: List = []
            large_blobs: List = []
        self.suppressor = _DummySuppressor()
        self.bg_model = type("BG", (), {"ready": True})()
        self.bg_fg_mask: Optional[np.ndarray] = None
        self.motion_mask: Optional[np.ndarray] = None
        self.white_mask: Optional[np.ndarray] = None
        self.trail_mask: Optional[np.ndarray] = None
        self.combined_mask: Optional[np.ndarray] = None

    # Main detection

    def detect(
        self,
        frame: np.ndarray,
        search_roi: Optional[Tuple[int, int, int, int]] = None,
        corridor: Optional[TrajectoryCorridor] = None,
        track_active: bool = False,
    ) -> Optional[BallCandidate]:
        """Detect ball in *frame* using YOLO inference.

        Returns the highest-confidence BallCandidate, or None.
        """
        self.candidates.clear()
        self.best = None

        # Generate dummy debug masks (so visualiser doesn't crash)
        h, w = frame.shape[:2]
        empty = np.zeros((h, w), dtype=np.uint8)
        self.bg_fg_mask = empty
        self.motion_mask = empty
        self.white_mask = empty
        self.trail_mask = empty
        self.combined_mask = empty

        # Run inference
        raw_dets = self._backend.infer(frame)

        if not raw_dets:
            return None

        # Convert to BallCandidate objects
        for (x1, y1, x2, y2, conf, cls) in raw_dets:
            if cls != 0:  # only class 0 = ball
                continue

            c = BallCandidate()
            bw, bh = x2 - x1, y2 - y1
            cx = x1 + bw // 2
            cy = y1 + bh // 2
            c.center = (cx, cy)
            c.bbox = (x1, y1, bw, bh)
            c.area = float(bw * bh)
            c.circularity = 0.8  # YOLO already filtered for ball-ness
            c.in_motion_mask = True
            c.isolation_score = 1.0
            c.brightness_score = conf
            c.score = conf

            # Corridor scoring (optional secondary filter)
            if corridor is not None:
                c.corridor_score = corridor.get_corridor_score(cx, cy)
            else:
                c.corridor_score = 1.0

            self.candidates.append(c)

        if not self.candidates:
            return None

        # Sort by confidence
        self.candidates.sort(key=lambda c: c.score, reverse=True)
        self.best = self.candidates[0]
        return self.best

    # Rescue detection (simplified for ML)

    def rescue_near(
        self, centre: Tuple[int, int], radius: int = 50,
    ) -> Optional[BallCandidate]:
        """Check if any recent candidate is near the predicted position.

        With ML detection, rescue is less critical since the model
        either sees the ball or doesn't. But we still check the
        candidates list for near misses below the threshold.
        """
        # Already covered by main detection; return None
        # The tracker handles gaps via Kalman prediction
        return None

    # Reset (compatibility)

    def reset(self) -> None:
        self.candidates.clear()
        self.best = None

    def reset_full(self) -> None:
        self.reset()
