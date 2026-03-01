"""
Pitch recorder — save a pitch sequence (frames + detections) to disk.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from msb.config import Config
from msb.detector import BallCandidate


class PitchRecorder:
    """Capture frames and detection metadata during a pitch."""

    def __init__(self, cfg: Optional[Config] = None) -> None:
        self.cfg = cfg or Config()
        self.root = Path(self.cfg.pitches_dir)
        self.recording: bool = False
        self.frames: List[np.ndarray] = []
        self.detections: List[Optional[Dict[str, Any]]] = []
        self._start_time: float = 0.0

    def start(self) -> None:
        self.recording = True
        self.frames.clear()
        self.detections.clear()
        self._start_time = time.time()
        print("[REC] Recording started...")

    def stop(self) -> None:
        self.recording = False
        print(f"[REC] Stopped. {len(self.frames)} frames captured.")

    def add_frame(self, frame: np.ndarray,
                  ball: Optional[BallCandidate]) -> bool:
        self.frames.append(frame.copy())
        if ball is not None:
            self.detections.append({
                "center": list(ball.center),
                "area": ball.area,
                "circularity": ball.circularity,
                "bbox": list(ball.bbox),
                "in_motion": ball.in_motion_mask,
                "isolation": ball.isolation_score,
                "corridor": ball.corridor_score,
                "score": ball.score,
            })
        else:
            self.detections.append(None)
        if len(self.frames) >= self.cfg.record_max_frames:
            self.stop()
            return False
        return True

    def save(self) -> Optional[str]:
        if not self.frames:
            print("[REC] Nothing to save.")
            return None
        ts = time.strftime("%Y%m%d_%H%M%S")
        pitch_dir = self.root / ts
        pitch_dir.mkdir(parents=True, exist_ok=True)
        for i, frame in enumerate(self.frames):
            cv2.imwrite(str(pitch_dir / f"frame_{i:04d}.png"), frame)
        meta = {
            "timestamp": ts,
            "n_frames": len(self.frames),
            "fps": self.cfg.target_fps,
            "screen_roi": list(self.cfg.screen_roi),
            "detections": self.detections,
        }
        with open(pitch_dir / "pitch_meta.json", "w") as fh:
            json.dump(meta, fh, indent=2)
        print(f"[REC] Saved {len(self.frames)} frames to {pitch_dir}")
        return str(pitch_dir)
