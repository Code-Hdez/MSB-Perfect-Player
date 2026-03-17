"""Mario Superstar Baseball Ball Tracking Pipeline

Core library for real-time pitched-ball detection and tracking.

Quick start::

    from msb import Config, BallDetector, BallTracker, TrajectoryCorridor

    cfg = Config.load("config.toml")
    detector = BallDetector(cfg)
    tracker  = BallTracker(cfg)
    corridor = TrajectoryCorridor(cfg)
"""

from msb.config import Config
from msb.detector import (
    BallCandidate, BackgroundModel, PitcherSuppressor, BallDetector,
)
from msb.tracker import TrackState, BallTrack, BallTracker
from msb.corridor import TrajectoryCorridor
from msb.game_phase import GamePhase, GamePhaseManager
from msb.swing_controller import SwingState, SwingController
from msb.metrics import FrameMetrics, SessionMetrics
from msb.predictor import TrajectoryPredictor
from msb.recorder import PitchRecorder
from msb.detector_ml import MLBallDetector
from msb.tracker_ml import MLBallTracker, MLTrackState, MLBallTrack

from msb.sources import CaptureThread

__all__ = [
    "Config",
    "BallCandidate", "BackgroundModel", "PitcherSuppressor", "BallDetector",
    "MLBallDetector",
    "TrackState", "BallTrack", "BallTracker",
    "MLBallTracker", "MLTrackState", "MLBallTrack",
    "TrajectoryCorridor",
    "TrajectoryPredictor",
    "PitchRecorder",
    "GamePhase", "GamePhaseManager",
    "SwingState", "SwingController",
    "FrameMetrics", "SessionMetrics",
    "CaptureThread",
]
