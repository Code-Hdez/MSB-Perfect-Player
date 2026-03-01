"""
Configuration management for the MSB tracking pipeline.

Supports TOML files (Python 3.11+ built-in, or ``pip install tomli``),
JSON fallback, and programmatic overrides.

Usage::

    from msb.config import Config

    cfg = Config()                      # all defaults
    cfg = Config.load("config.toml")    # from file
    cfg.screen_roi = (100, 100, 1380, 920)  # override
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:
    import tomllib                          # Python 3.11+
except ModuleNotFoundError:
    try:
        import tomli as tomllib             # pip install tomli
    except ModuleNotFoundError:
        tomllib = None                      # type: ignore[assignment]


@dataclass
class Config:
    """Complete configuration for the MSB ball tracking pipeline.

    Every detection threshold, zone definition, tracker lifecycle
    parameter, and display option lives here.  Values match the
    original ``pitch_analyzer.py`` module-level constants.
    """

    #Capture
    screen_roi: Tuple[int, int, int, int] = (378, 127, 1542, 1019)
    monitor_index: int = 0
    target_fps: int = 60

    # Ball detection — HSV
    ball_hsv_lower: Tuple[int, int, int] = (0, 0, 140)
    ball_hsv_upper: Tuple[int, int, int] = (180, 80, 255)
    trail_hsv_lower: Tuple[int, int, int] = (0, 0, 100)
    trail_hsv_upper: Tuple[int, int, int] = (180, 100, 255)

    # Contour filtering
    ball_min_area: int = 6
    ball_max_area: int = 500
    ball_flight_max_area: int = 300
    ball_min_circularity: float = 0.15

    # Background model
    bg_alpha: float = 0.02
    bg_warmup_frames: int = 25
    bg_fg_threshold: int = 25

    # Frame differencing
    diff_threshold: int = 25
    diff_dilate_iter: int = 2

    # Pitcher-body suppression
    pitcher_body_min_area: int = 200
    isolation_zone_scale: float = 0.6
    isolation_inner: float = 0.5
    isolation_outer: float = 2.0

    # Corridor
    corridor_default: Optional[Tuple[int, int, int, int]] = (380, 80, 820, 900)
    corridor_margin: int = 50
    corridor_penalty_dist: float = 60.0

    # Tracker
    track_max_dist_min: float = 50.0
    track_max_dist_speed_k: float = 3.0
    track_gap_expand: float = 0.35
    track_lost_frames: int = 8
    track_tentative_lost: int = 5
    trajectory_history: int = 60
    track_min_confirmations: int = 4
    track_size_ratio: float = 5.0
    min_pitch_vy: float = 3.0
    max_pitcher_zone_frames: int = 15
    min_departure_dist: float = 60.0
    max_vy_sign_changes: int = 2

    # Kalman filter
    kf_process_noise: float = 16.0
    kf_measurement_noise: float = 4.0

    # Zones
    pitcher_zone: Tuple[int, int, int, int] = (520, 80, 680, 260)
    reacq_zone: Tuple[int, int, int, int] = (480, 180, 710, 600)
    reacq_window: int = 30

    # Static-element suppression
    static_cell_size: int = 12
    static_hit_threshold: int = 3

    # Recording
    record_max_frames: int = 120
    pitches_dir: str = "./pitches"

    # Display
    display_scale: float = 0.85

    #  Serialisation helpers

    @classmethod
    def load(cls, path: str) -> "Config":
        """Load from a TOML or JSON file.  Unknown keys are ignored."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        raw: Dict[str, Any]
        if p.suffix in (".toml", ".tml"):
            if tomllib is None:
                raise ImportError(
                    "TOML support requires Python 3.11+ or "
                    "'pip install tomli'.")
            with open(p, "rb") as fh:
                raw = tomllib.load(fh)
        else:
            with open(p) as fh:
                raw = json.load(fh)

        flat = cls._flatten(raw)
        return cls._from_flat(flat)

    def save_json(self, path: str) -> None:
        """Persist current values as JSON (always available)."""
        data: Dict[str, Any] = {}
        for f in fields(self):
            val = getattr(self, f.name)
            # Convert tuples to lists for JSON
            if isinstance(val, tuple):
                val = list(val)
            data[f.name] = val
        with open(path, "w") as fh:
            json.dump(data, fh, indent=2)

    def apply_overrides(self, overrides: Dict[str, Any]) -> None:
        """Apply a dict of key→value overrides (e.g. from CLI args).

        Only known field names are accepted; unknown keys are silently
        skipped.  Lists are converted to tuples where the field type
        expects a tuple.
        """
        valid = {f.name for f in fields(self)}
        for k, v in overrides.items():
            if k not in valid or v is None:
                continue
            if isinstance(v, list):
                v = tuple(v)
            setattr(self, k, v)

    def to_dict(self) -> Dict[str, Any]:
        """Return all config values as a flat dict."""
        out: Dict[str, Any] = {}
        for f in fields(self):
            out[f.name] = getattr(self, f.name)
        return out

    # Internal helpers

    @staticmethod
    def _flatten(d: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively flatten nested dicts (TOML sections) into a
        single-level dict.  Leaf values are kept; section keys are
        dropped (all field names are globally unique)."""
        flat: Dict[str, Any] = {}
        for k, v in d.items():
            if isinstance(v, dict):
                flat.update(Config._flatten(v))
            else:
                flat[k] = v
        return flat

    @classmethod
    def _from_flat(cls, flat: Dict[str, Any]) -> "Config":
        valid = {f.name for f in fields(cls)}
        kwargs: Dict[str, Any] = {}
        for k, v in flat.items():
            if k not in valid:
                continue
            if isinstance(v, list):
                v = tuple(v)
            kwargs[k] = v
        return cls(**kwargs)
