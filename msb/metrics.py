"""
Structured per-frame telemetry and session-level summaries.

Collects timing breakdowns (capture, inference, tracking, prediction,
control), contact-accuracy statistics, and phase-transition events.
Supports CSV export for post-session analysis.

Usage::

    from msb.metrics import FrameMetrics, SessionMetrics

    session = SessionMetrics()
    # inside main loop:
    fm = FrameMetrics()
    fm.mark("capture")
    ...
    fm.mark("inference")
    ...
    fm.mark("tracking")
    ...
    session.record(fm, extras={...})
    # at end:
    session.print_summary()
    session.export_csv("session.csv")
"""

from __future__ import annotations

import csv
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# PER-FRAME METRICS

class FrameMetrics:
    """Lightweight wall-clock timer for a single frame.

    Call :meth:`mark` at each pipeline boundary.  The deltas between
    consecutive marks are stored as named durations.
    """

    __slots__ = ("_marks", "_durations", "extras")

    def __init__(self) -> None:
        self._marks: List[tuple] = []
        self._durations: Dict[str, float] = {}
        self.extras: Dict[str, Any] = {}

    def mark(self, label: str) -> None:
        """Record a timestamp for *label*.

        The elapsed time since the previous mark is stored as
        ``prev_label → label`` or just ``label`` (first mark).
        """
        t = time.perf_counter()
        if self._marks:
            prev_label, prev_t = self._marks[-1]
            self._durations[label] = t - prev_t
        self._marks.append((label, t))

    @property
    def total_sec(self) -> float:
        if len(self._marks) < 2:
            return 0.0
        return self._marks[-1][1] - self._marks[0][1]

    @property
    def durations(self) -> Dict[str, float]:
        return dict(self._durations)


# SESSION METRICS

@dataclass
class _SwingEvent:
    """Record of a single swing attempt."""
    timestamp: float
    error_px: float
    ttc_sec: float
    hit: bool
    phase: str = ""


class SessionMetrics:
    """Accumulates per-frame telemetry over an entire session.

    Maintains rolling FPS percentiles, phase-transition counts,
    and swing-accuracy statistics.
    """

    def __init__(self, fps_window: int = 300) -> None:
        self._frame_count: int = 0
        self._start_time: float = time.perf_counter()

        self._frame_times: deque = deque(maxlen=fps_window)

        self._stage_totals: Dict[str, float] = {}
        self._stage_counts: Dict[str, int] = {}

        self.phase_transitions: Dict[str, int] = {}

        self._swings: List[_SwingEvent] = []

        self._rows: List[Dict[str, Any]] = []

    # Per-frame recording

    def record(self, fm: FrameMetrics,
               extras: Optional[Dict[str, Any]] = None) -> None:
        """Record one frame's metrics."""
        self._frame_count += 1
        self._frame_times.append(fm.total_sec)

        for stage, dur in fm.durations.items():
            self._stage_totals[stage] = (
                self._stage_totals.get(stage, 0.0) + dur)
            self._stage_counts[stage] = (
                self._stage_counts.get(stage, 0) + 1)

        row: Dict[str, Any] = {"frame": self._frame_count}
        row.update(fm.durations)
        row["total_ms"] = fm.total_sec * 1000.0
        if extras:
            row.update(extras)
        if fm.extras:
            row.update(fm.extras)
        self._rows.append(row)

    def record_phase_transition(self, from_phase: str,
                                to_phase: str) -> None:
        key = f"{from_phase}->{to_phase}"
        self.phase_transitions[key] = (
            self.phase_transitions.get(key, 0) + 1)

    def record_swing(self, error_px: float, ttc_sec: float,
                     hit: bool, phase: str = "") -> None:
        self._swings.append(_SwingEvent(
            timestamp=time.perf_counter(),
            error_px=error_px,
            ttc_sec=ttc_sec,
            hit=hit,
            phase=phase,
        ))

    # Queries

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def elapsed_sec(self) -> float:
        return time.perf_counter() - self._start_time

    def fps_percentiles(self) -> Dict[str, float]:
        """Return p50, p95, p99 FPS from the rolling window."""
        if not self._frame_times:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
        import numpy as np
        ft = np.array(self._frame_times)
        ft = ft[ft > 0]
        if len(ft) == 0:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
        fps = 1.0 / ft
        return {
            "p50": float(np.percentile(fps, 50)),
            "p95": float(np.percentile(fps, 5)),   # p95 = 5th-percentile fps
            "p99": float(np.percentile(fps, 1)),
        }

    def stage_avg_ms(self) -> Dict[str, float]:
        """Average milliseconds per pipeline stage."""
        out: Dict[str, float] = {}
        for stage, total in self._stage_totals.items():
            cnt = self._stage_counts.get(stage, 1)
            out[stage] = (total / cnt) * 1000.0
        return out

    def swing_accuracy(self) -> Dict[str, Any]:
        """Swing hit-rate and mean error."""
        if not self._swings:
            return {"total": 0, "hits": 0, "rate": 0.0,
                    "mean_error_px": 0.0}
        hits = sum(1 for s in self._swings if s.hit)
        errors = [s.error_px for s in self._swings]
        import numpy as np
        return {
            "total": len(self._swings),
            "hits": hits,
            "rate": hits / len(self._swings),
            "mean_error_px": float(np.mean(errors)),
            "median_error_px": float(np.median(errors)),
        }

    # Output

    def print_summary(self) -> None:
        """Print a concise session summary to stdout."""
        elapsed = self.elapsed_sec
        fps = self.fps_percentiles()
        stages = self.stage_avg_ms()
        sa = self.swing_accuracy()

        print("\n" + "=" * 60)
        print("  SESSION SUMMARY")
        print("=" * 60)
        print(f"  Frames:  {self._frame_count:,}  "
              f"({elapsed:.1f}s)")
        print(f"  FPS:     p50={fps['p50']:.1f}  "
              f"p95={fps['p95']:.1f}  p99={fps['p99']:.1f}")
        if stages:
            print("  Stage avg (ms):")
            for s, ms in stages.items():
                print(f"    {s:16s} {ms:6.2f}")
        if sa["total"] > 0:
            print(f"  Swings:  {sa['total']}  "
                  f"hits={sa['hits']}  rate={sa['rate']:.1%}  "
                  f"mean_err={sa['mean_error_px']:.1f}px")
        if self.phase_transitions:
            print("  Phase transitions:")
            for k, v in sorted(self.phase_transitions.items()):
                print(f"    {k}: {v}")
        print("=" * 60 + "\n")

    def export_csv(self, path: str) -> None:
        """Write per-frame rows to CSV."""
        if not self._rows:
            return
        p = Path(path)
        all_keys: List[str] = []
        seen: set = set()
        for row in self._rows:
            for k in row:
                if k not in seen:
                    all_keys.append(k)
                    seen.add(k)

        with open(p, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys,
                                    extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self._rows)
        print(f"[METRICS] Exported {len(self._rows)} rows → {p}")
