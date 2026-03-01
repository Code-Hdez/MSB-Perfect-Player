"""
Shared utility functions and visual constants for the MSB pipeline.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

# BGR colours

COL_GREEN   = (0, 255, 0)
COL_RED     = (0, 0, 255)
COL_YELLOW  = (0, 255, 255)
COL_CYAN    = (255, 255, 0)
COL_WHITE   = (255, 255, 255)
COL_BLACK   = (0, 0, 0)
COL_MAGENTA = (255, 0, 255)
COL_ORANGE  = (0, 165, 255)

FONT = cv2.FONT_HERSHEY_SIMPLEX

# Helper functions

def crop(frame: np.ndarray, rect: Tuple[int, int, int, int]) -> np.ndarray:
    """Crop *frame* to *(x1, y1, x2, y2)* rectangle."""
    x1, y1, x2, y2 = rect
    return frame[max(0, y1):y2, max(0, x1):x2]


def put_text(
    img: np.ndarray,
    text: str,
    org: Tuple[int, int],
    scale: float = 0.6,
    color: Tuple[int, ...] = COL_GREEN,
    thickness: int = 2,
) -> None:
    """Draw text with a dark shadow for readability."""
    x, y = org
    cv2.putText(img, text, (x + 1, y + 1), FONT, scale,
                COL_BLACK, thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), FONT, scale,
                color, thickness, cv2.LINE_AA)


def in_rect(x: int, y: int, rect: Tuple[int, int, int, int]) -> bool:
    """Return True if *(x, y)* is inside *(x1, y1, x2, y2)*."""
    return rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]
