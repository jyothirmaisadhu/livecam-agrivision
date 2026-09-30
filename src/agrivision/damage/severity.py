"""Damage assessment: severity % from a segmentation mask, and the base paper's
OpenCV threshold method kept as a baseline for comparison."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from agrivision.config import SeverityLevels, load_settings

BACKGROUND, LEAF, LESION = 0, 1, 2


@dataclass(frozen=True)
class Severity:
    percent: float
    level: str  # healthy | mild | moderate | severe


def classify_level(percent: float, levels: SeverityLevels) -> str:
    if percent < levels.healthy:
        return "healthy"
    if percent <= levels.mild:
        return "mild"
    if percent <= levels.moderate:
        return "moderate"
    return "severe"


def severity_from_mask(mask: np.ndarray, levels: SeverityLevels | None = None) -> Severity:
    """Severity % = lesion pixels / (leaf + lesion pixels) * 100.

    The denominator is the leaf area only, so background does not dilute the score
    (the base paper divided by total image pixels).
    """
    levels = levels or load_settings().severity
    lesion = int(np.count_nonzero(mask == LESION))
    leaf_area = lesion + int(np.count_nonzero(mask == LEAF))
    percent = 100.0 * lesion / leaf_area if leaf_area else 0.0
    return Severity(round(percent, 2), classify_level(percent, levels))


def baseline_opencv_severity(image_rgb: np.ndarray, threshold: int = 100) -> float:
    """Base paper (Ramana Reddy et al., 2025): gray -> Gaussian blur -> fixed threshold ->
    morphology -> white pixels / total pixels * 100. Values < 1% are reported as 0."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY_INV)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    percent = 100.0 * np.count_nonzero(binary) / binary.size
    return 0.0 if percent < 1.0 else round(percent, 2)
