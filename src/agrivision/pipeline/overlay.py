from __future__ import annotations

import cv2
import numpy as np

from agrivision.decision.engine import PestDetection, Report

LEVEL_COLOURS = {
    "healthy": (0, 180, 0),
    "mild": (0, 200, 255),
    "moderate": (0, 140, 255),
    "severe": (0, 0, 255),
}


def draw(
    frame_bgr: np.ndarray, report: Report | None, pests: list[PestDetection], mask: np.ndarray | None
) -> np.ndarray:
    out = frame_bgr.copy()
    if mask is not None:
        lesion = mask == 2
        out[lesion] = (0.5 * out[lesion] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
    for p in pests:
        x1, y1, x2, y2 = map(int, p.box)
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 200, 0), 2)
        cv2.putText(
            out,
            f"{p.label} {p.confidence:.2f}",
            (x1, max(15, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 200, 0),
            2,
        )
    if report is not None:
        cv2.rectangle(out, (0, 0), (out.shape[1], 58), (0, 0, 0), -1)
        cv2.putText(out, report.headline, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if report.severity_level:
            colour = LEVEL_COLOURS.get(report.severity_level, (255, 255, 255))
            cv2.putText(
                out,
                f"Severity: {report.severity_percent:.1f}% ({report.severity_level})",
                (10, 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                colour,
                2,
            )
    return out
