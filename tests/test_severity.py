import numpy as np
import pytest

from agrivision.config import SeverityLevels
from agrivision.damage.severity import baseline_opencv_severity, classify_level, severity_from_mask

LEVELS = SeverityLevels(healthy=1.0, mild=10.0, moderate=25.0)


@pytest.mark.parametrize(
    "pct,level",
    [(0.0, "healthy"), (0.9, "healthy"), (5, "mild"), (10, "mild"), (20, "moderate"), (60, "severe")],
)
def test_levels(pct, level):
    assert classify_level(pct, LEVELS) == level


def test_severity_uses_leaf_area_not_image_area():
    mask = np.zeros((100, 100), np.uint8)
    mask[:, :50] = 1  # 5000 leaf px
    mask[:10, :50] = 2  # 500 of them lesion
    s = severity_from_mask(mask, LEVELS)
    assert s.percent == 10.0 and s.level == "mild"


def test_empty_mask():
    assert severity_from_mask(np.zeros((10, 10), np.uint8), LEVELS).percent == 0.0


def test_baseline_grows_with_lesion(leaf):
    assert baseline_opencv_severity(leaf(lesion_radius=30)) > baseline_opencv_severity(leaf(lesion_radius=5))
