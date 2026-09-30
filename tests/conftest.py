import cv2
import numpy as np
import pytest


def synthetic_leaf(size: int = 128, lesion_radius: int = 0, seed: int = 0) -> np.ndarray:
    """Green ellipse on a light grey background with optional brown lesion (RGB)."""
    rng = np.random.default_rng(seed)
    img = np.full((size, size, 3), 225, np.uint8)
    cv2.ellipse(img, (size // 2, size // 2), (size // 3, size // 4), 0, 0, 360, (40, 150, 40), -1)
    if lesion_radius:
        cv2.circle(img, (size // 2, size // 2), lesion_radius, (120, 70, 30), -1)
    noise = rng.integers(-5, 6, img.shape)
    return np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)


@pytest.fixture
def leaf():
    return synthetic_leaf
