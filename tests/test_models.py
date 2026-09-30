"""Build tiny untrained models and round-trip them through TFLite (no downloads)."""

import json

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from agrivision.damage.unet import build_unet, dice_ce_loss  # noqa: E402
from agrivision.disease.model import build_classifier  # noqa: E402
from agrivision.pipeline.models import DamageSegmenter, DiseaseClassifier  # noqa: E402
from agrivision.tflite_export import export_tflite  # noqa: E402


def test_classifier_tflite_roundtrip(tmp_path, leaf):
    model, _ = build_classifier(4, image_size=96, variant="small", weights=None)
    assert model.output_shape == (None, 4)
    path = export_tflite(model, tmp_path / "d.tflite", quantize="none")
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps(["a_healthy", "a_x", "b_healthy", "b_y"]))
    result = DiseaseClassifier(path, labels).predict(leaf(size=200))
    assert result.label in {"a_healthy", "a_x", "b_healthy", "b_y"} and 0 <= result.confidence <= 1


def test_unet_shapes_loss_and_tflite(tmp_path, leaf):
    model, _ = build_unet(3, image_size=64, weights=None)
    assert model.output_shape == (None, 64, 64, 3)
    y_true = np.random.randint(0, 3, (2, 64, 64, 1))
    loss = dice_ce_loss(y_true, model(np.zeros((2, 64, 64, 3), np.float32)))
    assert np.isfinite(float(loss))
    path = export_tflite(model, tmp_path / "u.tflite", quantize="float16")
    mask = DamageSegmenter(path).predict(leaf(size=100))
    assert mask.shape == (100, 100) and mask.max() <= 2
