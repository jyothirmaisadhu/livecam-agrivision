"""Thin wrappers that load the exported models and return plain Python results."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from agrivision.decision.engine import DiseaseResult, PestDetection


class TFLiteModel:
    def __init__(self, path: Path):
        self.interpreter = tf.lite.Interpreter(model_path=str(path))
        self.interpreter.allocate_tensors()
        self.input = self.interpreter.get_input_details()[0]
        self.output = self.interpreter.get_output_details()[0]
        self.size = int(self.input["shape"][1])

    def run(self, image_rgb: np.ndarray) -> np.ndarray:
        x = cv2.resize(image_rgb, (self.size, self.size), interpolation=cv2.INTER_AREA)
        x = np.expand_dims(x.astype(np.float32), 0)  # models expect raw 0-255 RGB
        self.interpreter.set_tensor(self.input["index"], x)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output["index"])[0]


class DiseaseClassifier:
    def __init__(self, tflite_path: Path, labels_path: Path):
        self.model = TFLiteModel(tflite_path)
        self.labels: list[str] = json.loads(labels_path.read_text())

    def predict(self, image_rgb: np.ndarray) -> DiseaseResult:
        probs = self.model.run(image_rgb)
        i = int(np.argmax(probs))
        return DiseaseResult(self.labels[i], float(probs[i]))


class DamageSegmenter:
    def __init__(self, tflite_path: Path):
        self.model = TFLiteModel(tflite_path)

    def predict(self, image_rgb: np.ndarray) -> np.ndarray:
        """Returns an HxW uint8 mask (0 bg, 1 leaf, 2 lesion) at the input image size."""
        mask = np.argmax(self.model.run(image_rgb), axis=-1).astype(np.uint8)
        h, w = image_rgb.shape[:2]
        return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)


class PestDetector:
    def __init__(self, weights: Path, confidence: float):
        from ultralytics import YOLO  # heavy import, only when pests are enabled

        self.model = YOLO(str(weights), task="detect")
        self.confidence = confidence

    def predict(self, image_rgb: np.ndarray) -> list[PestDetection]:
        result = self.model.predict(image_rgb[:, :, ::-1], conf=self.confidence, verbose=False)[0]
        names = result.names
        return [
            PestDetection(names[int(c)], float(s), tuple(float(v) for v in xyxy))
            for xyxy, c, s in zip(
                result.boxes.xyxy.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist(), strict=True
            )
        ]
