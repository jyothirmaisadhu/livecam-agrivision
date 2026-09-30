from __future__ import annotations

from pathlib import Path

import tensorflow as tf


def export_tflite(model: tf.keras.Model, out_path: Path, quantize: str = "float16") -> Path:
    """quantize: 'none' | 'float16' | 'dynamic' (int8 weights)."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if quantize in ("float16", "dynamic"):
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if quantize == "float16":
        converter.target_spec.supported_types = [tf.float16]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(converter.convert())
    return out_path
