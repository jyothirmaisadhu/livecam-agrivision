"""Evaluate the disease classifier on data/disease/test: accuracy, per-class
precision/recall/F1 and a confusion-matrix image (same metrics as the base paper)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix  # noqa: E402

from agrivision.config import load_settings  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("data/disease"))
    ap.add_argument("--model-dir", type=Path, default=Path("models/disease"))
    args = ap.parse_args()

    class_names = json.loads((args.model_dir / "labels.json").read_text())
    size = load_settings().disease_size
    model = tf.keras.models.load_model(args.model_dir / "best.keras")
    test = tf.keras.utils.image_dataset_from_directory(
        args.data / "test",
        label_mode="categorical",
        class_names=class_names,
        image_size=(size, size),
        batch_size=32,
        shuffle=False,
    )
    y_true, y_pred = [], []
    for x, y in test:
        y_true.extend(np.argmax(y, 1))
        y_pred.extend(np.argmax(model.predict(x, verbose=0), 1))

    labels = list(range(len(class_names)))
    report = classification_report(
        y_true, y_pred, labels=labels, target_names=class_names, digits=4, zero_division=0
    )
    print(report)
    (args.model_dir / "test_report.txt").write_text(report)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(14, 14))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=ax, xticks_rotation=90, colorbar=False)
    fig.tight_layout()
    fig.savefig(args.model_dir / "confusion_matrix.png", dpi=120)
    print(f"Saved {args.model_dir / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
