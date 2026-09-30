"""Step 2: train the MobileNetV3 disease classifier (transfer learning, 2 phases).

Phase 1: backbone frozen, train only the new head (fast, stable).
Phase 2: unfreeze the top of the backbone and fine-tune with a small learning rate.

Expects data/disease/{train,val}/<class>/*.jpg made by agrivision.data.prepare_classification.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from agrivision.config import load_classes, load_settings
from agrivision.disease.model import augmentation, build_classifier
from agrivision.tflite_export import export_tflite

AUTOTUNE = tf.data.AUTOTUNE


def load_split(root: Path, split: str, class_names: list[str], size: int, batch: int, shuffle: bool):
    return tf.keras.utils.image_dataset_from_directory(
        root / split,
        labels="inferred",
        label_mode="categorical",
        class_names=class_names,
        image_size=(size, size),
        batch_size=batch,
        shuffle=shuffle,
        seed=42,
    )


def present_classes(root: Path, all_classes: list[str]) -> list[str]:
    """Train only on classes that actually have images (missing ones are reported)."""
    present = [
        c for c in all_classes if (root / "train" / c).is_dir() and any((root / "train" / c).iterdir())
    ]
    missing = sorted(set(all_classes) - set(present))
    if missing:
        print(f"[warn] no training images for {len(missing)} classes: {missing}")
    return present


def class_weights(ds, num_classes: int) -> dict[int, float]:
    counts = np.zeros(num_classes)
    for _, y in ds.unbatch():
        counts[int(tf.argmax(y))] += 1
    total = counts.sum()
    return {i: float(total / (num_classes * c)) if c else 0.0 for i, c in enumerate(counts)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/disease"))
    ap.add_argument("--out", type=Path, default=Path("models/disease"))
    ap.add_argument("--variant", choices=["large", "small"], default="large")
    ap.add_argument("--epochs-head", type=int, default=10)
    ap.add_argument("--epochs-finetune", type=int, default=15)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--unfreeze-last", type=int, default=60, help="backbone layers to fine-tune in phase 2")
    ap.add_argument("--no-pretrained", action="store_true", help="random init (only for quick smoke tests)")
    args = ap.parse_args()

    size = load_settings().disease_size
    class_names = present_classes(args.data, load_classes().disease)
    train = load_split(args.data, "train", class_names, size, args.batch, shuffle=True)
    val = load_split(args.data, "val", class_names, size, args.batch, shuffle=False)
    weights = class_weights(train, len(class_names))

    aug = augmentation()
    train = train.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    val = val.prefetch(AUTOTUNE)

    model, backbone = build_classifier(
        len(class_names), size, args.variant, weights=None if args.no_pretrained else "imagenet"
    )
    args.out.mkdir(parents=True, exist_ok=True)
    best = args.out / "best.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(best, monitor="val_accuracy", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2),
        tf.keras.callbacks.CSVLogger(args.out / "history.csv", append=True),
    ]

    backbone.trainable = False
    model.compile(tf.keras.optimizers.Adam(1e-3), "categorical_crossentropy", metrics=["accuracy"])
    model.fit(train, validation_data=val, epochs=args.epochs_head, class_weight=weights, callbacks=callbacks)

    if args.epochs_finetune > 0:
        backbone.trainable = True
        for layer in backbone.layers[: -args.unfreeze_last]:
            layer.trainable = False
        for layer in backbone.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
        model.compile(tf.keras.optimizers.Adam(1e-5), "categorical_crossentropy", metrics=["accuracy"])
        model.fit(
            train, validation_data=val, epochs=args.epochs_finetune, class_weight=weights, callbacks=callbacks
        )

    model.save(best)
    (args.out / "labels.json").write_text(json.dumps(class_names, indent=2))
    (args.out / "labels.txt").write_text("\n".join(class_names))
    tfl = export_tflite(model, args.out / "disease_mobilenetv3.tflite", quantize="float16")
    print(f"Saved {best} and {tfl}")


if __name__ == "__main__":
    main()
