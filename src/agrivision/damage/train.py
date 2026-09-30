"""Step 4: train the U-Net damage segmenter.

Expects data/damage/images/*.jpg and data/damage/masks/*.png (same stem, values 0/1/2),
e.g. produced by agrivision.data.make_masks and corrected by hand.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf

from agrivision.config import load_classes, load_settings
from agrivision.damage.unet import build_unet, dice_ce_loss
from agrivision.data.split import split_items
from agrivision.tflite_export import export_tflite

AUTOTUNE = tf.data.AUTOTUNE


def pairs(root: Path) -> list[tuple[str, str]]:
    out = []
    for img in sorted((root / "images").glob("*")):
        mask = root / "masks" / f"{img.stem}.png"
        if mask.exists():
            out.append((str(img), str(mask)))
    return out


def make_dataset(items: list[tuple[str, str]], size: int, batch: int, train: bool) -> tf.data.Dataset:
    def load(img_path, mask_path):
        img = tf.image.decode_image(tf.io.read_file(img_path), channels=3, expand_animations=False)
        mask = tf.image.decode_png(tf.io.read_file(mask_path), channels=1)
        img = tf.image.resize(img, (size, size))
        mask = tf.image.resize(mask, (size, size), method="nearest")
        if train:
            if tf.random.uniform(()) > 0.5:
                img, mask = tf.image.flip_left_right(img), tf.image.flip_left_right(mask)
            if tf.random.uniform(()) > 0.5:
                img, mask = tf.image.flip_up_down(img), tf.image.flip_up_down(mask)
            img = tf.clip_by_value(tf.image.random_brightness(img, 25.0), 0, 255)
        return img, tf.cast(mask, tf.int32)

    imgs, masks = zip(*items, strict=True)
    ds = tf.data.Dataset.from_tensor_slices((list(imgs), list(masks)))
    if train:
        ds = ds.shuffle(len(items), seed=42)
    return ds.map(load, num_parallel_calls=AUTOTUNE).batch(batch).prefetch(AUTOTUNE)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/damage"))
    ap.add_argument("--out", type=Path, default=Path("models/damage"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--no-pretrained", action="store_true")
    args = ap.parse_args()

    size = load_settings().damage_size
    split = load_settings().split
    num_classes = len(load_classes().segmentation)
    train_items, val_items, _ = split_items(pairs(args.data), split.train, split.val, split.seed)
    if not train_items or not val_items:
        raise SystemExit(f"Not enough image/mask pairs in {args.data}")

    model, _ = build_unet(num_classes, size, weights=None if args.no_pretrained else "imagenet")
    model.compile(
        tf.keras.optimizers.Adam(1e-3),
        loss=dice_ce_loss,
        metrics=[tf.keras.metrics.MeanIoU(num_classes, sparse_y_pred=False, name="mean_iou")],
    )
    args.out.mkdir(parents=True, exist_ok=True)
    model.fit(
        make_dataset(train_items, size, args.batch, train=True),
        validation_data=make_dataset(val_items, size, args.batch, train=False),
        epochs=args.epochs,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                args.out / "best.keras", monitor="val_mean_iou", mode="max", save_best_only=True
            ),
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
            tf.keras.callbacks.CSVLogger(args.out / "history.csv"),
        ],
    )
    model.save(args.out / "best.keras")
    tfl = export_tflite(model, args.out / "damage_unet.tflite", quantize="float16")
    print(f"Saved {args.out / 'best.keras'} and {tfl}")


if __name__ == "__main__":
    main()
