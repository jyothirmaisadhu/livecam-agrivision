"""Generate a tiny synthetic dataset in all three formats so every training script and the
live demo can be smoke-tested end-to-end before the real datasets are downloaded.
The resulting models are meaningless; this only checks that the plumbing works."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml

from agrivision.config import load_classes


def leaf(rng: np.random.Generator, size: int, lesions: int) -> tuple[np.ndarray, np.ndarray]:
    img = np.full((size, size, 3), rng.integers(190, 240), np.uint8)
    mask = np.zeros((size, size), np.uint8)
    center = (size // 2 + int(rng.integers(-10, 10)), size // 2 + int(rng.integers(-10, 10)))
    axes = (size // 3, size // 4)
    angle = int(rng.integers(0, 180))
    cv2.ellipse(img, center, axes, angle, 0, 360, (40, int(rng.integers(120, 180)), 40), -1)
    cv2.ellipse(mask, center, axes, angle, 0, 360, 1, -1)
    for _ in range(lesions):
        p = (
            center[0] + int(rng.integers(-axes[1], axes[1])),
            center[1] + int(rng.integers(-axes[1], axes[1])),
        )
        r = int(rng.integers(3, size // 12))
        cv2.circle(img, p, r, (120, 70, 30), -1)
        cv2.circle(mask, p, r, 2, -1)
    mask[(mask == 2) & (cv2.ellipse(np.zeros_like(mask), center, axes, angle, 0, 360, 1, -1) == 0)] = 0
    return img, mask


def save_rgb(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("data/demo"))
    ap.add_argument("--per-class", type=int, default=12)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    classes = load_classes()

    demo_diseases = ["tomato_healthy", "tomato_early_blight", "corn_healthy", "corn_common_rust"]
    for cls in demo_diseases:
        for split, n in (("train", args.per_class), ("val", 4), ("test", 4)):
            for i in range(n):
                img, _ = leaf(rng, 160, 0 if cls.endswith("healthy") else int(rng.integers(2, 6)))
                save_rgb(args.out / "disease" / split / cls / f"{i}.jpg", img)

    (args.out / "damage" / "masks").mkdir(parents=True, exist_ok=True)
    for i in range(args.per_class * 2):
        img, mask = leaf(rng, 160, int(rng.integers(0, 6)))
        save_rgb(args.out / "damage" / "images" / f"{i:04d}.jpg", img)
        cv2.imwrite(str(args.out / "damage" / "masks" / f"{i:04d}.png"), mask)

    pest_dir = args.out / "pest"
    for split, n in (("train", args.per_class), ("val", 4), ("test", 4)):
        for i in range(n):
            img, _ = leaf(rng, 320, 0)
            lines = []
            for _ in range(int(rng.integers(1, 4))):
                cls_id = int(rng.integers(0, 2))
                x, y = int(rng.integers(40, 280)), int(rng.integers(40, 280))
                colour = (250, 250, 250) if cls_id == 0 else (20, 20, 20)
                cv2.ellipse(img, (x, y), (14, 8), 30, 0, 360, colour, -1)
                lines.append(f"{[6, 5][cls_id]} {x / 320:.4f} {y / 320:.4f} {32 / 320:.4f} {22 / 320:.4f}")
            save_rgb(pest_dir / "images" / split / f"{i}.jpg", img)
            (pest_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
            (pest_dir / "labels" / split / f"{i}.txt").write_text("\n".join(lines))
    data_yaml = {
        "path": str(pest_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": dict(enumerate(classes.pest)),
    }
    (pest_dir / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False))

    frames = [leaf(rng, 320, k % 5)[0] for k in range(60)]
    writer = cv2.VideoWriter(str(args.out / "demo.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 320))
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"Demo data written to {args.out}")


if __name__ == "__main__":
    main()
