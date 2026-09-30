"""Step 3: train YOLO11n for pest detection with Ultralytics and export to TFLite.

Expects data/pest/data.yaml made by agrivision.data.prepare_yolo.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

from agrivision.config import load_settings


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/pest/data.yaml"))
    ap.add_argument("--out", type=Path, default=Path("models/pest"))
    ap.add_argument("--weights", default="yolo11n.pt", help="pretrained COCO weights to start from")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=load_settings().pest_size)
    ap.add_argument("--device", default=None, help="0 for GPU, 'cpu' for CPU")
    ap.add_argument("--no-export", action="store_true")
    args = ap.parse_args()

    model = YOLO(args.weights)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(args.out.resolve()),
        name="train",
        exist_ok=True,
        patience=20,
        close_mosaic=10,
    )
    best = args.out / "train" / "weights" / "best.pt"
    metrics = YOLO(str(best)).val(
        data=str(args.data),
        split="test",
        imgsz=args.imgsz,
        device=args.device,
        project=str(args.out.resolve()),
        name="test",
        exist_ok=True,
    )
    print(f"test mAP50={metrics.box.map50:.4f}  mAP50-95={metrics.box.map:.4f}")
    shutil.copy2(best, args.out / "pest_yolo11n.pt")

    if not args.no_export:
        exported = YOLO(str(best)).export(format="tflite", imgsz=args.imgsz, half=True)
        shutil.copy2(exported, args.out / "pest_yolo11n.tflite")
        print(f"Saved {args.out / 'pest_yolo11n.tflite'}")


if __name__ == "__main__":
    main()
