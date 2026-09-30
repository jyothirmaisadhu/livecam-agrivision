"""Copy trained models, labels, knowledge base and settings into the Flutter app's assets.

  python scripts/export_app_assets.py --models models

Missing models are skipped; the app shows which ones are bundled.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from agrivision.config import KB_DIR, load_classes, load_settings

APP_ASSETS = Path(__file__).resolve().parents[1] / "app" / "assets"


def app_config() -> dict:
    s = load_settings()
    return {
        "crop_pests": {k: list(v) for k, v in load_classes().crop_pests.items()},
        "severity_levels": {
            "healthy": s.severity.healthy,
            "mild": s.severity.mild,
            "moderate": s.severity.moderate,
        },
        "live": {
            "process_every_n_frames": s.live.process_every_n_frames,
            "smoothing_window": s.live.smoothing_window,
            "disease_confidence": s.live.disease_confidence,
            "pest_confidence": s.live.pest_confidence,
            "speak_cooldown_seconds": s.live.speak_cooldown_seconds,
        },
    }


def export_knowledge(out: Path = APP_ASSETS) -> None:
    kb_out = out / "knowledge_base"
    kb_out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(KB_DIR / "recommendations.json", kb_out / "recommendations.json")
    (kb_out / "app_config.json").write_text(json.dumps(app_config(), indent=2) + "\n")


def export_models(models: Path, out: Path = APP_ASSETS) -> list[str]:
    model_out = out / "models"
    model_out.mkdir(parents=True, exist_ok=True)
    copies = {
        models / "disease" / "disease_mobilenetv3.tflite": "disease_mobilenetv3.tflite",
        models / "disease" / "labels.json": "disease_labels.json",
        models / "damage" / "damage_unet.tflite": "damage_unet.tflite",
        models / "pest" / "pest_yolo11n.tflite": "pest_yolo11n.tflite",
    }
    copied = []
    for src, name in copies.items():
        if src.exists():
            shutil.copy2(src, model_out / name)
            copied.append(name)
        else:
            print(f"skip (not found): {src}")
    (model_out / "pest_labels.txt").write_text("\n".join(load_classes().pest) + "\n")
    return copied


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", type=Path, default=Path("models"))
    ap.add_argument("--out", type=Path, default=APP_ASSETS)
    args = ap.parse_args()
    export_knowledge(args.out)
    copied = export_models(args.models, args.out)
    print(f"Exported knowledge base + {len(copied)} model files to {args.out}")


if __name__ == "__main__":
    main()
