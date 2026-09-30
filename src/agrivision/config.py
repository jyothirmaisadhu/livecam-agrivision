from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"
KB_DIR = ROOT / "knowledge_base"


@dataclass(frozen=True)
class SeverityLevels:
    healthy: float
    mild: float
    moderate: float


@dataclass(frozen=True)
class LiveSettings:
    process_every_n_frames: int
    smoothing_window: int
    disease_confidence: float
    pest_confidence: float
    speak_cooldown_seconds: float


@dataclass(frozen=True)
class SplitSettings:
    train: float
    val: float
    test: float
    seed: int


@dataclass(frozen=True)
class Settings:
    disease_size: int
    pest_size: int
    damage_size: int
    severity: SeverityLevels
    live: LiveSettings
    split: SplitSettings


@dataclass(frozen=True)
class Classes:
    crops: list[str]
    disease: list[str]
    pest: list[str]
    crop_pests: dict[str, list[str]]
    segmentation: list[str]


def _load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


@lru_cache
def load_classes(path: Path = CONFIG_DIR / "classes.yaml") -> Classes:
    raw = _load_yaml(path)
    return Classes(
        crops=raw["crops"],
        disease=raw["disease_classes"],
        pest=raw["pest_classes"],
        crop_pests=raw["crop_pests"],
        segmentation=raw["segmentation_classes"],
    )


@lru_cache
def load_settings(path: Path = CONFIG_DIR / "settings.yaml") -> Settings:
    raw = _load_yaml(path)
    return Settings(
        disease_size=raw["image_size"]["disease"],
        pest_size=raw["image_size"]["pest"],
        damage_size=raw["image_size"]["damage"],
        severity=SeverityLevels(**raw["severity_levels"]),
        live=LiveSettings(**raw["live"]),
        split=SplitSettings(**raw["split"]),
    )


def crop_of(disease_class: str) -> str:
    return disease_class.split("_", 1)[0]
