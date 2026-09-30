"""Step 5: turn model outputs into farmer-facing advice (screen text + spoken sentence)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from agrivision.config import KB_DIR, crop_of, load_classes


@dataclass(frozen=True)
class DiseaseResult:
    label: str
    confidence: float


@dataclass(frozen=True)
class PestDetection:
    label: str
    confidence: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels


@dataclass(frozen=True)
class Advice:
    title: str
    detail: str
    treatment: str = ""
    prevention: str = ""


@dataclass
class Report:
    crop: str | None
    headline: str
    severity_percent: float | None
    severity_level: str | None
    items: list[Advice] = field(default_factory=list)
    spoken: str = ""


@lru_cache
def load_kb(path: Path = KB_DIR / "recommendations.json") -> dict:
    return json.loads(path.read_text())


def _pest_counts(pests: list[PestDetection]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for p in pests:
        counts[p.label] = counts.get(p.label, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def build_report(
    disease: DiseaseResult | None,
    pests: list[PestDetection],
    severity_percent: float | None = None,
    severity_level: str | None = None,
    kb: dict | None = None,
) -> Report:
    kb = kb or load_kb()
    crop = crop_of(disease.label) if disease else None
    relevant = set(load_classes().crop_pests.get(crop, [])) if crop else None
    items: list[Advice] = []
    spoken: list[str] = []

    if disease is None:
        headline = "No confident disease result"
    elif disease.label in kb["healthy"]:
        headline = kb["healthy"][disease.label]["message"]
        spoken.append(headline)
    else:
        info = kb["diseases"][disease.label]
        headline = f"{crop.title()}: {info['name']} ({disease.confidence:.0%})"
        items.append(Advice(info["name"], info["symptoms"], info["treatment"], info["prevention"]))
        spoken.append(f"{info['name']} detected on {crop}.")
        if severity_level and severity_level != "healthy":
            spoken.append(f"Severity is {severity_level}, about {severity_percent:.0f} percent of the leaf.")
            items.append(Advice(f"Severity: {severity_level}", kb["severity_advice"][severity_level]))
        spoken.append(f"Treatment: {info['treatment']}")

    counts = _pest_counts(pests)
    for label, n in counts.items():
        info = kb["pests"][label]
        note = "" if relevant is None or label in relevant else f" (not a common pest of {crop})"
        items.append(
            Advice(
                f"Pest: {info['name']} x{n}{note}", info["symptoms"], info["treatment"], info["prevention"]
            )
        )
        spoken.append(f"{n} {info['name']} found. {info['treatment']}")
        for linked in kb["vector_links"].get(label, []):
            if crop is None or crop_of(linked) == crop:
                virus = kb["diseases"][linked]["name"]
                items.append(Advice("Virus risk", f"{info['name']} can spread {virus}. Control it early."))

    if not spoken:
        spoken.append("Point the camera at a leaf, closer and in good light.")
    return Report(crop, headline, severity_percent, severity_level, items, " ".join(spoken))
