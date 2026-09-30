import json
import sys
from pathlib import Path

from agrivision.config import KB_DIR, load_classes

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import export_app_assets  # noqa: E402

ASSETS = ROOT / "app" / "assets"


def test_flutter_assets_match_python_sources():
    """Re-run `python scripts/export_app_assets.py` if this fails."""
    kb = json.loads((ASSETS / "knowledge_base" / "recommendations.json").read_text())
    assert kb == json.loads((KB_DIR / "recommendations.json").read_text())
    config = json.loads((ASSETS / "knowledge_base" / "app_config.json").read_text())
    assert config == export_app_assets.app_config()
    labels = (ASSETS / "models" / "pest_labels.txt").read_text().split()
    assert labels == load_classes().pest


def test_export_writes_assets(tmp_path):
    export_app_assets.export_knowledge(tmp_path)
    assert export_app_assets.export_models(tmp_path / "no-models", tmp_path) == []
    assert (tmp_path / "knowledge_base" / "app_config.json").exists()
    assert (tmp_path / "models" / "pest_labels.txt").exists()
