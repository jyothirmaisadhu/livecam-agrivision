from agrivision.config import load_classes
from agrivision.decision.engine import DiseaseResult, PestDetection, build_report, load_kb


def test_kb_covers_every_class():
    kb, c = load_kb(), load_classes()
    for d in c.disease:
        assert d in (kb["healthy"] if d.endswith("_healthy") else kb["diseases"]), d
    assert set(kb["pests"]) == set(c.pest)


def test_disease_with_severity_and_vector_pest():
    r = build_report(
        DiseaseResult("tomato_early_blight", 0.93),
        [PestDetection("whitefly", 0.8, (0, 0, 10, 10)), PestDetection("whitefly", 0.7, (5, 5, 20, 20))],
        severity_percent=18.0,
        severity_level="moderate",
    )
    assert r.crop == "tomato"
    assert "Early Blight" in r.headline
    titles = [i.title for i in r.items]
    assert "Severity: moderate" in titles
    assert "Pest: Whitefly x2" in titles
    assert "Virus risk" in titles
    assert "moderate" in r.spoken and "2 Whitefly" in r.spoken


def test_off_crop_pest_is_flagged():
    r = build_report(DiseaseResult("paddy_healthy", 0.9), [PestDetection("pink_bollworm", 0.6, (0, 0, 1, 1))])
    assert "not a common pest of paddy" in r.items[0].title


def test_nothing_detected():
    r = build_report(None, [])
    assert r.items == [] and "camera" in r.spoken
