from agrivision.config import crop_of, load_classes, load_settings


def test_classes_consistent():
    c = load_classes()
    assert len(c.disease) == 25 and len(set(c.disease)) == 25
    assert {crop_of(d) for d in c.disease} == set(c.crops)
    for crop, pests in c.crop_pests.items():
        assert crop in c.crops
        assert set(pests) <= set(c.pest)


def test_settings():
    s = load_settings()
    assert s.severity.healthy < s.severity.mild < s.severity.moderate
    assert abs(s.split.train + s.split.val + s.split.test - 1.0) < 1e-6
