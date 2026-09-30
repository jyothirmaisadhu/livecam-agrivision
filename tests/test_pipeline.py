from agrivision.pipeline.smoothing import MajorityVote


def test_majority_vote():
    v = MajorityVote(5)
    assert v.update("a") == "a"
    assert v.update("b") is None
    assert v.update("a") == "a"
    for _ in range(3):
        v.update("b")
    assert v.update("b") == "b"
