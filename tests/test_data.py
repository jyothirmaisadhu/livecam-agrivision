import cv2
import numpy as np

from agrivision.data.images import average_hash
from agrivision.data.make_masks import LEAF, LESION, draft_mask
from agrivision.data.prepare_yolo import remap_label_lines
from agrivision.data.split import split_items


def test_split_deterministic_and_complete():
    items = list(range(100))
    a = split_items(items, 0.7, 0.15, seed=1)
    b = split_items(items, 0.7, 0.15, seed=1)
    assert a == b
    assert [len(x) for x in a] == [70, 15, 15]
    assert sorted(a[0] + a[1] + a[2]) == items


def test_average_hash_ignores_tiny_changes(leaf):
    img = leaf(seed=0)
    assert average_hash(img) == average_hash(cv2.GaussianBlur(img, (3, 3), 0))
    assert average_hash(img) != average_hash(np.rot90(leaf(lesion_radius=30)).copy())


def test_remap_label_lines():
    lines = ["0 0.5 0.5 0.1 0.1", "1 0.2 0.2 0.1 0.1", "bad line"]
    out = remap_label_lines(
        lines, ["whitefly", "ant"], {"whitefly": "whitefly"}, {"aphids": 0, "whitefly": 6}
    )
    assert out == ["6 0.5 0.5 0.1 0.1"]


def test_draft_mask_finds_leaf_and_lesion(leaf):
    mask = draft_mask(leaf(size=128, lesion_radius=12))
    assert np.count_nonzero(mask == LEAF) > 1500
    lesion = np.count_nonzero(mask == LESION)
    assert 0.5 * np.pi * 12**2 < lesion < 1.5 * np.pi * 12**2
    assert mask[64, 64] == LESION and mask[0, 0] == 0
