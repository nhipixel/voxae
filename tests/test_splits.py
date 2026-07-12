"""Split assignment: stratification, determinism, no image leakage."""

from voxae.data.splits import make_splits


def test_splits_are_deterministic():
    ids = {"ds": [f"img{i}" for i in range(50)]}
    assert make_splits(ids, seed=7) == make_splits(ids, seed=7)
    assert make_splits(ids, seed=7) != make_splits(ids, seed=8)


def test_split_proportions_per_dataset():
    ids = {"a": [f"a{i}" for i in range(100)], "b": [f"b{i}" for i in range(20)]}
    assignment = make_splits(ids, ratios=(0.8, 0.1, 0.1), seed=0)
    for ds, n in (("a", 100), ("b", 20)):
        counts = {"train": 0, "val": 0, "test": 0}
        for image_id, split in assignment.items():
            if image_id.startswith(ds[0]):
                counts[split] += 1
        assert counts["train"] == round(n * 0.8)
        assert counts["val"] == round(n * 0.1)
        assert sum(counts.values()) == n


def test_every_image_assigned_exactly_once():
    ids = {"ds": [f"img{i}" for i in range(13)]}
    assignment = make_splits(ids)
    assert len(assignment) == 13
    assert set(assignment.values()) <= {"train", "val", "test"}


def test_bad_ratios_raise():
    import pytest

    with pytest.raises(ValueError, match="sum to 1"):
        make_splits({"ds": ["a"]}, ratios=(0.5, 0.2, 0.2))
