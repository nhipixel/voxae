"""iter_samples pairing strategies against synthetic directory trees."""

from dataclasses import replace

import pytest

from voxae.data.registry import REGISTRY, PairingError, iter_samples


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x00")


def test_mirrored_pairing_disambiguates_repeated_stems_by_path(tmp_path):
    # Two sequences both have a frame "000" — mirrored pairing must not cross-pair them.
    _touch(tmp_path / "seq1" / "Images" / "000.png")
    _touch(tmp_path / "seq1" / "Labels" / "000.png")
    _touch(tmp_path / "seq2" / "Images" / "000.png")
    _touch(tmp_path / "seq2" / "Labels" / "000.png")

    spec = REGISTRY["uavid"]  # mirrored, Images/Labels hints
    pairs = {(img.parent.parent.name, img.stem) for img, _ in iter_samples(spec, tmp_path)}
    assert pairs == {("seq1", "000"), ("seq2", "000")}
    for img, mask in iter_samples(spec, tmp_path):
        assert img.parent.parent == mask.parent.parent  # same sequence, not cross-paired


def test_stem_pairing_handles_asymmetric_depth(tmp_path):
    # Mirrors the real Kaggle SDD layout: images/ is flat, masks are nested
    # three levels deeper under gt/semantic/label_images/.
    _touch(tmp_path / "semantic_drone_dataset" / "training_set" / "images" / "000.jpg")
    _touch(
        tmp_path
        / "semantic_drone_dataset"
        / "training_set"
        / "gt"
        / "semantic"
        / "label_images"
        / "000.png"
    )

    spec = REGISTRY["sdd"]
    assert spec.pairing == "stem"
    pairs = list(iter_samples(spec, tmp_path))
    assert len(pairs) == 1
    img, mask = pairs[0]
    assert img.name == "000.jpg"
    assert mask.name == "000.png"


def test_stem_pairing_raises_on_duplicate_stems(tmp_path):
    _touch(tmp_path / "a" / "label_images" / "000.png")
    _touch(tmp_path / "b" / "label_images" / "000.png")
    _touch(tmp_path / "images" / "000.jpg")

    spec = REGISTRY["sdd"]
    with pytest.raises(PairingError, match="duplicate mask stem"):
        list(iter_samples(spec, tmp_path))


def test_mirrored_pairing_requires_symmetric_trees(tmp_path):
    # Sanity check that mirrored mode genuinely needs symmetry: an
    # asymmetric tree under a mirrored spec finds nothing (not an error —
    # just zero pairs, which is what prepare() surfaces to the user).
    _touch(tmp_path / "training_set" / "images" / "000.jpg")
    _touch(tmp_path / "training_set" / "gt" / "semantic" / "label_images" / "000.png")

    mirrored_sdd = replace(REGISTRY["sdd"], pairing="mirrored")
    assert list(iter_samples(mirrored_sdd, tmp_path)) == []
