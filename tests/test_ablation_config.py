"""The ablation arms have to differ from the control in one thing each.

An arm that quietly changes two settings measures neither of them, and nothing
in the run itself would say so. These read the shipped YAML rather than a copy,
so a later edit that breaks the invariant fails here.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

CONFIGS = Path(__file__).parent.parent / "voxae" / "train" / "configs"
ABLATIONS = CONFIGS / "ablations"

# The one key each arm is allowed to move, beside its own output directory.
ARMS = {
    "no_ce": "supervise_text",
    "dice_light": "dice_weight",
    "projector_1layer": "projector_layers",
    "sam_frozen": "train_mask_decoder",
}


def load(name: str) -> dict:
    return yaml.safe_load((ABLATIONS / f"{name}.yaml").read_text(encoding="utf-8"))


def test_every_arm_has_a_config():
    found = {p.stem for p in ABLATIONS.glob("*.yaml")}
    assert found == {"base", *ARMS}


@pytest.mark.parametrize(("arm", "key"), sorted(ARMS.items()))
def test_arm_changes_exactly_one_setting(arm: str, key: str):
    base = load("base")
    cfg = load(arm)
    moved = {k for k, v in cfg.items() if base.get(k) != v}
    assert moved == {key, "output_dir"}, f"{arm} also moved {moved - {key, 'output_dir'}}"


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_arm_writes_to_its_own_directory(arm: str):
    assert load(arm)["output_dir"] == f"outputs/ablations/{arm}"


@pytest.mark.parametrize("name", ["base", *sorted(ARMS)])
def test_budget_and_seed_are_shared(name: str):
    """Comparability rests on these three being identical across arms."""
    cfg = load(name)
    assert cfg["max_steps"] == 800
    assert cfg["grad_accum"] == 2
    assert cfg["seed"] == 0


def test_no_ce_withholds_labels_rather_than_zeroing_the_weight():
    """ce_weight 0 leaves a zero gradient on lm_head, which AdamW still decays."""
    cfg = load("no_ce")
    assert cfg["supervise_text"] is False
    assert cfg["ce_weight"] == 1.0


def test_full_run_is_seeded():
    cfg = yaml.safe_load((CONFIGS / "full_2b.yaml").read_text(encoding="utf-8"))
    assert cfg["seed"] == 0
