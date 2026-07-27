"""End-to-end bridge shape/backward tests on tiny random-init models.

No weight downloads: SAM2 from a default random config, the LLM from a tiny
Qwen2 text-only config. Slow-ish on CPU (~tens of seconds) but fully offline.
"""

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

import torch

from voxae.model.sam2_head import Sam2DecoderHead
from voxae.model.seg_bridge import SegProjector, VoxaeSegModel

SEG = 5


@pytest.fixture(scope="module")
def sam_head():
    return Sam2DecoderHead.from_config()


@pytest.fixture(scope="module")
def tiny_lm():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    config = Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    return Qwen2ForCausalLM(config)


def test_projector_maps_to_prompt_dim(sam_head):
    proj = SegProjector(llm_hidden=32, prompt_dim=sam_head.prompt_dim)
    out = proj(torch.randn(3, 32))
    assert out.shape == (3, sam_head.prompt_dim)


def test_projector_accepts_lower_precision_input(sam_head):
    """A bfloat16 backbone feeds a full-precision projector."""
    proj = SegProjector(llm_hidden=32, prompt_dim=sam_head.prompt_dim)
    out = proj(torch.randn(3, 32, dtype=torch.bfloat16))
    assert out.shape == (3, sam_head.prompt_dim)


def test_encode_decode_shapes(sam_head):
    size = sam_head.input_image_size
    feats = sam_head.encode_images(torch.randn(2, 3, size, size))
    assert len(feats) == 3
    assert all(f.shape[0] == 2 for f in feats)

    pred = sam_head.decode(feats, torch.randn(2, sam_head.prompt_dim))
    assert pred.ndim == 3 and pred.shape[0] == 2


def test_full_model_losses_and_backward(sam_head, tiny_lm):
    model = VoxaeSegModel(tiny_lm, sam_head, seg_token_id=SEG, llm_hidden=32)
    size = sam_head.input_image_size
    with torch.no_grad():
        feats = sam_head.encode_images(torch.randn(2, 3, size, size))

    ids = torch.tensor([[1, 2, SEG, 3], [4, SEG, 2, 1]])
    attn = torch.ones_like(ids)
    gt = torch.zeros(2, 64, 64)
    gt[:, :32] = 1.0

    out = model(ids, attn, feats, gt_masks=gt, labels=ids.clone())
    assert {"loss", "loss_ce", "loss_bce", "loss_dice", "loss_mask"} <= set(out)
    assert out["pred_masks"].shape[0] == 2

    out["loss"].backward()
    grads = [p.grad for p in model.projector.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)


def test_mask_only_forward_no_labels(sam_head, tiny_lm):
    model = VoxaeSegModel(tiny_lm, sam_head, seg_token_id=SEG, llm_hidden=32)
    size = sam_head.input_image_size
    with torch.no_grad():
        feats = sam_head.encode_images(torch.randn(1, 3, size, size))
    ids = torch.tensor([[1, SEG]])
    out = model(ids, torch.ones_like(ids), feats)
    assert "loss_ce" not in out
    assert out["loss"].item() == 0.0
