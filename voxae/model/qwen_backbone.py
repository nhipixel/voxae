"""LLM backbone utilities: <SEG> token management, LoRA, and loading.

The backbone is any causal (V)LM whose final hidden states can be read at
<SEG> token positions. Quantization and LoRA are opt-in so the same builder
serves GPU training (4-bit + LoRA) and CPU tests (tiny random-init config).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

SEG_TOKEN = "<SEG>"


@dataclass
class LoraSettings:
    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )


def add_seg_token(tokenizer, model) -> int:
    """Register <SEG> in the vocabulary and resize embeddings; returns its id."""
    added = tokenizer.add_special_tokens({"additional_special_tokens": [SEG_TOKEN]})
    if added:
        model.resize_token_embeddings(len(tokenizer))
    return tokenizer.convert_tokens_to_ids(SEG_TOKEN)


def seg_hidden_states(
    hidden_states: torch.Tensor,  # (B, T, d) final layer
    input_ids: torch.Tensor,  # (B, T)
    seg_token_id: int,
) -> torch.Tensor:
    """Hidden state at each sample's LAST <SEG> position -> (B, d).

    Samples without a <SEG> token raise: the collator always appends one, so
    absence indicates a data bug rather than a condition to paper over.
    """
    mask = input_ids == seg_token_id
    if not bool(mask.any(dim=1).all()):
        raise ValueError("every sample must contain a <SEG> token")
    # index of the last occurrence per row
    positions = torch.arange(input_ids.shape[1], device=input_ids.device)
    last = torch.where(mask, positions, torch.full_like(positions, -1)).max(dim=1).values
    return hidden_states[torch.arange(hidden_states.shape[0]), last]


def apply_lora(model, settings: LoraSettings):
    """Wrap the LM with PEFT LoRA; embeddings/lm_head stay trainable for <SEG>."""
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=settings.r,
        lora_alpha=settings.alpha,
        lora_dropout=settings.dropout,
        target_modules=list(settings.target_modules),
        modules_to_save=["embed_tokens", "lm_head"],
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, config)


def load_backbone(
    model_id: str,
    load_in_4bit: bool = False,
    dtype: str = "bfloat16",
):
    """Load a Qwen2/2.5-VL conditional-generation backbone + its processor."""
    from transformers import AutoModelForImageTextToText, AutoProcessor

    kwargs: dict = {"dtype": getattr(torch, dtype)}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=getattr(torch, dtype),
        )
    model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor
