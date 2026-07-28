"""Inference pipeline for the trained <SEG> bridge.

Training always supervises the fixed assistant response ("It is <SEG>."), so
inference runs one teacher-forced forward with that response appended and
reads the <SEG> hidden state — no autoregressive generation, which keeps
latency low and exactly matches the training formulation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from voxae.model.qwen_backbone import SEG_TOKEN


class VoxaeSegPipeline:
    """Loads a trained checkpoint and predicts masks for (image, query) pairs."""

    def __init__(self, model, processor, device: str = "cpu"):
        self.model = model.eval()
        self.processor = processor
        self.device = torch.device(device)
        self.model.to(self.device)
        self.name = "voxae-trained"

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: str | Path,
        backbone_id: str,
        sam2_id: str,
        device: str = "cpu",
        dtype: str = "float32",
    ) -> VoxaeSegPipeline:
        """Rebuild the model exactly as trained and load trainable weights."""
        from voxae.model.qwen_backbone import (
            LoraSettings,
            add_seg_token,
            apply_lora,
            load_backbone,
        )
        from voxae.model.sam2_head import Sam2DecoderHead
        from voxae.model.seg_bridge import VoxaeSegModel

        state = torch.load(
            Path(checkpoint_dir) / "state.pt", map_location="cpu", weights_only=False
        )
        cfg = state.get("config", {})

        backbone, processor = load_backbone(backbone_id, dtype=dtype)
        seg_token_id = add_seg_token(processor.tokenizer, backbone)
        # LoRA renames the backbone's parameters, so it has to be reapplied at the
        # same rank before the saved tensors can find their targets.
        if cfg.get("lora", any("lora_" in name for name in state["trainable"])):
            backbone = apply_lora(backbone, LoraSettings(**cfg.get("lora_settings", {})))
        sam_head = Sam2DecoderHead.from_pretrained(sam2_id)
        model = VoxaeSegModel(
            backbone,
            sam_head,
            seg_token_id=seg_token_id,
            llm_hidden=backbone.config.get_text_config().hidden_size,
        )

        current = dict(model.named_parameters())
        missed = [
            name
            for name, tensor in state["trainable"].items()
            if name not in current or current[name].shape != tensor.shape
        ]
        if missed:
            raise RuntimeError(
                f"{len(missed)} of {len(state['trainable'])} saved tensors did not match the "
                f"rebuilt model (first: {missed[0]}). The checkpoint was trained under a "
                "different configuration."
            )
        for name, tensor in state["trainable"].items():
            current[name].data.copy_(tensor)
        return cls(model, processor, device=device)

    def predict(self, image: Image.Image, query: str) -> np.ndarray:
        """(image, query) -> full-resolution boolean mask."""
        return self.predict_logits(image, query) > 0

    @torch.inference_mode()
    def predict_logits(self, image: Image.Image, query: str) -> np.ndarray:
        """(image, query) -> full-resolution mask logits.

        The decision threshold is a display choice, not a model property, so
        callers that want to move it keep the field rather than re-running a
        forward pass per threshold.
        """
        from voxae.train.collate import ASSISTANT_TEMPLATE, vlm_image

        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": query}],
            },
            {"role": "assistant", "content": [{"type": "text", "text": ASSISTANT_TEMPLATE}]},
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False)
        inputs = self.processor(text=[text], images=[vlm_image(image)], return_tensors="pt").to(
            self.device
        )

        size = self.model.sam_head.input_image_size
        sam_px = self._sam_pixels(image, size).unsqueeze(0).to(self.device)
        sam_feats = self.model.sam_head.encode_images(sam_px)

        out = self.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            sam_features=sam_feats,
            **{k: v for k, v in inputs.items() if k not in {"input_ids", "attention_mask"}},
        )
        low_res = out["pred_masks"][0:1].unsqueeze(1).float()
        full = torch.nn.functional.interpolate(
            low_res, size=(image.height, image.width), mode="bilinear", align_corners=False
        )[0, 0]
        return full.cpu().numpy()

    @staticmethod
    def _sam_pixels(image: Image.Image, size: int) -> torch.Tensor:
        img = image.convert("RGB").resize((size, size), Image.BILINEAR)
        arr = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return (arr - mean) / std

    def __contains__(self, token: str) -> bool:
        return token == SEG_TOKEN
