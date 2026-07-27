"""Frozen SAM 2.1 vision encoder + mask decoder, adapted for bridge training.

The reasoning-seg model injects a projected LLM embedding as an EXTRA sparse
prompt token (LISA-style); everything SAM-side stays frozen. This module
replicates transformers' Sam2Model.forward composition so the decoder can be
driven with custom sparse embeddings:

    encode_images: pixel_values -> [fpn0, fpn1, final] feature maps
                   (no_memory_embedding added, reshaped to spatial maps)
    decode:        features + bridge embedding -> low-res mask logits
"""

from __future__ import annotations

import torch
from torch import nn


class Sam2DecoderHead(nn.Module):
    """Wraps a (frozen) transformers Sam2Model for prompt-injected decoding."""

    def __init__(self, sam2_model, train_mask_decoder: bool = False):
        super().__init__()
        self.sam = sam2_model
        self.sam.requires_grad_(False)
        if train_mask_decoder:
            self.sam.mask_decoder.requires_grad_(True)
        self.train_mask_decoder = train_mask_decoder

    @classmethod
    def from_pretrained(cls, model_id: str, train_mask_decoder: bool = False, **kw):
        from transformers import Sam2Model

        return cls(Sam2Model.from_pretrained(model_id, **kw), train_mask_decoder)

    @classmethod
    def from_config(cls, config=None, train_mask_decoder: bool = False):
        """Random-init head for offline tests (no weight download)."""
        from transformers import Sam2Config, Sam2Model

        return cls(Sam2Model(config or Sam2Config()), train_mask_decoder)

    @property
    def prompt_dim(self) -> int:
        return self.sam.config.mask_decoder_config.hidden_size

    @property
    def input_image_size(self) -> int:
        """Expected square input size; the finest FPN level has stride 4."""
        return int(self.sam.config.vision_config.backbone_feature_sizes[0][0]) * 4

    @torch.no_grad()
    def encode_images(self, pixel_values: torch.Tensor) -> list[torch.Tensor]:
        """Pixel values -> the 3 FPN feature maps the decoder consumes.

        Mirrors Sam2Model.forward: fpn_hidden_states + no_memory_embedding on
        the last level, reshaped from (seq, B, C) to (B, C, H, W).
        """
        batch = pixel_values.shape[0]
        image_outputs = self.sam.get_image_features(pixel_values, return_dict=True)
        feature_maps = list(image_outputs.fpn_hidden_states)
        feature_maps[-1] = feature_maps[-1] + self.sam.no_memory_embedding
        return [
            feat.permute(1, 2, 0).view(batch, -1, *feat_size)
            for feat, feat_size in zip(feature_maps, self.sam.backbone_feature_sizes, strict=True)
        ]

    def _empty_prompts(self, batch: int, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """Baseline sparse/dense embeddings for a no-prompt decode (padded point)."""
        points = torch.zeros(batch, 1, 1, 2, dtype=dtype, device=device)
        labels = -torch.ones(batch, 1, 1, dtype=torch.int32, device=device)
        return self.sam.prompt_encoder(
            input_points=points, input_labels=labels, input_boxes=None, input_masks=None
        )

    def decode(
        self,
        image_features: list[torch.Tensor],
        bridge_embeddings: torch.Tensor,  # (B, prompt_dim) projected LLM states
    ) -> torch.Tensor:
        """Decode one mask per sample, prompted by the bridge embedding.

        Returns low-resolution mask logits (B, H_low, W_low).
        """
        batch = bridge_embeddings.shape[0]
        device, dtype = bridge_embeddings.device, image_features[-1].dtype

        sparse, dense = self._empty_prompts(batch, device, dtype)
        bridge_token = bridge_embeddings.to(dtype).view(batch, 1, 1, -1)
        sparse = torch.cat([sparse, bridge_token], dim=2)

        pos = self.sam.get_image_wide_positional_embeddings().repeat(batch, 1, 1, 1)
        low_res_masks, _iou, _, _obj = self.sam.mask_decoder(
            image_embeddings=image_features[-1],
            image_positional_embeddings=pos.to(device=device, dtype=dtype),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
            high_resolution_features=image_features[:-1],
        )
        return low_res_masks[:, 0, 0]
