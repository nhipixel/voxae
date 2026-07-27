"""The <SEG> bridge: LLM hidden state -> SAM2 prompt space -> mask.

Trainables: the projector (always), LoRA adapters on the LLM (when applied),
and optionally SAM2's mask decoder. Vision encoders stay frozen. Loss is
LM cross-entropy (from the LM head, when labels are given) plus BCE + Dice
on the decoded mask against ground truth.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from voxae.model.losses import mask_loss
from voxae.model.qwen_backbone import seg_hidden_states
from voxae.model.sam2_head import Sam2DecoderHead


class SegProjector(nn.Module):
    """Two-layer MLP from LLM hidden size into SAM2's prompt-embedding space."""

    def __init__(self, llm_hidden: int, prompt_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(llm_hidden, llm_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(llm_hidden, prompt_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VoxaeSegModel(nn.Module):
    """LLM backbone + projector + SAM2 decoder head, trained end to end."""

    def __init__(
        self,
        backbone: nn.Module,
        sam_head: Sam2DecoderHead,
        seg_token_id: int,
        llm_hidden: int,
        ce_weight: float = 1.0,
        bce_weight: float = 2.0,
        dice_weight: float = 0.5,
    ):
        super().__init__()
        self.backbone = backbone
        self.sam_head = sam_head
        self.projector = SegProjector(llm_hidden, sam_head.prompt_dim)
        self.seg_token_id = seg_token_id
        self.ce_weight = ce_weight
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        sam_features: list[torch.Tensor],
        gt_masks: torch.Tensor | None = None,  # (B, H, W) float {0,1}
        labels: torch.Tensor | None = None,
        **backbone_inputs,
    ) -> dict[str, torch.Tensor]:
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            **backbone_inputs,
        )
        seg_states = seg_hidden_states(out.hidden_states[-1], input_ids, self.seg_token_id)
        pred = self.sam_head.decode(sam_features, self.projector(seg_states))

        result: dict[str, torch.Tensor] = {"pred_masks": pred}
        total = torch.zeros((), device=pred.device, dtype=pred.dtype)
        if labels is not None and out.loss is not None:
            result["loss_ce"] = out.loss
            total = total + self.ce_weight * out.loss
        if gt_masks is not None:
            target = gt_masks.unsqueeze(1).float()
            target = F.interpolate(target, size=pred.shape[-2:], mode="nearest")[:, 0]
            losses = mask_loss(pred.float(), target, self.bce_weight, self.dice_weight)
            result.update(losses)
            total = total + losses["loss_mask"]
        result["loss"] = total
        return result

    def trainable_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)
