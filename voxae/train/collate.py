"""Batch assembly for bridge training.

Each sample pairs a chat-formatted conversation (user: image + query,
assistant: an answer ending in <SEG>) with the ground-truth mask decoded from
its RLE. CE labels are masked so only assistant tokens contribute; the mask
loss supervises the <SEG> embedding through the SAM2 decoder.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from voxae.data import rle as rle_codec
from voxae.data.schemas import QuerySample
from voxae.model.qwen_backbone import SEG_TOKEN

ASSISTANT_TEMPLATE = f"It is {SEG_TOKEN}."


def load_samples(jsonl_path: Path, split: str | None = None) -> list[QuerySample]:
    samples = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            s = QuerySample.model_validate_json(line)
            if split is None or s.split == split:
                samples.append(s)
    return samples


def mask_ce_labels(input_ids: torch.Tensor, assistant_start: int) -> torch.Tensor:
    """Labels with everything before the assistant span ignored (-100)."""
    labels = input_ids.clone()
    labels[:assistant_start] = -100
    return labels


@dataclass
class VoxaeCollator:
    """Turns QuerySamples into model inputs using a HF chat processor."""

    processor: object  # AutoProcessor for the backbone
    data_root: Path
    sam_image_size: int = 1024
    max_length: int = 512

    def __call__(self, samples: list[QuerySample]) -> dict:
        images, texts, gt_masks, sam_pixels = [], [], [], []
        for s in samples:
            image = Image.open(self.data_root / s.rel_path).convert("RGB")
            images.append(image)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": s.text},
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": ASSISTANT_TEMPLATE}]},
            ]
            texts.append(self.processor.apply_chat_template(messages, tokenize=False))
            mask = rle_codec.decode(s.rle)
            gt_masks.append(torch.from_numpy(mask.astype(np.float32)))
            sam_pixels.append(self._sam_pixels(image))

        batch = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        batch["gt_masks"] = torch.stack(
            [self._resize_mask(m, self.sam_image_size) for m in gt_masks]
        )
        batch["sam_pixel_values"] = torch.stack(sam_pixels)
        batch["labels"] = batch["input_ids"].clone()
        batch["labels"][batch["attention_mask"] == 0] = -100
        return batch

    def _sam_pixels(self, image: Image.Image) -> torch.Tensor:
        """Square-resize + ImageNet-normalize for the SAM2 encoder."""
        img = image.resize((self.sam_image_size, self.sam_image_size), Image.BILINEAR)
        arr = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return (arr - mean) / std

    @staticmethod
    def _resize_mask(mask: torch.Tensor, size: int) -> torch.Tensor:
        m = mask.unsqueeze(0).unsqueeze(0)
        return torch.nn.functional.interpolate(m, size=(size, size), mode="nearest")[0, 0]
