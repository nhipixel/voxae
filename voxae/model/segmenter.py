"""Segmentation backends: pixel-space prompts (bbox + points) -> binary mask.

The real backend is SAM 2.1 via Hugging Face transformers, loaded lazily so
that importing this module never pulls torch (CI stays CPU-light with mocks).
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import Any, Protocol

import numpy as np
from PIL import Image

from voxae.config import Settings, get_settings


class SegmenterError(RuntimeError):
    """Raised when a segmentation backend fails."""


class Segmenter(Protocol):
    name: str

    def segment(
        self,
        image: Image.Image,
        bbox_px: tuple[float, float, float, float],
        points_px: list[tuple[float, float]],
    ) -> np.ndarray: ...


class Sam2Segmenter:
    """SAM 2.1 image segmentation through transformers (CPU-friendly at demo scale).

    The vision encoder dominates the cost and depends only on the image, so
    its output is cached: asking several questions about one photo, which is
    what the demo invites, pays for the encoder once.
    """

    def __init__(self, settings: Settings | None = None, cache_size: int = 4):
        self.settings = settings or get_settings()
        self.name = f"sam2:{self.settings.sam2_model}"
        self._model: Any = None
        self._processor: Any = None
        self._cache_size = cache_size
        self._embeddings: OrderedDict[str, list] = OrderedDict()

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch  # noqa: F401
            from transformers import Sam2Model, Sam2Processor
        except ImportError as e:  # pragma: no cover - environment-specific
            raise SegmenterError(
                f"SAM2 backend requires the 'ml' extra: uv sync --extra ml (import failed: {e})"
            ) from e
        self._processor = Sam2Processor.from_pretrained(self.settings.sam2_model)
        self._model = Sam2Model.from_pretrained(self.settings.sam2_model)
        self._model.to(self.settings.device).eval()

    @staticmethod
    def _cache_key(image: Image.Image) -> str:
        return hashlib.md5(image.tobytes()).hexdigest()  # identity, not security

    def _image_embeddings(self, image: Image.Image, pixel_values) -> list:
        """Encoder output for this image, computed once and reused.

        Mirrors what Sam2Model.forward does internally before prompting, so the
        cached value is exactly what it would have produced.
        """
        key = self._cache_key(image)
        if key in self._embeddings:
            self._embeddings.move_to_end(key)
            return self._embeddings[key]

        import torch

        with torch.inference_mode():
            features = self._model.get_image_features(pixel_values, return_dict=True)
        maps = list(features.fpn_hidden_states)
        maps[-1] = maps[-1] + self._model.no_memory_embedding
        embeddings = [
            feat.permute(1, 2, 0).view(pixel_values.shape[0], -1, *size)
            for feat, size in zip(maps, self._model.backbone_feature_sizes, strict=True)
        ]

        self._embeddings[key] = embeddings
        if len(self._embeddings) > self._cache_size:
            self._embeddings.popitem(last=False)
        return embeddings

    def segment(
        self,
        image: Image.Image,
        bbox_px: tuple[float, float, float, float],
        points_px: list[tuple[float, float]],
    ) -> np.ndarray:
        self._load()
        import torch

        img = image.convert("RGB")
        input_points = [[[list(p) for p in points_px]]]  # batch x obj x pts x 2
        input_labels = [[[1] * len(points_px)]]
        input_boxes = [[list(bbox_px)]]
        inputs = self._processor(
            images=img,
            input_points=input_points,
            input_labels=input_labels,
            input_boxes=input_boxes,
            return_tensors="pt",
        ).to(self.settings.device)

        # forward() accepts embeddings in place of pixels, and rejects both.
        embeddings = self._image_embeddings(img, inputs["pixel_values"])
        with torch.inference_mode():
            outputs = self._model(
                image_embeddings=embeddings,
                input_points=inputs["input_points"],
                input_labels=inputs["input_labels"],
                input_boxes=inputs["input_boxes"],
                multimask_output=False,
            )
        masks = self._processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"]
        )[0]
        mask = masks[0, 0].numpy() > 0.5
        return mask.astype(bool)


class MockSegmenter:
    """Deterministic segmenter for tests/keyless demo: fills the prompt bbox."""

    name = "mock"

    def segment(
        self,
        image: Image.Image,
        bbox_px: tuple[float, float, float, float],
        points_px: list[tuple[float, float]],
    ) -> np.ndarray:
        w, h = image.size
        mask = np.zeros((h, w), dtype=bool)
        x1, y1, x2, y2 = (round(v) for v in bbox_px)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        mask[y1:y2, x1:x2] = True
        return mask
