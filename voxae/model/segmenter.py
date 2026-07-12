"""Segmentation backends: pixel-space prompts (bbox + points) -> binary mask.

The real backend is SAM 2.1 via Hugging Face transformers, loaded lazily so
that importing this module never pulls torch (CI stays CPU-light with mocks).
"""

from __future__ import annotations

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
    """SAM 2.1 image segmentation through transformers (CPU-friendly at demo scale)."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.name = f"sam2:{self.settings.sam2_model}"
        self._model: Any = None
        self._processor: Any = None

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
        with torch.inference_mode():
            outputs = self._model(**inputs, multimask_output=False)
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
