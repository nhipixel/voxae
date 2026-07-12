"""Zero-shot reasoning-segmentation baseline (Seg-Zero-style, decoupled).

    image + query --VLM--> {bbox, points, rationale} --SAM2--> mask

This is the baseline every trained model must beat, and it is fully
GPU-optional: hosted VLM + CPU SAM2.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from PIL import Image

from voxae.data.schemas import ZeroShotTrace
from voxae.model.grounder import Grounder
from voxae.model.segmenter import Segmenter


@dataclass
class ZeroShotResult:
    mask: np.ndarray  # HxW bool
    trace: ZeroShotTrace


class ZeroShotPipeline:
    def __init__(self, grounder: Grounder, segmenter: Segmenter):
        self.grounder = grounder
        self.segmenter = segmenter

    def run(self, image: Image.Image, query: str) -> ZeroShotResult:
        query = query.strip()
        if not query:
            raise ValueError("query must be non-empty")
        w, h = image.size

        t0 = time.perf_counter()
        grounding = self.grounder.ground(image, query)
        t1 = time.perf_counter()

        bbox_px = grounding.bbox.to_pixels(w, h)
        points_px = [p.to_pixels(w, h) for p in grounding.points]
        mask = self.segmenter.segment(image, bbox_px, points_px)
        t2 = time.perf_counter()

        trace = ZeroShotTrace(
            query=query,
            model=f"{self.grounder.name} + {self.segmenter.name}",
            grounding=grounding,
            image_width=w,
            image_height=h,
            vlm_latency_ms=(t1 - t0) * 1000,
            sam_latency_ms=(t2 - t1) * 1000,
            total_latency_ms=(t2 - t0) * 1000,
        )
        return ZeroShotResult(mask=mask, trace=trace)
