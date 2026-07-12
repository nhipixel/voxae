"""Grounding backends: query + image -> structured region proposal (bbox + points).

Uses a hosted VLM through any OpenAI-compatible chat API so the pipeline
needs no local GPU. Tests use MockGrounder.
"""

from __future__ import annotations

import base64
import io
import json
import re
import time
from typing import Protocol

import httpx
from PIL import Image

from voxae.config import Settings, get_settings
from voxae.data.schemas import NORM_MAX, BBoxNorm, GroundingResult, PointNorm


class GrounderError(RuntimeError):
    """Raised when a grounding backend cannot produce a valid result."""


class Grounder(Protocol):
    name: str

    def ground(self, image: Image.Image, query: str) -> GroundingResult: ...


SYSTEM_PROMPT = (
    "You are a precise visual grounding assistant for aerial and outdoor scenes. "
    "Given an image and a query, locate the SINGLE region that best answers the query. "
    "Coordinates use a normalized space where the image spans 0-1000 in both axes "
    "(x right, y down). Respond with ONLY a JSON object, no code fences, matching: "
    + GroundingResult.json_schema_prompt()
    + " Points must lie INSIDE the bbox, on the target region."
)


def extract_json(text: str) -> dict:
    """Pull the first JSON object out of possibly-noisy model output."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        brace = re.search(r"\{.*\}", text, re.DOTALL)
        if brace is None:
            raise GrounderError(f"no JSON object found in model output: {text[:200]!r}")
        candidate = brace.group(0)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as e:
        raise GrounderError(f"malformed JSON from model: {e}: {candidate[:200]!r}") from e


def _image_to_data_uri(image: Image.Image, max_px: int = 1536) -> str:
    img = image.convert("RGB")
    if max(img.size) > max_px:
        img.thumbnail((max_px, max_px))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


class QwenAPIGrounder:
    """Grounding via a hosted VLM behind an OpenAI-compatible /chat/completions API."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        if not self.settings.vlm_api_key:
            raise GrounderError("VOXAE_VLM_API_KEY is not set — configure .env or Space secrets")
        self.name = f"api:{self.settings.vlm_model}"

    def ground(self, image: Image.Image, query: str) -> GroundingResult:
        payload = {
            "model": self.settings.vlm_model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": _image_to_data_uri(image)}},
                        {"type": "text", "text": f"Query: {query}"},
                    ],
                },
            ],
            "temperature": 0.1,
        }
        headers = {"Authorization": f"Bearer {self.settings.vlm_api_key}"}
        last_err: Exception | None = None
        for _attempt in range(1 + self.settings.vlm_max_retries):
            try:
                resp = httpx.post(
                    f"{self.settings.vlm_base_url.rstrip('/')}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=self.settings.vlm_timeout_s,
                )
                resp.raise_for_status()
                text = resp.json()["choices"][0]["message"]["content"]
                return GroundingResult.model_validate(extract_json(text))
            except (httpx.HTTPError, KeyError, GrounderError, ValueError) as e:
                last_err = e
                time.sleep(0.5)
        raise GrounderError(f"grounding failed after retries: {last_err}") from last_err


class MockGrounder:
    """Deterministic grounder for tests and keyless demo mode: centered box."""

    name = "mock"

    def ground(self, image: Image.Image, query: str) -> GroundingResult:
        q = NORM_MAX // 4
        return GroundingResult(
            bbox=BBoxNorm(x1=q, y1=q, x2=NORM_MAX - q, y2=NORM_MAX - q),
            points=[PointNorm(x=NORM_MAX // 2, y=NORM_MAX // 2)],
            rationale="mock grounder: centered region",
        )
