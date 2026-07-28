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


class GrounderRequestError(GrounderError):
    """The request itself is wrong (unknown model, oversized payload).

    Separate from GrounderError because retrying an identical bad request only
    burns the retry budget and delays a message the caller needs to see.
    """


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


def _first_json_object(text: str) -> str | None:
    """First balanced {...} in the text, ignoring braces inside strings.

    Models asked for one region sometimes return several, comma-separated. A
    greedy regex spans them all and fails to parse; taking the first complete
    object keeps the answer the prompt asked for.
    """
    start = depth = 0
    in_string = escaped = False
    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


# Models reliably write points as {"x": 900, 280} or {"x": 900, 280], dropping
# the second key. A point has exactly two integer fields, so the reading is
# unambiguous; this is a syntax slip over a valid answer, not a guess about
# intent. Applied only after a strict parse fails, so well-formed output is
# never rewritten.
_BARE_SECOND_COORD = re.compile(r'\{\s*"x"\s*:\s*(-?\d+)\s*,\s*(-?\d+)\s*[}\]]')


def _repair_known_slips(text: str) -> str:
    return _BARE_SECOND_COORD.sub(r'{"x": \1, "y": \2}', text)


_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def extract_json(text: str) -> dict:
    """Pull the first JSON object out of possibly-noisy model output.

    A fence is unwrapped rather than matched against, because models sometimes
    fence a whole array of regions; scanning the body then yields the first
    object instead of a fragment cut at the wrong brace.
    """
    fence = _FENCE.search(text)
    body = fence.group(1) if fence else text
    last_error: Exception | None = None
    for source in (body, _repair_known_slips(body)):
        candidate = _first_json_object(source)
        if candidate is None:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as e:
            last_error = e
    if last_error is None:
        raise GrounderError(f"no JSON object found in model output: {text[:200]!r}")
    raise GrounderError(f"malformed JSON from model: {last_error}: {text[:200]!r}") from last_error


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
                if resp.status_code >= 400:
                    # The status alone is not actionable; the provider names the
                    # unknown model, the size limit, or the quota in the body.
                    detail = f"HTTP {resp.status_code}: {resp.text[:300]}"
                    if resp.status_code == 429 or resp.status_code >= 500:
                        raise GrounderError(detail)
                    raise GrounderRequestError(detail)
                text = resp.json()["choices"][0]["message"]["content"]
                return GroundingResult.model_validate(extract_json(text))
            except GrounderRequestError:
                raise
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
