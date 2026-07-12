# Voxae

**Language-grounded, metric 3D scene understanding for physical AI.**

Ask a 3D scene *what's where, how big, and how far* — grounded in real-world measurements.
A perception primitive for embodied agents (drones, robots, autonomous systems); demonstrated on drone-captured scenes.

[![CI](https://github.com/nhipixel/voxae/actions/workflows/ci.yml/badge.svg)](https://github.com/nhipixel/voxae/actions)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

> **Status:** live zero-shot demo. The trained model and the metric 3D lift come next. Build log: TBD.

## What it does

```
"highlight everything that would block a 2.5 m-wide fire truck"
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│ v0  reasoning segmentation:  VLM ──<SEG>──▶ SAM 2.1 ──▶ mask │
│ v1  metric 3D lift:  masks ──▶ Gaussian splats ──▶ browser   │
│     (query → 3D highlight, click → distance in meters)       │
└─────────────────────────────────────────────────────────────┘
```

Three query families, one system:
- **Referring** — "the building with the red roof"
- **Affordance** — "where could a small drone land safely?"
- **Metric** — "is this gap wide enough for a 2.5 m vehicle?" *(answers computed from ground-sample distance)*
## Architecture (v0 — zero-shot baseline)

`image + query → Qwen2.5-VL (structured grounding JSON) → SAM 2.1 → mask`

The baseline is deliberately decoupled (Seg-Zero-style) and runs with **zero GPUs** — hosted VLM + CPU SAM. It is the floor every trained checkpoint must beat. Training the bridge replaces the text-coordinate handoff with a learned `<SEG>`-token bridge: the LLM's hidden state is projected directly into SAM's prompt space, and the bridge is trained end-to-end (QLoRA) on Voxae-Reason (datasheet: TBD).

## Quickstart

```bash
git clone https://github.com/nhipixel/voxae && cd voxae
uv sync --extra dev                 # core + tests (CPU, no weights)
uv run pytest                       # verify
cp .env.example .env                # add VOXAE_VLM_API_KEY for live grounding
uv run voxae segment path/to/aerial.jpg "where could a drone land?" --mock
uv sync --extra ml --extra demo     # full local demo
uv run python app.py                # Gradio UI
```

## Honest novelty

Reasoning segmentation exists (LISA), including for satellites (SegEarth-R1) and drones (PixDLM/DRSeg, RIS-LAD). What Voxae adds: **metric-grounded queries** (answers depend on physical dimensions computed from GSD), a **fully-open, single-GPU-budget reproducible pipeline**, and the **metric 3D lift** for outdoor scenes. Full prior-art map with citations: TBD.

## License

Code: [Apache-2.0](LICENSE). Dataset annotations: CC BY-NC-SA 4.0 (inherits non-commercial seed-data licenses — see the datasheet). Demo gallery images: openly licensed (CC0/CC BY), attribution in `voxae/demo/assets/ATTRIBUTION.md`.
