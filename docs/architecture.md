# Architecture

## v1

```
             ┌────────────────────────────┐
 image ─────▶│ Grounder (protocol)        │   hosted Qwen2.5-VL via any
 query ─────▶│  QwenAPIGrounder | Mock    │   OpenAI-compatible endpoint
             └─────────────┬──────────────┘
                           │ GroundingResult (bbox + points, 0-1000 norm)
                           ▼
             ┌────────────────────────────┐
             │ Segmenter (protocol)       │   SAM 2.1 via transformers,
             │  Sam2Segmenter | Mock      │   CPU-friendly at demo scale
             └─────────────┬──────────────┘
                           │ HxW bool mask
                           ▼
              overlay + ZeroShotTrace (audit record)
```

Design rules:
- **Protocols + mocks everywhere** — CI runs with zero network, zero weights.
- **Lazy heavy imports** — importing any module never pulls torch; the `ml`
  extra is only needed to *run* the real segmenter.
- **Structured outputs only** — the VLM must return schema-validated JSON
  (normalized 0–1000 coords survive model-side image resizing).
