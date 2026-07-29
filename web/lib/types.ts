/** Mirrors the payload from `api_predict` in voxae/demo/gradio_app.py. */

export type TrainedResult = {
  model: string;
  /** 8-bit greyscale PNG data URI: the probability field, not a decision. */
  probs_png: string;
  /** 1-bit PNG data URI at the server's chosen threshold. */
  mask_png: string;
  area_pct: number;
  latency_s: number;
  threshold: number;
  mean_confidence_in_mask: number;
};

export type BaselineResult = {
  model: string;
  mask_png: string;
  area_pct: number;
  latency_s: number;
  bbox_norm: [number, number, number, number];
  rationale: string;
  /** False when the Space has no API key and is returning mock grounding. */
  live: boolean;
};

export type GroundTruth = {
  sample_id: string;
  family: string;
  mask_png: string;
  trained_iou: number | null;
  baseline_iou: number | null;
};

export type Prediction = {
  query: string;
  image: { width: number; height: number };
  trained: TrainedResult | null;
  baseline: BaselineResult | null;
  ground_truth: GroundTruth | null;
  agreement_iou: number | null;
  /** Present when the third-party grounding API failed but the bridge did not. */
  baseline_error?: string;
  /** Present when the caller asked for the bridge alone. */
  baseline_skipped?: boolean;
};

export type LayerKey = "trained" | "baseline" | "ground_truth";
