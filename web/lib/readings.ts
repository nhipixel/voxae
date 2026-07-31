/**
 * Cached readings: real predictions captured once from the deployed model.
 *
 * A visitor should see a complete scored reading the moment they arrive, not
 * after a GPU allocation. Each file under /readings is the verbatim response
 * of the live endpoint for one worked example, stamped with when it was
 * captured. "Read the scene" always reruns live; this is the zero-second
 * first impression, and it keeps the demo standing when the GPU quota is
 * spent.
 */

import type { Prediction } from "./types";

export type CachedReading = {
  captured_at: string;
  captured_from: string;
  prediction: Prediction;
};

/** sample_id per example query; grown by scripts/capture_readings.mjs. */
export const READING_IDS: Record<string, string> = {
  "Identify all ground surfaces, paved or otherwise cleared, that a heavy vehicle could drive across.":
    "uavid-000900-q04-s0",
};

export async function loadReading(query: string): Promise<CachedReading | null> {
  const id = READING_IDS[query.trim()];
  if (!id) return null;
  try {
    const res = await fetch(`/readings/${id}.json`);
    if (!res.ok) return null;
    return (await res.json()) as CachedReading;
  } catch {
    return null;
  }
}
