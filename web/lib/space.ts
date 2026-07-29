/**
 * Client for the Space's `gr.api` endpoint, over plain HTTP.
 *
 *   1. POST /gradio_api/upload            -> a server side path for the file
 *   2. POST /gradio_api/call/predict      -> an event id
 *   3. GET  /gradio_api/call/predict/{id} -> SSE, terminated by `complete`
 *
 * Server side only, so the token stays out of the browser.
 */

import type { Prediction } from "./types";

const BASE = (process.env.VOXAE_SPACE_URL ?? "https://nhibuilds-voxae.hf.space").replace(/\/$/, "");

export class SpaceError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly kind: "quota" | "upstream" | "protocol" = "upstream",
  ) {
    super(message);
  }
}

function authHeaders(): HeadersInit {
  const token = process.env.HF_TOKEN;
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function upload(image: Blob, filename: string): Promise<string> {
  const form = new FormData();
  form.append("files", image, filename);

  const res = await fetch(`${BASE}/gradio_api/upload`, {
    method: "POST",
    headers: authHeaders(),
    body: form,
  });
  if (!res.ok) {
    throw new SpaceError(`upload failed (${res.status})`, 502);
  }
  const paths = (await res.json()) as string[];
  if (!paths?.length) throw new SpaceError("upload returned no path", 502, "protocol");
  return paths[0];
}

/** Reads the SSE stream to its `complete` event, or throws what it reports. */
async function awaitResult(eventId: string, signal?: AbortSignal): Promise<Prediction> {
  const res = await fetch(`${BASE}/gradio_api/call/predict/${eventId}`, {
    headers: authHeaders(),
    signal,
  });
  if (!res.ok || !res.body) {
    throw new SpaceError(`stream failed (${res.status})`, 502);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let event = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // SSE frames are newline-delimited; keep any partial line for the next read.
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";

    for (const line of lines) {
      if (line.startsWith("event:")) {
        event = line.slice(6).trim();
      } else if (line.startsWith("data:")) {
        const raw = line.slice(5).trim();
        if (event === "complete") {
          reader.cancel();
          const parsed = JSON.parse(raw) as [Prediction];
          return parsed[0];
        }
        if (event === "error") {
          reader.cancel();
          let message = raw;
          try {
            message = (JSON.parse(raw)?.error as string) ?? raw;
          } catch {
            /* not JSON: the raw line is the best message available */
          }
          const quota = /quota/i.test(message);
          throw new SpaceError(message, quota ? 429 : 502, quota ? "quota" : "upstream");
        }
      }
    }
  }
  throw new SpaceError("stream ended without a result", 502, "protocol");
}

export const DEFAULT_THRESHOLD = 0.5;

export async function predict(
  image: Blob,
  filename: string,
  query: string,
  signal?: AbortSignal,
  threshold: number = DEFAULT_THRESHOLD,
): Promise<Prediction> {
  const path = await upload(image, filename);

  const started = await fetch(`${BASE}/gradio_api/call/predict`, {
    method: "POST",
    headers: { ...authHeaders(), "Content-Type": "application/json" },
    // All three values are required; omitting threshold fails with a null error.
    body: JSON.stringify({
      data: [{ path, meta: { _type: "gradio.FileData" } }, query, threshold],
    }),
    signal,
  });
  if (!started.ok) {
    throw new SpaceError(`call failed (${started.status})`, 502);
  }
  const { event_id: eventId } = (await started.json()) as { event_id?: string };
  if (!eventId) throw new SpaceError("call returned no event id", 502, "protocol");

  return awaitResult(eventId, signal);
}

/** Whether the Space is awake, so the UI can warn about a cold start. */
export async function spaceStage(): Promise<string> {
  try {
    const res = await fetch(`https://huggingface.co/api/spaces/nhibuilds/voxae/runtime`, {
      headers: authHeaders(),
      next: { revalidate: 30 },
    });
    if (!res.ok) return "unknown";
    return ((await res.json()) as { stage?: string }).stage ?? "unknown";
  } catch {
    return "unknown";
  }
}
