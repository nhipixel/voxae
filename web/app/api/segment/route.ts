import { NextResponse } from "next/server";

import { predict, SpaceError } from "@/lib/space";

// Node rather than Edge, so the SSE read is not cut short.
export const runtime = "nodejs";
export const maxDuration = 120;

const MAX_BYTES = 10 * 1024 * 1024;
const ACCEPTED = new Set(["image/png", "image/jpeg", "image/webp"]);

// Per instance throttle. Serverless instances do not share memory, so the real
// ceiling is the ZeroGPU quota upstream.
const WINDOW_MS = 60_000;
const PER_WINDOW = 12;
const hits: number[] = [];

function throttled(): boolean {
  const now = Date.now();
  while (hits.length && now - hits[0] > WINDOW_MS) hits.shift();
  if (hits.length >= PER_WINDOW) return true;
  hits.push(now);
  return false;
}

export async function POST(request: Request) {
  if (throttled()) {
    return NextResponse.json(
      { error: "Too many requests in the last minute. Give it a moment." },
      { status: 429 },
    );
  }

  let form: FormData;
  try {
    form = await request.formData();
  } catch {
    return NextResponse.json({ error: "Expected multipart form data." }, { status: 400 });
  }

  const image = form.get("image");
  const query = String(form.get("query") ?? "").trim();

  if (!(image instanceof File)) {
    return NextResponse.json({ error: "No image provided." }, { status: 400 });
  }
  if (!query) {
    return NextResponse.json({ error: "Ask something about the scene." }, { status: 400 });
  }
  if (image.size > MAX_BYTES) {
    return NextResponse.json(
      { error: `Image is ${(image.size / 1e6).toFixed(1)} MB; the limit is 10 MB.` },
      { status: 413 },
    );
  }
  if (image.type && !ACCEPTED.has(image.type)) {
    return NextResponse.json(
      { error: `Unsupported type ${image.type}. Use PNG, JPEG, or WebP.` },
      { status: 415 },
    );
  }

  try {
    const result = await predict(image, image.name || "upload.png", query, request.signal);
    return NextResponse.json(result);
  } catch (error) {
    if (error instanceof SpaceError) {
      const message =
        error.kind === "quota"
          ? "The demo's GPU quota for today is used up. It resets on a rolling 24 hour window."
          : error.message;
      return NextResponse.json({ error: message, kind: error.kind }, { status: error.status });
    }
    if (error instanceof Error && error.name === "AbortError") {
      return NextResponse.json({ error: "Cancelled." }, { status: 499 });
    }
    return NextResponse.json({ error: "The model backend did not respond." }, { status: 502 });
  }
}
