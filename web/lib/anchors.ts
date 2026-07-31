/**
 * Picks where the use-case props act, from the confidence field itself.
 *
 * Coarse on purpose: these are illustration anchors, not measurements.
 */

import type { Raster } from "./raster";

type UV = [number, number];

function coarse(field: Raster, gw = 64): { g: Float32Array; gw: number; gh: number } {
  const gh = Math.max(8, Math.round((gw * field.height) / field.width));
  const g = new Float32Array(gw * gh);
  for (let y = 0; y < gh; y++) {
    for (let x = 0; x < gw; x++) {
      const px = Math.floor(((x + 0.5) / gw) * field.width);
      const py = Math.floor(((y + 0.5) / gh) * field.height);
      g[y * gw + x] = field.data[(py * field.width + px) * 4] / 255;
    }
  }
  return { g, gw, gh };
}

/** The single most confident spot, away from the edges. */
export function droneAnchor(field: Raster): UV[] {
  const { g, gw, gh } = coarse(field);
  let best = 0;
  let bx = gw / 2;
  let by = gh / 2;
  for (let y = 2; y < gh - 2; y++) {
    for (let x = 2; x < gw - 2; x++) {
      let s = 0;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) s += g[(y + dy) * gw + x + dx];
      if (s > best) {
        best = s;
        bx = x;
        by = y;
      }
    }
  }
  return [[(bx + 0.5) / gw, (by + 0.5) / gh]];
}

/** A left-to-right path through confident ground. */
export function routeAnchors(field: Raster, points = 7): UV[] {
  const { g, gw, gh } = coarse(field);
  const out: UV[] = [];
  for (let i = 0; i < points; i++) {
    const x0 = Math.floor((i / points) * gw);
    const x1 = Math.floor(((i + 1) / points) * gw);
    let best = -1;
    let by = gh / 2;
    for (let y = 1; y < gh - 1; y++) {
      let s = 0;
      for (let x = x0; x < x1; x++) s += g[y * gw + x];
      if (s > best) {
        best = s;
        by = y;
      }
    }
    out.push([(x0 + x1) / 2 / gw, (by + 0.5) / gh]);
  }
  return out;
}

/** Centres of the largest confident regions, for survey rings. */
export function ringAnchors(field: Raster, threshold = 0.3, count = 3): UV[] {
  const { g, gw, gh } = coarse(field);
  const seen = new Uint8Array(gw * gh);
  const blobs: { size: number; cx: number; cy: number }[] = [];
  for (let i = 0; i < g.length; i++) {
    if (seen[i] || g[i] < threshold) continue;
    let size = 0;
    let sx = 0;
    let sy = 0;
    const stack = [i];
    seen[i] = 1;
    while (stack.length) {
      const j = stack.pop()!;
      size++;
      sx += j % gw;
      sy += Math.floor(j / gw);
      const x = j % gw;
      const y = Math.floor(j / gw);
      for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]] as const) {
        const nx = x + dx;
        const ny = y + dy;
        const k = ny * gw + nx;
        if (nx >= 0 && ny >= 0 && nx < gw && ny < gh && !seen[k] && g[k] >= threshold) {
          seen[k] = 1;
          stack.push(k);
        }
      }
    }
    blobs.push({ size, cx: sx / size, cy: sy / size });
  }
  blobs.sort((a, b) => b.size - a.size);
  return blobs.slice(0, count).map((b) => [(b.cx + 0.5) / gw, (b.cy + 0.5) / gh]);
}
