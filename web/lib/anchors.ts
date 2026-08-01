/**
 * Where the use-case props may stand, derived from the data rather than by eye.
 *
 * Occupancy comes from the annotated mask when a frame has one: it is drawn
 * around every parked car and planted bed, so its holes are exactly the
 * obstacles a device must avoid. Without an annotation it falls back to the
 * confidence field and the surface model, which agree on open ground but do
 * not resolve individual cars.
 *
 * Clearance, not the centre point, decides. A gap between two parked cars
 * reads as clear at its midpoint while a truck's body still overlaps them, so
 * every prop asks for room measured in cells, not for a coordinate.
 */

import type { Raster } from "./raster";

export type UV = [number, number];

/** Confidence a cell must carry to count as part of the answer. */
const MIN_CONFIDENCE = 0.55;
/** Height above the ground plane that still counts as clear, 0 to 1. */
const MAX_RELIEF = 0.05;

// Roughly eight source pixels per cell, so a car spans several.
const GRID_W = 192;

type Survey = { clear: Uint8Array; w: number; h: number; clearance: Float32Array };

function sample(r: Raster, u: number, v: number): number {
  const x = Math.min(r.width - 1, Math.max(0, Math.round(u * (r.width - 1))));
  const y = Math.min(r.height - 1, Math.max(0, Math.round(v * (r.height - 1))));
  return r.data[(y * r.width + x) * 4] / 255;
}

/**
 * Marks every cell that is both confident and flat, then measures how far
 * each sits from the nearest obstruction. That clearance is what lets a prop
 * claim room rather than merely a point.
 */
function survey(field: Raster, relief: Raster | null, occupancy: Raster | null): Survey {
  const w = GRID_W;
  const h = Math.max(16, Math.round((GRID_W * field.height) / field.width));
  const clear = new Uint8Array(w * h);

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const u = (x + 0.5) / w;
      const v = (y + 0.5) / h;
      if (occupancy) {
        clear[y * w + x] = sample(occupancy, u, v) >= 0.5 ? 1 : 0;
        continue;
      }
      const confident = sample(field, u, v) >= MIN_CONFIDENCE;
      const flat = !relief || sample(relief, u, v) <= MAX_RELIEF;
      clear[y * w + x] = confident && flat ? 1 : 0;
    }
  }

  // Chamfer distance transform: two sweeps give clearance accurately enough.
  const clearance = new Float32Array(w * h);
  const BIG = 1e6;
  for (let i = 0; i < clearance.length; i++) clearance[i] = clear[i] ? BIG : 0;
  const at = (x: number, y: number) =>
    x < 0 || y < 0 || x >= w || y >= h ? 0 : clearance[y * w + x];
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (!clear[y * w + x]) continue;
      clearance[y * w + x] = Math.min(
        clearance[y * w + x],
        at(x - 1, y) + 1,
        at(x, y - 1) + 1,
        at(x - 1, y - 1) + 1.414,
        at(x + 1, y - 1) + 1.414,
      );
    }
  }
  for (let y = h - 1; y >= 0; y--) {
    for (let x = w - 1; x >= 0; x--) {
      if (!clear[y * w + x]) continue;
      clearance[y * w + x] = Math.min(
        clearance[y * w + x],
        at(x + 1, y) + 1,
        at(x, y + 1) + 1,
        at(x + 1, y + 1) + 1.414,
        at(x - 1, y + 1) + 1.414,
      );
    }
  }
  return { clear, w, h, clearance };
}

const toUV = (x: number, y: number, f: Survey): UV => [(x + 0.5) / f.w, (y + 0.5) / f.h];

function roomiest(f: Survey, margin = 2): { x: number; y: number; room: number } {
  let room = -1;
  let x = f.w >> 1;
  let y = f.h >> 1;
  for (let yy = margin; yy < f.h - margin; yy++) {
    for (let xx = margin; xx < f.w - margin; xx++) {
      const c = f.clearance[yy * f.w + xx];
      if (c > room) {
        room = c;
        x = xx;
        y = yy;
      }
    }
  }
  return { x, y, room };
}

/** The roomiest clear cell: a landing pad wants space on every side. */
export function droneAnchor(
  field: Raster,
  relief: Raster | null,
  occupancy: Raster | null = null,
): UV[] {
  const f = survey(field, relief, occupancy);
  const best = roomiest(f);
  return [toUV(best.x, best.y, f)];
}

/**
 * A corridor down the clearest ground.
 *
 * Walks outward from the roomiest cell, each step taking the clearest cell
 * within reach of the last, so the route bends around parked cars instead of
 * driving over them.
 */
export function routeAnchors(
  field: Raster,
  relief: Raster | null,
  occupancy: Raster | null = null,
): UV[] {
  const f = survey(field, relief, occupancy);
  const start = roomiest(f, 1);
  // Enough room for a vehicle body, not merely for its centre line.
  const MIN_ROOM = 3;
  if (start.room < MIN_ROOM) return [];

  const REACH = 8;
  const walk = (dir: 1 | -1): UV[] => {
    const out: UV[] = [];
    let x = start.x;
    for (let y = start.y + dir * 3; y > 0 && y < f.h - 1; y += dir * 3) {
      let pick = -1;
      let pickScore = -1;
      for (let dx = -REACH; dx <= REACH; dx++) {
        const nx = x + dx;
        if (nx < 1 || nx >= f.w - 1) continue;
        const c = f.clearance[y * f.w + nx];
        if (c < MIN_ROOM) continue;
        // Prefer room, but resist wandering: a route should look driven.
        const score = c - Math.abs(dx) * 0.35;
        if (score > pickScore) {
          pickScore = score;
          pick = nx;
        }
      }
      if (pick < 0) break;
      x = pick;
      out.push(toUV(x, y, f));
    }
    return out;
  };

  const path = [...walk(-1).reverse(), toUV(start.x, start.y, f), ...walk(1)];
  return path.length >= 2 ? path : [];
}

/** Well-separated clear sectors, so the sweeps do not overlap. */
export function ringAnchors(
  field: Raster,
  relief: Raster | null,
  occupancy: Raster | null = null,
  count = 3,
): UV[] {
  const f = survey(field, relief, occupancy);
  const MIN_ROOM = 3;

  // Spread the beacons, but never return fewer than asked for: a separation
  // this greedy loop cannot satisfy is relaxed rather than silently dropping
  // sectors, which is what left the triage panel showing one beacon.
  for (let spread = 0.22; spread >= 0.05; spread -= 0.04) {
    const minApart = f.w * spread;
    const picked: { x: number; y: number }[] = [];

    for (let n = 0; n < count; n++) {
      let best = -1;
      let bx = -1;
      let by = -1;
      for (let y = 2; y < f.h - 2; y++) {
        for (let x = 2; x < f.w - 2; x++) {
          const c = f.clearance[y * f.w + x];
          if (c < MIN_ROOM) continue;
          if (!picked.every((q) => Math.hypot(q.x - x, q.y - y) >= minApart)) continue;
          if (c > best) {
            best = c;
            bx = x;
            by = y;
          }
        }
      }
      if (bx < 0) break;
      picked.push({ x: bx, y: by });
    }
    if (picked.length >= count) return picked.map((q) => toUV(q.x, q.y, f));
  }
  return [];
}
