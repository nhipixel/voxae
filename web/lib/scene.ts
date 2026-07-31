/**
 * Scene geometry: the sheared relief that stands the photograph up.
 *
 * The sheet stays rectangular and every pixel keeps its colour and its
 * confidence; only position moves. A pixel of height H is lifted by H and
 * pulled back toward its true footprint by H times the shear factor, which
 * is the classic relief-displacement correction for an oblique photograph.
 * Facade pixels ramp in height between base and roof, so the wall texture
 * lands on the wall.
 */

import type { Raster } from "./raster";
import { decode } from "./raster";

export type SceneMeta = { shear: number; depressionDeg: number };

export type SceneAssets = { depth: Raster; meta: SceneMeta };

const HEIGHT_SCALE = 0.15;
// The metric shear assumes metric height units; the normalized surface model
// is not metric, so the correction is trimmed to what stands facades upright
// against their own image-space extent.
const SHEAR_GAIN = 0.5;

export async function loadScene(stem: string): Promise<SceneAssets | null> {
  try {
    const res = await fetch(`/scene/${stem}.json`, { cache: "no-store" });
    if (!res.ok) return null;
    const meta = (await res.json()) as SceneMeta;
    if (typeof meta.shear !== "number") return null;
    // Regenerated offline without a name change, so the cache must not win.
    const depth = await decode(`/scene/${stem}-height.png`, { cache: "no-store" });
    return { depth, meta };
  } catch {
    return null;
  }
}

export type SceneGeometry = {
  /** vec3 world position per grid vertex, row major, (gridW+1)x(gridH+1). */
  positions: Float32Array;
  gridW: number;
  gridH: number;
  /** World position for any texture coordinate, for placing props. */
  worldAt: (u: number, v: number) => [number, number, number];
};

/** Bilinear sample of the surface model's red channel at (u, v). */
function sampleH(depth: Raster, u: number, v: number): number {
  const x = Math.min(depth.width - 1.001, Math.max(0, u * (depth.width - 1)));
  const y = Math.min(depth.height - 1.001, Math.max(0, v * (depth.height - 1)));
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const fx = x - x0;
  const fy = y - y0;
  const at = (xx: number, yy: number) => depth.data[(yy * depth.width + xx) * 4] / 255;
  return (
    at(x0, y0) * (1 - fx) * (1 - fy) +
    at(x0 + 1, y0) * fx * (1 - fy) +
    at(x0, y0 + 1) * (1 - fx) * fy +
    at(x0 + 1, y0 + 1) * fx * fy
  );
}

export function buildSceneGeometry(assets: SceneAssets, gridW = 220, gridH = 150): SceneGeometry {
  const { depth, meta } = assets;
  const aspect = depth.height / depth.width;

  const worldAt = (u: number, v: number): [number, number, number] => {
    const h = sampleH(depth, u, v) * HEIGHT_SCALE;
    return [u - 0.5, (0.5 - v) * aspect - h * meta.shear * SHEAR_GAIN, h];
  };

  const positions = new Float32Array((gridW + 1) * (gridH + 1) * 3);
  let k = 0;
  for (let gy = 0; gy <= gridH; gy++) {
    for (let gx = 0; gx <= gridW; gx++) {
      const [x, y, z] = worldAt(gx / gridW, gy / gridH);
      positions[k++] = x;
      positions[k++] = y;
      positions[k++] = z;
    }
  }

  return { positions, gridW, gridH, worldAt };
}
