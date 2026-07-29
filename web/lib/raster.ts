/**
 * Draws the endpoint's probability field as terrain.
 *
 * The field is a confidence per pixel, which is an elevation surface: the
 * threshold is a waterline, and everything above it is the answer. Rendering
 * it that way needs the raw values, which is why the API sends a field rather
 * than a finished picture.
 */

export type Raster = { data: Uint8ClampedArray; width: number; height: number };

/** Decodes a data URI into raw pixels via an offscreen canvas. */
export async function decode(dataUri: string): Promise<Raster> {
  const bitmap = await createImageBitmap(await (await fetch(dataUri)).blob());
  const canvas = document.createElement("canvas");
  canvas.width = bitmap.width;
  canvas.height = bitmap.height;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) throw new Error("2d context unavailable");
  ctx.drawImage(bitmap, 0, 0);
  bitmap.close();
  const { data, width, height } = ctx.getImageData(0, 0, canvas.width, canvas.height);
  return { data, width, height };
}

type Rgb = [number, number, number];

export function hexToRgb(hex: string): Rgb {
  const n = parseInt(hex.replace("#", ""), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

/**
 * Hypsometric tint: lowland green through tan to a pale summit.
 *
 * Follows the convention of a relief map, which also happens to keep the most
 * confident regions lightest and so most legible over dark aerial imagery.
 */
const RELIEF: Rgb[] = [
  [143, 160, 122],
  [201, 178, 126],
  [226, 211, 168],
  [244, 239, 226],
];

function rampAt(t: number): Rgb {
  const scaled = Math.min(0.999, Math.max(0, t)) * (RELIEF.length - 1);
  const i = Math.floor(scaled);
  const f = scaled - i;
  const a = RELIEF[i];
  const b = RELIEF[Math.min(RELIEF.length - 1, i + 1)];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}

/** Land above the waterline, tinted by elevation. Below it, nothing. */
export function relief(field: Raster, waterline: number, alpha = 0.62): ImageData {
  const out = new ImageData(field.width, field.height);
  const sea = waterline * 255;
  const span = Math.max(1, 255 - sea);

  for (let i = 0; i < field.data.length; i += 4) {
    const value = field.data[i];
    if (value < sea) continue;
    const [r, g, b] = rampAt((value - sea) / span);
    out.data[i] = r;
    out.data[i + 1] = g;
    out.data[i + 2] = b;
    out.data[i + 3] = Math.round(255 * alpha);
  }
  return out;
}

/**
 * Iso-confidence contours.
 *
 * A pixel is on a contour when it sits at or above a level and touches a
 * neighbour below it. Index contours, drawn every fourth level, are darker,
 * the same way a survey sheet emphasises them.
 */
export function contours(field: Raster, levels: number[], colour: Rgb): ImageData {
  const { width, height, data } = field;
  const out = new ImageData(width, height);
  const [r, g, b] = colour;

  levels.forEach((level, n) => {
    const cut = level * 255;
    const index = n % 4 === 0;
    const alpha = index ? 235 : 150;

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const o = (y * width + x) * 4;
        if (data[o] < cut) continue;
        const left = x > 0 && data[o - 4] >= cut;
        const right = x < width - 1 && data[o + 4] >= cut;
        const up = y > 0 && data[o - width * 4] >= cut;
        const down = y < height - 1 && data[o + width * 4] >= cut;
        if (left && right && up && down) continue;

        out.data[o] = r;
        out.data[o + 1] = g;
        out.data[o + 2] = b;
        out.data[o + 3] = alpha;
        if (index && y < height - 1) {
          const below = o + width * 4;
          out.data[below] = r;
          out.data[below + 1] = g;
          out.data[below + 2] = b;
          out.data[below + 3] = alpha;
        }
      }
    }
  });
  return out;
}

/**
 * Diagonal hatch, the way a revision is overprinted on a survey sheet.
 *
 * Gives the baseline an encoding that is not just a different colour, so the
 * two predictions stay distinguishable without relying on hue.
 */
export function hatch(mask: Raster, colour: Rgb, pitch = 7): ImageData {
  const { width, height, data } = mask;
  const out = new ImageData(width, height);
  const [r, g, b] = colour;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const o = (y * width + x) * 4;
      if (data[o] < 128) continue;
      if ((x + y) % pitch > 1) continue;
      out.data[o] = r;
      out.data[o + 1] = g;
      out.data[o + 2] = b;
      out.data[o + 3] = 210;
    }
  }
  return out;
}

/**
 * The boundary alone, for ground truth, which is a reference not an opinion.
 *
 * `radius` is a dilation radius, so the drawn line is 2 * radius + 1 pixels.
 * Kept thin on purpose: an annotated region is full of holes around cars and
 * trees, and a heavy line turns that detail into scribble over the answer.
 */
export function neatline(mask: Raster, colour: Rgb, radius = 1, alpha = 200): ImageData {
  const { width, height, data } = mask;
  const out = new ImageData(width, height);
  const [r, g, b] = colour;
  const on = (x: number, y: number) =>
    x >= 0 && y >= 0 && x < width && y < height && data[(y * width + x) * 4] >= 128;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (!on(x, y)) continue;
      if (on(x - 1, y) && on(x + 1, y) && on(x, y - 1) && on(x, y + 1)) continue;
      for (let dy = -radius; dy <= radius; dy++) {
        for (let dx = -radius; dx <= radius; dx++) {
          const px = x + dx;
          const py = y + dy;
          if (px < 0 || py < 0 || px >= width || py >= height) continue;
          const o = (py * width + px) * 4;
          out.data[o] = r;
          out.data[o + 1] = g;
          out.data[o + 2] = b;
          out.data[o + 3] = alpha;
        }
      }
    }
  }
  return out;
}

export function toCanvas(image: ImageData): HTMLCanvasElement {
  const canvas = document.createElement("canvas");
  canvas.width = image.width;
  canvas.height = image.height;
  canvas.getContext("2d")!.putImageData(image, 0, 0);
  return canvas;
}

/** Land as a share of the sheet, recomputed so it tracks the waterline. */
export function coverage(field: Raster, waterline: number): number {
  const sea = waterline * 255;
  let count = 0;
  for (let i = 0; i < field.data.length; i += 4) {
    if (field.data[i] >= sea) count++;
  }
  return (count / (field.width * field.height)) * 100;
}

/** IoU between the land above the waterline and a reference mask. */
export function iouAgainst(field: Raster, reference: Raster, waterline: number): number | null {
  if (field.width !== reference.width || field.height !== reference.height) return null;
  const sea = waterline * 255;
  let inter = 0;
  let union = 0;
  for (let i = 0; i < field.data.length; i += 4) {
    const a = field.data[i] >= sea;
    const b = reference.data[i] >= 128;
    if (a && b) inter++;
    if (a || b) union++;
  }
  return union === 0 ? null : inter / union;
}
