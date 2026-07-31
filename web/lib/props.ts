/**
 * Solid illustration models for the use-case vignettes.
 *
 * Small flat-shaded meshes built from primitives, animated per frame: a
 * drone descending to the safest cell, a truck tracing a confident route,
 * beacons sweeping candidate regions. Shade rides in the fourth component so
 * faces read as form without a lighting pass.
 */

export type PropKind = "drone" | "route" | "rings";
type V3 = [number, number, number];
export type PropGroup = { data: Float32Array; color: [number, number, number, number] };

const INK: [number, number, number, number] = [0.15, 0.17, 0.18, 1];
const HULL: [number, number, number, number] = [0.92, 0.93, 0.9, 1];
const ACCENT: [number, number, number, number] = [0.56, 0.23, 0.15, 1];
const GLOW: [number, number, number, number] = [0.17, 0.42, 0.54, 0.4];

class Mesh {
  tris: number[] = [];

  /** Axis-aligned box centred at c, half-sizes h, optional yaw about z. */
  box(c: V3, h: V3, yaw = 0) {
    const cs = Math.cos(yaw);
    const sn = Math.sin(yaw);
    const corner = (sx: number, sy: number, sz: number): V3 => {
      const x = sx * h[0];
      const y = sy * h[1];
      return [c[0] + x * cs - y * sn, c[1] + x * sn + y * cs, c[2] + sz * h[2]];
    };
    const p = [
      corner(-1, -1, -1), corner(1, -1, -1), corner(1, 1, -1), corner(-1, 1, -1),
      corner(-1, -1, 1), corner(1, -1, 1), corner(1, 1, 1), corner(-1, 1, 1),
    ];
    const face = (a: number, b: number, cc: number, d: number, shade: number) => {
      this.tri(p[a], p[b], p[cc], shade);
      this.tri(p[a], p[cc], p[d], shade);
    };
    face(4, 5, 6, 7, 1.0);   // top
    face(0, 3, 2, 1, 0.55);  // bottom
    face(0, 1, 5, 4, 0.85);  // front
    face(2, 3, 7, 6, 0.7);   // back
    face(1, 2, 6, 5, 0.78);  // right
    face(3, 0, 4, 7, 0.78);  // left
  }

  /** Regular n-gon prism, for rotor hubs, poles and wheels. */
  prism(c: V3, r: number, hz: number, n = 8, shade = 0.8, axis: "z" | "x" = "z") {
    const ring = (z: number): V3[] =>
      Array.from({ length: n }, (_, i) => {
        const a = (i / n) * Math.PI * 2;
        return axis === "z"
          ? ([c[0] + r * Math.cos(a), c[1] + r * Math.sin(a), c[2] + z] as V3)
          : ([c[0] + z, c[1] + r * Math.cos(a), c[2] + r * Math.sin(a)] as V3);
      });
    const lo = ring(-hz);
    const hi = ring(hz);
    const capHi: V3 = axis === "z" ? [c[0], c[1], c[2] + hz] : [c[0] + hz, c[1], c[2]];
    const capLo: V3 = axis === "z" ? [c[0], c[1], c[2] - hz] : [c[0] - hz, c[1], c[2]];
    for (let i = 0; i < n; i++) {
      const j = (i + 1) % n;
      this.tri(lo[i], lo[j], hi[j], shade);
      this.tri(lo[i], hi[j], hi[i], shade);
      this.tri(hi[i], hi[j], capHi, 1.0);
      this.tri(lo[j], lo[i], capLo, 0.6);
    }
  }

  /** Flat annulus on the ground, for sweep rings and pads. */
  disc(c: V3, r0: number, r1: number, n = 28, shade = 1.0) {
    for (let i = 0; i < n; i++) {
      const a0 = (i / n) * Math.PI * 2;
      const a1 = ((i + 1) / n) * Math.PI * 2;
      const p = (r: number, a: number): V3 => [c[0] + r * Math.cos(a), c[1] + r * Math.sin(a), c[2]];
      this.tri(p(r0, a0), p(r1, a0), p(r1, a1), shade);
      this.tri(p(r0, a0), p(r1, a1), p(r0, a1), shade);
    }
  }

  /** A flat ribbon along a polyline, hovering just off the ground. */
  ribbon(points: V3[], width: number, lift = 0.004, shade = 1.0) {
    for (let i = 0; i < points.length - 1; i++) {
      const a = points[i];
      const b = points[i + 1];
      const dx = b[0] - a[0];
      const dy = b[1] - a[1];
      const l = Math.hypot(dx, dy) || 1;
      const nx = (-dy / l) * width;
      const ny = (dx / l) * width;
      const q = (p: V3, s: number): V3 => [p[0] + nx * s, p[1] + ny * s, p[2] + lift];
      this.tri(q(a, -1), q(b, -1), q(b, 1), shade);
      this.tri(q(a, -1), q(b, 1), q(a, 1), shade);
    }
  }

  tri(a: V3, b: V3, c: V3, shade: number) {
    this.tris.push(a[0], a[1], a[2], shade, b[0], b[1], b[2], shade, c[0], c[1], c[2], shade);
  }

  build(color: [number, number, number, number]): PropGroup {
    return { data: new Float32Array(this.tris), color };
  }
}

function droneProps(anchor: V3, t: number): PropGroup[] {
  const [ax, ay, az] = anchor;
  const hz = az + 0.16 + 0.01 * Math.sin(t * 2.2);
  const hull = new Mesh();
  const dark = new Mesh();
  const glow = new Mesh();

  hull.box([ax, ay, hz], [0.016, 0.016, 0.006]);
  for (const q of [Math.PI / 4, (3 * Math.PI) / 4, (5 * Math.PI) / 4, (7 * Math.PI) / 4]) {
    const ex = ax + Math.cos(q) * 0.03;
    const ey = ay + Math.sin(q) * 0.03;
    hull.box([(ax + ex) / 2, (ay + ey) / 2, hz], [0.014, 0.0035, 0.0018], q);
    dark.prism([ex, ey, hz + 0.006], 0.0035, 0.004, 6, 0.9);
    // Two-blade propeller, spinning.
    dark.box([ex, ey, hz + 0.011], [0.016, 0.0022, 0.0008], t * 14 + q);
  }
  dark.box([ax - 0.01, ay, hz - 0.009], [0.0018, 0.013, 0.003]);
  dark.box([ax + 0.01, ay, hz - 0.009], [0.0018, 0.013, 0.003]);

  // Descent dashes and the landing pad.
  for (let z = hz - 0.028; z > az + 0.012; z -= 0.022) {
    dark.box([ax, ay, z], [0.0016, 0.0016, 0.005]);
  }
  glow.disc([ax, ay, az + 0.004], 0.018, 0.03);
  dark.disc([ax, ay, az + 0.005], 0.028, 0.0305, 28, 1.0);

  return [hull.build(HULL), dark.build(INK), glow.build(GLOW)];
}

function routeProps(anchors: V3[], t: number): PropGroup[] {
  const path = new Mesh();
  path.ribbon(anchors, 0.006);

  const lengths = [0];
  for (let i = 0; i < anchors.length - 1; i++) {
    const a = anchors[i];
    const b = anchors[i + 1];
    lengths.push(lengths[i] + Math.hypot(b[0] - a[0], b[1] - a[1]));
  }
  const total = lengths[lengths.length - 1] || 1;
  const along = ((t * 0.055) % 1) * total;
  let i = 0;
  while (i < lengths.length - 2 && lengths[i + 1] < along) i++;
  const f = (along - lengths[i]) / Math.max(lengths[i + 1] - lengths[i], 1e-6);
  const a = anchors[i];
  const b = anchors[i + 1];
  const yaw = Math.atan2(b[1] - a[1], b[0] - a[0]);
  const cx = a[0] + (b[0] - a[0]) * f;
  const cy = a[1] + (b[1] - a[1]) * f;
  const cz = a[2] + (b[2] - a[2]) * f + 0.011;

  const truck = new Mesh();
  const dark = new Mesh();
  truck.box([cx, cy, cz + 0.004], [0.015, 0.0085, 0.0055], yaw);
  truck.box(
    [cx + Math.cos(yaw) * 0.017, cy + Math.sin(yaw) * 0.017, cz + 0.001],
    [0.0065, 0.0075, 0.0075],
    yaw,
  );
  for (const s of [-1, 1]) {
    for (const w of [-0.009, 0.011]) {
      dark.prism(
        [
          cx + Math.cos(yaw) * w - Math.sin(yaw) * 0.0085 * s,
          cy + Math.sin(yaw) * w + Math.cos(yaw) * 0.0085 * s,
          cz - 0.006,
        ],
        0.0042,
        0.0022,
        8,
        0.7,
        "x",
      );
    }
  }
  return [path.build(GLOW), truck.build(ACCENT), dark.build(INK)];
}

function ringProps(anchors: V3[], t: number): PropGroup[] {
  const pylons = new Mesh();
  const heads = new Mesh();
  const sweep = new Mesh();
  anchors.forEach((a, i) => {
    pylons.box([a[0], a[1], a[2] + 0.03], [0.0022, 0.0022, 0.03]);
    const pulse = 0.006 + 0.002 * Math.sin(t * 3 + i);
    heads.box([a[0], a[1], a[2] + 0.066], [pulse, pulse, pulse], t * 0.8);
    const r = ((t * 0.08 + i * 0.37) % 1) * 0.09 + 0.014;
    sweep.disc([a[0], a[1], a[2] + 0.004], r - 0.004, r);
  });
  return [pylons.build(INK), heads.build(ACCENT), sweep.build(GLOW)];
}

export function buildProps(kind: PropKind, anchors: V3[], t: number): PropGroup[] {
  if (!anchors.length) return [];
  if (kind === "drone") return droneProps(anchors[0], t);
  if (kind === "route") return routeProps(anchors, t);
  return ringProps(anchors, t);
}
