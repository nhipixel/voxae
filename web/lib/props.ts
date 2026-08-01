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

// Against a dark table the structural parts must read light; the accent and
// glow lift to stay legible over night-toned photography.
const INK: [number, number, number, number] = [0.42, 0.47, 0.5, 1];
const HULL: [number, number, number, number] = [0.93, 0.95, 0.94, 1];
const ACCENT: [number, number, number, number] = [0.88, 0.42, 0.26, 1];
const GLOW: [number, number, number, number] = [0.35, 0.72, 0.9, 0.42];
const BLUR: [number, number, number, number] = [0.72, 0.8, 0.84, 0.1];
const BEAM: [number, number, number, number] = [0.55, 0.84, 0.98, 0.07];

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

  /** An open cone pointing down, for a landing beam. */
  cone(apex: V3, radius: number, depth: number, n = 22, shade = 1) {
    for (let i = 0; i < n; i++) {
      const a0 = (i / n) * Math.PI * 2;
      const a1 = ((i + 1) / n) * Math.PI * 2;
      const rim = (a: number): V3 => [
        apex[0] + radius * Math.cos(a),
        apex[1] + radius * Math.sin(a),
        apex[2] - depth,
      ];
      this.tri(apex, rim(a0), rim(a1), shade);
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

  // A landing cycle, not a hover: descend, settle on the pad, lift off again.
  const phase = (t * 0.075) % 1;
  const ease = (x: number) => 1 - Math.pow(1 - Math.min(1, Math.max(0, x)), 3);
  let lift: number;
  if (phase < 0.44) lift = 1 - ease(phase / 0.44);
  else if (phase < 0.62) lift = 0;
  else lift = ease((phase - 0.62) / 0.38);
  const grounded = lift < 0.02;
  const hz = az + 0.016 + lift * (0.115 + 0.006 * Math.sin(t * 2.2));
  const spin = grounded ? t * 3 : t * 18;
  // Oversized against a real drone so the airframe reads, but small enough
  // that the pad sits on the plaza instead of swallowing it. The panel
  // zooms in rather than the model scaling up.
  const S = 0.5;

  const hull = new Mesh();
  const dark = new Mesh();
  const glow = new Mesh();
  const blur = new Mesh();
  const beam = new Mesh();
  const port = new Mesh();
  const star = new Mesh();

  // Fuselage: a tapered body, a raised avionics deck, and a battery pack.
  hull.box([ax, ay, hz], [0.019 * S, 0.024 * S, 0.006 * S]);
  hull.box([ax, ay, hz + 0.008 * S], [0.013 * S, 0.016 * S, 0.004 * S]);
  hull.box([ax, ay + 0.004 * S, hz + 0.013 * S], [0.009 * S, 0.011 * S, 0.002 * S]);
  dark.box([ax, ay - 0.002 * S, hz - 0.008 * S], [0.012 * S, 0.015 * S, 0.003 * S]);

  // Nose gimbal: yoke, ball, and lens, panning gently as it scans.
  const pan = Math.sin(t * 0.9) * 0.35;
  const gy = ay - 0.026 * S;
  dark.box([ax, gy + 0.004 * S, hz - 0.004 * S], [0.007 * S, 0.003 * S, 0.004 * S]);
  hull.prism([ax, gy, hz - 0.009 * S], 0.0062 * S, 0.005 * S, 10, 0.95);
  dark.prism(
    [ax + Math.sin(pan) * 0.005 * S, gy - Math.cos(pan) * 0.004 * S, hz - 0.009 * S],
    0.0034 * S, 0.0035 * S, 10, 0.65, "x",
  );

  // Antennae.
  dark.box([ax - 0.014 * S, ay + 0.02 * S, hz + 0.008 * S], [0.0011 * S, 0.0011 * S, 0.008 * S]);
  dark.box([ax + 0.014 * S, ay + 0.02 * S, hz + 0.008 * S], [0.0011 * S, 0.0011 * S, 0.008 * S]);

  const arms = [Math.PI / 4, (3 * Math.PI) / 4, (5 * Math.PI) / 4, (7 * Math.PI) / 4];
  for (const q of arms) {
    const ex = ax + Math.cos(q) * 0.04 * S;
    const ey = ay + Math.sin(q) * 0.04 * S;
    hull.box([(ax + ex) / 2, (ay + ey) / 2, hz], [0.019 * S, 0.0042 * S, 0.0022 * S], q);
    hull.prism([ex, ey, hz + 0.004 * S], 0.005 * S, 0.005 * S, 10, 0.95);
    dark.prism([ex, ey, hz + 0.0105 * S], 0.0032 * S, 0.0022 * S, 8, 0.8);

    // Blades, plus a translucent disc that reads as rotor blur in flight.
    for (const k of [0, 1, 2]) {
      dark.box(
        [ex, ey, hz + 0.0135 * S],
        [0.021 * S, 0.0024 * S, 0.0008 * S],
        spin + q + (k * Math.PI) / 3,
      );
    }
    if (!grounded) blur.disc([ex, ey, hz + 0.0138 * S], 0.006 * S, 0.021 * S, 20);

    // Motor bells carry the navigation lights: port red, starboard green.
    const blink = Math.sin(t * 6) > 0;
    const lamp: V3 = [ex, ey, hz + 0.001 * S];
    if (Math.cos(q) < 0 && blink) port.prism(lamp, 0.0026 * S, 0.0026 * S, 6, 1);
    if (Math.cos(q) > 0 && !blink) star.prism(lamp, 0.0026 * S, 0.0026 * S, 6, 1);
  }

  // Landing gear: angled legs onto flat feet, compressing on touchdown.
  const squat = grounded ? 0.0018 * S : 0;
  for (const sx of [-1, 1]) {
    const lx = ax + 0.013 * S * sx;
    dark.box([lx, ay, hz - 0.014 * S + squat], [0.0022 * S, 0.019 * S, 0.007 * S]);
    hull.box([lx, ay, hz - 0.02 * S + squat], [0.0055 * S, 0.021 * S, 0.0014 * S]);
  }

  // A landing beam while airborne; the pad answers with a pulse once down.
  if (!grounded) {
    beam.cone([ax, ay, hz - 0.021 * S], 0.03 * S, hz - 0.021 * S - (az + 0.006));
    for (let z = hz - 0.03 * S; z > az + 0.016; z -= 0.02) {
      dark.box([ax, ay, z], [0.0016 * S, 0.0016 * S, 0.0045 * S]);
    }
  }
  // Contact shadow, so the drone never looks pasted on.
  const shadowR = 0.02 * S + lift * 0.016;
  glow.disc([ax, ay, az + 0.0035], 0, shadowR, 24, 0.35);

  const pulse = grounded ? 0.004 * (1 + Math.sin(t * 5)) : 0;
  glow.disc([ax, ay, az + 0.004], 0.019 * S - pulse * 0.4, 0.03 * S + pulse);
  hull.disc([ax, ay, az + 0.005], 0.029 * S, 0.031 * S, 28, 1.0);
  hull.box([ax - 0.0065 * S, ay, az + 0.005], [0.0016 * S, 0.009 * S, 0.0008]);
  hull.box([ax + 0.0065 * S, ay, az + 0.005], [0.0016 * S, 0.009 * S, 0.0008]);
  hull.box([ax, ay, az + 0.005], [0.0055 * S, 0.0016 * S, 0.0008]);

  return [
    hull.build(HULL),
    dark.build(INK),
    blur.build(BLUR),
    beam.build(BEAM),
    glow.build(GLOW),
    port.build([0.95, 0.28, 0.22, 1]),
    star.build([0.32, 0.9, 0.45, 1]),
  ];
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
    const r = ((t * 0.07 + i * 0.37) % 1) * 0.048 + 0.012;
    sweep.disc([a[0], a[1], a[2] + 0.004], r - 0.003, r);
  });
  return [pylons.build(INK), heads.build(ACCENT), sweep.build(GLOW)];
}

export function buildProps(kind: PropKind, anchors: V3[], t: number): PropGroup[] {
  if (!anchors.length) return [];
  if (kind === "drone") return droneProps(anchors[0], t);
  if (kind === "route") return routeProps(anchors, t);
  return ringProps(anchors, t);
}
