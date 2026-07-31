/**
 * A heightfield renderer for the confidence surface. Raw WebGL1, no library.
 *
 * The model's output is a probability per pixel, which this draws literally:
 * elevation is confidence, the photograph is draped over the relief, and the
 * waterline is a translucent sea plane. Everything the flat sheet encodes in
 * tint and contour lines becomes shape and light.
 *
 * WebGL1 with no required extensions: normals come from finite differences of
 * the height texture in the fragment shader, so it runs on anything that has
 * vertex texture fetch, which is effectively every device since 2012.
 */

export type FieldRaster = { data: Uint8ClampedArray; width: number; height: number };

export type ViewState = {
  /** Radians. 0 faces north; positive turns the sheet clockwise. */
  yaw: number;
  /** Radians above the horizon. PI/2 is straight down, the flat-sheet view. */
  pitch: number;
  zoom: number;
  waterline: number;
};

export const HOME_VIEW: Omit<ViewState, "waterline"> = {
  yaw: -0.24,
  pitch: 1.0,
  zoom: 1.85,
};

const HEIGHT_SCALE = 0.17;
const GRID = 220;

const TERRAIN_VS = `
attribute vec2 aUv;
uniform mat4 uMvp;
uniform sampler2D uHeight;
uniform sampler2D uDepth;
uniform mediump float uUseDepth;
uniform float uHScale;
uniform float uAspect;
varying vec2 vUv;
void main() {
  vUv = aUv;
  float lift = mix(texture2D(uHeight, aUv).r, texture2D(uDepth, aUv).r, uUseDepth);
  vec3 pos = vec3(aUv.x - 0.5, (0.5 - aUv.y) * uAspect, lift * uHScale);
  gl_Position = uMvp * vec4(pos, 1.0);
}`;

const SCENE_VS = `
attribute vec2 aUv;
attribute vec3 aPos;
uniform mat4 uMvp;
varying vec2 vUv;
void main() {
  vUv = aUv;
  gl_Position = uMvp * vec4(aPos, 1.0);
}`;

const TERRAIN_FS = `
precision mediump float;
uniform sampler2D uPhoto;
uniform sampler2D uHeight;
uniform sampler2D uDepth;
uniform mediump float uUseDepth;
uniform vec2 uTexel;
uniform float uWater;
uniform mediump float uHasField;
varying vec2 vUv;

float lift(vec2 uv) {
  return mix(texture2D(uHeight, uv).r, texture2D(uDepth, uv).r, uUseDepth);
}

void main() {
  float vP = texture2D(uHeight, vUv).r;
  float hL = lift(vUv - vec2(uTexel.x, 0.0));
  float hR = lift(vUv + vec2(uTexel.x, 0.0));
  float hT = lift(vUv - vec2(0.0, uTexel.y));
  float hB = lift(vUv + vec2(0.0, uTexel.y));
  // Height scale times a tuned gain, baked so no uniform crosses stages:
  // shared uniforms must match precision between shaders, and ANGLE enforces
  // it at link time.
  const float s = 20.4;
  vec3 n = normalize(vec3((hL - hR) * s, (hB - hT) * s, 2.0));
  float light = 0.66 + 0.34 * max(dot(n, normalize(vec3(-0.45, 0.55, 0.7))), 0.0);

  vec3 photo = texture2D(uPhoto, vUv).rgb;
  float back = smoothstep(0.06, 0.22, hB - hT);
  photo = mix(photo, vec3(0.70, 0.71, 0.66), back * 0.75);
  if (uHasField < 0.5) {
    gl_FragColor = vec4(photo * light, 1.0);
    return;
  }
  vec3 col;
  if (vP >= uWater) {
    float t = (vP - uWater) / max(1.0 - uWater, 1e-3);
    vec3 low  = vec3(0.561, 0.627, 0.478);
    vec3 mid  = vec3(0.788, 0.698, 0.494);
    vec3 high = vec3(0.957, 0.937, 0.886);
    vec3 ramp = t < 0.5 ? mix(low, mid, t * 2.0) : mix(mid, high, t * 2.0 - 1.0);
    col = mix(photo, ramp, mix(0.40, 0.26, uUseDepth)) * light;

    float d = abs(fract(vP * 10.0) - 0.5);
    float line = 1.0 - smoothstep(0.0, 0.14, d);
    col = mix(col, vec3(0.29, 0.204, 0.086), line * 0.42);

    float foam = 1.0 - smoothstep(0.0, 0.018, vP - uWater);
    col = mix(col, vec3(0.94, 0.95, 0.93), foam * 0.75);
  } else {
    col = photo * vec3(0.42, 0.52, 0.60) * light;
  }
  gl_FragColor = vec4(col, 1.0);
}`;

const FLAT_VS = `
attribute vec3 aPos;
attribute float aShade;
uniform mat4 uMvp;
varying float vShade;
void main() { vShade = aShade; gl_Position = uMvp * vec4(aPos, 1.0); }`;

const FLAT_FS = `
precision mediump float;
uniform vec4 uColor;
varying float vShade;
void main() { gl_FragColor = vec4(uColor.rgb * vShade, uColor.a); }`;

function compile(gl: WebGLRenderingContext, type: number, src: string): WebGLShader {
  const shader = gl.createShader(type)!;
  gl.shaderSource(shader, src);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    throw new Error(`shader: ${gl.getShaderInfoLog(shader)}`);
  }
  return shader;
}

function program(gl: WebGLRenderingContext, vs: string, fs: string): WebGLProgram {
  const p = gl.createProgram()!;
  gl.attachShader(p, compile(gl, gl.VERTEX_SHADER, vs));
  gl.attachShader(p, compile(gl, gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
    throw new Error(`link: ${gl.getProgramInfoLog(p)}`);
  }
  return p;
}

/* Just enough matrix math; a library would be larger than this file. */

type Mat4 = Float32Array;

function perspective(fovY: number, aspect: number, near: number, far: number): Mat4 {
  const f = 1 / Math.tan(fovY / 2);
  const nf = 1 / (near - far);
  const m = new Float32Array(16);
  m[0] = f / aspect;
  m[5] = f;
  m[10] = (far + near) * nf;
  m[11] = -1;
  m[14] = 2 * far * near * nf;
  return m;
}

function lookAt(eye: number[], target: number[], up: number[]): Mat4 {
  const z = norm3(sub3(eye, target));
  const x = norm3(cross3(up, z));
  const y = cross3(z, x);
  return new Float32Array([
    x[0], y[0], z[0], 0,
    x[1], y[1], z[1], 0,
    x[2], y[2], z[2], 0,
    -dot3(x, eye), -dot3(y, eye), -dot3(z, eye), 1,
  ]);
}

function mul4(a: Mat4, b: Mat4): Mat4 {
  const out = new Float32Array(16);
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      out[c * 4 + r] =
        a[r] * b[c * 4] + a[4 + r] * b[c * 4 + 1] + a[8 + r] * b[c * 4 + 2] + a[12 + r] * b[c * 4 + 3];
    }
  }
  return out;
}

const sub3 = (a: number[], b: number[]) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const cross3 = (a: number[], b: number[]) => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
const dot3 = (a: number[], b: number[]) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const norm3 = (a: number[]) => {
  const l = Math.hypot(a[0], a[1], a[2]) || 1;
  return [a[0] / l, a[1] / l, a[2] / l];
};

/**
 * A light separable box blur on the red channel, for the height texture only.
 *
 * Mid-grey speckle in low confidence areas extrudes into rubble that reads as
 * false detail. One radius-1 pass calms the geometry; the analytic field the
 * sheet, the scores, and the export use is never touched.
 */
export function smoothField(field: FieldRaster): FieldRaster {
  const { width: w, height: h, data } = field;
  const tmp = new Float32Array(w * h);
  const out = new Uint8ClampedArray(w * h * 4);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const l = data[(y * w + Math.max(0, x - 1)) * 4];
      const c = data[(y * w + x) * 4];
      const r = data[(y * w + Math.min(w - 1, x + 1)) * 4];
      tmp[y * w + x] = (l + c + r) / 3;
    }
  }
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const t = tmp[Math.max(0, y - 1) * w + x];
      const c = tmp[y * w + x];
      const b = tmp[Math.min(h - 1, y + 1) * w + x];
      const v = Math.round((t + c + b) / 3);
      const o = (y * w + x) * 4;
      out[o] = out[o + 1] = out[o + 2] = v;
      out[o + 3] = 255;
    }
  }
  return { data: out, width: w, height: h };
}

export class TerrainRenderer {
  private gl: WebGLRenderingContext;
  private terrain: WebGLProgram;
  private scene: WebGLProgram;
  private flat: WebGLProgram;
  private photoTex: WebGLTexture;
  private heightTex: WebGLTexture;
  private depthTex: WebGLTexture;
  private uvBuffer: WebGLBuffer;
  private indexBuffer: WebGLBuffer;
  private indexCount = 0;
  private skirtBuffer: WebGLBuffer;
  private skirtCount = 0;
  private waterBuffer: WebGLBuffer;
  private posBuffer: WebGLBuffer;
  private propBuffer: WebGLBuffer;
  private scenePositions: Float32Array | null = null;
  private hasField = false;
  private aspect = 9 / 16;
  private texel: [number, number] = [1 / 256, 1 / 256];
  ready = false;

  constructor(private canvas: HTMLCanvasElement) {
    const gl = canvas.getContext("webgl", {
      antialias: true,
      alpha: true,
      // Kept on so the view can be read back for export and verification.
      preserveDrawingBuffer: true,
    });
    if (!gl) throw new Error("webgl unavailable");
    this.gl = gl;
    this.terrain = program(gl, TERRAIN_VS, TERRAIN_FS);
    this.scene = program(gl, SCENE_VS, TERRAIN_FS);
    this.flat = program(gl, FLAT_VS, FLAT_FS);
    this.photoTex = gl.createTexture()!;
    this.heightTex = gl.createTexture()!;
    this.depthTex = gl.createTexture()!;
    this.uvBuffer = gl.createBuffer()!;
    this.indexBuffer = gl.createBuffer()!;
    this.skirtBuffer = gl.createBuffer()!;
    this.waterBuffer = gl.createBuffer()!;
    this.posBuffer = gl.createBuffer()!;
    this.propBuffer = gl.createBuffer()!;
    const blank = { data: new Uint8ClampedArray([0, 0, 0, 255]), width: 1, height: 1 };
    this.uploadTexture(this.heightTex, null, blank);
    this.uploadTexture(this.depthTex, null, blank);
    gl.enable(gl.DEPTH_TEST);
    gl.clearColor(0, 0, 0, 0);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
  }

  private uploadTexture(tex: WebGLTexture, source: TexImageSource | null, raw?: FieldRaster) {
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    if (raw) {
      gl.texImage2D(
        gl.TEXTURE_2D, 0, gl.RGBA, raw.width, raw.height, 0,
        gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array(raw.data),
      );
    } else if (source) {
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, source);
    }
    // Non power of two, so clamp and no mips.
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  }

  setPhoto(photo: TexImageSource) {
    this.uploadTexture(this.photoTex, photo);
  }

  private field: FieldRaster | null = null;
  private depthField: FieldRaster | null = null;

  setField(field: FieldRaster | null) {
    this.hasField = field != null;
    if (!field) {
      // A 1x1 zero keeps every sampler bound and every branch simple.
      this.field = null;
      this.uploadTexture(this.heightTex, null, {
        data: new Uint8ClampedArray([0, 0, 0, 255]),
        width: 1,
        height: 1,
      });
      if (this.scenePositions) this.ready = true;
      return;
    }
    this.field = field;
    this.uploadTexture(this.heightTex, null, field);
    this.texel = [1 / field.width, 1 / field.height];
    if (!this.scenePositions) {
      this.aspect = field.height / field.width;
      this.buildGeometry();
    }
    this.ready = true;
  }

  /** True vertex positions from the depth mesh; the sheet becomes a scene. */
  setScene(positions: Float32Array, gridW: number, gridH: number, depth: FieldRaster) {
    this.scenePositions = positions;
    this.depthField = depth;
    this.uploadTexture(this.depthTex, null, depth);
    this.texel = [1 / depth.width, 1 / depth.height];
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.posBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);
    this.buildSceneGrid(gridW, gridH, positions);
    this.ready = true;
  }

  /** Animated illustration meshes: interleaved x,y,z,shade triangles. */
  private propGroups: { count: number; offset: number; color: [number, number, number, number] }[] = [];

  setProps(groups: { data: Float32Array; color: [number, number, number, number] }[] | null) {
    const gl = this.gl;
    this.propGroups = [];
    if (!groups || !groups.length) return;
    let total = 0;
    for (const g of groups) total += g.data.length;
    const packed = new Float32Array(total);
    let cursor = 0;
    for (const g of groups) {
      packed.set(g.data, cursor);
      this.propGroups.push({ offset: cursor / 4, count: g.data.length / 4, color: g.color });
      cursor += g.data.length;
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, this.propBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, packed, gl.DYNAMIC_DRAW);
  }

  private buildSceneGrid(w: number, h: number, positions: Float32Array) {
    const gl = this.gl;
    const uvs = new Float32Array((w + 1) * (h + 1) * 2);
    let k = 0;
    for (let y = 0; y <= h; y++) {
      for (let x = 0; x <= w; x++) {
        uvs[k++] = x / w;
        uvs[k++] = y / h;
      }
    }
    const idx = new Uint16Array(w * h * 6);
    k = 0;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const a = y * (w + 1) + x;
        const b = a + 1;
        const c = a + w + 1;
        const d = c + 1;
        idx[k++] = a; idx[k++] = c; idx[k++] = b;
        idx[k++] = b; idx[k++] = c; idx[k++] = d;
      }
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, this.uvBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, uvs, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.indexBuffer);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, idx, gl.STATIC_DRAW);
    this.indexCount = idx.length;

    // Skirt ring straight from the border vertices, down to a base.
    const at = (x: number, y: number) => {
      const i = (y * (w + 1) + x) * 3;
      return [positions[i], positions[i + 1], positions[i + 2]] as const;
    };
    const ring: (readonly [number, number, number])[] = [];
    for (let x = 0; x <= w; x++) ring.push(at(x, h));
    for (let y = h - 1; y >= 0; y--) ring.push(at(w, y));
    for (let x = w - 1; x >= 0; x--) ring.push(at(x, 0));
    for (let y = 1; y <= h - 1; y++) ring.push(at(0, y));
    ring.push(ring[0]);
    const base = -0.05;
    const skirt: number[] = [];
    for (let i = 0; i < ring.length - 1; i++) {
      const [x0, y0, z0] = ring[i];
      const [x1, y1, z1] = ring[i + 1];
      skirt.push(
        x0, y0, z0, 1.0, x0, y0, base, 0.78, x1, y1, z1, 1.0,
        x1, y1, z1, 1.0, x0, y0, base, 0.78, x1, y1, base, 0.78,
      );
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, this.skirtBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(skirt), gl.STATIC_DRAW);
    this.skirtCount = skirt.length / 4;
  }

  /** Real scene structure to stand the photograph up on; null falls back to
      extruding the confidence field itself. */
  setDepth(depth: FieldRaster | null) {
    this.depthField = depth;
    if (depth) this.uploadTexture(this.depthTex, null, depth);
    if (this.field) this.buildGeometry();
  }

  /** Displacement at uv, nearest sample; the skirt uses it to meet the rim. */
  private sample(u: number, v: number): number {
    const f = this.depthField ?? this.field;
    if (!f) return 0;
    const x = Math.min(f.width - 1, Math.max(0, Math.round(u * (f.width - 1))));
    const y = Math.min(f.height - 1, Math.max(0, Math.round(v * (f.height - 1))));
    return f.data[(y * f.width + x) * 4] / 255;
  }

  private buildGeometry() {
    const gl = this.gl;
    const w = GRID;
    const h = Math.max(24, Math.round(GRID * this.aspect));

    const uvs = new Float32Array((w + 1) * (h + 1) * 2);
    let k = 0;
    for (let y = 0; y <= h; y++) {
      for (let x = 0; x <= w; x++) {
        uvs[k++] = x / w;
        uvs[k++] = y / h;
      }
    }
    const idx = new Uint32Array(w * h * 6);
    k = 0;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const a = y * (w + 1) + x;
        const b = a + 1;
        const c = a + w + 1;
        const d = c + 1;
        idx[k++] = a; idx[k++] = c; idx[k++] = b;
        idx[k++] = b; idx[k++] = c; idx[k++] = d;
      }
    }
    // Uint32 indices exceed WebGL1's default 16-bit limit past ~256 grid;
    // this grid stays under 65k vertices, so Uint16 is enough.
    const idx16 = new Uint16Array(idx);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.uvBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, uvs, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.indexBuffer);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, idx16, gl.STATIC_DRAW);
    this.indexCount = idx.length;

    // The cut edge of the model: a skirt from the sheet border down to the
    // base, shaded like paper, so a tilted view reads as a solid object.
    const skirt: number[] = [];
    const base = -0.045;
    const edge = (u: number, v: number) => [u - 0.5, (0.5 - v) * this.aspect];
    const ring: [number, number][] = [];
    for (let x = 0; x <= w; x++) ring.push([x / w, 0]);
    for (let y = 1; y <= h; y++) ring.push([1, y / h]);
    for (let x = w - 1; x >= 0; x--) ring.push([x / w, 1]);
    for (let y = h - 1; y >= 1; y--) ring.push([0, y / h]);
    ring.push(ring[0]);
    for (let i = 0; i < ring.length - 1; i++) {
      const [u0, v0] = ring[i];
      const [u1, v1] = ring[i + 1];
      const [x0, y0] = edge(u0, v0);
      const [x1, y1] = edge(u1, v1);
      // The top edge follows the terrain rim so the model reads as solid.
      const t0 = this.sample(u0, v0) * 0.17;
      const t1 = this.sample(u1, v1) * 0.17;
      skirt.push(
        x0, y0, t0, 1.0, x0, y0, base, 0.78, x1, y1, t1, 1.0,
        x1, y1, t1, 1.0, x0, y0, base, 0.78, x1, y1, base, 0.78,
      );
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, this.skirtBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(skirt), gl.STATIC_DRAW);
    this.skirtCount = skirt.length / 4;

    // Sea: one quad, slightly wider than the sheet so it reads as water, not
    // a decal.
    const m = 1.045;
    const sx = 0.5 * m;
    const sy = 0.5 * this.aspect * m;
    const water = new Float32Array([
      -sx, -sy, 0, 1, sx, -sy, 0, 1, -sx, sy, 0, 1,
      -sx, sy, 0, 1, sx, -sy, 0, 1, sx, sy, 0, 1,
    ]);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.waterBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, water, gl.STATIC_DRAW);
  }

  resize(cssWidth: number, cssHeight: number, dpr: number) {
    const w = Math.max(1, Math.round(cssWidth * dpr));
    const h = Math.max(1, Math.round(cssHeight * dpr));
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
  }

  render(view: ViewState) {
    const gl = this.gl;
    if (!this.ready) return;
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    const target = [0, 0, HEIGHT_SCALE * 0.35];
    const dist = 1.42 / view.zoom;
    const eye = [
      target[0] + dist * Math.cos(view.pitch) * Math.sin(view.yaw),
      target[1] - dist * Math.cos(view.pitch) * Math.cos(view.yaw),
      target[2] + dist * Math.sin(view.pitch),
    ];
    const canvasAspect = this.canvas.width / Math.max(1, this.canvas.height);
    const mvp = mul4(perspective(0.62, canvasAspect, 0.02, 12), lookAt(eye, target, [0, 0, 1]));

    // Terrain: the scene program when true positions exist.
    const prog = this.scenePositions ? this.scene : this.terrain;
    gl.useProgram(prog);
    gl.uniformMatrix4fv(gl.getUniformLocation(prog, "uMvp"), false, mvp);
    if (!this.scenePositions) {
      gl.uniform1f(gl.getUniformLocation(prog, "uHScale"), HEIGHT_SCALE);
      gl.uniform1f(gl.getUniformLocation(prog, "uAspect"), this.aspect);
      gl.uniform1f(gl.getUniformLocation(prog, "uUseDepth"), 0);
    }
    gl.uniform1f(gl.getUniformLocation(prog, "uWater"), view.waterline);
    gl.uniform1f(gl.getUniformLocation(prog, "uHasField"), this.hasField ? 1 : 0);
    gl.uniform2f(gl.getUniformLocation(prog, "uTexel"), this.texel[0], this.texel[1]);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.photoTex);
    gl.uniform1i(gl.getUniformLocation(prog, "uPhoto"), 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.heightTex);
    gl.uniform1i(gl.getUniformLocation(prog, "uHeight"), 1);
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, this.depthTex);
    gl.uniform1i(gl.getUniformLocation(prog, "uDepth"), 2);
    // The fragment's lift() feeds normals; in scene mode they come from depth.
    gl.uniform1f(gl.getUniformLocation(prog, "uUseDepth"), this.depthField ? 1 : 0);

    const aUv = gl.getAttribLocation(prog, "aUv");
    gl.bindBuffer(gl.ARRAY_BUFFER, this.uvBuffer);
    gl.enableVertexAttribArray(aUv);
    gl.vertexAttribPointer(aUv, 2, gl.FLOAT, false, 0, 0);
    let aScenePos = -1;
    if (this.scenePositions) {
      aScenePos = gl.getAttribLocation(prog, "aPos");
      gl.bindBuffer(gl.ARRAY_BUFFER, this.posBuffer);
      gl.enableVertexAttribArray(aScenePos);
      gl.vertexAttribPointer(aScenePos, 3, gl.FLOAT, false, 0, 0);
    }
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.indexBuffer);
    gl.drawElements(gl.TRIANGLES, this.indexCount, gl.UNSIGNED_SHORT, 0);
    gl.disableVertexAttribArray(aUv);
    if (aScenePos >= 0) gl.disableVertexAttribArray(aScenePos);

    // Skirt.
    gl.useProgram(this.flat);
    gl.uniformMatrix4fv(gl.getUniformLocation(this.flat, "uMvp"), false, mvp);
    gl.uniform4f(gl.getUniformLocation(this.flat, "uColor"), 0.726, 0.745, 0.699, 1.0);
    const aPos = gl.getAttribLocation(this.flat, "aPos");
    const aShade = gl.getAttribLocation(this.flat, "aShade");
    gl.bindBuffer(gl.ARRAY_BUFFER, this.skirtBuffer);
    gl.enableVertexAttribArray(aPos);
    gl.enableVertexAttribArray(aShade);
    gl.vertexAttribPointer(aPos, 3, gl.FLOAT, false, 16, 0);
    gl.vertexAttribPointer(aShade, 1, gl.FLOAT, false, 16, 12);
    gl.drawArrays(gl.TRIANGLES, 0, this.skirtCount);

    // Sea, translucent, at the waterline's elevation. Over real structure the
    // waterline is a colour cut, not a physical plane, so the sea stays home.
    if (this.depthField || this.scenePositions) {
      this.drawProps(mvp, aPos, aShade);
      gl.disableVertexAttribArray(aPos);
      gl.disableVertexAttribArray(aShade);
      return;
    }
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.depthMask(false);
    gl.uniform4f(gl.getUniformLocation(this.flat, "uColor"), 0.196, 0.435, 0.565, 0.44);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.waterBuffer);
    gl.vertexAttribPointer(aPos, 3, gl.FLOAT, false, 16, 0);
    gl.vertexAttribPointer(aShade, 1, gl.FLOAT, false, 16, 12);
    // The quad's z is authored at 0; lift it to the waterline in clip space by
    // re-uploading is wasteful, so draw it via a tiny model hack: the shader
    // has no model matrix, so bake the lift into the buffer each frame only
    // when the waterline changes. Cheaper: 6 vertices, rewrite is trivial.
    this.liftWater(view.waterline);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    gl.depthMask(true);
    gl.disable(gl.BLEND);
    this.drawProps(mvp, aPos, aShade);
    gl.disableVertexAttribArray(aPos);
    gl.disableVertexAttribArray(aShade);
  }

  private drawProps(mvp: Mat4, aPos: number, aShade: number) {
    if (!this.propGroups.length) return;
    const gl = this.gl;
    gl.useProgram(this.flat);
    gl.uniformMatrix4fv(gl.getUniformLocation(this.flat, "uMvp"), false, mvp);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.propBuffer);
    gl.enableVertexAttribArray(aPos);
    gl.enableVertexAttribArray(aShade);
    gl.vertexAttribPointer(aPos, 3, gl.FLOAT, false, 16, 0);
    gl.vertexAttribPointer(aShade, 1, gl.FLOAT, false, 16, 12);
    for (const g of this.propGroups) {
      const [r, gr, b, a] = g.color;
      if (a < 1) {
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
        gl.depthMask(false);
      }
      gl.uniform4f(gl.getUniformLocation(this.flat, "uColor"), r, gr, b, a);
      gl.drawArrays(gl.TRIANGLES, g.offset, g.count);
      if (a < 1) {
        gl.depthMask(true);
        gl.disable(gl.BLEND);
      }
    }
  }

  private lastWater = -1;
  private liftWater(waterline: number) {
    if (waterline === this.lastWater) return;
    this.lastWater = waterline;
    const gl = this.gl;
    const z = waterline * HEIGHT_SCALE;
    const m = 1.045;
    const sx = 0.5 * m;
    const sy = 0.5 * this.aspect * m;
    const water = new Float32Array([
      -sx, -sy, z, 1, sx, -sy, z, 1, -sx, sy, z, 1,
      -sx, sy, z, 1, sx, -sy, z, 1, sx, sy, z, 1,
    ]);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.waterBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, water, gl.STATIC_DRAW);
  }

  dispose() {
    // Free objects but keep the context alive: React StrictMode remounts the
    // component on the same canvas, and a killed context stays killed.
    const gl = this.gl;
    gl.deleteProgram(this.terrain);
    gl.deleteProgram(this.flat);
    gl.deleteTexture(this.photoTex);
    gl.deleteTexture(this.heightTex);
    gl.deleteTexture(this.depthTex);
    gl.deleteBuffer(this.uvBuffer);
    gl.deleteBuffer(this.indexBuffer);
    gl.deleteBuffer(this.skirtBuffer);
    gl.deleteBuffer(this.waterBuffer);
    gl.deleteBuffer(this.posBuffer);
    gl.deleteBuffer(this.propBuffer);
    this.ready = false;
  }
}
