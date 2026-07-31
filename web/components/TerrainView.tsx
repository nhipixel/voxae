"use client";

import { useCallback, useEffect, useRef } from "react";

import { buildProps, type PropKind } from "@/lib/props";
import type { SceneAssets, SceneGeometry } from "@/lib/scene";
import { buildSceneGeometry } from "@/lib/scene";
import type { FieldRaster } from "@/lib/terrain3d";
import { HOME_VIEW, TerrainRenderer, smoothField } from "@/lib/terrain3d";

export type ScenePropSpec = { kind: PropKind; anchors: [number, number][] } | null;

type Props = {
  photo: HTMLImageElement | null;
  /** The confidence field; null shows the bare scene, unpainted. */
  field: FieldRaster | null;
  /** Depth mesh assets; when present the photograph stands up on true
      geometry. Without them the field itself is extruded. */
  scene?: SceneAssets | null;
  waterline: number;
  /** Ambient mode: a slow self-orbit with no controls, for illustration. */
  ambient?: boolean;
  /** Animated illustration lines, anchored in texture coordinates. */
  sceneProps?: ScenePropSpec;
  className?: string;
};

const PITCH_MIN = 0.5;
const PITCH_MAX = 1.53;

function prefersStill() {
  return typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

/**
 * The relief view. With scene assets the photograph is draped over a depth
 * mesh in a ground-levelled world, so facades stand vertical; confidence is
 * the paint. Drag to orbit, scroll to zoom, double click to return home.
 */
export default function TerrainView({
  photo,
  field,
  scene = null,
  waterline,
  ambient = false,
  sceneProps = null,
  className = "",
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const renderer = useRef<TerrainRenderer | null>(null);
  const geometry = useRef<SceneGeometry | null>(null);
  const propSpec = useRef<ScenePropSpec>(null);
  const view = useRef({ ...HOME_VIEW, waterline });
  const raf = useRef<number | null>(null);
  const anim = useRef<number | null>(null);
  const drift = useRef<number | null>(null);
  const interacted = useRef(false);

  const draw = useCallback(() => {
    raf.current = null;
    const r = renderer.current;
    if (!r) return;
    const spec = propSpec.current;
    if (spec && geometry.current) {
      const world = spec.anchors.map(([u, v]) => geometry.current!.worldAt(u, v));
      r.setProps(buildProps(spec.kind, world, performance.now() / 1000));
    } else {
      r.setProps(null);
    }
    r.render(view.current);
    const canvas = canvasRef.current;
    if (canvas) {
      canvas.dataset.frames = String(Number(canvas.dataset.frames ?? 0) + 1);
      canvas.dataset.ready = String(r.ready);
    }
  }, []);

  const invalidate = useCallback(() => {
    if (raf.current == null) raf.current = requestAnimationFrame(draw);
  }, [draw]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    let r: TerrainRenderer | null = null;
    try {
      r = new TerrainRenderer(canvas);
    } catch (e) {
      // No WebGL: the parent keeps the flat sheet available. The reason is
      // recorded on the element so a blank view is diagnosable in the field.
      canvas.dataset.glError = e instanceof Error ? e.message : String(e);
      return;
    }
    canvas.dataset.glError = "";
    renderer.current = r;

    const parent = canvas.parentElement!;
    const observer = new ResizeObserver(() => {
      const rect = parent.getBoundingClientRect();
      canvas.dataset.observed = `${Math.round(rect.width)}x${Math.round(rect.height)}`;
      r!.resize(rect.width, rect.height, Math.min(2, window.devicePixelRatio || 1));
      invalidate();
    });
    observer.observe(parent);

    // Synchronous render hooks for tests and headless environments, where
    // frame callbacks never fire. Harmless in normal use.
    (canvas as HTMLCanvasElement & { __renderNow?: (w: number, h: number) => void }).__renderNow =
      (w: number, h: number) => {
        r!.resize(w, h, 1);
        draw();
      };
    (canvas as HTMLCanvasElement & {
      __captureView?: (yaw: number, pitch: number, zoom: number, w: number, h: number) => string;
    }).__captureView = (yaw, pitch, zoom, w, h) => {
      view.current = { ...view.current, yaw, pitch, zoom };
      r!.resize(w, h, 1);
      draw();
      return canvas.toDataURL("image/jpeg", 0.6);
    };

    return () => {
      observer.disconnect();
      if (raf.current != null) cancelAnimationFrame(raf.current);
      if (anim.current != null) cancelAnimationFrame(anim.current);
      if (drift.current != null) {
        cancelAnimationFrame(drift.current);
        drift.current = null;
      }
      r!.dispose();
      renderer.current = null;
    };
  }, [draw, invalidate]);

  useEffect(() => {
    if (photo && renderer.current) {
      renderer.current.setPhoto(photo);
      invalidate();
    }
  }, [photo, invalidate]);

  /** A slow turn, running until the visitor interacts or the tab hides. */
  const startDrift = useCallback(() => {
    if (drift.current != null) {
      cancelAnimationFrame(drift.current);
      drift.current = null;
    }
    if (prefersStill()) return;
    let lastT = performance.now();
    let phase = 0;
    const spin = (now: number) => {
      drift.current = null;
      if (interacted.current) return;
      if (!document.hidden) {
        // A pendulum across the photographed side: a full orbit would parade
        // the sheared back of every building.
        phase += ((now - lastT) / 1000) * (ambient ? 0.22 : 0.07);
        view.current.yaw = HOME_VIEW.yaw + Math.sin(phase) * 0.42;
        draw();
      }
      lastT = now;
      drift.current = requestAnimationFrame(spin);
    };
    drift.current = requestAnimationFrame(spin);
  }, [ambient, draw]);

  useEffect(
    () => () => {
      if (drift.current != null) {
        cancelAnimationFrame(drift.current);
        drift.current = null;
      }
    },
    [],
  );

  // New geometry is the moment the scene stands up: tilt in from overhead.
  useEffect(() => {
    const canvas = canvasRef.current;
    const r = renderer.current;
    if (canvas) canvas.dataset.fieldState = r ? (scene || field ? "set" : "none") : "no-renderer";
    if (!r || (!scene && !field)) return;

    if (scene) {
      geometry.current = buildSceneGeometry(scene);
      r.setScene(
        geometry.current.positions,
        geometry.current.gridW,
        geometry.current.gridH,
        scene.depth,
      );
      r.setField(field ? smoothField(field) : null);
    } else {
      geometry.current = null;
      r.setField(smoothField(field!));
    }
    if (canvas) canvas.dataset.depth = scene ? "scene" : "field";

    if (anim.current != null) cancelAnimationFrame(anim.current);
    if (ambient || prefersStill()) {
      view.current = { ...view.current, ...HOME_VIEW, pitch: ambient ? 0.92 : HOME_VIEW.pitch };
      invalidate();
      startDrift();
      return;
    }
    const from = { yaw: 0, pitch: PITCH_MAX };
    const start = performance.now();
    const step = (now: number) => {
      const t = Math.min(1, (now - start) / 1100);
      const e = 1 - Math.pow(1 - t, 3);
      view.current.yaw = from.yaw + (HOME_VIEW.yaw - from.yaw) * e;
      view.current.pitch = from.pitch + (HOME_VIEW.pitch - from.pitch) * e;
      draw();
      if (t < 1) anim.current = requestAnimationFrame(step);
      else startDrift();
    };
    view.current = { ...view.current, ...from, zoom: HOME_VIEW.zoom };
    anim.current = requestAnimationFrame(step);
  }, [scene, field, ambient, draw, invalidate, startDrift]);

  // Props animate on their own clock, independent of camera drift.
  const propTick = useRef<number | null>(null);
  useEffect(() => {
    propSpec.current = sceneProps;
    if (propTick.current != null) {
      cancelAnimationFrame(propTick.current);
      propTick.current = null;
    }
    if (sceneProps && !prefersStill()) {
      const tick = () => {
        propTick.current = null;
        if (!propSpec.current) return;
        if (!document.hidden) draw();
        propTick.current = requestAnimationFrame(tick);
      };
      propTick.current = requestAnimationFrame(tick);
    } else {
      invalidate();
    }
    return () => {
      if (propTick.current != null) {
        cancelAnimationFrame(propTick.current);
        propTick.current = null;
      }
    };
  }, [sceneProps, draw, invalidate]);

  useEffect(() => {
    view.current.waterline = waterline;
    invalidate();
  }, [waterline, invalidate]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || ambient) return;
    let dragging = false;
    let last = [0, 0];

    const down = (e: PointerEvent) => {
      dragging = true;
      interacted.current = true;
      last = [e.clientX, e.clientY];
      canvas.setPointerCapture(e.pointerId);
      if (anim.current != null) cancelAnimationFrame(anim.current);
    };
    const move = (e: PointerEvent) => {
      if (!dragging) return;
      view.current.yaw += (e.clientX - last[0]) * 0.008;
      view.current.pitch = Math.min(
        PITCH_MAX,
        Math.max(PITCH_MIN, view.current.pitch + (e.clientY - last[1]) * 0.006),
      );
      last = [e.clientX, e.clientY];
      invalidate();
    };
    const up = () => {
      dragging = false;
    };
    const wheel = (e: WheelEvent) => {
      e.preventDefault();
      interacted.current = true;
      view.current.zoom = Math.min(
        4.5,
        Math.max(0.75, view.current.zoom * (e.deltaY < 0 ? 1.09 : 0.92)),
      );
      invalidate();
    };
    const home = () => {
      view.current = { ...view.current, ...HOME_VIEW };
      invalidate();
    };

    canvas.addEventListener("pointerdown", down);
    canvas.addEventListener("pointermove", move);
    canvas.addEventListener("pointerup", up);
    canvas.addEventListener("pointercancel", up);
    canvas.addEventListener("wheel", wheel, { passive: false });
    canvas.addEventListener("dblclick", home);
    return () => {
      canvas.removeEventListener("pointerdown", down);
      canvas.removeEventListener("pointermove", move);
      canvas.removeEventListener("pointerup", up);
      canvas.removeEventListener("pointercancel", up);
      canvas.removeEventListener("wheel", wheel);
      canvas.removeEventListener("dblclick", home);
    };
  }, [ambient, invalidate]);

  return (
    <canvas
      ref={canvasRef}
      className={`block h-full w-full touch-none ${ambient ? "" : "cursor-grab active:cursor-grabbing"} ${className}`}
      aria-label="Relief model: the photograph standing on estimated scene structure, painted by the model's confidence. Drag to tilt, scroll to zoom, double click to reset."
    />
  );
}
