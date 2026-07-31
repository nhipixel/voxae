"use client";

import { useCallback, useEffect, useRef } from "react";

import type { FieldRaster } from "@/lib/terrain3d";
import { HOME_VIEW, TerrainRenderer, smoothField } from "@/lib/terrain3d";

type Props = {
  photo: HTMLImageElement | null;
  field: FieldRaster | null;
  waterline: number;
  /** Ambient mode: a slow self-orbit with no controls, for illustration. */
  ambient?: boolean;
  className?: string;
};

const PITCH_MIN = 0.32;
const PITCH_MAX = 1.53;

function prefersStill() {
  return typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

/**
 * The relief view: the confidence surface as a physical terrain model.
 *
 * Arrives tilted by an entrance move from straight overhead, which is the
 * flat sheet's viewpoint, so the two views read as one object. Drag to orbit,
 * scroll to zoom, double click to return home.
 */
export default function TerrainView({ photo, field, waterline, ambient = false, className = "" }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const renderer = useRef<TerrainRenderer | null>(null);
  const view = useRef({ ...HOME_VIEW, waterline });
  const raf = useRef<number | null>(null);
  const anim = useRef<number | null>(null);
  const drift = useRef<number | null>(null);
  const interacted = useRef(false);

  const draw = useCallback(() => {
    raf.current = null;
    renderer.current?.render(view.current);
    const canvas = canvasRef.current;
    if (canvas) {
      canvas.dataset.frames = String(Number(canvas.dataset.frames ?? 0) + 1);
      canvas.dataset.ready = String(renderer.current?.ready ?? "no-renderer");
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

    // Synchronous render hook for tests and headless environments, where
    // frame callbacks never fire. Harmless in normal use.
    (canvas as HTMLCanvasElement & { __renderNow?: (w: number, h: number) => void }).__renderNow =
      (w: number, h: number) => {
        r!.resize(w, h, 1);
        draw();
      };

    return () => {
      observer.disconnect();
      if (raf.current != null) cancelAnimationFrame(raf.current);
      if (anim.current != null) cancelAnimationFrame(anim.current);
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
    if (drift.current != null || prefersStill()) return;
    let lastT = performance.now();
    const spin = (now: number) => {
      drift.current = null;
      if (interacted.current) return;
      if (!document.hidden) {
        view.current.yaw += ((now - lastT) / 1000) * (ambient ? 0.14 : 0.045);
        draw();
      }
      lastT = now;
      drift.current = requestAnimationFrame(spin);
    };
    drift.current = requestAnimationFrame(spin);
  }, [ambient, draw]);

  useEffect(() => () => {
    if (drift.current != null) cancelAnimationFrame(drift.current);
  }, []);

  // A new field is the moment the model answers: tilt up from overhead.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (canvas) canvas.dataset.fieldState = field ? (renderer.current ? "set" : "no-renderer") : "none";
    if (!field || !renderer.current) return;
    renderer.current.setField(smoothField(field));
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
  }, [field, ambient, draw, invalidate, startDrift]);

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
      view.current.zoom = Math.min(2.4, Math.max(0.65, view.current.zoom * (e.deltaY < 0 ? 1.08 : 0.93)));
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
      aria-label="Relief model of the confidence surface. Drag to tilt, scroll to zoom, double click to reset."
    />
  );
}
