"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import type { Raster } from "@/lib/raster";
import { decode } from "@/lib/raster";
import LoadBar from "./LoadBar";
import TerrainView from "./TerrainView";

/**
 * One reading, three operating points.
 *
 * The model is asked once; everything after that is policy. Each card sets a
 * different waterline on the same cached confidence surface, which is the
 * honest form of "what can you do with this": the applications differ in how
 * much doubt they can afford, not in the model that serves them.
 */

const READING = "/readings/uavid-000900-q04-s0.json";
const PHOTO = "/examples/uavid-000900.png";
const DEPTH = "/depth/uavid-000900.png";

const CASES = [
  {
    key: "drone",
    name: "Drone landing",
    waterline: 0.85,
    line: "A landing decision tolerates no doubt. Raise the waterline and only ground the model is nearly certain of stays above water.",
    Icon: IconDrone,
  },
  {
    key: "site",
    name: "Site access",
    waterline: 0.5,
    line: "Route heavy equipment across ground the model has never seen. This is the operating point the survey scored: 0.745 agreement.",
    Icon: IconHardHat,
  },
  {
    key: "triage",
    name: "Disaster triage",
    waterline: 0.25,
    line: "After a storm, cast a wide net and verify by eye. A low waterline surfaces every surface worth a second look.",
    Icon: IconBeacon,
  },
] as const;

function prefersStill() {
  return typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

export default function UseCases() {
  const [photo, setPhoto] = useState<HTMLImageElement | null>(null);
  const [field, setField] = useState<Raster | null>(null);
  const [depth, setDepth] = useState<Raster | null | undefined>(undefined);
  const [failed, setFailed] = useState(false);
  const [active, setActive] = useState(0);
  const [waterline, setWaterline] = useState<number>(CASES[0].waterline);
  const anim = useRef<number | null>(null);
  const cycle = useRef<number | null>(null);
  const touched = useRef(false);

  useEffect(() => {
    let live = true;
    const img = new Image();
    img.onload = () => live && setPhoto(img);
    img.src = PHOTO;
    (async () => {
      try {
        const res = await fetch(READING);
        if (!res.ok) throw new Error(String(res.status));
        const reading = await res.json();
        const probs = reading?.prediction?.trained?.probs_png;
        if (typeof probs !== "string") throw new Error("no field");
        const raster = await decode(probs);
        if (live) setField(raster);
      } catch {
        if (live) setFailed(true);
      }
    })();
    void decode(DEPTH)
      .then((raster) => live && setDepth(raster))
      .catch(() => live && setDepth(null));
    return () => {
      live = false;
      if (anim.current != null) cancelAnimationFrame(anim.current);
      if (cycle.current != null) window.clearTimeout(cycle.current);
    };
  }, []);

  const glideTo = useCallback((target: number) => {
    if (anim.current != null) cancelAnimationFrame(anim.current);
    if (prefersStill()) {
      setWaterline(target);
      return;
    }
    const start = performance.now();
    let from: number | null = null;
    const step = (now: number) => {
      setWaterline((current: number) => {
        if (from == null) from = current;
        const t = Math.min(1, (now - start) / 900);
        const e = 1 - Math.pow(1 - t, 3);
        if (t < 1) anim.current = requestAnimationFrame(step);
        return from + (target - from) * e;
      });
    };
    anim.current = requestAnimationFrame(step);
  }, []);

  const pick = useCallback(
    (index: number, byUser: boolean) => {
      if (byUser) touched.current = true;
      setActive(index);
      glideTo(CASES[index].waterline);
    },
    [glideTo],
  );

  // The cards take turns until the visitor chooses one.
  useEffect(() => {
    if (!field || prefersStill()) return;
    const tick = () => {
      cycle.current = window.setTimeout(() => {
        if (!touched.current && !document.hidden) {
          setActive((i) => {
            const next = (i + 1) % CASES.length;
            glideTo(CASES[next].waterline);
            return next;
          });
        }
        if (!touched.current) tick();
      }, 5200);
    };
    tick();
    return () => {
      if (cycle.current != null) window.clearTimeout(cycle.current);
    };
  }, [field, glideTo]);

  if (failed) return null;

  const loading = !field || depth === undefined;

  return (
    <section className="mt-14">
      <div className="h-px bg-ink" />
      <div className="flex flex-wrap items-baseline justify-between gap-x-8 gap-y-2 pt-5">
        <h2 className="sheet-title text-xl">What can you do with this</h2>
        <p className="datum text-[11px] text-faint">
          One inference. The waterline is the policy, not the model.
        </p>
      </div>

      <div className="mt-6 grid gap-8 lg:grid-cols-[minmax(0,1fr)_21rem]">
        <div className="relative border border-ink/25 bg-mylar" style={{ aspectRatio: "16 / 9" }}>
          {loading ? (
            <div className="absolute inset-0 flex items-center justify-center">
              <LoadBar
                label={!field ? "Fetching the cached reading" : "Standing the scene up"}
                value={!field ? 0.35 : 0.75}
              />
            </div>
          ) : (
            <>
              <div className="pointer-events-none absolute inset-0">
                <TerrainView
                  photo={photo}
                  field={field}
                  depth={depth ?? null}
                  waterline={waterline}
                  ambient
                />
              </div>
              <div className="pointer-events-none absolute bottom-2 left-2 flex items-baseline gap-3 bg-linen/85 px-2 py-1">
                <span className="sheet-label !text-ink">{CASES[active].name}</span>
                <span className="datum text-[11px] text-faint">
                  waterline {waterline.toFixed(2)}. Colour is confidence; the terrain is the scene
                </span>
              </div>
            </>
          )}
        </div>

        <ul className="flex flex-col gap-1.5">
          {CASES.map(({ key, name, line, waterline: w, Icon }, i) => (
            <li key={key}>
              <button
                type="button"
                aria-pressed={i === active}
                onClick={() => pick(i, true)}
                className={`w-full border px-4 py-3 text-left transition ${
                  i === active ? "border-ink bg-mylar/50" : "border-neat hover:border-ink"
                }`}
              >
                <span className="flex items-center gap-2.5">
                  <Icon />
                  <span className="sheet-label !text-ink">{name}</span>
                  <span className="datum ml-auto text-[10px] text-faint">{w.toFixed(2)}</span>
                </span>
                <span className="mt-1.5 block text-xs leading-relaxed text-faint">{line}</span>
              </button>
            </li>
          ))}
          <li className="mt-1 px-1 text-[11px] leading-relaxed text-faint">
            The same photograph and the same single forward pass serve all three. Retraining is
            replaced by choosing how much confidence the job demands.
          </li>
        </ul>
      </div>
    </section>
  );
}

const GLYPH = {
  width: 15,
  height: 15,
  viewBox: "0 0 16 16",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.3,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
  "aria-hidden": true,
};

function IconDrone() {
  return (
    <svg {...GLYPH}>
      <circle cx="3.2" cy="3.2" r="1.9" />
      <circle cx="12.8" cy="3.2" r="1.9" />
      <circle cx="3.2" cy="12.8" r="1.9" />
      <circle cx="12.8" cy="12.8" r="1.9" />
      <path d="M4.6 4.6 6.5 6.5m5-1.9L9.5 6.5m-5 5 1.9-1.9m5 1.9L9.5 9.5M6.5 6.5h3v3h-3z" />
    </svg>
  );
}

function IconHardHat() {
  return (
    <svg {...GLYPH}>
      <path d="M2 11.5a6 6 0 0 1 12 0" />
      <path d="M1.5 11.5h13v1.8h-13z" />
      <path d="M6.6 5.4V3.6h2.8v1.8" />
    </svg>
  );
}

function IconBeacon() {
  return (
    <svg {...GLYPH}>
      <path d="M5.5 13.5 6.6 7h2.8l1.1 6.5z" />
      <path d="M4 13.5h8" />
      <path d="M8 4.5V2.2M4.4 5.6 3 4.2m8.6 1.4L13 4.2" />
    </svg>
  );
}
