"use client";

import { useEffect, useMemo, useState } from "react";

import { droneAnchor, ringAnchors, routeAnchors, type UV } from "@/lib/anchors";
import type { Raster } from "@/lib/raster";
import { decode } from "@/lib/raster";
import type { SceneAssets } from "@/lib/scene";
import { loadScene } from "@/lib/scene";
import LoadBar from "./LoadBar";
import TerrainView, { type ScenePropSpec } from "./TerrainView";

/**
 * One reading, three jobs, three panels.
 *
 * Each panel studies its own patch of the same scene with its own camera,
 * device, and waterline: the model was asked once, and everything that
 * differs between the panels is policy.
 */

const READING = "/readings/uavid-000900-q04-s0.json";
const PHOTO = "/examples/uavid-000900.png";
const STEM = "uavid-000900";
const ASPECT = 0.5625;

const at = ([u, v]: UV) => ({ tx: (u - 0.5) * 0.92, ty: (0.5 - v) * ASPECT * 0.92 });

const CASES = [
  {
    key: "drone",
    name: "Drone landing",
    waterline: 0.85,
    line: "A landing decision tolerates no doubt. Only ground the model is nearly certain of stays above water, and the pad is placed where nothing is standing.",
    kind: "drone" as const,
    view: { yaw: -0.12, pitch: 0.78, zoom: 4.2 },
    Icon: IconDrone,
  },
  {
    key: "site",
    name: "Site access",
    waterline: 0.5,
    line: "Heavy equipment routed along the carriageway the model scored 0.745 against the survey. The route follows clear ground and bends around parked cars.",
    kind: "route" as const,
    view: { yaw: 0.12, pitch: 0.76, zoom: 3.1 },
    Icon: IconHardHat,
  },
  {
    key: "triage",
    name: "Disaster triage",
    waterline: 0.25,
    line: "After a storm, cast a wide net: beacons sweep the open sectors a low waterline surfaces, and a crew verifies by eye.",
    kind: "rings" as const,
    view: { yaw: -0.3, pitch: 0.95, zoom: 1.7 },
    Icon: IconBeacon,
  },
] as const;

export default function UseCases() {
  const [photo, setPhoto] = useState<HTMLImageElement | null>(null);
  const [field, setField] = useState<Raster | null>(null);
  // The annotated answer, when the frame has one: its holes are the cars.
  const [occupancy, setOccupancy] = useState<Raster | null>(null);
  const [scene, setScene] = useState<SceneAssets | null | undefined>(undefined);
  const [failed, setFailed] = useState(false);
  const [reliefFault, setReliefFault] = useState<string | null>(null);

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
        const gt = reading?.prediction?.ground_truth?.mask_png;
        if (typeof gt === "string") {
          const gtRaster = await decode(gt);
          if (live) setOccupancy(gtRaster);
        }
      } catch {
        if (live) setFailed(true);
      }
    })();
    void loadScene(STEM).then((assets) => live && setScene(assets));
    return () => {
      live = false;
    };
  }, []);

  if (failed) return null;

  // Measured once per reading: where the answer is confident and the surface
  // model says nothing is standing.
  const placements = useMemo(() => {
    if (!field) return null;
    const relief = scene?.depth ?? null;
    return {
      drone: droneAnchor(field, relief, occupancy),
      route: routeAnchors(field, relief, occupancy),
      rings: ringAnchors(field, relief, occupancy),
    };
  }, [field, scene, occupancy]);

  const loading = !field || scene === undefined || !placements;

  return (
    <section className="mt-14">
      <div className="h-px bg-ink" />
      <div className="flex flex-wrap items-baseline justify-between gap-x-8 gap-y-2 pt-5">
        <h2 className="sheet-title text-xl">What can you do with this</h2>
        <p className="datum text-[11px] text-faint">
          One inference. The waterline is the policy, not the model.
        </p>
      </div>

      {loading ? (
        <div
          className="mt-6 flex items-center justify-center border border-ink/25 bg-mylar"
          style={{ aspectRatio: "3 / 1" }}
        >
          <LoadBar
            label={!field ? "Fetching the cached reading" : "Standing the scene up"}
            value={!field ? 0.35 : 0.75}
          />
        </div>
      ) : (
        <div className="mt-6 grid gap-5 md:grid-cols-3">
          {CASES.map(({ key, name, waterline, line, kind, view, Icon }) => {
            const anchors = placements[kind === "route" ? "route" : kind === "rings" ? "rings" : "drone"];
            const focus = anchors[Math.floor(anchors.length / 2)] ?? [0.5, 0.5];
            // Beacons are a survey of the whole sheet; the other two panels
            // study one device and frame on it.
            const home = kind === "rings" ? { ...view } : { ...view, ...at(focus) };
            return (
            <figure key={key} className="group m-0">
              <div
                className="relative overflow-hidden border border-ink/25 bg-mylar transition group-hover:border-ink"
                style={{ aspectRatio: "4 / 3" }}
              >
                {reliefFault ? (
                  <img
                    src={PHOTO}
                    alt=""
                    className="absolute inset-0 h-full w-full object-cover opacity-80"
                  />
                ) : (
                  <div className="absolute inset-0">
                    <TerrainView
                      photo={photo}
                      field={field}
                      scene={scene ?? null}
                      waterline={waterline}
                      sceneProps={anchors.length ? { kind, anchors: anchors as [number, number][] } : null}
                      home={home}
                      onUnavailable={setReliefFault}
                    />
                  </div>
                )}
                <span className="datum pointer-events-none absolute bottom-1.5 right-2 bg-linen/85 px-1.5 py-0.5 text-[10px] text-faint">
                  waterline {waterline.toFixed(2)}
                </span>
              </div>
              <figcaption className="mt-2.5">
                <span className="flex items-center gap-2">
                  <Icon />
                  <span className="sheet-label !text-ink">{name}</span>
                </span>
                <p className="mt-1.5 text-xs leading-relaxed text-faint">{line}</p>
              </figcaption>
            </figure>
            );
          })}
        </div>
      )}
      {!loading && (
        <p className="mt-4 text-[11px] leading-relaxed text-faint">
          {reliefFault
            ? `Three panels, one photograph, one forward pass. The relief view is unavailable here (${reliefFault}), so the panels show the photograph flat.`
            : "Three panels, one photograph, one forward pass. Drag any panel to tilt it; scroll to zoom in on the device."}
        </p>
      )}
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
