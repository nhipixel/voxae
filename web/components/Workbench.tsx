"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { EXAMPLES, type Example } from "@/lib/examples";
import { loadReading } from "@/lib/readings";
import type { SceneAssets } from "@/lib/scene";
import { loadScene } from "@/lib/scene";
import type { Raster } from "@/lib/raster";
import { contours, coverage, decode, hatch, hexToRgb, iouAgainst, neatline, relief } from "@/lib/raster";
import type { LayerKey, Prediction } from "@/lib/types";
import LoadBar from "./LoadBar";
import Plate from "./Plate";
import TerrainView from "./TerrainView";

export const SHEET = {
  contour: "#7a5b2e",
  hydro: "#2c6b8a",
  revision: "#8e3a26",
} as const;

// Contour lines sit on the tint they describe, so they are drawn a few steps
// darker than the swatch that stands for the layer.
const CONTOUR_LINE = "#4a3416";
const LEVELS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];

const START: Record<LayerKey, boolean> = { trained: true, baseline: false, ground_truth: true };

const LAYERS: { key: LayerKey; name: string; note: string; swatch: string }[] = [
  { key: "trained", name: "Trained bridge", note: "tinted relief", swatch: SHEET.contour },
  { key: "baseline", name: "Zero-shot baseline", note: "diagonal hatch", swatch: SHEET.revision },
  { key: "ground_truth", name: "Surveyed answer", note: "outline only", swatch: SHEET.hydro },
];

// The bridge answers in about a second once the GPU is allocated; the baseline
// waits on a hosted model that has been taking far longer. The bar is paced
// against each rather than pretending to know real progress.
const EXPECTED_BRIDGE_S = 8;
const EXPECTED_BASELINE_S = 45;

function prefersStill() {
  return typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

export default function Workbench() {
  const [example, setExample] = useState<Example>(EXAMPLES[0]);
  const [imageUrl, setImageUrl] = useState(EXAMPLES[0].image);
  const [blob, setBlob] = useState<Blob | null>(null);
  const [filename, setFilename] = useState("uavid-000900.png");
  const [query, setQuery] = useState(EXAMPLES[0].query);

  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [pending, setPending] = useState(false);
  const [comparing, setComparing] = useState(false);
  const [waited, setWaited] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [elapsed, setElapsed] = useState<number | null>(null);
  const [cachedAt, setCachedAt] = useState<string | null>(null);
  const [sheetMode, setSheetMode] = useState<"relief" | "flat">("relief");
  const [reliefFault, setReliefFault] = useState<string | null>(null);
  // The relief opens as the scene alone; the answer is painted on request.
  const [paintConfidence, setPaintConfidence] = useState(false);
  const [booting, setBooting] = useState(true);

  const [layers, setLayers] = useState(START);
  const [waterline, setWaterline] = useState(0.5);
  const [base, setBase] = useState<HTMLImageElement | null>(null);
  const [rasters, setRasters] = useState<Partial<Record<LayerKey, Raster>>>({});
  const [scene, setScene] = useState<SceneAssets | null | undefined>(undefined);

  const uploadRef = useRef<HTMLInputElement>(null);
  const objectUrl = useRef<string | null>(null);
  const flood = useRef<number | null>(null);
  const sheetRef = useRef<HTMLCanvasElement>(null);

  useEffect(
    () => () => {
      if (objectUrl.current) URL.revokeObjectURL(objectUrl.current);
      if (flood.current) cancelAnimationFrame(flood.current);
    },
    [],
  );

  useEffect(() => {
    const img = new Image();
    img.onload = () => setBase(img);
    img.src = imageUrl;
    return () => {
      img.onload = null;
    };
  }, [imageUrl]);

  // Scene structure ships only for the worked example frames; an upload
  // falls back to extruding the confidence field itself.
  useEffect(() => {
    let cancelled = false;
    if (blob || !imageUrl.startsWith("/examples/")) {
      setScene(null);
      return;
    }
    setScene(undefined);
    const stem = imageUrl.slice("/examples/".length).replace(/\.[^.]+$/, "");
    // One retry, then a visible fallback: a missing 3D asset must never
    // silently impersonate a design change.
    void loadScene(stem)
      .then(async (assets) => assets ?? loadScene(stem))
      .then((assets) => {
        if (cancelled) return;
        setScene(assets);
        if (!assets) setError("The 3D scene assets did not load; showing the flat sheet.");
      });
    return () => {
      cancelled = true;
    };
  }, [blob, imageUrl]);

  // Decode once per response, not once per waterline change.
  useEffect(() => {
    let cancelled = false;
    if (!prediction) {
      setRasters({});
      return;
    }
    (async () => {
      const sources = [
        ["trained", prediction.trained?.probs_png],
        ["baseline", prediction.baseline?.mask_png],
        ["ground_truth", prediction.ground_truth?.mask_png],
      ] as const;
      const entries = await Promise.all(
        sources
          .filter(([, uri]) => Boolean(uri))
          .map(async ([key, uri]) => [key, await decode(uri as string)] as const),
      );
      if (!cancelled) setRasters(Object.fromEntries(entries));
    })();
    return () => {
      cancelled = true;
    };
  }, [prediction]);

  useEffect(() => {
    if (!pending) return;
    setWaited(0);
    const started = performance.now();
    const id = window.setInterval(() => setWaited((performance.now() - started) / 1000), 120);
    return () => window.clearInterval(id);
  }, [pending]);

  /** Drop the waterline to its final level so the answer rises out of the sea. */
  const floodTo = useCallback((target: number) => {
    if (flood.current) cancelAnimationFrame(flood.current);
    if (prefersStill()) {
      setWaterline(target);
      return;
    }
    const from = 0.95;
    const start = performance.now();
    const step = (now: number) => {
      const t = Math.min(1, (now - start) / 1100);
      setWaterline(from + (target - from) * (1 - Math.pow(1 - t, 3)));
      if (t < 1) flood.current = requestAnimationFrame(step);
    };
    setWaterline(from);
    flood.current = requestAnimationFrame(step);
  }, []);

  // Drawn layers. Contours and the two masks depend only on the response;
  // relief is the one that has to follow the waterline.
  const reliefLayer = useMemo(
    () => (rasters.trained ? relief(rasters.trained, waterline) : null),
    [rasters.trained, waterline],
  );
  const contourLayer = useMemo(
    () => (rasters.trained ? contours(rasters.trained, LEVELS, hexToRgb(CONTOUR_LINE)) : null),
    [rasters.trained],
  );
  const hatchLayer = useMemo(
    () => (rasters.baseline ? hatch(rasters.baseline, hexToRgb(SHEET.revision)) : null),
    [rasters.baseline],
  );
  const neatlineLayer = useMemo(
    () => (rasters.ground_truth ? neatline(rasters.ground_truth, hexToRgb(SHEET.hydro)) : null),
    [rasters.ground_truth],
  );

  const mainLayers = useMemo(
    () => [
      layers.baseline ? hatchLayer : null,
      layers.trained ? reliefLayer : null,
      layers.trained ? contourLayer : null,
      layers.ground_truth ? neatlineLayer : null,
    ],
    [layers, hatchLayer, reliefLayer, contourLayer, neatlineLayer],
  );

  const reading = useMemo(() => {
    const field = rasters.trained;
    if (!field) return { areaPct: null as number | null, iou: null as number | null };
    return {
      areaPct: coverage(field, waterline),
      iou: rasters.ground_truth ? iouAgainst(field, rasters.ground_truth, waterline) : null,
    };
  }, [rasters, waterline]);

  const showCached = useCallback(
    async (query: string) => {
      const reading = await loadReading(query);
      if (!reading) return false;
      setPrediction(reading.prediction);
      setCachedAt(reading.captured_at);
      setElapsed(null);
      floodTo(reading.prediction.trained?.threshold ?? 0.5);
      return true;
    },
    [floodTo],
  );

  const pickExample = (next: Example) => {
    setExample(next);
    setQuery(next.query);
    setImageUrl(next.image);
    setFilename(next.image.split("/").pop() ?? "frame.png");
    setBlob(null);
    setPrediction(null);
    setCachedAt(null);
    setError(null);
    void showCached(next.query);
  };

  // First load shows the scene itself, unpainted: no numbers a visitor never
  // asked for. Readings appear when they pick an example or run live.
  useEffect(() => {
    if (scene !== undefined || error) setBooting(false);
  }, [rasters.trained, scene, error]);

  const pickUpload = (file: File) => {
    if (objectUrl.current) URL.revokeObjectURL(objectUrl.current);
    objectUrl.current = URL.createObjectURL(file);
    setImageUrl(objectUrl.current);
    setBlob(file);
    setFilename(file.name);
    setPrediction(null);
    setError(null);
  };

  const read = useCallback(
    async (withBaseline = false) => {
      if (pending || !query.trim()) return;
      setPending(true);
      setComparing(withBaseline);
      setCachedAt(null);
      setError(null);
      if (!withBaseline) setPrediction(null);
      const started = performance.now();

      try {
        // Fetch examples too, so uploads and examples take one path.
        const payload = blob ?? (await (await fetch(imageUrl)).blob());
        const form = new FormData();
        form.append("image", new File([payload], filename, { type: payload.type || "image/png" }));
        form.append("query", query.trim());
        if (withBaseline) form.append("baseline", "1");

        const res = await fetch("/api/segment", { method: "POST", body: form });
        // A platform level failure answers in plain text, so parsing blind
        // turns a readable problem into a syntax error.
        const raw = await res.text();
        let body: Prediction & { error?: string };
        try {
          body = JSON.parse(raw);
        } catch {
          throw new Error(
            res.status === 504 || res.status === 502
              ? "The model did not answer in time. Try again."
              : `The server returned an unexpected response (${res.status}).`,
          );
        }
        if (!res.ok) throw new Error(body?.error ?? `The request failed (${res.status}).`);

        const result = body as Prediction;
        setPrediction(result);
        setElapsed((performance.now() - started) / 1000);
        if (!withBaseline) floodTo(result.trained?.threshold ?? 0.5);
        if (withBaseline) setLayers((l) => ({ ...l, baseline: true }));
      } catch (e) {
        setError(e instanceof Error ? e.message : "The request did not complete.");
      } finally {
        setPending(false);
        setComparing(false);
      }
    },
    [blob, filename, floodTo, imageUrl, pending, query],
  );

  const stem = () => {
    const id = prediction?.ground_truth?.sample_id;
    return id ?? (blob ? filename.replace(/\.[^.]+$/, "") : "voxae");
  };

  const save = (blob: Blob, name: string) => {
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = name;
    a.click();
    URL.revokeObjectURL(url);
  };

  /** The reading as it stands, including what the waterline currently implies. */
  const exportReading = () => {
    if (!prediction) return;
    const record = {
      query: prediction.query,
      image: prediction.image,
      waterline,
      bridge: prediction.trained && {
        model: prediction.trained.model,
        latency_s: prediction.trained.latency_s,
        server_threshold: prediction.trained.threshold,
        area_pct_at_waterline: reading.areaPct,
        agreement_at_waterline: reading.iou,
        mean_confidence_in_mask: prediction.trained.mean_confidence_in_mask,
      },
      baseline: prediction.baseline && {
        model: prediction.baseline.model,
        latency_s: prediction.baseline.latency_s,
        area_pct: prediction.baseline.area_pct,
        bbox_norm: prediction.baseline.bbox_norm,
        rationale: prediction.baseline.rationale,
        live: prediction.baseline.live,
      },
      baseline_skipped: prediction.baseline_skipped ?? false,
      baseline_error: prediction.baseline_error ?? null,
      ground_truth: prediction.ground_truth && {
        sample_id: prediction.ground_truth.sample_id,
        family: prediction.ground_truth.family,
        bridge_iou_at_server_threshold: prediction.ground_truth.trained_iou,
        baseline_iou: prediction.ground_truth.baseline_iou,
      },
      agreement_between_predictors: prediction.agreement_iou,
      round_trip_s: elapsed,
    };
    save(
      new Blob([JSON.stringify(record, null, 2)], { type: "application/json" }),
      `${stem()}-reading.json`,
    );
  };

  /** The sheet exactly as drawn, overlays and waterline included. */
  const exportSheet = () => {
    sheetRef.current?.toBlob((b) => b && save(b, `${stem()}-sheet.png`), "image/png");
  };

  const trained = prediction?.trained;
  const sceneLoading = sheetMode === "relief" && scene === undefined;
  const showRelief =
    sheetMode === "relief" &&
    !sceneLoading &&
    !reliefFault &&
    Boolean(scene || (prediction?.trained && rasters.trained));
  const baseline = prediction?.baseline;
  const gt = prediction?.ground_truth;
  // Prefer the locally recomputed values; they track the waterline.
  const shownIou = reading.iou ?? gt?.trained_iou ?? null;
  const shownArea = reading.areaPct ?? trained?.area_pct ?? null;
  const progress = Math.min(
    0.94,
    waited / (comparing ? EXPECTED_BASELINE_S : EXPECTED_BRIDGE_S),
  );

  return (
    <>
      {/* The sheet and the things you ask it, side by side. */}
      <div className="grid gap-x-9 gap-y-6 lg:grid-cols-[minmax(0,1fr)_21rem]">
        <figure className="m-0">
          {!booting && (
            <div className="mb-2 flex items-center justify-between">
              <div className="flex gap-1">
                {(
                  [
                    ["relief", "Relief model"],
                    ["flat", "Flat sheet"],
                  ] as const
                ).map(([mode, name]) => (
                  <button
                    key={mode}
                    type="button"
                    aria-pressed={sheetMode === mode}
                    onClick={() => setSheetMode(mode)}
                    className={`sheet-label border px-2.5 py-1 transition ${
                      sheetMode === mode
                        ? "border-ink !text-ink"
                        : "border-transparent hover:border-neat"
                    }`}
                  >
                    {name}
                  </button>
                ))}
              </div>
              {sheetMode === "relief" && (
                <button
                  type="button"
                  aria-pressed={paintConfidence && Boolean(rasters.trained)}
                  disabled={!rasters.trained}
                  title={
                    rasters.trained
                      ? "Paint the model's confidence onto the scene"
                      : "Read the scene first: there is no answer to paint yet"
                  }
                  onClick={() => setPaintConfidence((on) => !on)}
                  className={`sheet-label flex items-center gap-2 border px-2.5 py-1 transition ${
                    !rasters.trained
                      ? "cursor-not-allowed border-neat/50 opacity-40"
                      : paintConfidence
                        ? "border-ink !text-ink"
                        : "border-neat hover:border-ink"
                  }`}
                >
                  <span
                    aria-hidden
                    className="h-2.5 w-2.5 shrink-0 border border-ink/40"
                    style={{
                      background: paintConfidence && rasters.trained ? "#cfa63f" : "transparent",
                    }}
                  />
                  Confidence
                </button>
              )}
            </div>
          )}

          <div className="relative border border-ink/25 bg-mylar">
            {/* The flat plate stays mounted underneath: it is the export
                surface and the fallback when WebGL is unavailable. */}
            <div className={showRelief || booting ? "invisible" : ""}>
              <Plate base={base} layers={mainLayers} liftable canvasRef={sheetRef} />
            </div>
            {(booting || sceneLoading) && (
              <div className="absolute inset-0 flex items-center justify-center">
                <LoadBar
                  label={prediction ? "Decoding the confidence surface" : "Standing the scene up"}
                  value={prediction && !rasters.trained ? 0.55 : 0.8}
                />
              </div>
            )}
            {showRelief && !booting && (
              <div className="absolute inset-0">
                <TerrainView
                  photo={base}
                  field={paintConfidence ? (rasters.trained ?? null) : null}
                  scene={scene ?? null}
                  onUnavailable={setReliefFault}
                  waterline={waterline}
                />
              </div>
            )}
            {showRelief && !booting && prediction && paintConfidence && (
              <span className="sheet-label pointer-events-none absolute right-2 top-2 bg-linen/85 px-2 py-0.5">
                {scene ? "Colour is confidence" : "Elevation is confidence"}
              </span>
            )}
          </div>
          <figcaption className="sheet-label mt-2 flex justify-between">
            <span>
              {reliefFault
              ? `Relief view unavailable here (${reliefFault}); showing the flat sheet`
              : showRelief
                ? !prediction
                  ? "The scene stands ready. Read the scene to paint the answer onto it"
                  : scene
                    ? "Colour and waterline are the model’s confidence on real structure. Drag to tilt, scroll to zoom, double click resets"
                    : "Height is the model’s confidence. Drag to tilt, scroll to zoom, double click resets"
                : cachedAt
                  ? `Cached reading, ${cachedAt}. Read the scene reruns it live`
                  : prediction
                    ? "Hold the sheet to lift the overlay"
                    : "Nothing drawn yet"}
            </span>
            <span className="datum normal-case tracking-normal">
              {blob ? filename : example.image.includes("000900") ? "Frame A" : "Frame B"}
            </span>
          </figcaption>
        </figure>

        <aside className="flex flex-col gap-6">
          <section>
            <h2 className="sheet-label">Pick a photograph</h2>
            <div className="mt-2 h-px bg-neat" />
            <ul className="mt-2">
              {EXAMPLES.map((ex) => {
                const active = example.query === ex.query && !blob;
                return (
                  <li key={ex.query}>
                    <button
                      type="button"
                      aria-current={active ? "true" : undefined}
                      onClick={() => pickExample(ex)}
                      className={`flex w-full items-baseline gap-2 py-1.5 text-left text-xs transition ${
                        active ? "text-ink" : "text-faint hover:text-ink"
                      }`}
                    >
                      <span
                        aria-hidden
                        className={`h-1.5 w-1.5 shrink-0 rotate-45 border border-ink/40 ${active ? "bg-ink" : ""}`}
                      />
                      <span>{ex.label}</span>
                      <span className="datum ml-auto text-[10px]">
                        {ex.image.includes("000900") ? "A" : "B"}
                      </span>
                    </button>
                  </li>
                );
              })}
            </ul>
            <button
              type="button"
              onClick={() => uploadRef.current?.click()}
              className="sheet-label mt-2 flex w-full items-center justify-center gap-2 border border-neat py-2 transition hover:border-ink hover:!text-ink"
            >
              <IconUpload />
              Upload image
            </button>
            <input
              ref={uploadRef}
              type="file"
              accept="image/png,image/jpeg,image/webp"
              className="hidden"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) pickUpload(file);
                e.target.value = "";
              }}
            />
            <p className="mt-2 text-[11px] leading-relaxed text-faint">
              {blob
                ? "Your photograph has no annotated answer, so this run will not be scored."
                : "The first three scenes are the same photograph asked three different things."}
            </p>
          </section>

          <section>
            <label htmlFor="query" className="sheet-label">
              Ask it something
            </label>
            <div className="mt-2 h-px bg-neat" />
            <textarea
              id="query"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) read();
              }}
              rows={4}
              className="mt-3 w-full resize-none border border-neat bg-linen px-3 py-2 text-sm leading-relaxed outline-none placeholder:text-faint"
              placeholder="Where could a small drone land safely?"
            />
            <button
              type="button"
              onClick={() => read(false)}
              disabled={pending || !query.trim()}
              className="sheet-label mt-2 w-full bg-ink py-3 !text-linen transition disabled:opacity-40"
            >
              {pending && !comparing ? "Reading the scene" : "Read the scene"}
            </button>

            {prediction && !baseline && (
              <button
                type="button"
                onClick={() => read(true)}
                disabled={pending}
                className="sheet-label mt-2 w-full border border-neat py-2.5 transition hover:border-ink hover:!text-ink disabled:opacity-40"
              >
                {comparing ? "Asking the baseline" : "Compare against zero-shot"}
              </button>
            )}

            {pending && (
              <div className="mt-3" role="status" aria-live="polite">
                <div className="h-[3px] w-full bg-mylar">
                  <div
                    className="h-full bg-contour transition-[width] duration-150 ease-linear"
                    style={{ width: `${progress * 100}%` }}
                  />
                </div>
                <p className="datum mt-1.5 flex justify-between gap-2 text-[11px] text-faint">
                  <span>
                    {comparing
                      ? "Running the zero-shot baseline on the same query"
                      : waited < 4
                        ? "Sending the photograph"
                        : "The bridge is reading it"}
                  </span>
                  <span>{waited.toFixed(1)} s</span>
                </p>
              </div>
            )}

            {error && (
              <p className="mt-3 border-l-2 border-revision bg-revision/5 px-3 py-2 text-xs leading-relaxed">
                {error}
              </p>
            )}
          </section>
        </aside>
      </div>

      {/* A legend belongs with its map, so it sits directly beneath. */}
      {prediction && (
        <section className="mt-5">
          <div className="h-px bg-neat" />
          <div className="flex flex-wrap items-center gap-x-8 gap-y-2 pt-3">
            <h2 className="sheet-label">Showing</h2>
            {LAYERS.map(({ key, name, note, swatch }) => {
              const has = key === "trained" ? trained : key === "baseline" ? baseline : gt;
              return (
                <button
                  key={key}
                  type="button"
                  disabled={!has}
                  aria-pressed={Boolean(layers[key] && has)}
                  onClick={() => setLayers((l) => ({ ...l, [key]: !l[key] }))}
                  className="flex items-center gap-2 text-xs transition disabled:opacity-35"
                >
                  <span
                    aria-hidden
                    className="h-2.5 w-2.5 shrink-0 border border-ink/30"
                    style={{ background: layers[key] && has ? swatch : "transparent" }}
                  />
                  <span className={layers[key] && has ? "text-ink" : "text-faint"}>{name}</span>
                  <span className="datum text-[10px] text-faint">{note}</span>
                </button>
              );
            })}
          </div>
        </section>
      )}

      {trained && (
        <section className="mt-6">
          <div className="flex items-baseline justify-between gap-4">
            <label htmlFor="sea" className="sheet-label">
              Sea level
            </label>
            <p className="text-xs text-faint">
              Only ground the model is at least{" "}
              <span className="datum text-ink">{Math.round(waterline * 100)}%</span> sure of counts
              as the answer
            </p>
          </div>
          <div className="mt-2 h-px bg-neat" />
          <input
            id="sea"
            type="range"
            min={0.05}
            max={0.95}
            step={0.01}
            value={waterline}
            onChange={(e) => {
              if (flood.current) cancelAnimationFrame(flood.current);
              setWaterline(Number(e.target.value));
            }}
            className="gauge mt-4"
          />
          <p className="datum mt-1 flex justify-between text-[10px] text-faint">
            <span>5%, almost everything counts</span>
            <span>95%, only what it is certain of</span>
          </p>
        </section>
      )}

      {prediction && (
        <section className="mt-8">
          <div className="flex flex-wrap items-baseline justify-between gap-x-6 gap-y-2">
            <h2 className="sheet-label">Reading</h2>
            <div className="flex flex-wrap items-center gap-3">
              <button
                type="button"
                onClick={exportReading}
                className="sheet-label flex items-center gap-2 border border-neat px-3 py-1.5 transition hover:border-ink hover:!text-ink"
              >
                <IconDownload />
                Export reading
              </button>
              <button
                type="button"
                onClick={exportSheet}
                className="sheet-label flex items-center gap-2 border border-neat px-3 py-1.5 transition hover:border-ink hover:!text-ink"
              >
                <IconDownload />
                Export sheet
              </button>
              <span className="sheet-label">{gt ? `${gt.family} question` : "not surveyed"}</span>
            </div>
          </div>
          <div className="mt-2 h-px bg-neat" />
          <dl className="datum mt-4 grid grid-cols-2 gap-x-8 gap-y-5 sm:grid-cols-3 lg:grid-cols-6">
            <Cell label="Bridge agreement" value={shownIou?.toFixed(3) ?? "not surveyed"} />
            <Cell
              label="Baseline agreement"
              // Null for three different reasons: nobody asked for the
              // baseline, it was asked and failed, or there is no annotated
              // answer to score anything against.
              value={
                gt?.baseline_iou != null
                  ? gt.baseline_iou.toFixed(3)
                  : prediction.baseline_skipped
                    ? "not compared"
                    : gt
                      ? "did not run"
                      : "not surveyed"
              }
            />
            <Cell
              label="Ground above water"
              value={shownArea != null ? `${shownArea.toFixed(1)}%` : "-"}
            />
            <Cell label="Bridge took" value={trained ? `${trained.latency_s.toFixed(2)} s` : "-"} />
            <Cell
              label="Baseline took"
              value={
                baseline
                  ? `${baseline.latency_s.toFixed(2)} s`
                  : prediction.baseline_skipped
                    ? "not compared"
                    : "did not run"
              }
            />
            <Cell label="Round trip" value={elapsed != null ? `${elapsed.toFixed(1)} s` : "-"} />
          </dl>
          <p className="mt-4 max-w-3xl text-xs leading-relaxed text-faint">
            Agreement is the overlap between an answer and the annotated region, from 0 to 1. The
            first three numbers are recomputed here as you move the sea level; nothing is sent when
            you drag, because the whole confidence surface arrived with the answer.
          </p>

          {prediction.baseline_error && (
            <p className="mt-4 border-l-2 border-revision pl-3 text-xs leading-relaxed text-faint">
              The baseline&rsquo;s hosted grounding model did not answer. The reading above is
              unaffected.
            </p>
          )}
          {baseline && !baseline.live && (
            <p className="mt-4 border-l-2 border-revision pl-3 text-xs leading-relaxed text-faint">
              The baseline is returning placeholder boxes: no grounding key is set.
            </p>
          )}
        </section>
      )}

      {prediction && (
        <section className="mt-12">
          <div className="h-px bg-ink" />
          <div className="flex flex-wrap items-baseline justify-between gap-x-8 gap-y-2 pt-5">
            <h2 className="sheet-title text-xl">Each answer on its own sheet</h2>
            <p className="datum text-[11px] text-faint">
              Same photograph, same question, printed separately
            </p>
          </div>

          <div className="mt-6 grid gap-6 md:grid-cols-3">
            <Detail
              name="Trained bridge"
              swatch={SHEET.contour}
              score={shownIou}
              caption="Relief tinted by confidence, with contours at every tenth. This is the whole surface the model produced, not a decision about it."
              base={base}
              layers={[reliefLayer, contourLayer]}
              available={Boolean(trained)}
            />
            <Detail
              name="Zero-shot baseline"
              swatch={SHEET.revision}
              score={gt?.baseline_iou ?? null}
              caption={
                baseline?.rationale
                  ? `Asks a hosted model for a box, then segments inside it. It looked for: ${baseline.rationale}`
                  : prediction.baseline_skipped
                    ? "Asks a hosted model for a box, then segments inside it. Run the comparison to fill this in."
                    : "Asks a hosted model for a box, then segments inside it. It did not answer this time."
              }
              base={base}
              layers={[hatchLayer]}
              available={Boolean(baseline)}
            />
            <Detail
              name="Surveyed answer"
              swatch={SHEET.hydro}
              score={null}
              caption={
                gt
                  ? `The annotated region for ${gt.sample_id}. Both columns are scored against this outline.`
                  : "An uploaded photograph has no annotated region, so neither column can be scored."
              }
              base={base}
              layers={[neatlineLayer]}
              available={Boolean(gt)}
            />
          </div>
        </section>
      )}
    </>
  );
}

function Cell({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="sheet-label !text-[10px]">{label}</dt>
      <dd className="mt-1 text-xl">{value}</dd>
    </div>
  );
}

function Detail({
  name,
  swatch,
  score,
  caption,
  base,
  layers,
  available,
}: {
  name: string;
  swatch: string;
  score: number | null;
  caption: string;
  base: HTMLImageElement | null;
  layers: (ImageData | null | undefined)[];
  available: boolean;
}) {
  return (
    <figure className={`m-0 ${available ? "" : "opacity-45"}`}>
      <div className="border border-ink/25 bg-mylar">
        <Plate base={base} layers={available ? layers : []} />
      </div>
      <figcaption className="mt-2">
        <div className="flex items-baseline gap-2">
          <span aria-hidden className="h-2.5 w-2.5 shrink-0 border border-ink/30" style={{ background: swatch }} />
          <span className="sheet-label !text-ink">{name}</span>
          {score != null && <span className="datum ml-auto text-sm">{score.toFixed(3)}</span>}
        </div>
        <p className="mt-2 text-xs leading-relaxed text-faint">{caption}</p>
      </figcaption>
    </figure>
  );
}

/** Hairline glyphs, stroked in the current colour so they inherit button state. */
const GLYPH = {
  width: 13,
  height: 13,
  viewBox: "0 0 16 16",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.4,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
  "aria-hidden": true,
};

function IconUpload() {
  return (
    <svg {...GLYPH}>
      <path d="M8 10.5V2.5M8 2.5 5.2 5.3M8 2.5l2.8 2.8" />
      <path d="M2.5 10.5v3h11v-3" />
    </svg>
  );
}

/** The upload glyph with its arrow reversed, so the pair reads as one set. */
function IconDownload() {
  return (
    <svg {...GLYPH}>
      <path d="M8 2.5v8M8 10.5 5.2 7.7M8 10.5l2.8-2.8" />
      <path d="M2.5 10.5v3h11v-3" />
    </svg>
  );
}
