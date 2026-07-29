"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { EXAMPLES, type Example } from "@/lib/examples";
import type { Raster } from "@/lib/raster";
import { contours, coverage, decode, hatch, hexToRgb, iouAgainst, neatline, relief } from "@/lib/raster";
import type { LayerKey, Prediction } from "@/lib/types";
import Plate from "./Plate";

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
  { key: "trained", name: "Trained bridge", note: "relief", swatch: SHEET.contour },
  { key: "baseline", name: "Zero-shot baseline", note: "overprint", swatch: SHEET.revision },
  { key: "ground_truth", name: "Surveyed answer", note: "neatline", swatch: SHEET.hydro },
];

// A cold call runs about 20 s and a warm one about 8 s. The bar is paced
// against that rather than pretending to know real progress.
const EXPECTED_S = 22;

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
  const [waited, setWaited] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [elapsed, setElapsed] = useState<number | null>(null);

  const [layers, setLayers] = useState(START);
  const [waterline, setWaterline] = useState(0.5);
  const [base, setBase] = useState<HTMLImageElement | null>(null);
  const [rasters, setRasters] = useState<Partial<Record<LayerKey, Raster>>>({});

  const uploadRef = useRef<HTMLInputElement>(null);
  const objectUrl = useRef<string | null>(null);
  const flood = useRef<number | null>(null);

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

  const pickExample = (next: Example) => {
    setExample(next);
    setQuery(next.query);
    setImageUrl(next.image);
    setFilename(next.image.split("/").pop() ?? "frame.png");
    setBlob(null);
    setPrediction(null);
    setError(null);
  };

  const pickUpload = (file: File) => {
    if (objectUrl.current) URL.revokeObjectURL(objectUrl.current);
    objectUrl.current = URL.createObjectURL(file);
    setImageUrl(objectUrl.current);
    setBlob(file);
    setFilename(file.name);
    setPrediction(null);
    setError(null);
  };

  const read = useCallback(async () => {
    if (pending || !query.trim()) return;
    setPending(true);
    setError(null);
    setPrediction(null);
    const started = performance.now();

    try {
      // Fetch examples too, so uploads and examples take one path.
      const payload = blob ?? (await (await fetch(imageUrl)).blob());
      const form = new FormData();
      form.append("image", new File([payload], filename, { type: payload.type || "image/png" }));
      form.append("query", query.trim());

      const res = await fetch("/api/segment", { method: "POST", body: form });
      const body = await res.json();
      if (!res.ok) throw new Error(body?.error ?? `The request failed (${res.status}).`);

      const result = body as Prediction;
      setPrediction(result);
      setElapsed((performance.now() - started) / 1000);
      floodTo(result.trained?.threshold ?? 0.5);
    } catch (e) {
      setError(e instanceof Error ? e.message : "The request did not complete.");
    } finally {
      setPending(false);
    }
  }, [blob, filename, floodTo, imageUrl, pending, query]);

  const trained = prediction?.trained;
  const baseline = prediction?.baseline;
  const gt = prediction?.ground_truth;
  // Prefer the locally recomputed values; they track the waterline.
  const shownIou = reading.iou ?? gt?.trained_iou ?? null;
  const shownArea = reading.areaPct ?? trained?.area_pct ?? null;
  const progress = Math.min(0.94, waited / EXPECTED_S);

  return (
    <>
      <section className="mb-4">
        <label htmlFor="query" className="sheet-label">
          Question
        </label>
        <div className="mt-2 h-px bg-neat" />
        <div className="mt-3 flex flex-col gap-2 md:flex-row md:items-stretch">
          <textarea
            id="query"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) read();
            }}
            rows={2}
            className="flex-1 resize-none border border-neat bg-linen px-3 py-2 text-sm leading-relaxed outline-none placeholder:text-faint"
            placeholder="Where could a small drone land safely?"
          />
          <div className="flex gap-2 md:flex-col">
            <button
              type="button"
              onClick={read}
              disabled={pending || !query.trim()}
              className="sheet-label flex-1 bg-ink px-5 py-2.5 !text-linen transition disabled:opacity-40 md:flex-none"
            >
              {pending ? "Reading the scene" : "Read the scene"}
            </button>
            <button
              type="button"
              onClick={() => uploadRef.current?.click()}
              className="sheet-label border border-neat px-5 py-2.5 transition hover:border-ink hover:!text-ink md:flex-none"
            >
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
          </div>
        </div>

        {pending && (
          <div className="mt-3" role="status" aria-live="polite">
            <div className="h-[3px] w-full bg-mylar">
              <div
                className="h-full bg-contour transition-[width] duration-150 ease-linear"
                style={{ width: `${progress * 100}%` }}
              />
            </div>
            <p className="datum mt-1.5 flex justify-between text-[11px] text-faint">
              <span>
                {waited < 6
                  ? "Sending the photograph"
                  : waited < 14
                    ? "The bridge is reading it"
                    : "Waiting on the baseline's hosted model"}
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

      <figure className="m-0">
        <div className="border border-ink/25 bg-mylar">
          <Plate base={base} layers={mainLayers} liftable />
        </div>
        <figcaption className="sheet-label mt-2 flex justify-between">
          <span>Hold the sheet to lift the overlay</span>
          {prediction && (
            <span className="datum normal-case tracking-normal">
              {prediction.image.width} x {prediction.image.height} px
            </span>
          )}
        </figcaption>
      </figure>

      {trained && (
        <section className="mt-6">
          <div className="flex items-baseline justify-between">
            <label htmlFor="sea" className="sheet-label">
              Sea level
            </label>
            <span className="datum text-sm">{waterline.toFixed(2)}</span>
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
          <p className="mt-3 max-w-2xl text-xs leading-relaxed text-faint">
            Raise it and only the model&rsquo;s most confident ground stays above water. Nothing is
            sent when you drag: the whole confidence surface arrived with the answer.
          </p>
        </section>
      )}

      <section className="mt-9 grid gap-x-10 gap-y-8 md:grid-cols-2 lg:grid-cols-[minmax(0,15rem)_minmax(0,1fr)_minmax(0,17rem)]">
        <div>
          <h2 className="sheet-label">Layers</h2>
          <div className="mt-2 h-px bg-neat" />
          <ul className="mt-3">
            {LAYERS.map(({ key, name, note, swatch }) => {
              const has = key === "trained" ? trained : key === "baseline" ? baseline : gt;
              return (
                <li key={key}>
                  <button
                    type="button"
                    disabled={!has}
                    aria-pressed={Boolean(layers[key] && has)}
                    onClick={() => setLayers((l) => ({ ...l, [key]: !l[key] }))}
                    className="flex w-full items-center gap-2.5 py-1.5 text-left text-xs transition disabled:opacity-35"
                  >
                    <span
                      aria-hidden
                      className="h-2.5 w-2.5 shrink-0 border border-ink/30"
                      style={{ background: layers[key] && has ? swatch : "transparent" }}
                    />
                    <span className={layers[key] && has ? "text-ink" : "text-faint"}>{name}</span>
                    <span className="datum ml-auto text-[10px] text-faint">{note}</span>
                  </button>
                </li>
              );
            })}
          </ul>
        </div>

        <div>
          <div className="flex items-baseline justify-between">
            <h2 className="sheet-label">Reading</h2>
            <span className="sheet-label">{gt ? gt.family : prediction ? "not surveyed" : ""}</span>
          </div>
          <div className="mt-2 h-px bg-neat" />
          {prediction ? (
            <dl className="datum mt-4 grid grid-cols-2 gap-x-8 gap-y-3 sm:grid-cols-3">
              <Cell label="Agreement with survey" value={shownIou?.toFixed(3) ?? "not surveyed"} big />
              <Cell
                label="Baseline agreement"
                value={gt?.baseline_iou != null ? gt.baseline_iou.toFixed(3) : "not surveyed"}
                big
              />
              <Cell
                label="Ground above water"
                value={shownArea != null ? `${shownArea.toFixed(1)}%` : "-"}
                big
              />
              <Cell label="Bridge" value={trained ? `${trained.latency_s.toFixed(2)} s` : "-"} />
              <Cell label="Baseline" value={baseline ? `${baseline.latency_s.toFixed(2)} s` : "did not run"} />
              <Cell label="Round trip" value={elapsed != null ? `${elapsed.toFixed(1)} s` : "-"} />
            </dl>
          ) : (
            <p className="mt-3 max-w-md text-xs leading-relaxed text-faint">
              Pick a scene or upload a photograph, ask it something, and the numbers land here.
            </p>
          )}

          {prediction?.baseline_error && (
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
        </div>

        <div>
          <h2 className="sheet-label">Scenes</h2>
          <div className="mt-2 h-px bg-neat" />
          <ul className="mt-3">
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
                      className={`mt-1 h-1.5 w-1.5 shrink-0 rotate-45 border border-ink/40 ${active ? "bg-ink" : ""}`}
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
          <p className="mt-3 text-xs leading-relaxed text-faint">
            The first three read the same photograph. Identical pixels, three different answers.
          </p>
        </div>
      </section>

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
                  ? `Asked a hosted model for a box, then segmented inside it. It looked for: ${baseline.rationale}`
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

function Cell({ label, value, big }: { label: string; value: string; big?: boolean }) {
  return (
    <div>
      <dt className="sheet-label !text-[10px]">{label}</dt>
      <dd className={big ? "mt-1 text-xl" : "mt-1 text-sm"}>{value}</dd>
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
