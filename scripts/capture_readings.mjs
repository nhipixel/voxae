// Capture one cached reading per worked example from the deployed endpoint.
//
//   node scripts/capture_readings.mjs https://voxae.vercel.app
//
// Each response is written verbatim to web/public/readings/<sample_id>.json,
// stamped with the capture date. Rerun whenever the checkpoint changes; the
// files are what the app serves as its zero-second first impression. Uses the
// public route (baseline included), so expect roughly a minute per example
// and a healthy share of the day's GPU quota.

import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const base = (process.argv[2] ?? "https://voxae.vercel.app").replace(/\/$/, "");

// Mirrors EXAMPLES in web/lib/examples.ts.
const EXAMPLES = [
  ["Identify all ground surfaces, paved or otherwise cleared, that a heavy vehicle could drive across.", "uavid-000900"],
  ["Find an open stretch of road big enough to stage equipment or park several vehicles side by side.", "uavid-000900"],
  ["Show road surface that stays clear of overhanging tree canopy, suitable for a vehicle to idle without branch clearance issues.", "uavid-000900"],
  ["Mark the ground surfaces firm enough for a heavy vehicle to drive across, ignoring grass and trees.", "uavid-000400"],
  ["Highlight the large building block occupying the upper-left portion of the frame.", "uavid-000400"],
];

mkdirSync(join(root, "web/public/readings"), { recursive: true });

let manifest = {};
for (const [query, stem] of EXAMPLES) {
  const image = readFileSync(join(root, `web/public/examples/${stem}.png`));
  const form = new FormData();
  form.append("image", new File([image], `${stem}.png`, { type: "image/png" }));
  form.append("query", query);
  form.append("baseline", "1");

  process.stdout.write(`${stem}  ${query.slice(0, 50)}... `);
  const t0 = Date.now();
  const res = await fetch(`${base}/api/segment`, { method: "POST", body: form });
  const body = await res.json();
  if (!res.ok || !body.trained) {
    console.log(`FAILED (${res.status}) ${body.error ?? ""}`);
    continue;
  }
  const id = body.ground_truth?.sample_id;
  if (!id) {
    console.log("no ground truth returned; refusing to cache an unscored reading");
    continue;
  }
  writeFileSync(
    join(root, `web/public/readings/${id}.json`),
    JSON.stringify({
      captured_at: new Date().toISOString().slice(0, 10),
      captured_from: "the deployed model, via the same endpoint the live path uses",
      prediction: body,
    }),
  );
  manifest[query] = id;
  console.log(`ok in ${((Date.now() - t0) / 1000).toFixed(0)}s -> ${id}.json`);
}

console.log("\nPaste into READING_IDS in web/lib/readings.ts:\n");
console.log(JSON.stringify(manifest, null, 2));
