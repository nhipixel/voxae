import { ImageResponse } from "next/og";

export const size = { width: 1200, height: 630 };
export const contentType = "image/png";
export const alt =
  "Voxae. Ask an aerial photograph a question and get the region that answers it, with the model's confidence drawn as terrain.";

/**
 * Archivo as a TTF, which is what Satori can parse.
 *
 * The stylesheet only serves woff2 to a modern user agent, so this asks as an
 * old one. A failure here must not fail the build, so the card falls back to
 * the default face.
 */
async function archivo(): Promise<ArrayBuffer | null> {
  try {
    const css = await fetch(
      "https://fonts.googleapis.com/css2?family=Archivo:wght@600&display=swap",
      { headers: { "User-Agent": "Mozilla/5.0 (Windows NT 6.1)" } },
    ).then((r) => r.text());
    const url = /src:\s*url\((.+?)\)/.exec(css)?.[1];
    if (!url) return null;
    return await fetch(url).then((r) => r.arrayBuffer());
  } catch {
    return null;
  }
}

const LINEN = "#dce0d8";
const INK = "#22282a";
const FAINT = "#5f6a64";
const CONTOUR = "#4a3416";
const HYDRO = "#2c6b8a";

export default async function Image() {
  const font = await archivo();

  return new ImageResponse(
    (
      <div
        style={{
          position: "relative",
          width: "100%",
          height: "100%",
          display: "flex",
          background: LINEN,
          backgroundImage: `linear-gradient(to right, rgba(34,40,42,.05) 1px, transparent 1px), linear-gradient(to bottom, rgba(34,40,42,.05) 1px, transparent 1px)`,
          backgroundSize: "40px 40px",
          color: INK,
          fontFamily: font ? "Archivo" : "sans-serif",
        }}
      >
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            width: 712,
            height: "100%",
            padding: 64,
          }}
        >
          <div style={{ display: "flex", height: 3, width: "100%", background: INK }} />

          <div
            style={{ display: "flex", flexDirection: "column", flex: 1, justifyContent: "center" }}
          >
            <div style={{ display: "flex", fontSize: 108, letterSpacing: -2, lineHeight: 1 }}>
              Voxae
            </div>
            <div style={{ display: "flex", marginTop: 24, fontSize: 31, lineHeight: 1.3 }}>
              Ask an aerial photograph a question in plain language and get the region that
              answers it.
            </div>
            <div style={{ display: "flex", marginTop: 18, fontSize: 21, color: FAINT }}>
              Confidence is drawn as terrain. You set the waterline.
            </div>
          </div>

          <div style={{ display: "flex", height: 1, width: "100%", background: "#a9b1a5" }} />
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              width: "100%",
              marginTop: 16,
              fontSize: 19,
              color: FAINT,
            }}
          >
            <div style={{ display: "flex" }}>Qwen2-VL 2B to SAM 2.1</div>
            <div style={{ display: "flex" }}>Affordance gIoU 0.421 against 0.290</div>
          </div>
        </div>

        {/* A map inset on the sheet: terrain above a waterline, bleeding off
            three edges, kept clear of the type by a hard division. */}
        <div style={{ display: "flex", position: "relative" }}>
          <div style={{ display: "flex", width: 1, height: "100%", background: "#a9b1a5" }} />
          <svg width="487" height="630" viewBox="0 0 487 630">
            <path
              d="M0 452c58-16 88 10 146 3s88-26 150-15 76 20 138 10 63-13 63-13V630H0z"
              fill={HYDRO}
              opacity=".9"
            />
            <g fill="none" stroke={CONTOUR} strokeWidth="3" strokeLinecap="round">
              <path d="M0 450c58-16 88 10 146 3s88-26 150-15 76 20 138 10 63-13 63-13" />
              <path d="M0 386c60-18 80-56 150-66s114 38 168 17 96-40 119-42" opacity=".78" />
              <path d="M26 322c58-20 84-50 146-60s104 31 154 12 84-33 107-35" opacity=".6" />
              <path d="M84 258c50-18 78-42 130-50s88 24 132 7 70-26 91-28" opacity=".44" />
              <path d="M158 198c40-14 65-31 105-38s70 19 105 5" opacity=".3" />
              <path d="M238 144c29-10 49-22 81-27s51 13 78 3" opacity=".18" />
            </g>
          </svg>
        </div>
      </div>
    ),
    {
      ...size,
      fonts: font ? [{ name: "Archivo", data: font, weight: 600, style: "normal" }] : undefined,
    },
  );
}
