import UseCases from "@/components/UseCases";
import Workbench from "@/components/Workbench";

const RECORD = [
  { split: "Affordance questions", trained: "0.421", zero: "0.290", n: "222" },
  { split: "Referring questions", trained: "0.399", zero: "0.538", n: "84" },
  { split: "All", trained: "0.414", zero: "0.371", n: "306" },
];

export default function Home() {
  return (
    <main className="mx-auto max-w-[1440px] px-6 py-10 lg:px-10">
      <header className="mb-9">
        <div className="rule-in h-px bg-ink" />
        <div className="settle flex flex-wrap items-start justify-between gap-x-8 gap-y-4 pt-5">
          <div className="max-w-xl">
            <h1 className="sheet-title text-[2.6rem] leading-[1.05]">Voxae</h1>
            <p className="sheet-label mt-2 !text-ink">
              A vision language model that segments what you ask for, for drones and robots that
              have to read a scene at runtime
            </p>
            <p className="mt-3 text-[0.95rem] leading-relaxed">
              Ask an aerial photograph a question in plain language. It answers with a region rather
              than a label, and its confidence is drawn as terrain: contours mark equal certainty,
              and you set the waterline that decides what counts.
            </p>
          </div>
          <dl className="datum grid grid-cols-2 gap-x-6 gap-y-1 text-[11px] text-faint">
            <dt>Sheet</dt>
            <dd className="text-ink">UAVid, held out</dd>
            <dt>Bridge</dt>
            <dd className="text-ink">Qwen2-VL 2B to SAM 2.1</dd>
            <dt>Survey</dt>
            <dd className="text-ink">306 annotated samples</dd>
          </dl>
        </div>
      </header>

      <Workbench />

      <section className="mt-14">
        <div className="h-px bg-ink" />
        <div className="grid gap-10 pt-6 md:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
          <div>
            <h2 className="sheet-label">Survey record</h2>
            <table className="datum mt-4 w-full text-xs">
              <thead className="text-faint">
                <tr>
                  <th className="pb-2 text-left font-normal">Question type</th>
                  <th className="pb-2 text-right font-normal">Bridge</th>
                  <th className="pb-2 text-right font-normal">Baseline</th>
                  <th className="pb-2 text-right font-normal">n</th>
                </tr>
              </thead>
              <tbody>
                {RECORD.map((row) => (
                  <tr key={row.split} className="border-t border-neat">
                    <td className="py-2 font-sans">{row.split}</td>
                    <td className="py-2 text-right">{row.trained}</td>
                    <td className="py-2 text-right text-faint">{row.zero}</td>
                    <td className="py-2 text-right text-faint">{row.n}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="mt-4 text-xs leading-relaxed text-faint">
              Mean IoU against the annotated answer. The baseline wins on referring questions,
              which name a thing you can point at. The bridge wins on affordance questions, which
              ask what a surface is good for. Both are printed here because the second result only
              means something beside the first.
            </p>
          </div>

          <div className="space-y-4 text-xs leading-relaxed text-faint">
            <h2 className="sheet-label">How the sheet is drawn</h2>
            <p>
              Your browser sends the photograph to a route handler on this site, which passes it to
              the model and reads the answer back off a stream. The model is never addressed from
              the browser and no credentials reach it.
            </p>
            <p>
              What comes back is the confidence surface itself, about 60 KB of greyscale, not a
              finished picture. Every contour, every tint, and every number that moves when you drag
              the waterline is computed here from that one response. The answer arrives once and
              then costs nothing to re-read.
            </p>
            <p>
              An uploaded photograph has no annotated answer, so it is drawn but not scored. The
              five listed scenes are held-out evaluation samples and carry theirs.
            </p>
          </div>
        </div>
      </section>

      <UseCases />

      <footer className="datum mt-12 flex flex-wrap justify-between gap-4 border-t border-neat pt-4 text-[11px] text-faint">
        <span>Photography from the UAVid dataset, used under its research licence.</span>
        <span className="flex gap-5">
          <a className="underline underline-offset-4 hover:text-ink" href="https://github.com/nhipixel/voxae">
            Source
          </a>
          <a
            className="underline underline-offset-4 hover:text-ink"
            href="https://huggingface.co/spaces/nhibuilds/voxae"
          >
            Model
          </a>
        </span>
      </footer>
    </main>
  );
}
