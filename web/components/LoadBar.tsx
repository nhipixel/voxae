/** A thin staged progress bar, so a working load never looks like a blank. */
export default function LoadBar({ label, value }: { label: string; value: number }) {
  return (
    <div className="w-56 max-w-[70%]" role="status" aria-live="polite">
      <div className="h-[3px] w-full bg-neat/60">
        <div
          className="h-full bg-contour transition-[width] duration-300 ease-out"
          style={{ width: `${Math.round(Math.min(1, Math.max(0, value)) * 100)}%` }}
        />
      </div>
      <p className="sheet-label mt-2 text-center">{label}</p>
    </div>
  );
}
