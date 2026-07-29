"use client";

import { useEffect, useRef, useState } from "react";

import { toCanvas } from "@/lib/raster";

type Props = {
  base: HTMLImageElement | null;
  layers: (ImageData | null | undefined)[];
  /** Lets the reader press and hold to see the photograph underneath. */
  liftable?: boolean;
  className?: string;
  /** Hands the drawn canvas back, so the sheet can be saved as it appears. */
  canvasRef?: React.RefObject<HTMLCanvasElement | null>;
};

/** One photograph with overlays printed on top, drawn at the image's own size. */
export default function Plate({
  base,
  layers,
  liftable = false,
  className = "",
  canvasRef,
}: Props) {
  const own = useRef<HTMLCanvasElement>(null);
  const ref = canvasRef ?? own;
  const [lifted, setLifted] = useState(false);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas || !base) return;
    canvas.width = base.naturalWidth;
    canvas.height = base.naturalHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(base, 0, 0, canvas.width, canvas.height);
    if (lifted) return;

    ctx.imageSmoothingEnabled = false;
    for (const layer of layers) {
      if (layer) ctx.drawImage(toCanvas(layer), 0, 0, canvas.width, canvas.height);
    }
  }, [base, layers, lifted]);

  return (
    <canvas
      ref={ref}
      className={`block h-auto w-full select-none ${className}`}
      onPointerDown={liftable ? () => setLifted(true) : undefined}
      onPointerUp={liftable ? () => setLifted(false) : undefined}
      onPointerLeave={liftable ? () => setLifted(false) : undefined}
    />
  );
}
