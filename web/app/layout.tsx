import type { Metadata } from "next";
import { Archivo, IBM_Plex_Mono, Public_Sans } from "next/font/google";

import "./globals.css";

const display = Archivo({
  subsets: ["latin"],
  axes: ["wdth"],
  variable: "--font-display",
  display: "swap",
});

const body = Public_Sans({
  subsets: ["latin"],
  variable: "--font-body",
  display: "swap",
});

const data = IBM_Plex_Mono({
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  variable: "--font-data",
  display: "swap",
});

export const metadata: Metadata = {
  title: "Voxae — read a scene by asking it a question",
  description:
    "Ask an aerial scene a question in plain language and get back the region that answers it. The model's confidence is drawn as terrain, and you set the waterline.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${display.variable} ${body.variable} ${data.variable}`}>
      <body>{children}</body>
    </html>
  );
}
