import path from "node:path";

import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Without this, a lockfile further up the filesystem is picked as the root
  // and the build resolves modules from outside the app.
  turbopack: { root: path.resolve(".") },
};

export default nextConfig;
