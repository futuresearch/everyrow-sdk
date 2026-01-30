import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "export",
  trailingSlash: false,
  images: { unoptimized: true },
  basePath: "/docs",

  // Rewrites for local dev - ignored in static export since public/ files
  // are copied directly to out/
  async rewrites() {
    return {
      // beforeFiles runs before the filesystem (including public/) is checked,
      // which lets us serve notebook HTML before the catch-all route matches
      beforeFiles: [
        {
          source: "/notebooks/:path*",
          destination: "/notebooks/:path*/index.html",
        },
      ],
    };
  },
};

export default nextConfig;
