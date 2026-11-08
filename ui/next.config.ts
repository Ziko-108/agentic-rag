import type { NextConfig } from "next";

const config: NextConfig = {
  output: "standalone", // a small self-contained server for the Docker image
  compress: false, // gzip would buffer the server-sent events of /chat/stream
};

export default config;
