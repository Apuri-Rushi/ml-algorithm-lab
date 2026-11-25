import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Requests to /api go to the local Python model server (website/server.py).
const api = { "/api": { target: "http://127.0.0.1:8000", changeOrigin: true } };

export default defineConfig({
  plugins: [react()],
  base: "./",
  server: { proxy: api },
  preview: { proxy: api },
});
