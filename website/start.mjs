// Starts the Python model server and the website together:  npm start
// Press Ctrl+C once to stop both.
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const win = process.platform === "win32";

const server = spawn("python", [join(here, "server.py")], { stdio: "inherit", shell: win });
const viteArgs = process.env.NO_OPEN ? ["vite"] : ["vite", "--open"];
const site = spawn(win ? "npx.cmd" : "npx", viteArgs, { cwd: here, stdio: "inherit", shell: win });

const stop = () => {
  for (const p of [server, site]) {
    if (p.exitCode === null) {
      if (win) spawn("taskkill", ["/pid", String(p.pid), "/T", "/F"], { stdio: "ignore" });
      else p.kill("SIGINT");
    }
  }
};
process.on("SIGINT", () => { stop(); process.exit(0); });
server.on("exit", (code) => { if (code) console.error(`Model server stopped (exit ${code}).`); });
site.on("exit", () => { stop(); process.exit(0); });
