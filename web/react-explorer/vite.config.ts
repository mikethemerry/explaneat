import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react";
import { createHash } from "crypto";
import { existsSync, readdirSync, readFileSync, statSync } from "fs";
import { join, relative, resolve, sep } from "path";

// Files whose content determines the built bundle. The same list and hash are
// computed in Python by explaneat/install.py (ui_source_hash), so `explaneat
// doctor` and the tests can tell whether the committed bundle in
// explaneat/static/react_explorer was built from the current source.
const ROOT_FILES = [
  "index.html",
  "package.json",
  "package-lock.json",
  "tsconfig.json",
  "tsconfig.node.json",
  "vite.config.ts",
];
const SOURCE_DIRS = ["src", "public"];

function listFiles(dir: string): string[] {
  if (!existsSync(dir)) return [];
  const out: string[] = [];
  for (const name of readdirSync(dir)) {
    if (name.startsWith(".")) continue; // .DS_Store, editor swap files
    const p = join(dir, name);
    if (statSync(p).isDirectory()) out.push(...listFiles(p));
    else out.push(p);
  }
  return out;
}

export function uiSourceHash(root: string): { sha256: string; files: number } {
  const files = [
    ...ROOT_FILES.map((f) => join(root, f)).filter((p) => existsSync(p)),
    ...SOURCE_DIRS.flatMap((d) => listFiles(join(root, d))),
  ];
  const rel = files.map((p) => relative(root, p).split(sep).join("/")).sort();
  const h = createHash("sha256");
  for (const r of rel) {
    // CRLF -> LF, so a Windows checkout hashes like macOS/Linux
    const text = readFileSync(join(root, r)).toString("latin1").replace(/\r\n/g, "\n");
    h.update(r, "utf8");
    h.update("\0");
    h.update(Buffer.from(text, "latin1"));
    h.update("\0");
  }
  return { sha256: h.digest("hex"), files: rel.length };
}

function buildInfo(): Plugin {
  return {
    name: "explaneat-build-info",
    generateBundle() {
      const { sha256, files } = uiSourceHash(__dirname);
      this.emitFile({
        type: "asset",
        fileName: "build-info.json",
        source:
          JSON.stringify(
            { source_sha256: sha256, source_files: files, source_dir: "web/react-explorer" },
            null,
            1,
          ) + "\n",
      });
    },
  };
}

export default defineConfig({
  base: "/",
  plugins: [react(), buildInfo()],
  server: {
    port: 3001,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
    rollupOptions: {
      input: resolve(__dirname, "index.html"),
    },
  },
});
