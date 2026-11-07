// Render LaTeX strings with KaTeX, for make_packets.py.
//   node katex_render.js <path/to/katex.js> < in.json > out.json
// in.json: ["latex", ...]. out.json: [{"ok": bool, "mathml": "...", "error": "..."}, ...]
// MathML output needs no fonts or CSS, so packets.html works offline.
const katex = require(process.argv[2]);
let buf = "";
process.stdin.on("data", (d) => (buf += d));
process.stdin.on("end", () => {
  const out = JSON.parse(buf).map((s) => {
    try {
      katex.renderToString(s, { displayMode: true, throwOnError: true, strict: "ignore" });
      return { ok: true, mathml: katex.renderToString(s, { displayMode: true, output: "mathml", strict: "ignore" }) };
    } catch (e) {
      return { ok: false, error: String(e.message || e) };
    }
  });
  process.stdout.write(JSON.stringify(out));
});
