/**
 * Pairs plot (viz type `pairs_plot`): a k x k grid over chosen inputs of an annotation or the
 * whole model, drawn as ONE svg (so "Save Snapshot" captures it).
 *
 * - diagonal: the exit's 1-D response to that input (others fixed, see `fix`) over the data's
 *   range, on a histogram of the data. Binary / discrete inputs: dots at the values the data
 *   takes only, never a line between them.
 * - lower triangle: the data rows (input j vs input i), coloured by the exit's actual value.
 *   Binary / discrete axes are jittered.
 * - upper triangle: the 2-D response heatmap of the exit over (i, j), with the pair's
 *   interaction strength H in the corner; the strongest pairs are outlined.
 * One diverging colour scale is shared by every panel and the legend.
 */
import * as d3 from "d3";

type Hist = { edges?: number[]; levels?: number[]; counts: number[] };
export type PairsInput = {
  dim: number;
  name: string;
  kind: "binary" | "discrete" | "continuous" | "constant";
  grid: number[];
  weights: number[];
  median: number;
  /** the value the other panels hold this input at (the median, or the reference case) */
  held?: number;
  min: number;
  max: number;
  hist: Hist;
  pd: number[];
  scatter: number[];
  sensitivity: number;
  units?: string;
};
export type PairsPair = {
  a: number;
  b: number;
  z: number[][];
  h: number;
  h_pd: number;
  interaction_sd: number;
  rank: number;
  /** rows nearest each heatmap cell (0: a combination the data does not take) */
  counts?: number[][];
  /** the two columns are identical in the data (split copies of one input) */
  same_input?: boolean;
};
export type PairsData = {
  dims: number[];
  inputs: PairsInput[];
  pairs: PairsPair[];
  color: number[];
  exit_label: string;
  color_domain: [number, number, number];
  value_range: [number, number];
  reference: number;
  fix: "median" | "average";
  n_rows: number;
  n_scatter: number;
  n_background: number | null;
  grid_size: number;
  units?: string;
  center_rule?: string;
  seconds?: number;
  /** "median" (each input at its median), "reference" (a reference case) or "average" */
  held_rule?: string;
};

const SVGNS = "http://www.w3.org/2000/svg";
const fmt = (v: number) => {
  const a = Math.abs(v);
  if (a < 1e-9) return "0"; // float dust (e.g. 3e-7 from a sigmoid, 5e-20 from unscaling)
  if (a >= 1000) return d3.format(".3s")(v);
  if (a >= 1) return d3.format(".3g")(v);
  if (a >= 0.01) return d3.format(".2g")(v);
  return d3.format(".1e")(v);
};

/** Seeded PRNG so the jitter is the same on every render. */
function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Position helpers for one input on one cell axis of length `len` (0 at the low end). */
type Axis = {
  isBand: boolean;
  pos: (v: number) => number;
  /** pixel extent of grid point i (heatmap cell) */
  cell: (i: number) => [number, number];
  band: number;
  ticks: { v: number; p: number; label: string }[];
};

function makeAxis(inp: PairsInput, len: number, pad: number): Axis {
  const isBand = inp.kind !== "continuous";
  if (isBand) {
    const levels = inp.grid;
    const sc = d3.scaleBand<number>().domain(levels.map((_, i) => i)).range([pad, len - pad]).paddingInner(0.08);
    const nearest = (v: number) => {
      let best = 0;
      for (let i = 1; i < levels.length; i++) if (Math.abs(levels[i] - v) < Math.abs(levels[best] - v)) best = i;
      return best;
    };
    return {
      isBand,
      band: sc.bandwidth(),
      pos: (v) => (sc(nearest(v)) ?? 0) + sc.bandwidth() / 2,
      cell: (i) => [sc(i) ?? 0, (sc(i) ?? 0) + sc.bandwidth()],
      // every level when there is room, else every other one
      ticks: levels
        .map((v, i) => ({ v, p: (sc(i) ?? 0) + sc.bandwidth() / 2, label: fmt(v) }))
        .filter((_, i) => levels.length * 22 <= len || i % 2 === 0),
    };
  }
  const lo = inp.grid[0];
  const hi = inp.grid[inp.grid.length - 1];
  const sc = d3.scaleLinear().domain([lo, hi === lo ? lo + 1 : hi]).range([pad, len - pad]);
  const g = inp.grid;
  return {
    isBand,
    band: 0,
    pos: (v) => sc(v),
    cell: (i) => {
      const left = i === 0 ? g[0] : (g[i - 1] + g[i]) / 2;
      const right = i === g.length - 1 ? g[i] : (g[i] + g[i + 1]) / 2;
      return [sc(left), sc(right)];
    },
    ticks: sc.ticks(len < 80 ? 2 : 3).map((v) => ({ v, p: sc(v), label: fmt(v) })),
  };
}

function el<K extends keyof SVGElementTagNameMap>(tag: K, attrs: Record<string, string | number>, parent?: Element) {
  const e = document.createElementNS(SVGNS, tag) as SVGElementTagNameMap[K];
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, String(v));
  if (parent) parent.appendChild(e);
  return e;
}

function titled(e: Element, text: string) {
  const t = document.createElementNS(SVGNS, "title");
  t.textContent = text;
  e.appendChild(t);
}

function truncate(s: string, px: number, fontPx: number) {
  const max = Math.max(3, Math.floor(px / (fontPx * 0.56)));
  return s.length > max ? s.slice(0, max - 1) + "…" : s;
}

export function renderPairsPlot(
  data: PairsData,
  opts: { width: number; labelFor?: (name: string) => string },
): SVGSVGElement {
  const labelFor = opts.labelFor ?? ((n: string) => n);
  const inputs = data.inputs;
  const k = inputs.length;
  const W = Math.max(320, opts.width);
  const left = 40;
  const right = 36;
  const top = 6;
  const gap = 3;
  const cell = Math.max(44, Math.floor((W - left - right - gap * (k - 1)) / k));
  const gridW = cell * k + gap * (k - 1);
  const pad = 3;
  const smallFont = cell < 70 ? 8 : 9;

  const [c0, cMid, c1] = data.color_domain;
  const span = Math.max(Math.abs(c1 - c0), 1e-300);
  const fmtS = (v: number) => fmt(Math.abs(v) < span * 1e-4 ? 0 : v);
  const color = d3.scaleDiverging<string>().domain([c0, cMid, c1]).interpolator((t) => d3.interpolateRdBu(1 - t));
  const [v0, v1] = data.value_range;
  const vLo = Math.min(v0, cMid);
  const vHi = Math.max(v1, cMid);

  const xAxes = inputs.map((inp) => makeAxis(inp, cell, pad));
  // y axes run bottom-up: position = cell - x position
  const pairByKey = new Map<string, PairsPair>();
  for (const p of data.pairs) pairByKey.set(`${p.a}-${p.b}`, p);
  const ranked = data.pairs.filter((p) => !p.same_input);
  const nTop = ranked.filter((p) => p.h >= 0.05).slice(0, 3).length;

  // ── below-grid text block height
  const keyLines = k;
  const rankLines = Math.min(5, ranked.length);
  const legendH = 34;
  const textBlock = 16 + keyLines * 13 + 8 + 14 + rankLines * 13 + 8 + 4 * 13;
  const H = top + gridW + 30 + legendH + textBlock;
  const totalW = left + gridW + right;

  const svg = el("svg", {
    xmlns: SVGNS,
    width: totalW,
    height: H,
    viewBox: `0 0 ${totalW} ${H}`,
    "font-family": "system-ui, -apple-system, sans-serif",
    "font-size": smallFont,
  }) as SVGSVGElement;
  svg.style.background = "white";
  el("rect", { x: 0, y: 0, width: totalW, height: H, fill: "white" }, svg);

  const cellX = (c: number) => left + c * (cell + gap);
  const cellY = (r: number) => top + r * (cell + gap);
  const rnd = mulberry32(12345);
  const jitter = data.color.map(() => [rnd() - 0.5, rnd() - 0.5]);

  for (let r = 0; r < k; r++) {
    for (let c = 0; c < k; c++) {
      const g = el("g", { transform: `translate(${cellX(c)},${cellY(r)})` }, svg);
      el("rect", { x: 0, y: 0, width: cell, height: cell, fill: r === c ? "#f8fafc" : "white", stroke: "#cbd5e1", "stroke-width": 0.75 }, g);
      const ax = xAxes[c];
      const ay = xAxes[r];
      const yPos = (v: number) => cell - ay.pos(v);

      if (r === c) {
        drawDiagonal(g, inputs[r], ax, cell, pad, color, vLo, vHi, cMid, smallFont, labelFor, data, fmtS);
      } else if (r > c) {
        // lower: scatter of input c (x) vs input r (y), coloured by the exit
        const xs = inputs[c].scatter;
        const ys = inputs[r].scatter;
        const nBins = 15;
        const paths: string[][] = Array.from({ length: nBins }, () => []);
        const rad = cell < 70 ? 1.1 : 1.5;
        for (let n = 0; n < xs.length; n++) {
          let x = ax.pos(xs[n]);
          let y = yPos(ys[n]);
          if (ax.isBand) x += jitter[n][0] * ax.band * 0.7;
          if (ay.isBand) y += jitter[n][1] * ay.band * 0.7;
          const t = (data.color[n] - c0) / (c1 - c0 || 1);
          const bi = Math.max(0, Math.min(nBins - 1, Math.floor(t * nBins)));
          paths[bi].push(`M${(x - rad).toFixed(1)},${y.toFixed(1)}a${rad},${rad} 0 1,0 ${2 * rad},0a${rad},${rad} 0 1,0 ${-2 * rad},0`);
        }
        for (let bi = 0; bi < nBins; bi++) {
          if (!paths[bi].length) continue;
          const mid = c0 + ((bi + 0.5) / nBins) * (c1 - c0);
          el("path", { d: paths[bi].join(""), fill: color(mid), "fill-opacity": 0.75 }, g);
        }
        titled(g, `${labelFor(inputs[c].name)} (x) vs ${labelFor(inputs[r].name)} (y)\n${xs.length} data rows, coloured by ${data.exit_label}`);
      } else {
        // upper: 2-D response heatmap over (input c on x, input r on y)
        const p = pairByKey.get(`${r}-${c}`);
        if (!p) continue;
        for (let ii = 0; ii < p.z.length; ii++) {
          const [y0, y1] = ay.cell(ii);
          for (let jj = 0; jj < p.z[ii].length; jj++) {
            const [x0, x1] = ax.cell(jj);
            // a combination of two categorical values no row takes: faded (continuous axes:
            // the mirrored scatter shows where the data lies)
            const empty = ax.isBand && ay.isBand && p.counts ? p.counts[ii][jj] === 0 : false;
            el("rect", {
              x: x0.toFixed(1), y: (cell - y1).toFixed(1),
              width: Math.max(0.5, x1 - x0 + 0.3).toFixed(1), height: Math.max(0.5, y1 - y0 + 0.3).toFixed(1),
              fill: color(p.z[ii][jj]),
              ...(empty ? { "fill-opacity": 0.22 } : {}),
            }, g);
          }
        }
        const strong = !p.same_input && p.rank <= nTop;
        if (strong) el("rect", { x: 0.75, y: 0.75, width: cell - 1.5, height: cell - 1.5, fill: "none", stroke: "#111827", "stroke-width": 2 }, g);
        const label = p.same_input ? "same input" : `H ${p.h.toFixed(2)}`;
        const tw = label.length * smallFont * 0.58 + 6;
        el("rect", { x: cell - tw - 2, y: 2, width: tw, height: smallFont + 5, rx: 2, fill: "white", "fill-opacity": 0.85 }, g);
        const t = el("text", { x: cell - 5, y: smallFont + 4, "text-anchor": "end", "font-weight": strong ? 700 : 400, fill: "#111827" }, g);
        t.textContent = label;
        if (data.held_rule === "reference") {
          // the reference case's own point on this surface
          const hx = inputs[c].held, hy = inputs[r].held;
          if (hx != null && hy != null) {
            const px = ax.pos(hx), py = cell - ay.pos(hy);
            el("circle", { cx: px, cy: py, r: 3.5, fill: "none", stroke: "#111827", "stroke-width": 2 }, g);
            el("circle", { cx: px, cy: py, r: 3.5, fill: "none", stroke: "white", "stroke-width": 0.8 }, g);
          }
        }
        titled(g,
          `${labelFor(inputs[r].name)} × ${labelFor(inputs[c].name)}\n` +
          (p.same_input ? "The two columns are identical in the data (split copies of one input): only the diagonal combinations occur.\n" : "") +
          `H = ${p.h.toFixed(3)} (rank ${p.rank} of ${data.pairs.length}): share of this surface's variation, where the data lies, that is not one effect of each input added\n` +
          `H vs the 1-D curves = ${p.h_pd.toFixed(3)}; interaction SD = ${fmt(p.interaction_sd)} (${data.exit_label} units)`);
      }

      // outer ticks: x under the bottom row; y left of column 0 (rows >= 1), right of the last column for row 0
      if (r === k - 1) {
        for (const tk of ax.ticks) {
          el("line", { x1: tk.p, x2: tk.p, y1: cell, y2: cell + 3, stroke: "#64748b" }, g);
          const t = el("text", { x: tk.p, y: cell + 4 + smallFont, "text-anchor": "middle", fill: "#475569" }, g);
          t.textContent = tk.label;
        }
      }
      if ((c === 0 && r >= 1) || (r === 0 && c === k - 1)) {
        const onLeft = c === 0;
        for (const tk of ay.ticks) {
          const y = cell - tk.p;
          el("line", { x1: onLeft ? -3 : cell, x2: onLeft ? 0 : cell + 3, y1: y, y2: y, stroke: "#64748b" }, g);
          const t = el("text", { x: onLeft ? -5 : cell + 5, y: y + smallFont / 3, "text-anchor": onLeft ? "end" : "start", fill: "#475569" }, g);
          t.textContent = tk.label;
        }
      }
    }
  }

  // ── colour legend
  let y = top + gridW + 26;
  const gradId = `pairs-grad-${Math.random().toString(36).slice(2, 8)}`;
  const defs = el("defs", {}, svg);
  const lg = el("linearGradient", { id: gradId, x1: "0%", x2: "100%", y1: "0%", y2: "0%" }, defs);
  for (let i = 0; i <= 10; i++) el("stop", { offset: `${i * 10}%`, "stop-color": color(c0 + (i / 10) * (c1 - c0)) }, lg);
  const legW = Math.min(260, gridW);
  el("rect", { x: left, y, width: legW, height: 10, fill: `url(#${gradId})`, stroke: "#cbd5e1" }, svg);
  const lt = el("text", { x: left + legW + 8, y: y + 9, fill: "#334155" }, svg);
  lt.textContent = `${data.exit_label} (one colour scale for every panel)`;
  [[c0, "start"], [cMid, "middle"], [c1, "end"]].forEach(([v, anchor], i) => {
    const t = el("text", { x: left + (i * legW) / 2, y: y + 22, "text-anchor": anchor as string, fill: "#475569" }, svg);
    t.textContent = fmtS(v as number) + (i === 1 ? ` (centre: ${data.center_rule ?? "reference"})` : "");
  });
  y += legendH;

  const line = (text: string, opts2: { bold?: boolean; color?: string } = {}) => {
    const t = el("text", { x: left, y, fill: opts2.color ?? "#334155", "font-size": smallFont + 1, "font-weight": opts2.bold ? 600 : 400 }, svg);
    t.textContent = truncate(text, totalW - left - 4, smallFont + 1);
    y += 13;
  };
  line("Inputs (rows / columns, in order):", { bold: true });
  inputs.forEach((inp, i) => line(`${i + 1}. ${labelFor(inp.name)} — ${inp.kind}${inp.units === "source" ? ", data units" : ""}; sensitivity ${fmt(inp.sensitivity)}`));
  y += 8;
  line("Strongest interactions (H: 0 = the two effects just add):", { bold: true });
  ranked.slice(0, rankLines).forEach((p) =>
    line(`${p.rank}. ${labelFor(inputs[p.a].name)} × ${labelFor(inputs[p.b].name)}   H ${p.h.toFixed(2)}   (interaction SD ${fmt(p.interaction_sd)})`),
  );
  y += 8;
  const fixText = data.fix === "average"
    ? `Curves and heatmaps: other inputs averaged over ${data.n_background} seeded rows (partial dependence).`
    : data.held_rule === "reference"
      ? "Curves and heatmaps: other inputs held at the reference case (ringed on each heatmap)."
      : "Curves and heatmaps: other inputs fixed at their median (as the other plots).";
  line(fixText, { color: "#64748b" });
  line(`Scatter: ${data.n_scatter.toLocaleString()} of ${data.n_rows.toLocaleString()} rows${data.n_scatter < data.n_rows ? " (seeded sample)" : ""}; continuous inputs on a ${data.grid_size}-point grid, binary / discrete inputs at their values only.`, { color: "#64748b" });
  line("Faded cells: a combination of values no row has. H weighs each cell by its share of the rows.", { color: "#64748b" });
  line(`Axes in ${data.units === "source" ? "data units where the input is a dataset column" : "network (scaled) units"}; outlined heatmaps: the strongest pairs.`, { color: "#64748b" });
  return svg;
}

function drawDiagonal(
  g: SVGGElement, inp: PairsInput, ax: Axis, cell: number, pad: number,
  color: d3.ScaleDiverging<string>, vLo: number, vHi: number, cMid: number, font: number,
  labelFor: (n: string) => string, data: PairsData, fmtS: (v: number) => string,
) {
  const histTop = cell * 0.62;
  const counts = inp.hist.counts;
  const maxC = Math.max(1, ...counts);
  if (inp.hist.edges) {
    const e = inp.hist.edges;
    for (let i = 0; i < counts.length; i++) {
      const x0 = ax.pos(e[i]);
      const x1 = ax.pos(e[i + 1]);
      const h = (counts[i] / maxC) * (cell - histTop - pad);
      el("rect", { x: x0, y: cell - pad - h, width: Math.max(0.5, x1 - x0 - 0.4), height: h, fill: "#cbd5e1" }, g);
    }
  } else {
    const lv = inp.hist.levels ?? [];
    for (let i = 0; i < lv.length; i++) {
      const [x0, x1] = ax.cell(i);
      const h = (counts[i] / maxC) * (cell - histTop - pad);
      const bw = (x1 - x0) * 0.5;
      el("rect", { x: (x0 + x1) / 2 - bw / 2, y: cell - pad - h, width: bw, height: h, fill: "#cbd5e1" }, g);
    }
  }
  // response on a y scale shared by every diagonal panel
  const top = font + 8;
  const ys = d3.scaleLinear().domain([vLo, vHi === vLo ? vLo + 1 : vHi]).range([cell - pad, top]);
  el("line", { x1: pad, x2: cell - pad, y1: ys(cMid), y2: ys(cMid), stroke: "#94a3b8", "stroke-dasharray": "2,2" }, g);
  if (inp.kind === "continuous") {
    const d = inp.grid.map((x, i) => `${i ? "L" : "M"}${ax.pos(x).toFixed(1)},${ys(inp.pd[i]).toFixed(1)}`).join("");
    el("path", { d, fill: "none", stroke: "#111827", "stroke-width": 1.6 }, g);
  } else {
    inp.grid.forEach((x, i) => {
      el("circle", { cx: ax.pos(x), cy: ys(inp.pd[i]), r: 3.2, fill: color(inp.pd[i]), stroke: "#111827", "stroke-width": 1 }, g);
    });
  }
  // response scale (min / max) on the inside right edge
  [[vHi, top + font / 2], [vLo, cell - pad - 1]].forEach(([v, yy]) => {
    const t = el("text", { x: cell - 3, y: yy, "text-anchor": "end", fill: "#94a3b8", "font-size": font - 1 }, g);
    t.textContent = fmtS(v);
  });
  const name = labelFor(inp.name);
  const t = el("text", { x: 4, y: font + 2, "font-weight": 600, fill: "#0f172a" }, g);
  t.textContent = truncate(name, cell - 8, font);
  titled(g,
    `${name}\n${inp.kind} input; ${data.exit_label} vs ${inp.name} with the other inputs ${data.fix === "average" ? "averaged over the data" : data.held_rule === "reference" ? "at the reference case" : "at their median"}` +
    `\nrange ${fmt(inp.min)} … ${fmt(inp.max)}, median ${fmt(inp.median)}; grey: the data's distribution`);
}
