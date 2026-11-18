/**
 * Static-data mode: the explorer as a static site (the thesis appendix on GitHub
 * Pages), with no server. Built with VITE_STATIC_DATA=1; `explaneat export site`
 * writes the JSON every view fetches under data/ next to index.html.
 *
 * installStaticFetch() replaces window.fetch for /api/... URLs:
 *   GET  /api/<path>?<query>  -> data/<path>.json, or data/<path>@k=v,k2=v2.json
 *   POST viz-data / performance -> data/genomes/<g>/viz/... and performance@split=...
 *   GET  /model?collapsed=...   -> data/genomes/<g>/model.json, collapsed here (collapse.ts)
 *   everything else             -> 404 "not included in this static appendix",
 *                                  writes -> 403 (read only)
 * The same key rules are in explaneat/export/site_keys.py (keep them in step).
 */
import { collapseModel, type CollapseAnnotation } from "./collapse";
import type { ModelState } from "./client";

export const STATIC_MODE = import.meta.env.VITE_STATIC_DATA === "1";

export const NOT_INCLUDED = "Not included in this static appendix.";
export const READ_ONLY_STATIC = "This is a static, read-only appendix: changes cannot be saved.";

const DATA_BASE = `${import.meta.env.BASE_URL}data/`;

/** Characters outside [A-Za-z0-9_.-] become ~xx (hex), as in site_keys.safe(). */
export function safe(v: string): string {
  return v.replace(/[^A-Za-z0-9_.-]/g, (c) => "~" + c.charCodeAt(0).toString(16).padStart(2, "0"));
}

// Query parameters that do not change the static answer
const IGNORED_PARAMS: [RegExp, string[] | "all"][] = [
  [/^experiments$/, "all"],
  [/^genomes$/, "all"],
  [/^records$/, "all"],
  [/^review$/, "all"],
  [/^genomes\/[^/]+\/tools\/check$/, ["closed_form"]],
  [/^genomes\/[^/]+\/evidence\/formula$/, ["force"]],
];

/** data/ file for GET /api/<path>?<query> (path without the /api/ prefix). */
export function getKey(path: string, params: URLSearchParams): string {
  const p = path.replace(/^\/+|\/+$/g, "");
  const rule = IGNORED_PARAMS.find(([re]) => re.test(p));
  const pairs: [string, string][] = [];
  params.forEach((v, k) => {
    if (rule && (rule[1] === "all" || rule[1].includes(k))) return;
    pairs.push([k, v]);
  });
  pairs.sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0));
  const segs = p.split("/").map(safe).join("/");
  return pairs.length ? `${segs}@${pairs.map(([k, v]) => `${safe(k)}=${safe(v)}`).join(",")}.json` : `${segs}.json`;
}

type Body = Record<string, any>;

/** data/ file for a POST that only computes (viz-data, performance), or null. */
export function postKey(path: string, body: Body): string | null {
  const p = path.replace(/^\/+|\/+$/g, "");
  let m = p.match(/^genomes\/([^/]+)\/evidence\/viz-data$/);
  if (m) {
    const g = safe(m[1]);
    const scope = body.annotation_id ? `ann-${safe(String(body.annotation_id))}` : body.node_id ? `node-${safe(String(body.node_id))}` : "model";
    const t = String(body.viz_type);
    const q = (body.params ?? {}) as Body;
    const o = Number(q.output_dim ?? 0);
    let key: string;
    if (["line", "ice", "feature_output_scatter", "partial_dependence"].includes(t)) key = `i${Number(q.input_dim ?? 0)}_o${o}`;
    else if (t === "heatmap") key = `i${(q.input_dims ?? [0, 1]).join("-")}_o${o}`;
    else if (t === "pairs_plot") key = `o${o}_${q.fix ?? "median"}${Array.isArray(q.dims) ? `_d${q.dims.join("-")}` : ""}`;
    else if (t === "sensitivity") key = "all";
    else key = `o${o}`;
    if (body.view === "source") key += "_source";
    return `genomes/${g}/viz/${scope}/${safe(t)}@${key}.json`;
  }
  m = p.match(/^genomes\/([^/]+)\/evidence\/performance$/);
  if (m) return `genomes/${safe(m[1])}/performance@split=${safe(String(body.split ?? "train"))}.json`;
  m = p.match(/^genomes\/([^/]+)\/evidence\/input-distribution$/);
  if (m) return `genomes/${safe(m[1])}/input-distribution@f=${((body.feature_indices ?? []) as number[]).join("-")}.json`;
  m = p.match(/^genomes\/([^/]+)\/tools\/linearise$/);
  if (m && !(body.annotation_names ?? []).length) return `genomes/${safe(m[1])}/tools/linearise.json`;
  return null;
}

const COMPUTE_POST = [
  /^genomes\/[^/]+\/evidence\/(viz-data|performance|shap|input-distribution)$/,
  /^genomes\/[^/]+\/analyze\/(split-detection|classify-nodes|auto-classify)$/,
  /^genomes\/[^/]+\/operations\/validate$/,
  /^genomes\/[^/]+\/tools\/(linearise|operations\/import-preview)$/,
];

const cache = new Map<string, Promise<unknown | null>>();
const lists = new Map<string, Promise<Set<string>>>();

/** The exported files: data/_files.json, and data/genomes/<g>/_files.json per network. */
function fileList(dir: string): Promise<Set<string>> {
  if (!lists.has(dir)) {
    const realFetch = (window as any).__realFetch as typeof fetch;
    lists.set(
      dir,
      realFetch(`${DATA_BASE}${dir}_files.json`)
        .then((r) => (r.ok ? r.json() : []))
        .then((a: string[]) => new Set(a))
        .catch(() => new Set<string>()),
    );
  }
  return lists.get(dir)!;
}

/** Whether a data/ file is part of the export (asked first, so left-out views make no 404s). */
async function exported(file: string): Promise<boolean> {
  const m = file.match(/^(genomes\/[^/]+\/)(.+)$/);
  return m ? (await fileList(m[1])).has(m[2]) : (await fileList("")).has(file);
}

/** Load a data/ file (null when it is not part of the export). */
export function loadData<T = unknown>(file: string): Promise<T | null> {
  if (!cache.has(file)) {
    const realFetch = (window as any).__realFetch as typeof fetch;
    cache.set(
      file,
      exported(file)
        .then((ok) => (ok ? realFetch(DATA_BASE + file) : null))
        .then(async (r) => {
          if (!r || !r.ok) return null;
          const ct = r.headers.get("content-type") || "";
          const text = await r.text();
          // a static server's SPA fallback may answer with index.html
          if (!ct.includes("json") && text.trimStart().startsWith("<")) return null;
          const obj = JSON.parse(text);
          // shared parts (e.g. per-row class labels of the whole-model plots) are stored once
          if (obj && typeof obj === "object" && Array.isArray(obj.__merge)) {
            const parts = await Promise.all((obj.__merge as string[]).map((f) => loadData<Body>(f)));
            delete obj.__merge;
            for (const part of parts) if (part) Object.assign(obj, part);
          }
          return obj;
        })
        .catch(() => null),
    );
  }
  return cache.get(file) as Promise<T | null>;
}

function json(status: number, obj: unknown): Response {
  return new Response(JSON.stringify(obj), { status, headers: { "Content-Type": "application/json" } });
}

type CollapseMeta = { annotations: CollapseAnnotation[] };

async function modelResponse(g: string, params: URLSearchParams): Promise<Response> {
  const upto = params.get("upto");
  const file = upto != null ? `genomes/${safe(g)}/model@upto=${safe(upto)}.json` : `genomes/${safe(g)}/model.json`;
  const model = await loadData<ModelState>(file);
  if (!model) return json(404, { detail: NOT_INCLUDED });
  const collapsed = (params.get("collapsed") || "").split(",").map((s) => s.trim()).filter(Boolean);
  if (!collapsed.length) return json(200, model);
  const meta = await loadData<CollapseMeta>(`genomes/${safe(g)}/collapse-meta.json`);
  if (!meta) return json(200, model);
  let anns = meta.annotations;
  if (upto != null) {
    const k = Number(upto);
    anns = anns
      .filter((a) => a.op_index < k)
      .map((a) => ({ ...a, parent: a.parent && anns.some((p) => p.name === a.parent && p.op_index < k) ? a.parent : null }));
  }
  return json(200, collapseModel(model, anns, collapsed));
}

async function handle(url: string, init?: RequestInit): Promise<Response> {
  const u = new URL(url, "http://static.invalid");
  const path = decodeURIComponent(u.pathname.replace(/^\/api\/?/, ""));
  const method = (init?.method || "GET").toUpperCase();

  if (method === "GET") {
    if (path === "config") return json(200, { read_only: true, static: true });
    const mm = path.match(/^genomes\/([^/]+)\/model$/);
    if (mm) return modelResponse(mm[1], u.searchParams);
    const obj = await loadData(getKey(path, u.searchParams));
    return obj == null ? json(404, { detail: NOT_INCLUDED }) : json(200, obj);
  }
  if (method === "POST" && COMPUTE_POST.some((re) => re.test(path))) {
    let body: Body = {};
    try {
      body = init?.body ? JSON.parse(String(init.body)) : {};
    } catch {
      /* not JSON */
    }
    const file = postKey(path, body);
    const obj = file ? await loadData(file) : null;
    return obj == null ? json(404, { detail: NOT_INCLUDED }) : json(200, obj);
  }
  return json(403, { detail: READ_ONLY_STATIC });
}

/** In static mode, answer /api/... requests from data/ (call once, before rendering). */
export function installStaticFetch() {
  if (!STATIC_MODE || (window as any).__realFetch) return;
  const realFetch = window.fetch.bind(window);
  (window as any).__realFetch = realFetch;
  window.fetch = (input: RequestInfo | URL, init?: RequestInit) => {
    const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
    const local = url.startsWith("/api") ? url : url.startsWith(window.location.origin + "/api") ? url.slice(window.location.origin.length) : null;
    if (local === null) return realFetch(input as RequestInfo, init);
    return handle(local, init);
  };
}
