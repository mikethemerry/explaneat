/**
 * Lightweight URL-based routing via History API.
 * No external router dependency — just reads/writes window.location.
 *
 * The static appendix (VITE_STATIC_DATA=1) routes on the hash instead
 * (index.html#/genome/<id>?...), so it works from any folder, e.g. a GitHub
 * Pages subpath, with no server-side fallback.
 */
import { STATIC_MODE } from "../api/staticData";

/** The routed path and query: from the URL, or from the hash in static mode. */
function current(): { path: string; search: string } {
  if (!STATIC_MODE) return { path: window.location.pathname, search: window.location.search };
  const h = window.location.hash.replace(/^#/, "") || "/";
  const q = h.indexOf("?");
  return q < 0 ? { path: h, search: "" } : { path: h.slice(0, q), search: h.slice(q) };
}

/** An href for a view's URL (a hash link in static mode). */
export function hrefFor(url: string): string {
  return STATIC_MODE ? `#${url}` : url;
}

/** Whether the app is at its root (redirected to /experiments). */
export function atRoot(): boolean {
  const p = current().path;
  return p === "/" || p === "";
}

export type UrlView =
  | { type: "experiments" }
  | { type: "datasets" }
  | { type: "templates" }
  | { type: "review" }
  | { type: "review_label"; label: string }
  | { type: "portal" }
  | { type: "features" }
  | { type: "genome"; genomeId: string; experimentId: string; experimentName: string; annotationId?: string; pin?: string };

/** Parse the current URL into a View. */
export function parseUrl(): UrlView {
  const { path, search } = current();
  const params = new URLSearchParams(search);

  // /genome/:genomeId
  const genomeMatch = path.match(/^\/genome\/([^/]+)/);
  if (genomeMatch) {
    return {
      type: "genome",
      genomeId: genomeMatch[1],
      experimentId: params.get("experiment") || "",
      experimentName: decodeURIComponent(params.get("name") || ""),
      annotationId: params.get("annotation") || undefined,
      ...(params.get("pin") ? { pin: params.get("pin") as string } : {}),
    };
  }

  if (path.startsWith("/datasets")) return { type: "datasets" };
  if (path.startsWith("/templates")) return { type: "templates" };
  const reviewMatch = path.match(/^\/review\/([^/]+)/);
  if (reviewMatch) return { type: "review_label", label: decodeURIComponent(reviewMatch[1]) };
  if (path.startsWith("/review")) return { type: "review" };
  if (path.startsWith("/portal")) return { type: "portal" };
  if (path.startsWith("/features")) return { type: "features" };
  return { type: "experiments" };
}

/** Build a URL string from a View. */
export function viewToUrl(view: UrlView): string {
  switch (view.type) {
    case "experiments":
      return "/experiments";
    case "datasets":
      return "/datasets";
    case "templates":
      return "/templates";
    case "review":
      return "/review";
    case "review_label":
      return `/review/${encodeURIComponent(view.label)}`;
    case "portal":
      return "/portal";
    case "features":
      return "/features";
    case "genome": {
      const params = new URLSearchParams();
      if (view.experimentId) params.set("experiment", view.experimentId);
      if (view.experimentName) params.set("name", view.experimentName);
      if (view.annotationId) params.set("annotation", view.annotationId);
      if (view.pin) params.set("pin", view.pin);
      const qs = params.toString();
      return `/genome/${view.genomeId}${qs ? `?${qs}` : ""}`;
    }
  }
}

/** Push a new view to browser history. */
export function pushView(view: UrlView) {
  const url = viewToUrl(view);
  // Don't push if already at this URL
  const cur = current();
  if (cur.path + cur.search === url) return;
  window.history.pushState(null, "", hrefFor(url));
}

/** Replace current history entry (for initial load / redirects). */
export function replaceView(view: UrlView) {
  window.history.replaceState(null, "", hrefFor(viewToUrl(view)));
}
