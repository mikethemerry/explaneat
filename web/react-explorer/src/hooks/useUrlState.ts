/**
 * Lightweight URL-based routing via History API.
 * No external router dependency — just reads/writes window.location.
 */

export type UrlView =
  | { type: "experiments" }
  | { type: "datasets" }
  | { type: "templates" }
  | { type: "genome"; genomeId: string; experimentId: string; experimentName: string };

/** Parse the current URL into a View. */
export function parseUrl(): UrlView {
  const path = window.location.pathname;
  const params = new URLSearchParams(window.location.search);

  // /genome/:genomeId
  const genomeMatch = path.match(/^\/genome\/([^/]+)/);
  if (genomeMatch) {
    return {
      type: "genome",
      genomeId: genomeMatch[1],
      experimentId: params.get("experiment") || "",
      experimentName: decodeURIComponent(params.get("name") || ""),
    };
  }

  if (path.startsWith("/datasets")) return { type: "datasets" };
  if (path.startsWith("/templates")) return { type: "templates" };
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
    case "genome": {
      const params = new URLSearchParams();
      if (view.experimentId) params.set("experiment", view.experimentId);
      if (view.experimentName) params.set("name", view.experimentName);
      const qs = params.toString();
      return `/genome/${view.genomeId}${qs ? `?${qs}` : ""}`;
    }
  }
}

/** Push a new view to browser history. */
export function pushView(view: UrlView) {
  const url = viewToUrl(view);
  // Don't push if already at this URL
  if (window.location.pathname + window.location.search === url) return;
  window.history.pushState(null, "", url);
}

/** Replace current history entry (for initial load / redirects). */
export function replaceView(view: UrlView) {
  window.history.replaceState(null, "", viewToUrl(view));
}
