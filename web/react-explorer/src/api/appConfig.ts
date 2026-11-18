/**
 * Server configuration for the UI (GET /api/config): read-only mode hides the
 * editing controls. The static appendix answers { read_only: true, static: true }.
 */
import { useEffect, useState } from "react";
import { STATIC_MODE } from "./staticData";

export type AppConfig = { read_only: boolean; static: boolean };

const DEFAULT: AppConfig = { read_only: STATIC_MODE, static: STATIC_MODE };
let current: AppConfig = DEFAULT;
let loading: Promise<AppConfig> | null = null;

export function loadAppConfig(): Promise<AppConfig> {
  if (!loading) {
    loading = fetch("/api/config")
      .then((r) => (r.ok ? r.json() : DEFAULT))
      .then((c) => {
        current = { read_only: !!c.read_only || STATIC_MODE, static: !!c.static || STATIC_MODE };
        document.body.classList.toggle("read-only", current.read_only);
        document.body.classList.toggle("static-site", current.static);
        return current;
      })
      .catch(() => DEFAULT);
  }
  return loading;
}

/** The app configuration (the default until /api/config has answered). */
export function useAppConfig(): AppConfig {
  const [cfg, setCfg] = useState<AppConfig>(current);
  useEffect(() => {
    let live = true;
    loadAppConfig().then((c) => live && setCfg(c));
    return () => {
      live = false;
    };
  }, []);
  return cfg;
}

/** True when nothing can be saved (read-only server or the static appendix). */
export function useReadOnly(): boolean {
  return useAppConfig().read_only;
}
