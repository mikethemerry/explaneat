import { useCallback, useEffect, useState } from "react";
import { ExperimentList } from "./components/ExperimentList";
import { GenomeExplorer } from "./components/GenomeExplorer";
import { NavBar } from "./components/NavBar";
import { DatasetList } from "./components/DatasetList";
import { TemplatesPage } from "./components/TemplatesPage";
import { parseUrl, pushView, replaceView, type UrlView } from "./hooks/useUrlState";

export default function App() {
  const [view, setViewState] = useState<UrlView>(parseUrl);

  // Wrap setView to also push to browser history
  const setView = useCallback((v: UrlView) => {
    setViewState(v);
    pushView(v);
  }, []);

  // Listen for browser back/forward
  useEffect(() => {
    const onPopState = () => setViewState(parseUrl());
    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  }, []);

  // On mount, if at "/", redirect to /experiments
  useEffect(() => {
    if (window.location.pathname === "/" || window.location.pathname === "") {
      replaceView({ type: "experiments" });
    }
  }, []);

  if (view.type === "genome") {
    return (
      <GenomeExplorer
        genomeId={view.genomeId}
        experimentId={view.experimentId}
        experimentName={view.experimentName}
        onBack={() => setView({ type: "experiments" })}
      />
    );
  }

  return (
    <>
      <NavBar
        activeTab={view.type}
        onTabChange={(tab) => setView({ type: tab })}
      />
      {view.type === "experiments" ? (
        <ExperimentList
          onSelectGenome={(genomeId, experimentId, experimentName) =>
            setView({ type: "genome", genomeId, experimentId, experimentName })
          }
        />
      ) : view.type === "datasets" ? (
        <DatasetList />
      ) : (
        <TemplatesPage />
      )}
    </>
  );
}
