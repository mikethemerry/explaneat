import { useCallback, useEffect, useState } from "react";
import { ExperimentList } from "./components/ExperimentList";
import { GenomeExplorer } from "./components/GenomeExplorer";
import { NavBar } from "./components/NavBar";
import { DatasetList } from "./components/DatasetList";
import { TemplatesPage } from "./components/TemplatesPage";
import { RecordsPage } from "./components/RecordsPage";
import { ReviewPage } from "./components/ReviewPage";
import { PortalPage } from "./components/PortalPage";
import { FeaturesPage } from "./components/FeaturesPage";
import { atRoot, parseUrl, pushView, replaceView, type UrlView } from "./hooks/useUrlState";

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
    // static mode routes on the hash: links and typed URLs change it
    window.addEventListener("hashchange", onPopState);
    return () => {
      window.removeEventListener("popstate", onPopState);
      window.removeEventListener("hashchange", onPopState);
    };
  }, []);

  // On mount, if at "/", redirect to /experiments
  useEffect(() => {
    if (atRoot()) {
      replaceView({ type: "experiments" });
    }
  }, []);

  if (view.type === "genome") {
    return (
      <GenomeExplorer
        key={view.genomeId}
        genomeId={view.genomeId}
        pinId={view.pin}
        onOpenPin={(pinId) =>
          setView({ type: "genome", genomeId: view.genomeId, experimentId: view.experimentId, experimentName: view.experimentName, pin: pinId })
        }
        onOpenFeatures={() => setView({ type: "features" })}
        experimentId={view.experimentId}
        experimentName={view.experimentName}
        initialAnnotationId={view.annotationId}
        onBack={() => setView({ type: "experiments" })}
        onOpenGenome={(gid) =>
          setView({ type: "genome", genomeId: gid, experimentId: view.experimentId, experimentName: view.experimentName })
        }
      />
    );
  }

  return (
    <>
      <NavBar
        activeTab={view.type === "review_label" ? "review" : view.type}
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
      ) : view.type === "portal" ? (
        <PortalPage
          onOpenGenome={(genomeId, experimentId, experimentName) =>
            setView({ type: "genome", genomeId, experimentId, experimentName })
          }
        />
      ) : view.type === "review_label" ? (
        <ReviewPage label={view.label} onBack={() => setView({ type: "review" })} />
      ) : view.type === "features" ? (
        <FeaturesPage
          onOpen={(genomeId, experimentId, experimentName, pinId) =>
            setView({ type: "genome", genomeId, experimentId, experimentName, pin: pinId })
          }
        />
      ) : view.type === "review" ? (
        <RecordsPage
          onOpen={(genomeId, experimentId, experimentName) =>
            setView({ type: "genome", genomeId, experimentId, experimentName })
          }
          onVerify={(label) => setView({ type: "review_label", label })}
        />
      ) : (
        <TemplatesPage />
      )}
    </>
  );
}
