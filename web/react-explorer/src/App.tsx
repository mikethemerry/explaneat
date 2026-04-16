import { useState } from "react";
import { ExperimentList } from "./components/ExperimentList";
import { GenomeExplorer } from "./components/GenomeExplorer";
import { NavBar } from "./components/NavBar";
import { DatasetList } from "./components/DatasetList";
import { TemplatesPage } from "./components/TemplatesPage";

type View =
  | { type: "experiments" }
  | { type: "datasets" }
  | { type: "templates" }
  | { type: "genome"; genomeId: string; experimentId: string; experimentName: string };

export default function App() {
  const [view, setView] = useState<View>({ type: "experiments" });

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
