import { useState } from "react";
import { ExperimentList } from "./components/ExperimentList";
import { GenomeExplorer } from "./components/GenomeExplorer";

type Selection = {
  genomeId: string;
  experimentId: string;
  experimentName: string;
};

export default function App() {
  const [selection, setSelection] = useState<Selection | null>(null);

  if (selection) {
    return (
      <GenomeExplorer
        genomeId={selection.genomeId}
        experimentId={selection.experimentId}
        experimentName={selection.experimentName}
        onBack={() => setSelection(null)}
      />
    );
  }

  return (
    <ExperimentList
      onSelectGenome={(genomeId, experimentId, experimentName) =>
        setSelection({ genomeId, experimentId, experimentName })
      }
    />
  );
}




