import { useState } from "react";
import { ExperimentList } from "./components/ExperimentList";
import { GenomeExplorer } from "./components/GenomeExplorer";

type Selection = {
  genomeId: string;
  experimentName: string;
};

export default function App() {
  const [selection, setSelection] = useState<Selection | null>(null);

  if (selection) {
    return (
      <GenomeExplorer
        genomeId={selection.genomeId}
        experimentName={selection.experimentName}
        onBack={() => setSelection(null)}
      />
    );
  }

  return (
    <ExperimentList
      onSelectGenome={(genomeId, experimentName) =>
        setSelection({ genomeId, experimentName })
      }
    />
  );
}




