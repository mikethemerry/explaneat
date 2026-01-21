import { useState } from "react";
import { GenomeList } from "./components/GenomeList";
import { GenomeExplorer } from "./components/GenomeExplorer";

export default function App() {
  const [selectedGenomeId, setSelectedGenomeId] = useState<string | null>(null);

  if (selectedGenomeId) {
    return (
      <GenomeExplorer
        genomeId={selectedGenomeId}
        onBack={() => setSelectedGenomeId(null)}
      />
    );
  }

  return <GenomeList onSelectGenome={setSelectedGenomeId} />;
}




