import { useEffect, useState } from "react";
import { listGenomes, type GenomeListItem } from "../api/client";

type GenomeListProps = {
  onSelectGenome: (genomeId: string) => void;
};

export function GenomeList({ onSelectGenome }: GenomeListProps) {
  const [genomes, setGenomes] = useState<GenomeListItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [total, setTotal] = useState(0);

  useEffect(() => {
    async function load() {
      try {
        setLoading(true);
        setError(null);
        const response = await listGenomes(50, 0);
        setGenomes(response.genomes);
        setTotal(response.total);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load genomes");
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  if (loading) {
    return (
      <div className="genome-list-container">
        <div className="loading">Loading genomes...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="genome-list-container">
        <div className="error">
          <h3>Error loading genomes</h3>
          <p>{error}</p>
          <p className="hint">Make sure the API is running at http://localhost:8000</p>
        </div>
      </div>
    );
  }

  return (
    <div className="genome-list-container">
      <header className="genome-list-header">
        <h1>ExplaNEAT Explorer</h1>
        <p className="hint">
          {total} genome{total !== 1 ? "s" : ""} available
        </p>
      </header>

      <div className="genome-list">
        {genomes.length === 0 ? (
          <p className="hint">No genomes found. Run an experiment first.</p>
        ) : (
          <table className="genome-table">
            <thead>
              <tr>
                <th>Genome ID</th>
                <th>Fitness</th>
                <th>Nodes</th>
                <th>Connections</th>
                <th>Created</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {genomes.map((genome) => (
                <tr key={genome.id}>
                  <td>{genome.genome_id}</td>
                  <td>{genome.fitness?.toFixed(4) ?? "N/A"}</td>
                  <td>{genome.num_nodes}</td>
                  <td>{genome.num_connections}</td>
                  <td>{new Date(genome.created_at).toLocaleDateString()}</td>
                  <td>
                    <button
                      className="explore-btn"
                      onClick={() => onSelectGenome(genome.id)}
                    >
                      Explore
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
