import { useEffect, useState } from "react";
import {
  listExperiments,
  getBestGenome,
  type ExperimentListItem,
} from "../api/client";

type ExperimentListProps = {
  onSelectGenome: (genomeId: string, experimentName: string) => void;
};

export function ExperimentList({ onSelectGenome }: ExperimentListProps) {
  const [experiments, setExperiments] = useState<ExperimentListItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [total, setTotal] = useState(0);
  const [loadingExperiment, setLoadingExperiment] = useState<string | null>(null);

  useEffect(() => {
    async function load() {
      try {
        setLoading(true);
        setError(null);
        const response = await listExperiments(50, 0);
        setExperiments(response.experiments);
        setTotal(response.total);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load experiments");
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  const handleSelectExperiment = async (experiment: ExperimentListItem) => {
    try {
      setLoadingExperiment(experiment.id);
      setError(null);
      const bestGenome = await getBestGenome(experiment.id);
      onSelectGenome(bestGenome.genome_id, experiment.name);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load best genome");
    } finally {
      setLoadingExperiment(null);
    }
  };

  if (loading) {
    return (
      <div className="experiment-list-container">
        <div className="loading">Loading experiments...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="experiment-list-container">
        <div className="error">
          <h3>Error loading experiments</h3>
          <p>{error}</p>
          <p className="hint">Make sure the API is running at http://localhost:8000</p>
        </div>
      </div>
    );
  }

  return (
    <div className="experiment-list-container">
      <header className="experiment-list-header">
        <h1>ExplaNEAT Explorer</h1>
        <p className="hint">
          {total} experiment{total !== 1 ? "s" : ""} available
        </p>
      </header>

      <div className="experiment-list">
        {experiments.length === 0 ? (
          <p className="hint">No experiments found. Run an experiment first.</p>
        ) : (
          <table className="experiment-table">
            <thead>
              <tr>
                <th>#</th>
                <th>Name</th>
                <th>Status</th>
                <th>Gens</th>
                <th>Best Fitness</th>
                <th>Created</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {experiments.map((experiment, index) => (
                <tr key={experiment.id}>
                  <td className="index-col">{index}</td>
                  <td className="name-col" title={experiment.name}>
                    {experiment.name.length > 30
                      ? experiment.name.slice(0, 29) + "…"
                      : experiment.name}
                  </td>
                  <td>
                    <span className={`status-badge status-${experiment.status}`}>
                      {experiment.status}
                    </span>
                  </td>
                  <td>{experiment.generations}</td>
                  <td>
                    {experiment.best_fitness !== null
                      ? experiment.best_fitness.toFixed(3)
                      : "N/A"}
                  </td>
                  <td>{new Date(experiment.created_at).toLocaleDateString()}</td>
                  <td>
                    <button
                      className="explore-btn"
                      onClick={() => handleSelectExperiment(experiment)}
                      disabled={loadingExperiment !== null}
                    >
                      {loadingExperiment === experiment.id ? "Loading..." : "Explore"}
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      <div className="experiment-list-footer">
        <p className="hint">
          Select an experiment to explore its best genome (highest fitness)
        </p>
      </div>
    </div>
  );
}
