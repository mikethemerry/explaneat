import { useEffect, useState, useCallback } from "react";
import {
  listExperiments,
  getBestGenome,
  resumeExperiment,
  restartExperiment,
  type ExperimentListItem,
} from "../api/client";
import { DatasetSetupModal } from "./DatasetSetupModal";
import { ExportModal } from "./ExportPanel";
import { ExperimentCreateModal } from "./ExperimentCreateModal";

type ExperimentListProps = {
  onSelectGenome: (genomeId: string, experimentId: string, experimentName: string) => void;
};

export function ExperimentList({ onSelectGenome }: ExperimentListProps) {
  const [experiments, setExperiments] = useState<ExperimentListItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [total, setTotal] = useState(0);
  const [loadingExperiment, setLoadingExperiment] = useState<string | null>(null);
  const [setupExperiment, setSetupExperiment] = useState<ExperimentListItem | null>(null);
  const [showCreate, setShowCreate] = useState(false);
  const [exporting, setExporting] = useState<{ id: string; name: string } | null>(null);
  const [resumingExperiment, setResumingExperiment] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [appliedQuery, setAppliedQuery] = useState("");

  const loadExperiments = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await listExperiments(200, 0, appliedQuery || undefined);
      setExperiments(response.experiments);
      setTotal(response.total);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load experiments");
    } finally {
      setLoading(false);
    }
  }, [appliedQuery]);

  // Debounce the name filter
  useEffect(() => {
    const t = setTimeout(() => setAppliedQuery(query.trim()), 300);
    return () => clearTimeout(t);
  }, [query]);

  useEffect(() => {
    loadExperiments();
  }, [loadExperiments]);

  const handleSelectExperiment = async (experiment: ExperimentListItem) => {
    try {
      setLoadingExperiment(experiment.id);
      setError(null);
      const bestGenome = await getBestGenome(experiment.id);
      onSelectGenome(bestGenome.genome_id, experiment.id, experiment.name);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load best genome");
    } finally {
      setLoadingExperiment(null);
    }
  };

  const handleSetupComplete = useCallback(() => {
    setSetupExperiment(null);
    loadExperiments();
  }, [loadExperiments]);

  const handleResumeExperiment = useCallback(
    async (experiment: ExperimentListItem) => {
      try {
        setResumingExperiment(experiment.id);
        setError(null);
        await resumeExperiment(experiment.id);
        setExperiments((prev) =>
          prev.map((exp) =>
            exp.id === experiment.id ? { ...exp, status: "running" } : exp,
          ),
        );
        loadExperiments();
      } catch (err) {
        console.error("Failed to resume experiment", err);
        setError(
          err instanceof Error ? err.message : "Failed to resume experiment",
        );
      } finally {
        setResumingExperiment(null);
      }
    },
    [loadExperiments],
  );

  const handleRestartExperiment = useCallback(
    async (experiment: ExperimentListItem) => {
      if (!confirm(`Restart "${experiment.name}" from scratch? Existing populations will be deleted.`)) {
        return;
      }
      try {
        setResumingExperiment(experiment.id);
        setError(null);
        await restartExperiment(experiment.id);
        setExperiments((prev) =>
          prev.map((exp) =>
            exp.id === experiment.id ? { ...exp, status: "running" } : exp,
          ),
        );
        loadExperiments();
      } catch (err) {
        console.error("Failed to restart experiment", err);
        setError(
          err instanceof Error ? err.message : "Failed to restart experiment",
        );
      } finally {
        setResumingExperiment(null);
      }
    },
    [loadExperiments],
  );

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
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <h1>Experiments</h1>
          <button
            onClick={() => setShowCreate(true)}
            style={{
              padding: "8px 16px",
              background: "#2563eb",
              color: "white",
              border: "none",
              borderRadius: "4px",
              cursor: "pointer",
              fontWeight: 600,
              fontSize: "13px",
            }}
          >
            + New Experiment
          </button>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
          <input
            type="search"
            placeholder="Filter by name (e.g. 10seed-heart)"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            style={{ padding: "6px 10px", border: "1px solid #d1d5db", borderRadius: 4, minWidth: 280 }}
          />
          <p className="hint" style={{ margin: 0 }}>
            {total} experiment{total !== 1 ? "s" : ""}{appliedQuery ? ` matching "${appliedQuery}"` : " available"}
            {total > experiments.length ? ` (showing ${experiments.length})` : ""}
          </p>
        </div>
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
                <th>Dataset</th>
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
                      ? experiment.name.slice(0, 29) + "\u2026"
                      : experiment.name}
                  </td>
                  <td>
                    <span className={`status-badge status-${experiment.status}`}>
                      {experiment.status}
                    </span>
                  </td>
                  <td>
                    {experiment.has_split ? (
                      <span className="dataset-badge linked">
                        {experiment.dataset_name || "linked"}
                      </span>
                    ) : (
                      <button
                        className="dataset-setup-btn"
                        onClick={() => setSetupExperiment(experiment)}
                      >
                        Setup
                      </button>
                    )}
                  </td>
                  <td>{experiment.generations}</td>
                  <td>
                    {experiment.best_fitness !== null
                      ? experiment.best_fitness.toFixed(3)
                      : "N/A"}
                  </td>
                  <td>{new Date(experiment.created_at).toLocaleDateString()}</td>
                  <td>
                    <div style={{ display: "flex", gap: "0.4rem" }}>
                      {experiment.status === "interrupted" && (
                        <button
                          className="op-btn"
                          onClick={() => handleResumeExperiment(experiment)}
                          disabled={resumingExperiment !== null}
                        >
                          {resumingExperiment === experiment.id
                            ? "Resuming..."
                            : "Resume"}
                        </button>
                      )}
                      {(experiment.status === "interrupted" || experiment.status === "failed") && (
                        <button
                          className="op-btn"
                          onClick={() => handleRestartExperiment(experiment)}
                          disabled={resumingExperiment !== null}
                        >
                          Restart
                        </button>
                      )}
                      <button
                        className="op-btn"
                        title="Export every explained genome (JSON, LaTeX, TikZ, PDF, DOCX) with an index"
                        onClick={() => setExporting({ id: experiment.id, name: experiment.name })}
                      >
                        Export
                      </button>
                      <button
                        className="explore-btn"
                        onClick={() => handleSelectExperiment(experiment)}
                        disabled={loadingExperiment !== null}
                      >
                        {loadingExperiment === experiment.id ? "Loading..." : "Explore"}
                      </button>
                    </div>
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

      {exporting && (
        <ExportModal target={{ kind: "experiment", id: exporting.id, name: exporting.name }} onClose={() => setExporting(null)} />
      )}
      {setupExperiment && (
        <DatasetSetupModal
          experimentId={setupExperiment.id}
          experimentName={setupExperiment.name}
          datasetNameHint={setupExperiment.dataset_name}
          onComplete={handleSetupComplete}
          onClose={() => setSetupExperiment(null)}
        />
      )}

      {showCreate && (
        <ExperimentCreateModal
          onComplete={() => {
            setShowCreate(false);
            loadExperiments();
          }}
          onClose={() => setShowCreate(false)}
        />
      )}
    </div>
  );
}
