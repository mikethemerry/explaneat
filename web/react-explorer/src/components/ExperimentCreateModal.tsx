import { useState, useCallback, useEffect, useRef } from "react";
import {
  listDatasets,
  listSplits,
  createAndRunExperiment,
  getExperimentProgress,
  type DatasetResponse,
  type SplitResponse,
  type ExperimentProgressResponse,
} from "../api/client";

type ExperimentCreateModalProps = {
  onComplete: () => void;
  onClose: () => void;
};

export function ExperimentCreateModal({ onComplete, onClose }: ExperimentCreateModalProps) {
  // Form state
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [datasets, setDatasets] = useState<DatasetResponse[]>([]);
  const [selectedDatasetId, setSelectedDatasetId] = useState<string | null>(null);
  const [splits, setSplits] = useState<SplitResponse[]>([]);
  const [selectedSplitId, setSelectedSplitId] = useState<string | null>(null);

  // Config
  const [nGenerations, setNGenerations] = useState(10);
  const [nEpochs, setNEpochs] = useState(5);
  const [popSize, setPopSize] = useState(150);
  const [fitnessFunction, setFitnessFunction] = useState<"bce" | "auc">("bce");

  // Progress
  const [jobId, setJobId] = useState<string | null>(null);
  const [progress, setProgress] = useState<ExperimentProgressResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Load datasets
  useEffect(() => {
    listDatasets()
      .then((resp) => {
        setDatasets(resp.datasets.filter(d => d.has_data));
      })
      .catch(() => setError("Failed to load datasets"));
  }, []);

  // Load splits when dataset changes
  useEffect(() => {
    if (!selectedDatasetId) {
      setSplits([]);
      setSelectedSplitId(null);
      return;
    }
    listSplits(selectedDatasetId)
      .then((resp) => {
        setSplits(resp.splits);
        if (resp.splits.length > 0) {
          setSelectedSplitId(resp.splits[0].id);
        }
      })
      .catch(() => setSplits([]));
  }, [selectedDatasetId]);

  // Cleanup polling
  useEffect(() => {
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  const handleStart = useCallback(async () => {
    if (!name.trim() || !selectedDatasetId || !selectedSplitId) return;
    setError(null);
    setLoading(true);

    try {
      const resp = await createAndRunExperiment({
        name: name.trim(),
        description,
        dataset_id: selectedDatasetId,
        dataset_split_id: selectedSplitId,
        n_generations: nGenerations,
        n_epochs_backprop: nEpochs,
        population_size: popSize,
        fitness_function: fitnessFunction,
      });
      setJobId(resp.job_id);

      // Start polling
      const interval = setInterval(async () => {
        try {
          const prog = await getExperimentProgress(resp.job_id);
          setProgress(prog);
          if (prog.status === "completed" || prog.status === "failed" || prog.status === "cancelled") {
            clearInterval(interval);
            pollRef.current = null;
            if (prog.status === "completed") {
              setTimeout(() => onComplete(), 1500);
            }
          }
        } catch {
          clearInterval(interval);
          pollRef.current = null;
        }
      }, 1000);
      pollRef.current = interval;
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to start experiment");
    } finally {
      setLoading(false);
    }
  }, [name, description, selectedDatasetId, selectedSplitId, nGenerations, nEpochs, popSize, fitnessFunction, onComplete]);

  const isRunning = progress?.status === "running" || progress?.status === "pending";
  const isCompleted = progress?.status === "completed";
  const isFailed = progress?.status === "failed";
  const progressPct = progress
    ? (progress.current_generation / Math.max(progress.total_generations, 1)) * 100
    : 0;

  const selectedDataset = datasets.find(d => d.id === selectedDatasetId);

  return (
    <div style={{
      position: "fixed",
      inset: 0,
      background: "rgba(0,0,0,0.5)",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      zIndex: 1000,
    }}>
      <div style={{
        background: "white",
        borderRadius: "8px",
        padding: "24px",
        width: "480px",
        maxHeight: "80vh",
        overflowY: "auto",
      }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "16px" }}>
          <h2 style={{ margin: 0, fontSize: "18px" }}>New Experiment</h2>
          <button onClick={onClose} style={{ background: "none", border: "none", fontSize: "18px", cursor: "pointer" }}>
            x
          </button>
        </div>

        {!jobId ? (
          <>
            {/* Name */}
            <div style={{ marginBottom: "12px" }}>
              <label style={{ display: "block", marginBottom: "4px", fontWeight: 500, fontSize: "13px" }}>
                Name *
              </label>
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="My Experiment"
                style={{ width: "100%", padding: "6px", fontSize: "13px", boxSizing: "border-box" }}
              />
            </div>

            {/* Description */}
            <div style={{ marginBottom: "12px" }}>
              <label style={{ display: "block", marginBottom: "4px", fontWeight: 500, fontSize: "13px" }}>
                Description
              </label>
              <textarea
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                rows={2}
                style={{ width: "100%", padding: "6px", fontSize: "13px", boxSizing: "border-box", resize: "vertical" }}
              />
            </div>

            {/* Dataset */}
            <div style={{ marginBottom: "12px" }}>
              <label style={{ display: "block", marginBottom: "4px", fontWeight: 500, fontSize: "13px" }}>
                Dataset *
              </label>
              <select
                value={selectedDatasetId || ""}
                onChange={(e) => setSelectedDatasetId(e.target.value || null)}
                style={{ width: "100%", padding: "6px", fontSize: "13px" }}
              >
                <option value="">Select dataset...</option>
                {datasets.map((d) => (
                  <option key={d.id} value={d.id}>
                    {d.name} ({d.num_samples} samples, {d.num_features} features)
                  </option>
                ))}
              </select>
            </div>

            {/* Split */}
            {selectedDatasetId && splits.length > 0 && (
              <div style={{ marginBottom: "12px" }}>
                <label style={{ display: "block", marginBottom: "4px", fontWeight: 500, fontSize: "13px" }}>
                  Split
                </label>
                <select
                  value={selectedSplitId || ""}
                  onChange={(e) => setSelectedSplitId(e.target.value || null)}
                  style={{ width: "100%", padding: "6px", fontSize: "13px" }}
                >
                  {splits.map((s) => (
                    <option key={s.id} value={s.id}>
                      {s.split_type} (train: {s.train_size}, test: {s.test_size_actual})
                    </option>
                  ))}
                </select>
              </div>
            )}

            {selectedDatasetId && splits.length === 0 && (
              <div style={{ marginBottom: "12px", color: "#b45309", fontSize: "12px" }}>
                No splits found for this dataset. Create one first via dataset setup.
              </div>
            )}

            {/* Config */}
            <div style={{ marginBottom: "12px" }}>
              <div style={{ fontWeight: 500, fontSize: "13px", marginBottom: "8px" }}>Configuration</div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "8px" }}>
                <div>
                  <label style={{ display: "block", fontSize: "11px", color: "#64748b" }}>Generations</label>
                  <input
                    type="number"
                    value={nGenerations}
                    onChange={(e) => setNGenerations(Math.max(1, parseInt(e.target.value) || 1))}
                    min={1}
                    style={{ width: "100%", padding: "4px", fontSize: "12px", boxSizing: "border-box" }}
                  />
                </div>
                <div>
                  <label style={{ display: "block", fontSize: "11px", color: "#64748b" }}>Backprop Epochs</label>
                  <input
                    type="number"
                    value={nEpochs}
                    onChange={(e) => setNEpochs(Math.max(0, parseInt(e.target.value) || 0))}
                    min={0}
                    style={{ width: "100%", padding: "4px", fontSize: "12px", boxSizing: "border-box" }}
                  />
                </div>
                <div>
                  <label style={{ display: "block", fontSize: "11px", color: "#64748b" }}>Population Size</label>
                  <input
                    type="number"
                    value={popSize}
                    onChange={(e) => setPopSize(Math.max(2, parseInt(e.target.value) || 2))}
                    min={2}
                    style={{ width: "100%", padding: "4px", fontSize: "12px", boxSizing: "border-box" }}
                  />
                </div>
                <div>
                  <label style={{ display: "block", fontSize: "11px", color: "#64748b" }}>Fitness Function</label>
                  <select
                    value={fitnessFunction}
                    onChange={(e) => setFitnessFunction(e.target.value as "bce" | "auc")}
                    style={{ width: "100%", padding: "4px", fontSize: "12px" }}
                  >
                    <option value="bce">MSE-based (1/loss)</option>
                    <option value="auc">AUC</option>
                  </select>
                </div>
              </div>
            </div>

            {selectedDataset && (
              <div style={{ fontSize: "11px", color: "#64748b", marginBottom: "12px" }}>
                Inputs: {selectedDataset.num_features} | Outputs: {selectedDataset.num_classes && selectedDataset.num_classes > 2 ? selectedDataset.num_classes : 1}
              </div>
            )}

            {error && (
              <div style={{ background: "#fef2f2", color: "#dc2626", padding: "8px", borderRadius: "4px", fontSize: "12px", marginBottom: "12px" }}>
                {error}
              </div>
            )}

            <div style={{ display: "flex", gap: "8px", justifyContent: "flex-end" }}>
              <button
                onClick={onClose}
                style={{ padding: "8px 16px", background: "#e2e8f0", border: "none", borderRadius: "4px", cursor: "pointer" }}
              >
                Cancel
              </button>
              <button
                onClick={handleStart}
                disabled={!name.trim() || !selectedDatasetId || !selectedSplitId || loading}
                style={{
                  padding: "8px 16px",
                  background: name.trim() && selectedDatasetId && selectedSplitId ? "#2563eb" : "#94a3b8",
                  color: "white",
                  border: "none",
                  borderRadius: "4px",
                  cursor: name.trim() && selectedDatasetId && selectedSplitId ? "pointer" : "not-allowed",
                  fontWeight: 600,
                }}
              >
                {loading ? "Starting..." : "Create & Run"}
              </button>
            </div>
          </>
        ) : (
          /* Progress view */
          <div>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "8px" }}>
              <span style={{ fontWeight: 500 }}>
                {isRunning ? "Running..." : isCompleted ? "Experiment Complete!" : isFailed ? "Failed" : progress?.status || "Starting..."}
              </span>
              <span style={{ fontSize: "11px", color: "#64748b" }}>
                Gen {progress?.current_generation || 0}/{progress?.total_generations || nGenerations}
              </span>
            </div>

            {/* Progress bar */}
            <div style={{
              width: "100%",
              height: "8px",
              background: "#e2e8f0",
              borderRadius: "4px",
              overflow: "hidden",
              marginBottom: "12px",
            }}>
              <div style={{
                width: `${progressPct}%`,
                height: "100%",
                background: isFailed ? "#ef4444" : isCompleted ? "#22c55e" : "#2563eb",
                transition: "width 0.5s",
              }} />
            </div>

            {/* Metrics */}
            {progress && (
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "8px", fontSize: "12px", marginBottom: "12px" }}>
                <div>
                  <span style={{ color: "#64748b" }}>Best Fitness: </span>
                  <span style={{ fontWeight: 500 }}>
                    {progress.best_fitness?.toFixed(4) || "..."}
                  </span>
                </div>
                <div>
                  <span style={{ color: "#64748b" }}>Mean Fitness: </span>
                  <span style={{ fontWeight: 500 }}>
                    {progress.mean_fitness?.toFixed(4) || "..."}
                  </span>
                </div>
                <div>
                  <span style={{ color: "#64748b" }}>Species: </span>
                  <span>{progress.num_species}</span>
                </div>
                <div>
                  <span style={{ color: "#64748b" }}>Pop Size: </span>
                  <span>{progress.pop_size}</span>
                </div>
              </div>
            )}

            {isFailed && progress?.error && (
              <div style={{ background: "#fef2f2", color: "#dc2626", padding: "8px", borderRadius: "4px", fontSize: "12px", marginBottom: "12px" }}>
                {progress.error}
              </div>
            )}

            {isCompleted && (
              <div style={{ background: "#f0fdf4", color: "#16a34a", padding: "8px", borderRadius: "4px", fontSize: "12px", marginBottom: "12px" }}>
                Experiment completed! Refreshing experiment list...
              </div>
            )}

            <div style={{ display: "flex", justifyContent: "flex-end" }}>
              <button
                onClick={onClose}
                style={{
                  padding: "8px 16px",
                  background: isCompleted ? "#22c55e" : "#e2e8f0",
                  color: isCompleted ? "white" : "inherit",
                  border: "none",
                  borderRadius: "4px",
                  cursor: "pointer",
                }}
              >
                {isCompleted ? "Done" : "Close"}
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
