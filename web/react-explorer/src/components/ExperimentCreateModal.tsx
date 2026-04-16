import { useState, useCallback, useEffect, useRef } from "react";
import {
  listDatasets,
  listSplits,
  listConfigTemplates,
  createConfigTemplate,
  createAndRunExperiment,
  getExperimentProgress,
  type DatasetResponse,
  type SplitResponse,
  type ExperimentProgressResponse,
  type ConfigTemplateResponse,
  type ResolvedConfig,
  type ExperimentCreateRequest,
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

  // Config templates + resolved config
  const [templates, setTemplates] = useState<ConfigTemplateResponse[]>([]);
  const [selectedTemplateId, setSelectedTemplateId] = useState<string | null>(null);
  const [config, setConfig] = useState<ResolvedConfig | null>(null);
  const [advancedOpen, setAdvancedOpen] = useState(false);

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

  // Load config templates on mount
  useEffect(() => {
    listConfigTemplates()
      .then((res) => {
        setTemplates(res.templates);
        const defaultT = res.templates.find(t => t.name === "Default") || res.templates[0];
        if (defaultT) {
          setSelectedTemplateId(defaultT.id);
          setConfig(JSON.parse(JSON.stringify(defaultT.config)));
        }
      })
      .catch(() => setError("Failed to load templates"));
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

  const onTemplateChange = (id: string) => {
    const t = templates.find(t => t.id === id);
    if (!t) return;
    setSelectedTemplateId(id);
    setConfig(JSON.parse(JSON.stringify(t.config))); // deep copy
  };

  const handleSaveAsTemplate = async () => {
    if (!config) return;
    const templateName = prompt("Template name?");
    if (!templateName) return;
    try {
      const newTemplate = await createConfigTemplate(templateName, config);
      setTemplates([...templates, newTemplate]);
      setSelectedTemplateId(newTemplate.id);
    } catch {
      setError("Failed to save template");
    }
  };

  const renderNumberField = (
    label: string,
    group: "training" | "neat" | "backprop",
    key: string,
    min: number,
    step: number,
  ) => {
    if (!config) return null;
    const value = (config[group] as Record<string, unknown>)[key] as number;
    return (
      <div
        key={`${group}.${key}`}
        style={{ display: "grid", gridTemplateColumns: "1fr 110px", alignItems: "center", gap: "8px", marginBottom: "6px" }}
      >
        <label style={{ fontSize: "12px", color: "#374151" }}>{label}</label>
        <input
          type="number"
          min={min}
          step={step}
          value={value}
          onChange={(e) => {
            const newValue = step >= 1 ? parseInt(e.target.value) : parseFloat(e.target.value);
            setConfig({
              ...config,
              [group]: { ...config[group], [key]: isNaN(newValue) ? 0 : newValue },
            });
          }}
          style={{ width: "100%", padding: "4px", fontSize: "12px", boxSizing: "border-box" }}
        />
      </div>
    );
  };

  const handleStart = useCallback(async () => {
    if (!name.trim() || !selectedDatasetId || !selectedSplitId) return;
    setError(null);
    setLoading(true);

    try {
      const request: ExperimentCreateRequest = {
        name: name.trim(),
        description,
        dataset_id: selectedDatasetId,
        dataset_split_id: selectedSplitId,
        config_template_id: selectedTemplateId || undefined,
        config_overrides: config || undefined,
        // Keep the legacy fields for backwards compat (they'll be ignored by backend)
        n_generations: config?.training.n_generations ?? 10,
        n_epochs_backprop: config?.training.n_epochs_backprop ?? 5,
        population_size: config?.training.population_size ?? 150,
        fitness_function: config?.training.fitness_function ?? "bce",
      };
      const resp = await createAndRunExperiment(request);
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
  }, [name, description, selectedDatasetId, selectedSplitId, selectedTemplateId, config, onComplete]);

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

            {/* Template */}
            <div style={{ marginBottom: "12px" }}>
              <label style={{ display: "block", marginBottom: "4px", fontWeight: 500, fontSize: "13px" }}>
                Template
              </label>
              <select
                value={selectedTemplateId || ""}
                onChange={(e) => onTemplateChange(e.target.value)}
                style={{ width: "100%", padding: "6px", fontSize: "13px" }}
              >
                {templates.length === 0 && <option value="">Loading...</option>}
                {templates.map((t) => (
                  <option key={t.id} value={t.id}>{t.name}</option>
                ))}
              </select>
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

            {/* Advanced Config */}
            <div style={{ marginBottom: "12px" }}>
              <button
                type="button"
                className="op-btn"
                onClick={() => setAdvancedOpen(!advancedOpen)}
                style={{ marginBottom: "0.5rem" }}
              >
                {advancedOpen ? "▼" : "▶"} Advanced Config
              </button>

              {advancedOpen && config && (
                <div style={{ marginTop: "0.5rem", padding: "0.75rem", border: "1px solid #e5e7eb", borderRadius: "0.375rem" }}>
                  <h4 style={{ marginTop: 0, fontSize: "13px" }}>Training</h4>
                  {renderNumberField("Population size", "training", "population_size", 1, 1)}
                  {renderNumberField("Generations", "training", "n_generations", 1, 1)}
                  {renderNumberField("Backprop epochs per generation", "training", "n_epochs_backprop", 0, 1)}
                  <div
                    style={{ display: "grid", gridTemplateColumns: "1fr 110px", alignItems: "center", gap: "8px", marginBottom: "6px" }}
                  >
                    <label style={{ fontSize: "12px", color: "#374151" }}>Fitness function</label>
                    <select
                      value={config.training.fitness_function}
                      onChange={(e) => setConfig({
                        ...config,
                        training: { ...config.training, fitness_function: e.target.value as "bce" | "auc" },
                      })}
                      style={{ width: "100%", padding: "4px", fontSize: "12px" }}
                    >
                      <option value="bce">BCE (1/loss)</option>
                      <option value="auc">AUC</option>
                    </select>
                  </div>

                  <h4 style={{ fontSize: "13px" }}>NEAT Mutation & Topology</h4>
                  {renderNumberField("Bias mutate rate", "neat", "bias_mutate_rate", 0, 0.01)}
                  {renderNumberField("Bias mutate power", "neat", "bias_mutate_power", 0, 0.01)}
                  {renderNumberField("Bias replace rate", "neat", "bias_replace_rate", 0, 0.01)}
                  {renderNumberField("Weight mutate rate", "neat", "weight_mutate_rate", 0, 0.01)}
                  {renderNumberField("Weight mutate power", "neat", "weight_mutate_power", 0, 0.01)}
                  {renderNumberField("Weight replace rate", "neat", "weight_replace_rate", 0, 0.01)}
                  {renderNumberField("Enabled mutate rate", "neat", "enabled_mutate_rate", 0, 0.001)}
                  {renderNumberField("Node add prob", "neat", "node_add_prob", 0, 0.01)}
                  {renderNumberField("Node delete prob", "neat", "node_delete_prob", 0, 0.01)}
                  {renderNumberField("Conn add prob", "neat", "conn_add_prob", 0, 0.01)}
                  {renderNumberField("Conn delete prob", "neat", "conn_delete_prob", 0, 0.01)}
                  {renderNumberField("Compatibility threshold", "neat", "compatibility_threshold", 0, 0.1)}
                  {renderNumberField("Compatibility disjoint coef", "neat", "compatibility_disjoint_coefficient", 0, 0.1)}
                  {renderNumberField("Compatibility weight coef", "neat", "compatibility_weight_coefficient", 0, 0.1)}
                  {renderNumberField("Max stagnation", "neat", "max_stagnation", 1, 1)}
                  {renderNumberField("Species elitism", "neat", "species_elitism", 0, 1)}
                  {renderNumberField("Elitism", "neat", "elitism", 0, 1)}
                  {renderNumberField("Survival threshold", "neat", "survival_threshold", 0, 0.05)}

                  <h4 style={{ fontSize: "13px" }}>Backprop</h4>
                  {renderNumberField("Learning rate", "backprop", "learning_rate", 0, 0.01)}
                  <div
                    style={{ display: "grid", gridTemplateColumns: "1fr 110px", alignItems: "center", gap: "8px", marginBottom: "6px" }}
                  >
                    <label style={{ fontSize: "12px", color: "#374151" }}>Optimizer</label>
                    <select
                      value={config.backprop.optimizer}
                      onChange={(e) => setConfig({
                        ...config,
                        backprop: { ...config.backprop, optimizer: e.target.value },
                      })}
                      style={{ width: "100%", padding: "4px", fontSize: "12px" }}
                    >
                      <option value="adadelta">Adadelta</option>
                      <option value="adam">Adam</option>
                      <option value="sgd">SGD</option>
                    </select>
                  </div>

                  <button
                    type="button"
                    className="op-btn"
                    onClick={handleSaveAsTemplate}
                    style={{ marginTop: "0.75rem" }}
                  >
                    Save as new template
                  </button>
                </div>
              )}
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
                Gen {progress?.current_generation || 0}/{progress?.total_generations || (config?.training.n_generations ?? 10)}
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
