import { useEffect, useState, useCallback } from "react";
import {
  getExperimentSplit,
  listDatasets,
  listSplits,
  type DatasetResponse,
  type SplitResponse,
  type ExperimentSplitResponse,
} from "../api/client";

type DatasetSelectorProps = {
  experimentId: string;
  onSplitSelected: (splitId: string, splitChoice: "train" | "test" | "both") => void;
  onSampleFractionChange: (fraction: number) => void;
  sampleFraction: number;
};

export function DatasetSelector({
  experimentId,
  onSplitSelected,
  onSampleFractionChange,
  sampleFraction,
}: DatasetSelectorProps) {
  const [datasets, setDatasets] = useState<DatasetResponse[]>([]);
  const [splits, setSplits] = useState<SplitResponse[]>([]);
  const [selectedDatasetId, setSelectedDatasetId] = useState<string | null>(null);
  const [selectedSplitId, setSelectedSplitId] = useState<string | null>(null);
  const [splitChoice, setSplitChoice] = useState<"train" | "test" | "both">("both");
  const [loading, setLoading] = useState(false);

  // Auto-selection state
  const [autoSplit, setAutoSplit] = useState<ExperimentSplitResponse | null>(null);
  const [autoLoading, setAutoLoading] = useState(true);
  const [showManual, setShowManual] = useState(false);

  // Try to auto-select experiment's linked split
  useEffect(() => {
    setAutoLoading(true);
    getExperimentSplit(experimentId)
      .then((split) => {
        setAutoSplit(split);
        setSelectedSplitId(split.split_id);
        onSplitSelected(split.split_id, splitChoice);
      })
      .catch(() => {
        // No split linked - show manual selector
        setAutoSplit(null);
        setShowManual(true);
      })
      .finally(() => setAutoLoading(false));
  }, [experimentId]);

  // Load datasets for manual selection
  useEffect(() => {
    if (!showManual) return;
    listDatasets()
      .then((data) => setDatasets(data.datasets))
      .catch((err) => console.error("Failed to load datasets:", err));
  }, [showManual]);

  useEffect(() => {
    if (!selectedDatasetId || !showManual) {
      setSplits([]);
      return;
    }
    setLoading(true);
    listSplits(selectedDatasetId)
      .then((data) => {
        setSplits(data.splits);
        if (data.splits.length > 0 && !selectedSplitId) {
          setSelectedSplitId(data.splits[0].id);
          onSplitSelected(data.splits[0].id, splitChoice);
        }
      })
      .catch((err) => console.error("Failed to load splits:", err))
      .finally(() => setLoading(false));
  }, [selectedDatasetId, showManual]);

  const handleDatasetChange = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => {
      const id = e.target.value || null;
      setSelectedDatasetId(id);
      setSelectedSplitId(null);
    },
    [],
  );

  const handleSplitChange = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => {
      const id = e.target.value || null;
      setSelectedSplitId(id);
      if (id) onSplitSelected(id, splitChoice);
    },
    [onSplitSelected, splitChoice],
  );

  const handleSplitChoiceChange = useCallback(
    (choice: "train" | "test" | "both") => {
      setSplitChoice(choice);
      if (selectedSplitId) onSplitSelected(selectedSplitId, choice);
    },
    [onSplitSelected, selectedSplitId],
  );

  if (autoLoading) {
    return <div className="dataset-selector"><span className="hint">Loading dataset...</span></div>;
  }

  // Auto-selected: show compact summary with option to switch
  if (autoSplit && !showManual) {
    return (
      <div className="dataset-selector">
        <div className="selector-row">
          <label className="selector-label">Dataset</label>
          <span className="dataset-auto-badge">
            {autoSplit.dataset_name} ({autoSplit.train_size}+{autoSplit.test_size_actual} samples)
          </span>
        </div>

        <div className="selector-row">
          <label className="selector-label">Data</label>
          <div className="radio-group">
            {(["train", "test", "both"] as const).map((choice) => (
              <label key={choice} className="radio-label">
                <input
                  type="radio"
                  name="split-choice"
                  checked={splitChoice === choice}
                  onChange={() => handleSplitChoiceChange(choice)}
                />
                {choice}
              </label>
            ))}
          </div>
        </div>

        <div className="selector-row">
          <label className="selector-label">Sample</label>
          <input
            type="range"
            className="sample-slider"
            min={0.01}
            max={1}
            step={0.01}
            value={sampleFraction}
            onChange={(e) => onSampleFractionChange(parseFloat(e.target.value))}
          />
          <span className="sample-value">{Math.round(sampleFraction * 100)}%</span>
        </div>

        <button
          className="link-btn"
          onClick={() => setShowManual(true)}
        >
          Change dataset
        </button>
      </div>
    );
  }

  // No auto-split or manual mode: show full selector
  return (
    <div className="dataset-selector">
      {!autoSplit && (
        <div className="dataset-no-link-msg">
          <span className="hint">No dataset linked to this experiment. Select one manually or set up from the experiment list.</span>
        </div>
      )}

      <div className="selector-row">
        <label className="selector-label">Dataset</label>
        <select
          className="selector-select"
          value={selectedDatasetId || ""}
          onChange={handleDatasetChange}
        >
          <option value="">Select dataset...</option>
          {datasets.map((d) => (
            <option key={d.id} value={d.id}>
              {d.name} ({d.num_samples} samples, {d.num_features} features)
            </option>
          ))}
        </select>
      </div>

      {selectedDatasetId && (
        <div className="selector-row">
          <label className="selector-label">Split</label>
          <select
            className="selector-select"
            value={selectedSplitId || ""}
            onChange={handleSplitChange}
            disabled={loading || splits.length === 0}
          >
            {splits.length === 0 ? (
              <option value="">No splits available</option>
            ) : (
              splits.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.split_type} (train: {s.train_size}, test: {s.test_size_actual})
                </option>
              ))
            )}
          </select>
        </div>
      )}

      {selectedSplitId && (
        <>
          <div className="selector-row">
            <label className="selector-label">Data</label>
            <div className="radio-group">
              {(["train", "test", "both"] as const).map((choice) => (
                <label key={choice} className="radio-label">
                  <input
                    type="radio"
                    name="split-choice"
                    checked={splitChoice === choice}
                    onChange={() => handleSplitChoiceChange(choice)}
                  />
                  {choice}
                </label>
              ))}
            </div>
          </div>

          <div className="selector-row">
            <label className="selector-label">Sample</label>
            <input
              type="range"
              className="sample-slider"
              min={0.01}
              max={1}
              step={0.01}
              value={sampleFraction}
              onChange={(e) => onSampleFractionChange(parseFloat(e.target.value))}
            />
            <span className="sample-value">{Math.round(sampleFraction * 100)}%</span>
          </div>
        </>
      )}

      {autoSplit && (
        <button
          className="link-btn"
          onClick={() => {
            setShowManual(false);
            setSelectedSplitId(autoSplit.split_id);
            onSplitSelected(autoSplit.split_id, splitChoice);
          }}
        >
          Use linked dataset
        </button>
      )}
    </div>
  );
}
