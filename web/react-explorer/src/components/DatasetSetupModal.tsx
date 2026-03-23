import { useEffect, useState, useCallback } from "react";
import {
  listDatasets,
  downloadPMLBDataset,
  linkDatasetToExperiment,
  type DatasetResponse,
} from "../api/client";

type DatasetSetupModalProps = {
  experimentId: string;
  experimentName: string;
  datasetNameHint: string | null;
  onComplete: () => void;
  onClose: () => void;
};

/**
 * Try to extract a PMLB dataset identifier from a human-friendly dataset name.
 * e.g. "PMLB Backache (Working)" → "backache", "Iris Dataset" → "iris"
 */
function extractPmlbName(hint: string): string {
  let name = hint;
  // Strip "PMLB" prefix (case-insensitive)
  name = name.replace(/^pmlb\s+/i, "");
  // Strip parenthetical suffixes like "(Working)", "(Simple)"
  name = name.replace(/\s*\(.*?\)\s*$/, "");
  // Strip common trailing words
  name = name.replace(/\s+(dataset|data)$/i, "");
  // Convert to lowercase and replace spaces with underscores (PMLB convention)
  name = name.trim().toLowerCase().replace(/\s+/g, "_");
  return name;
}

export function DatasetSetupModal({
  experimentId,
  experimentName,
  datasetNameHint,
  onComplete,
  onClose,
}: DatasetSetupModalProps) {
  // Step 1: Choose dataset
  const [datasets, setDatasets] = useState<DatasetResponse[]>([]);
  const [selectedDatasetId, setSelectedDatasetId] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  // PMLB download - extract a plausible PMLB identifier from the hint
  const [pmlbName, setPmlbName] = useState(
    datasetNameHint ? extractPmlbName(datasetNameHint) : "",
  );
  const [downloading, setDownloading] = useState(false);
  const [downloadError, setDownloadError] = useState<string | null>(null);

  // Step 2: Split config
  const [step, setStep] = useState<1 | 2>(1);
  const [testFraction, setTestFraction] = useState(0.2);
  const [randomSeed, setRandomSeed] = useState(42);
  const [stratify, setStratify] = useState(false);
  const [linking, setLinking] = useState(false);
  const [linkError, setLinkError] = useState<string | null>(null);

  const loadDatasets = useCallback(async () => {
    setLoading(true);
    try {
      const data = await listDatasets();
      setDatasets(data.datasets);
      // Auto-select if a dataset matches the hint
      if (pmlbName && !selectedDatasetId) {
        const match = data.datasets.find(
          (d) => d.name.toLowerCase() === pmlbName.toLowerCase(),
        );
        if (match) {
          setSelectedDatasetId(match.id);
          setStep(2);
        }
      }
    } catch (err) {
      console.error("Failed to load datasets:", err);
    } finally {
      setLoading(false);
    }
  }, [pmlbName, selectedDatasetId]);

  useEffect(() => {
    loadDatasets();
  }, [loadDatasets]);

  const handleDownloadPMLB = useCallback(async () => {
    if (!pmlbName.trim()) return;
    setDownloading(true);
    setDownloadError(null);
    try {
      const dataset = await downloadPMLBDataset(pmlbName.trim());
      // Refresh dataset list and auto-select the new one
      await loadDatasets();
      setSelectedDatasetId(dataset.id);
      setStep(2);
    } catch (err) {
      setDownloadError(
        err instanceof Error ? err.message : "Failed to download dataset",
      );
    } finally {
      setDownloading(false);
    }
  }, [pmlbName, loadDatasets]);

  const handleSelectExisting = useCallback(
    (datasetId: string) => {
      setSelectedDatasetId(datasetId);
      setStep(2);
    },
    [],
  );

  const handleLink = useCallback(async () => {
    if (!selectedDatasetId) return;
    setLinking(true);
    setLinkError(null);
    try {
      await linkDatasetToExperiment(experimentId, {
        dataset_id: selectedDatasetId,
        test_proportion: testFraction,
        random_seed: randomSeed,
        stratify,
      });
      onComplete();
    } catch (err) {
      setLinkError(
        err instanceof Error ? err.message : "Failed to link dataset",
      );
    } finally {
      setLinking(false);
    }
  }, [experimentId, selectedDatasetId, testFraction, randomSeed, stratify, onComplete]);

  const selectedDataset = datasets.find((d) => d.id === selectedDatasetId);

  return (
    <div className="wizard-overlay dataset-setup-overlay" onClick={onClose}>
      <div
        className="wizard-content dataset-setup-modal"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="dataset-setup-header">
          <h3>Dataset Setup</h3>
          <span className="hint">{experimentName}</span>
        </div>

        {step === 1 && (
          <div className="wizard-step">
            <h4>Step 1: Choose Dataset</h4>

            {/* PMLB Download */}
            <div className="pmlb-download">
              <label className="selector-label">Download from PMLB</label>
              <div className="pmlb-row">
                <input
                  type="text"
                  className="text-input pmlb-input"
                  placeholder="Dataset name (e.g. backache)"
                  value={pmlbName}
                  onChange={(e) => setPmlbName(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") handleDownloadPMLB();
                  }}
                />
                <button
                  className="op-btn primary"
                  onClick={handleDownloadPMLB}
                  disabled={downloading || !pmlbName.trim()}
                >
                  {downloading ? "Downloading..." : "Download"}
                </button>
              </div>
              {datasetNameHint && (
                <span className="hint pmlb-hint">
                  Suggested from "{datasetNameHint}"
                </span>
              )}
              {downloadError && (
                <div className="error-message">{downloadError}</div>
              )}
            </div>

            {/* Existing Datasets */}
            <div className="existing-datasets">
              <label className="selector-label">Or select existing dataset</label>
              {loading ? (
                <span className="hint">Loading datasets...</span>
              ) : datasets.length === 0 ? (
                <span className="hint">No datasets in database. Download one from PMLB above.</span>
              ) : (
                <div className="dataset-list">
                  {datasets.map((d) => (
                    <div
                      key={d.id}
                      className={`dataset-list-item ${selectedDatasetId === d.id ? "selected" : ""}`}
                      onClick={() => handleSelectExisting(d.id)}
                    >
                      <span className="dataset-list-name">{d.name}</span>
                      <span className="dataset-list-meta">
                        {d.num_samples} samples, {d.num_features} features
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>

            <div className="wizard-actions">
              <button className="op-btn" onClick={onClose}>
                Cancel
              </button>
            </div>
          </div>
        )}

        {step === 2 && selectedDataset && (
          <div className="wizard-step">
            <h4>Step 2: Configure Split</h4>

            <div className="wizard-summary">
              <p>
                <strong>Dataset:</strong> {selectedDataset.name}
              </p>
              <p>
                {selectedDataset.num_samples} samples, {selectedDataset.num_features} features
                {selectedDataset.num_classes ? `, ${selectedDataset.num_classes} classes` : ""}
              </p>
            </div>

            <div className="split-config">
              <div className="split-config-row">
                <label className="selector-label">Test fraction</label>
                <input
                  type="range"
                  className="sample-slider"
                  min={0.05}
                  max={0.5}
                  step={0.05}
                  value={testFraction}
                  onChange={(e) => setTestFraction(parseFloat(e.target.value))}
                />
                <span className="sample-value">{Math.round(testFraction * 100)}%</span>
              </div>

              <div className="split-config-row">
                <label className="selector-label">Random seed</label>
                <input
                  type="number"
                  className="text-input seed-input"
                  value={randomSeed}
                  onChange={(e) => setRandomSeed(parseInt(e.target.value) || 42)}
                />
              </div>

              <div className="split-config-row">
                <label className="radio-label">
                  <input
                    type="checkbox"
                    checked={stratify}
                    onChange={(e) => setStratify(e.target.checked)}
                  />
                  Stratify on target
                </label>
              </div>
            </div>

            {linkError && <div className="error-message">{linkError}</div>}

            <div className="wizard-actions">
              <button className="op-btn" onClick={() => setStep(1)}>
                Back
              </button>
              <button
                className="op-btn primary"
                onClick={handleLink}
                disabled={linking}
              >
                {linking ? "Linking..." : "Link & Create Split"}
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
