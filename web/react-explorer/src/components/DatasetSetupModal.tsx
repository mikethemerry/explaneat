import { useEffect, useState, useCallback, useRef } from "react";
import {
  listDatasets,
  downloadPMLBDataset,
  downloadUCIDataset,
  linkDatasetToExperiment,
  searchDatasetCatalogs,
  type DatasetResponse,
  type DatasetSearchResult,
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

  // Catalog search
  const [searchQuery, setSearchQuery] = useState(
    datasetNameHint ? extractPmlbName(datasetNameHint) : "",
  );
  const [searchResults, setSearchResults] = useState<DatasetSearchResult[]>([]);
  const [searching, setSearching] = useState(false);
  const [showDropdown, setShowDropdown] = useState(false);
  const [selectedResult, setSelectedResult] = useState<DatasetSearchResult | null>(null);
  const [downloading, setDownloading] = useState(false);
  const [downloadError, setDownloadError] = useState<string | null>(null);
  const searchTimeout = useRef<ReturnType<typeof setTimeout> | null>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);

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
      if (searchQuery && !selectedDatasetId) {
        const match = data.datasets.find(
          (d) => d.name.toLowerCase() === searchQuery.toLowerCase(),
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
  }, [searchQuery, selectedDatasetId]);

  useEffect(() => {
    loadDatasets();
  }, [loadDatasets]);

  // Debounced catalog search
  const doSearch = useCallback(async (query: string) => {
    if (query.trim().length < 2) {
      setSearchResults([]);
      setShowDropdown(false);
      return;
    }
    setSearching(true);
    try {
      const data = await searchDatasetCatalogs(query.trim());
      setSearchResults(data.results);
      setShowDropdown(data.results.length > 0);
    } catch (err) {
      console.error("Search failed:", err);
    } finally {
      setSearching(false);
    }
  }, []);

  // Run initial search if we have a hint
  useEffect(() => {
    if (searchQuery.trim().length >= 2) {
      doSearch(searchQuery);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // only on mount

  const handleSearchChange = useCallback((value: string) => {
    setSearchQuery(value);
    setSelectedResult(null);
    setDownloadError(null);
    if (searchTimeout.current) clearTimeout(searchTimeout.current);
    searchTimeout.current = setTimeout(() => doSearch(value), 300);
  }, [doSearch]);

  // Close dropdown on outside click
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setShowDropdown(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const handleSelectSearchResult = useCallback((result: DatasetSearchResult) => {
    setSelectedResult(result);
    setSearchQuery(result.name);
    setShowDropdown(false);
  }, []);

  const handleDownloadSelected = useCallback(async () => {
    if (!selectedResult && !searchQuery.trim()) return;
    setDownloading(true);
    setDownloadError(null);
    try {
      let dataset: DatasetResponse;
      if (selectedResult?.source === "uci" && selectedResult.id) {
        dataset = await downloadUCIDataset(selectedResult.id, selectedResult.name);
      } else {
        const name = selectedResult?.name ?? searchQuery.trim();
        dataset = await downloadPMLBDataset(name);
      }
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
  }, [selectedResult, searchQuery, loadDatasets]);

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

            {/* Catalog Search */}
            <div className="pmlb-download">
              <label className="selector-label">Search PMLB & UCI catalogs</label>
              <div className="catalog-search-container" ref={dropdownRef}>
                <input
                  type="text"
                  className="text-input catalog-search-input"
                  placeholder="Search datasets (e.g. heart, iris, credit)"
                  value={searchQuery}
                  onChange={(e) => handleSearchChange(e.target.value)}
                  onFocus={() => { if (searchResults.length > 0) setShowDropdown(true); }}
                  onKeyDown={(e) => {
                    if (e.key === "Escape") setShowDropdown(false);
                  }}
                />
                {showDropdown && (
                  <div className="dataset-search-dropdown">
                    {searchResults.slice(0, 50).map((r, i) => (
                      <div
                        key={`${r.source}-${r.name}-${i}`}
                        className={`dataset-search-item ${selectedResult?.name === r.name && selectedResult?.source === r.source ? "selected" : ""}`}
                        onMouseDown={(e) => e.preventDefault()}
                        onClick={() => handleSelectSearchResult(r)}
                      >
                        <span className="dataset-search-name">{r.name}</span>
                        <span className="dataset-search-meta">
                          <span className={`dataset-source-badge ${r.source}`}>{r.source.toUpperCase()}</span>
                          {r.task_type && <span className="dataset-task-type">{r.task_type}</span>}
                        </span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
              {searching && <span className="hint">Searching catalogs...</span>}
              {selectedResult && (
                <div className="catalog-selected-result">
                  <span className="catalog-selected-info">
                    <strong>{selectedResult.name}</strong>
                    <span className={`dataset-source-badge ${selectedResult.source}`}>{selectedResult.source.toUpperCase()}</span>
                    {selectedResult.task_type && <span className="dataset-task-type">{selectedResult.task_type}</span>}
                  </span>
                  <button
                    className="op-btn primary"
                    onClick={handleDownloadSelected}
                    disabled={downloading}
                  >
                    {downloading ? "Downloading..." : "Download & Use"}
                  </button>
                </div>
              )}
              {!selectedResult && searchQuery.trim() && !showDropdown && !searching && (
                <div className="catalog-selected-result">
                  <span className="catalog-selected-info">
                    Direct PMLB name: <strong>{searchQuery.trim()}</strong>
                  </span>
                  <button
                    className="op-btn primary"
                    onClick={handleDownloadSelected}
                    disabled={downloading}
                  >
                    {downloading ? "Downloading..." : "Download & Use"}
                  </button>
                </div>
              )}
              {datasetNameHint && !selectedResult && !searchQuery.trim() && (
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
