import { useEffect, useState, useCallback, useRef } from "react";
import {
  listDatasets,
  downloadPMLBDataset,
  downloadUCIDataset,
  searchDatasetCatalogs,
  type DatasetResponse,
  type DatasetSearchResult,
} from "../api/client";
import { DatasetDetail } from "./DatasetDetail";

type DatasetListProps = {
  // currently no external props needed
};

function TaskTypeBadge({ dataset }: { dataset: DatasetResponse }) {
  const taskType =
    dataset.task_type ||
    (dataset.num_classes != null && dataset.num_classes >= 2
      ? "classification"
      : "regression");

  const isClassification = taskType === "classification";

  return (
    <span
      style={{
        display: "inline-block",
        padding: "0.2rem 0.5rem",
        borderRadius: "0.25rem",
        fontSize: "0.75rem",
        fontWeight: 600,
        textTransform: "uppercase",
        background: isClassification ? "#dbeafe" : "#fef3c7",
        color: isClassification ? "#1d4ed8" : "#d97706",
      }}
    >
      {taskType}
    </span>
  );
}

function SourceBadge({ source }: { source: string }) {
  const colors: Record<string, { bg: string; fg: string }> = {
    pmlb: { bg: "#e0e7ff", fg: "#4338ca" },
    uci: { bg: "#dcfce7", fg: "#166534" },
  };
  const c = colors[source.toLowerCase()] || { bg: "#f3f4f6", fg: "#374151" };
  return (
    <span
      style={{
        display: "inline-block",
        padding: "0.15rem 0.4rem",
        borderRadius: "0.25rem",
        fontSize: "0.7rem",
        fontWeight: 600,
        textTransform: "uppercase",
        background: c.bg,
        color: c.fg,
        marginLeft: "0.25rem",
      }}
    >
      {source}
    </span>
  );
}

export function DatasetList({}: DatasetListProps) {
  const [datasets, setDatasets] = useState<DatasetResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [total, setTotal] = useState(0);

  // Import
  const [showImport, setShowImport] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<DatasetSearchResult[]>([]);
  const [searching, setSearching] = useState(false);
  const [sourceFilter, setSourceFilter] = useState<"all" | "pmlb" | "uci">("all");
  const [importing, setImporting] = useState<string | null>(null); // name of dataset being imported
  const [importError, setImportError] = useState<string | null>(null);
  const searchTimeout = useRef<ReturnType<typeof setTimeout>>();

  // Detail view
  const [editingDatasetId, setEditingDatasetId] = useState<string | null>(null);

  const loadDatasets = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await listDatasets();
      setDatasets(response.datasets);
      setTotal(response.total);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load datasets");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadDatasets();
  }, [loadDatasets]);

  // Debounced search
  useEffect(() => {
    if (!showImport) return;

    clearTimeout(searchTimeout.current);

    if (!searchQuery.trim() && sourceFilter === "all") {
      setSearchResults([]);
      return;
    }

    searchTimeout.current = setTimeout(async () => {
      try {
        setSearching(true);
        const response = await searchDatasetCatalogs(searchQuery, sourceFilter);
        setSearchResults(response.results);
      } catch {
        // Silently fail search
      } finally {
        setSearching(false);
      }
    }, 300);

    return () => clearTimeout(searchTimeout.current);
  }, [searchQuery, sourceFilter, showImport]);

  const handleDownload = async (result: DatasetSearchResult) => {
    const key = `${result.source}:${result.name}`;
    try {
      setImporting(key);
      setImportError(null);
      if (result.source === "uci" && result.id != null) {
        await downloadUCIDataset(result.id, result.name);
      } else {
        await downloadPMLBDataset(result.name);
      }
      loadDatasets();
    } catch (err) {
      setImportError(
        err instanceof Error ? err.message : "Failed to import dataset"
      );
    } finally {
      setImporting(null);
    }
  };

  if (editingDatasetId) {
    return (
      <DatasetDetail
        datasetId={editingDatasetId}
        onBack={() => {
          setEditingDatasetId(null);
          loadDatasets();
        }}
      />
    );
  }

  if (loading) {
    return (
      <div className="experiment-list-container">
        <div className="loading">Loading datasets...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="experiment-list-container">
        <div className="error">
          <h3>Error loading datasets</h3>
          <p>{error}</p>
          <p className="hint">
            Make sure the API is running at http://localhost:8000
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="experiment-list-container">
      <header className="experiment-list-header">
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
          }}
        >
          <h1>Datasets</h1>
          <button
            onClick={() => setShowImport((v) => !v)}
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
            + Import Dataset
          </button>
        </div>
        <p className="hint">
          {total} dataset{total !== 1 ? "s" : ""} available
        </p>
      </header>

      {showImport && (
        <div
          style={{
            background: "white",
            border: "1px solid #e5e7eb",
            borderRadius: "0.5rem",
            padding: "1rem",
            marginBottom: "1rem",
            boxShadow: "0 1px 3px rgba(0,0,0,0.1)",
          }}
        >
          <div style={{ display: "flex", gap: "0.5rem", alignItems: "center", marginBottom: "0.75rem" }}>
            <input
              type="text"
              className="text-input"
              placeholder="Search datasets (e.g. heart, iris, adult)..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              autoFocus
              style={{ marginBottom: 0, flex: 1 }}
            />
            <select
              value={sourceFilter}
              onChange={(e) => setSourceFilter(e.target.value as "all" | "pmlb" | "uci")}
              style={{
                padding: "0.5rem",
                border: "1px solid #d1d5db",
                borderRadius: "0.25rem",
                fontSize: "0.85rem",
                background: "white",
              }}
            >
              <option value="all">All sources</option>
              <option value="pmlb">PMLB only</option>
              <option value="uci">UCI only</option>
            </select>
            <button
              className="op-btn secondary"
              onClick={() => {
                setShowImport(false);
                setImportError(null);
                setSearchQuery("");
                setSearchResults([]);
              }}
            >
              Close
            </button>
          </div>

          {importError && (
            <div className="error-message" style={{ marginBottom: "0.5rem" }}>
              {importError}
            </div>
          )}

          {searching && (
            <p className="hint" style={{ margin: "0.5rem 0" }}>Searching...</p>
          )}

          {!searching && searchQuery.trim() && searchResults.length === 0 && (
            <p className="hint" style={{ margin: "0.5rem 0" }}>
              No datasets found matching "{searchQuery}"
            </p>
          )}

          {searchResults.length > 0 && (
            <div style={{ maxHeight: "300px", overflowY: "auto", border: "1px solid #e5e7eb", borderRadius: "0.25rem" }}>
              <table className="experiment-table" style={{ fontSize: "0.85rem" }}>
                <thead>
                  <tr>
                    <th>Name</th>
                    <th>Source</th>
                    <th>Task</th>
                    <th></th>
                  </tr>
                </thead>
                <tbody>
                  {searchResults.slice(0, 50).map((result) => {
                    const key = `${result.source}:${result.name}`;
                    const isImporting = importing === key;
                    return (
                      <tr key={key}>
                        <td style={{ fontWeight: 500 }}>{result.name}</td>
                        <td><SourceBadge source={result.source} /></td>
                        <td>
                          {result.task_type && (
                            <span style={{
                              fontSize: "0.75rem",
                              color: result.task_type === "classification" ? "#1d4ed8" : "#d97706",
                            }}>
                              {result.task_type}
                            </span>
                          )}
                        </td>
                        <td>
                          <button
                            className="op-btn primary"
                            onClick={() => handleDownload(result)}
                            disabled={isImporting}
                            style={{ fontSize: "0.8rem", padding: "0.25rem 0.5rem" }}
                          >
                            {isImporting ? "Importing..." : "Import"}
                          </button>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
              {searchResults.length > 50 && (
                <p className="hint" style={{ padding: "0.5rem", margin: 0 }}>
                  Showing first 50 of {searchResults.length} results. Refine your search.
                </p>
              )}
            </div>
          )}

          {!searchQuery.trim() && (
            <p className="hint" style={{ margin: "0.5rem 0 0 0" }}>
              Search PMLB ({"\u2248"}284 datasets) and UCI ML Repository ({"\u2248"}200 datasets) by name.
              UCI datasets include feature type metadata (categorical, integer, etc).
            </p>
          )}
        </div>
      )}

      <div className="experiment-list">
        {datasets.length === 0 ? (
          <p className="hint">
            No datasets found. Import one to get started.
          </p>
        ) : (
          <table className="experiment-table">
            <thead>
              <tr>
                <th>Name</th>
                <th>Source</th>
                <th>Task</th>
                <th>Samples</th>
                <th>Features</th>
                <th>Classes</th>
                <th>Imported</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {datasets.map((dataset) => (
                <tr key={dataset.id}>
                  <td className="name-col" title={dataset.name}>
                    <span style={{ fontWeight: 500 }}>{dataset.name}</span>
                  </td>
                  <td>
                    <span style={{ color: "#6b7280", fontSize: "0.85rem" }}>
                      {dataset.source || "-"}
                    </span>
                  </td>
                  <td>
                    <TaskTypeBadge dataset={dataset} />
                  </td>
                  <td>{dataset.num_samples ?? "-"}</td>
                  <td>{dataset.num_features ?? "-"}</td>
                  <td>{dataset.num_classes ?? "-"}</td>
                  <td>
                    {dataset.created_at
                      ? new Date(dataset.created_at).toLocaleDateString()
                      : "-"}
                  </td>
                  <td>
                    <button
                      className="explore-btn"
                      onClick={() => setEditingDatasetId(dataset.id)}
                    >
                      Edit
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
          Edit a dataset to set task type, class names, and feature metadata
        </p>
      </div>
    </div>
  );
}
