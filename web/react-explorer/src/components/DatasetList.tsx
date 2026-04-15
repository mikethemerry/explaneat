import { useEffect, useState, useCallback } from "react";
import {
  listDatasets,
  downloadPMLBDataset,
  type DatasetResponse,
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

export function DatasetList({}: DatasetListProps) {
  const [datasets, setDatasets] = useState<DatasetResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [total, setTotal] = useState(0);

  // PMLB import
  const [showImport, setShowImport] = useState(false);
  const [pmlbName, setPmlbName] = useState("");
  const [importing, setImporting] = useState(false);
  const [importError, setImportError] = useState<string | null>(null);

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

  const handleImport = async () => {
    if (!pmlbName.trim()) return;
    try {
      setImporting(true);
      setImportError(null);
      await downloadPMLBDataset(pmlbName.trim());
      setPmlbName("");
      setShowImport(false);
      loadDatasets();
    } catch (err) {
      setImportError(
        err instanceof Error ? err.message : "Failed to import dataset"
      );
    } finally {
      setImporting(false);
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
            + Import from PMLB
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
          <div style={{ display: "flex", gap: "0.5rem", alignItems: "center" }}>
            <input
              type="text"
              className="text-input"
              placeholder="PMLB dataset name (e.g. backache)"
              value={pmlbName}
              onChange={(e) => setPmlbName(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleImport()}
              style={{ marginBottom: 0, flex: 1 }}
            />
            <button
              className="op-btn primary"
              onClick={handleImport}
              disabled={importing || !pmlbName.trim()}
              style={{ whiteSpace: "nowrap" }}
            >
              {importing ? "Downloading..." : "Download"}
            </button>
            <button
              className="op-btn secondary"
              onClick={() => {
                setShowImport(false);
                setImportError(null);
              }}
            >
              Cancel
            </button>
          </div>
          {importError && (
            <div className="error-message" style={{ marginTop: "0.5rem" }}>
              {importError}
            </div>
          )}
        </div>
      )}

      <div className="experiment-list">
        {datasets.length === 0 ? (
          <p className="hint">
            No datasets found. Import one from PMLB to get started.
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
