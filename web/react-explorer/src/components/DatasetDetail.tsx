import { useEffect, useState, useCallback } from "react";
import {
  getDataset,
  updateDataset,
  listSplits,
  createSplit,
  prepareDataset,
  type DatasetResponse,
  type DatasetUpdateRequest,
  type SplitResponse,
} from "../api/client";

type DatasetDetailProps = {
  datasetId: string;
  onBack: () => void;
};

export function DatasetDetail({ datasetId, onBack }: DatasetDetailProps) {
  const [dataset, setDataset] = useState<DatasetResponse | null>(null);
  const [splits, setSplits] = useState<SplitResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [saveSuccess, setSaveSuccess] = useState(false);

  // Editable fields
  const [description, setDescription] = useState("");
  const [taskType, setTaskType] = useState<"classification" | "regression">(
    "regression"
  );
  const [numClasses, setNumClasses] = useState<number>(2);
  const [classNames, setClassNames] = useState<string[]>([]);
  const [targetName, setTargetName] = useState("");
  const [targetDescription, setTargetDescription] = useState("");
  const [featureDescriptions, setFeatureDescriptions] = useState<
    Record<string, string>
  >({});
  const [featureTypes, setFeatureTypes] = useState<Record<string, string>>({});

  const loadDataset = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const [ds, splitsRes] = await Promise.all([
        getDataset(datasetId),
        listSplits(datasetId),
      ]);
      setDataset(ds);
      setSplits(splitsRes.splits);

      // Initialize editable state
      setDescription(ds.description || "");
      const inferredTaskType =
        ds.task_type ||
        (ds.num_classes != null && ds.num_classes >= 2
          ? "classification"
          : "regression");
      setTaskType(inferredTaskType as "classification" | "regression");
      setNumClasses(ds.num_classes ?? 2);
      setClassNames(ds.class_names || []);
      setTargetName(ds.target_name || "");
      setTargetDescription(ds.target_description || "");
      setFeatureDescriptions(ds.feature_descriptions || {});
      setFeatureTypes(ds.feature_types || {});
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load dataset");
    } finally {
      setLoading(false);
    }
  }, [datasetId]);

  useEffect(() => {
    loadDataset();
  }, [loadDataset]);

  // Sync class names array when numClasses changes
  useEffect(() => {
    if (taskType === "classification") {
      setClassNames((prev) => {
        if (prev.length === numClasses) return prev;
        const next = [...prev];
        while (next.length < numClasses) next.push("");
        return next.slice(0, numClasses);
      });
    }
  }, [numClasses, taskType]);

  const handleSave = async () => {
    if (!dataset) return;
    try {
      setSaving(true);
      setSaveSuccess(false);
      setError(null);

      const req: DatasetUpdateRequest = {
        description: description || undefined,
        task_type: taskType,
        target_name: targetName || undefined,
        target_description: targetDescription || undefined,
      };

      if (taskType === "classification") {
        req.num_classes = numClasses;
        req.class_names =
          classNames.some((n) => n.trim()) ? classNames : undefined;
      } else {
        req.num_classes = null;
        req.class_names = null;
      }

      // Only send feature metadata if there are changes
      const hasFeatureDescriptions = Object.values(featureDescriptions).some(
        (v) => v.trim()
      );
      if (hasFeatureDescriptions) {
        req.feature_descriptions = featureDescriptions;
      }
      const hasFeatureTypes = Object.values(featureTypes).some((v) => v.trim());
      if (hasFeatureTypes) {
        req.feature_types = featureTypes;
      }

      const updated = await updateDataset(datasetId, req);
      setDataset(updated);
      setSaveSuccess(true);
      setTimeout(() => setSaveSuccess(false), 3000);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to save");
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return (
      <div className="experiment-list-container">
        <div className="loading">Loading dataset...</div>
      </div>
    );
  }

  if (error && !dataset) {
    return (
      <div className="experiment-list-container">
        <div className="error">
          <h3>Error</h3>
          <p>{error}</p>
          <button className="op-btn" onClick={onBack}>
            Back
          </button>
        </div>
      </div>
    );
  }

  if (!dataset) return null;

  const featureNames = dataset.feature_names || [];

  return (
    <div className="experiment-list-container" style={{ maxWidth: "900px" }}>
      {/* Header */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "1rem",
          marginBottom: "1.5rem",
        }}
      >
        <button className="back-btn" onClick={onBack}>
          Back
        </button>
        <div style={{ flex: 1 }}>
          <h1 style={{ margin: 0 }}>{dataset.name}</h1>
          <span className="hint">
            {dataset.source || "Unknown source"}
            {dataset.version ? ` v${dataset.version}` : ""}
          </span>
        </div>
        <div style={{ display: "flex", gap: "0.5rem", alignItems: "center" }}>
          {saveSuccess && (
            <span style={{ color: "#059669", fontSize: "0.85rem", fontWeight: 500 }}>
              Saved
            </span>
          )}
          <button
            className="op-btn primary"
            onClick={handleSave}
            disabled={saving}
          >
            {saving ? "Saving..." : "Save Changes"}
          </button>
        </div>
      </div>

      {error && <div className="error-message">{error}</div>}

      {/* Description */}
      <Section title="Description">
        <textarea
          className="text-input"
          rows={3}
          placeholder="Dataset description..."
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          style={{ resize: "vertical" }}
        />
      </Section>

      {/* Task Configuration */}
      <Section title="Task Configuration">
        <div style={{ display: "flex", gap: "1.5rem", marginBottom: "1rem" }}>
          <label className="radio-label">
            <input
              type="radio"
              name="task_type"
              checked={taskType === "classification"}
              onChange={() => setTaskType("classification")}
            />
            Classification
          </label>
          <label className="radio-label">
            <input
              type="radio"
              name="task_type"
              checked={taskType === "regression"}
              onChange={() => setTaskType("regression")}
            />
            Regression
          </label>
        </div>

        {taskType === "classification" && (
          <div style={{ marginLeft: "0.5rem" }}>
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: "0.75rem",
                marginBottom: "0.75rem",
              }}
            >
              <label
                style={{ fontWeight: 500, fontSize: "0.85rem", color: "#374151" }}
              >
                Number of classes:
              </label>
              <input
                type="number"
                min={2}
                max={100}
                value={numClasses}
                onChange={(e) =>
                  setNumClasses(Math.max(2, parseInt(e.target.value) || 2))
                }
                style={{
                  width: "70px",
                  padding: "0.35rem 0.5rem",
                  border: "1px solid #e5e7eb",
                  borderRadius: "0.25rem",
                }}
              />
            </div>
            <div>
              <label
                style={{
                  fontWeight: 500,
                  fontSize: "0.85rem",
                  color: "#374151",
                  display: "block",
                  marginBottom: "0.35rem",
                }}
              >
                Class names:
              </label>
              <div
                style={{
                  display: "flex",
                  flexDirection: "column",
                  gap: "0.35rem",
                }}
              >
                {Array.from({ length: numClasses }, (_, i) => (
                  <div
                    key={i}
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: "0.5rem",
                    }}
                  >
                    <span
                      style={{
                        fontFamily: "monospace",
                        fontSize: "0.8rem",
                        color: "#6b7280",
                        minWidth: "20px",
                      }}
                    >
                      {i}:
                    </span>
                    <input
                      type="text"
                      className="text-input"
                      placeholder={`Class ${i}`}
                      value={classNames[i] || ""}
                      onChange={(e) => {
                        const next = [...classNames];
                        next[i] = e.target.value;
                        setClassNames(next);
                      }}
                      style={{ marginBottom: 0, flex: 1 }}
                    />
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </Section>

      {/* Target Variable */}
      <Section title="Target Variable">
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            gap: "0.5rem",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
            <label
              style={{
                fontWeight: 500,
                fontSize: "0.85rem",
                color: "#374151",
                minWidth: "50px",
              }}
            >
              Name:
            </label>
            <input
              type="text"
              className="text-input"
              placeholder="target"
              value={targetName}
              onChange={(e) => setTargetName(e.target.value)}
              style={{ marginBottom: 0, flex: 1 }}
            />
          </div>
          <div style={{ display: "flex", alignItems: "flex-start", gap: "0.75rem" }}>
            <label
              style={{
                fontWeight: 500,
                fontSize: "0.85rem",
                color: "#374151",
                minWidth: "50px",
                paddingTop: "0.35rem",
              }}
            >
              Desc:
            </label>
            <textarea
              className="text-input"
              rows={2}
              placeholder="Target variable description..."
              value={targetDescription}
              onChange={(e) => setTargetDescription(e.target.value)}
              style={{ marginBottom: 0, flex: 1, resize: "vertical" }}
            />
          </div>
        </div>
      </Section>

      {/* Feature Metadata */}
      {featureNames.length > 0 && (
        <Section title={`Feature Metadata (${featureNames.length} features)`}>
          <div
            style={{
              maxHeight: "400px",
              overflowY: "auto",
              border: "1px solid #e5e7eb",
              borderRadius: "0.25rem",
            }}
          >
            <table className="experiment-table" style={{ marginBottom: 0 }}>
              <thead>
                <tr>
                  <th style={{ width: "160px" }}>Feature</th>
                  <th style={{ width: "130px" }}>Type</th>
                  <th>Description</th>
                </tr>
              </thead>
              <tbody>
                {featureNames.map((name) => (
                  <tr key={name}>
                    <td>
                      <span
                        style={{
                          fontFamily: "monospace",
                          fontSize: "0.85rem",
                          fontWeight: 500,
                        }}
                      >
                        {name}
                      </span>
                    </td>
                    <td>
                      <select
                        value={featureTypes[name] || ""}
                        onChange={(e) =>
                          setFeatureTypes((prev) => ({
                            ...prev,
                            [name]: e.target.value,
                          }))
                        }
                        style={{
                          width: "100%",
                          padding: "0.3rem",
                          border: "1px solid #e5e7eb",
                          borderRadius: "0.25rem",
                          fontSize: "0.8rem",
                          background: "white",
                        }}
                      >
                        <option value="">-</option>
                        <option value="numeric">numeric</option>
                        <option value="integer">integer</option>
                        <option value="categorical">categorical</option>
                        <option value="binary">binary</option>
                        <option value="ordinal">ordinal</option>
                      </select>
                    </td>
                    <td>
                      <input
                        type="text"
                        className="text-input"
                        placeholder="Description..."
                        value={featureDescriptions[name] || ""}
                        onChange={(e) =>
                          setFeatureDescriptions((prev) => ({
                            ...prev,
                            [name]: e.target.value,
                          }))
                        }
                        style={{ marginBottom: 0 }}
                      />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Section>
      )}

      {/* Prepare Dataset */}
      <Section title="Prepare Dataset">
        <PrepareDatasetForm
          dataset={dataset}
          datasetId={datasetId}
          featureTypes={featureTypes}
          onPrepared={loadDataset}
          onSave={handleSave}
        />
      </Section>

      {/* Linked Splits */}
      <Section title="Linked Splits">
        {splits.length === 0 ? (
          <p className="hint" style={{ margin: "0 0 1rem 0" }}>
            No splits created yet.
          </p>
        ) : (
          <table className="experiment-table">
            <thead>
              <tr>
                <th>Name</th>
                <th>Type</th>
                <th>Train</th>
                <th>Test</th>
                <th>Seed</th>
              </tr>
            </thead>
            <tbody>
              {splits.map((split) => (
                <tr key={split.id}>
                  <td>
                    <span style={{ fontSize: "0.85rem" }}>
                      {split.name || `${Math.round((split.test_size ?? 0.2) * 100)}/${Math.round((1 - (split.test_size ?? 0.2)) * 100)} seed ${split.random_state ?? "?"}`}
                    </span>
                  </td>
                  <td>{split.split_type}</td>
                  <td>{split.train_size ?? "-"}</td>
                  <td>{split.test_size_actual ?? "-"}</td>
                  <td>{split.random_state ?? "-"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        <CreateSplitForm datasetId={datasetId} onCreated={loadDataset} />
      </Section>
    </div>
  );
}

function PrepareDatasetForm({
  dataset,
  datasetId,
  featureTypes,
  onPrepared,
  onSave,
}: {
  dataset: DatasetResponse;
  datasetId: string;
  featureTypes: Record<string, string>;
  onPrepared: () => void;
  onSave: () => Promise<void>;
}) {
  const [name, setName] = useState("");
  const [ordinalOnehot, setOrdinalOnehot] = useState<Set<string>>(new Set());
  const [preparing, setPreparing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  // If this is already a prepared dataset, show info instead of form
  if (dataset.source_dataset_id) {
    return (
      <p className="hint" style={{ margin: 0 }}>
        This is a prepared dataset (source: {dataset.source_dataset_id}).
      </p>
    );
  }

  const categoricalFeatures = Object.entries(featureTypes).filter(
    ([, t]) => t === "categorical"
  );
  const ordinalFeatures = Object.entries(featureTypes).filter(
    ([, t]) => t === "ordinal"
  );
  const hasEncodableFeatures =
    categoricalFeatures.length > 0 || ordinalFeatures.length > 0;

  const toggleOrdinalOnehot = (featureName: string) => {
    setOrdinalOnehot((prev) => {
      const next = new Set(prev);
      if (next.has(featureName)) {
        next.delete(featureName);
      } else {
        next.add(featureName);
      }
      return next;
    });
  };

  const handlePrepare = async () => {
    try {
      setPreparing(true);
      setError(null);
      setSuccess(null);
      // Save any pending feature-type changes first, so the prepare
      // endpoint sees the user's latest edits.
      await onSave();
      const prepared = await prepareDataset(
        datasetId,
        name || undefined,
        undefined,
        ordinalOnehot.size > 0 ? Array.from(ordinalOnehot) : undefined
      );
      setSuccess(`Created prepared dataset: ${prepared.name} (${prepared.id})`);
      onPrepared();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to prepare dataset");
    } finally {
      setPreparing(false);
    }
  };

  if (!hasEncodableFeatures) {
    return (
      <div>
        <p className="hint" style={{ margin: "0 0 0.5rem 0" }}>
          No categorical or ordinal features to encode.
        </p>
        <button className="op-btn" disabled>
          Prepare Dataset
        </button>
      </div>
    );
  }

  return (
    <div>
      {error && (
        <div className="error-message" style={{ marginBottom: "0.5rem" }}>
          {error}
        </div>
      )}
      {success && (
        <div
          style={{
            color: "#059669",
            fontSize: "0.85rem",
            fontWeight: 500,
            marginBottom: "0.5rem",
            padding: "0.5rem",
            background: "#ecfdf5",
            borderRadius: "0.25rem",
          }}
        >
          {success}
        </div>
      )}

      {/* Encoding summary */}
      <div style={{ marginBottom: "0.75rem" }}>
        <div
          style={{
            fontWeight: 500,
            fontSize: "0.85rem",
            color: "#374151",
            marginBottom: "0.35rem",
          }}
        >
          Features to encode:
        </div>
        <ul
          style={{
            margin: "0 0 0.5rem 0",
            padding: "0 0 0 1.25rem",
            fontSize: "0.85rem",
            color: "#4b5563",
            lineHeight: 1.6,
          }}
        >
          {categoricalFeatures.map(([feat]) => (
            <li key={feat}>
              <span style={{ fontFamily: "monospace", fontWeight: 500 }}>
                {feat}
              </span>{" "}
              — categorical (will be one-hot encoded)
            </li>
          ))}
          {ordinalFeatures.map(([feat]) => (
            <li key={feat}>
              <span style={{ fontFamily: "monospace", fontWeight: 500 }}>
                {feat}
              </span>{" "}
              — ordinal (
              {ordinalOnehot.has(feat)
                ? "will be one-hot encoded"
                : "will be rank-mapped"}
              )
            </li>
          ))}
        </ul>
      </div>

      {/* Ordinal options */}
      {ordinalFeatures.length > 0 && (
        <div style={{ marginBottom: "0.75rem" }}>
          <div
            style={{
              fontWeight: 500,
              fontSize: "0.85rem",
              color: "#374151",
              marginBottom: "0.35rem",
            }}
          >
            Ordinal encoding options:
          </div>
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              gap: "0.25rem",
              marginLeft: "0.25rem",
            }}
          >
            {ordinalFeatures.map(([feat]) => (
              <label
                key={feat}
                className="radio-label"
                style={{ fontSize: "0.85rem" }}
              >
                <input
                  type="checkbox"
                  checked={ordinalOnehot.has(feat)}
                  onChange={() => toggleOrdinalOnehot(feat)}
                />
                <span style={{ fontFamily: "monospace" }}>{feat}</span> — one-hot
                encode instead of rank
              </label>
            ))}
          </div>
        </div>
      )}

      {/* Name override */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "0.75rem",
          marginBottom: "0.75rem",
        }}
      >
        <label
          style={{
            fontWeight: 500,
            fontSize: "0.85rem",
            color: "#374151",
            minWidth: "50px",
          }}
        >
          Name:
        </label>
        <input
          type="text"
          className="text-input"
          placeholder={`${dataset.name} (prepared)`}
          value={name}
          onChange={(e) => setName(e.target.value)}
          style={{ marginBottom: 0, flex: 1 }}
        />
      </div>

      <button
        className="op-btn primary"
        onClick={handlePrepare}
        disabled={preparing}
      >
        {preparing ? "Preparing..." : "Prepare Dataset"}
      </button>
    </div>
  );
}

function CreateSplitForm({
  datasetId,
  onCreated,
}: {
  datasetId: string;
  onCreated: () => void;
}) {
  const [open, setOpen] = useState(false);
  const [testProportion, setTestProportion] = useState(0.2);
  const [randomSeed, setRandomSeed] = useState(42);
  const [stratify, setStratify] = useState(false);
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (!open) {
    return (
      <button
        className="op-btn"
        onClick={() => setOpen(true)}
        style={{ marginTop: "0.5rem" }}
      >
        + Create Split
      </button>
    );
  }

  const handleCreate = async () => {
    try {
      setCreating(true);
      setError(null);
      await createSplit(datasetId, testProportion, randomSeed, stratify);
      setOpen(false);
      onCreated();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create split");
    } finally {
      setCreating(false);
    }
  };

  const inputStyle = {
    padding: "0.35rem 0.5rem",
    border: "1px solid #e5e7eb",
    borderRadius: "0.25rem",
    fontSize: "0.85rem",
  };

  return (
    <div
      style={{
        marginTop: "0.75rem",
        padding: "0.75rem",
        border: "1px solid #e5e7eb",
        borderRadius: "0.375rem",
        background: "#f9fafb",
      }}
    >
      <div style={{ fontWeight: 500, fontSize: "0.85rem", marginBottom: "0.5rem" }}>
        Create New Split
      </div>

      {error && <div className="error-message" style={{ marginBottom: "0.5rem" }}>{error}</div>}

      <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
          <label style={{ fontSize: "0.85rem", color: "#374151", minWidth: "100px" }}>
            Test fraction:
          </label>
          <input
            type="number"
            min={0.05}
            max={0.5}
            step={0.05}
            value={testProportion}
            onChange={(e) => setTestProportion(parseFloat(e.target.value) || 0.2)}
            style={{ ...inputStyle, width: "80px" }}
          />
          <span style={{ fontSize: "0.8rem", color: "#6b7280" }}>
            ({Math.round(testProportion * 100)}% test)
          </span>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
          <label style={{ fontSize: "0.85rem", color: "#374151", minWidth: "100px" }}>
            Random seed:
          </label>
          <input
            type="number"
            min={0}
            value={randomSeed}
            onChange={(e) => setRandomSeed(parseInt(e.target.value) || 0)}
            style={{ ...inputStyle, width: "80px" }}
          />
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
          <label style={{ fontSize: "0.85rem", color: "#374151", minWidth: "100px" }}>
            Stratify:
          </label>
          <label className="radio-label">
            <input
              type="checkbox"
              checked={stratify}
              onChange={(e) => setStratify(e.target.checked)}
            />
            Stratified split (classification)
          </label>
        </div>
      </div>

      <div style={{ display: "flex", gap: "0.5rem", marginTop: "0.75rem" }}>
        <button
          className="op-btn primary"
          onClick={handleCreate}
          disabled={creating}
        >
          {creating ? "Creating..." : "Create"}
        </button>
        <button className="op-btn" onClick={() => setOpen(false)}>
          Cancel
        </button>
      </div>
    </div>
  );
}

function Section({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div
      style={{
        background: "white",
        border: "1px solid #e5e7eb",
        borderRadius: "0.5rem",
        padding: "1rem 1.25rem",
        marginBottom: "1rem",
        boxShadow: "0 1px 2px rgba(0,0,0,0.05)",
      }}
    >
      <h3
        style={{
          margin: "0 0 0.75rem 0",
          fontSize: "0.95rem",
          color: "#374151",
        }}
      >
        {title}
      </h3>
      {children}
    </div>
  );
}
