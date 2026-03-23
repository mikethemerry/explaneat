/**
 * DatasetInfoPanel - shows dataset overview, feature-to-node mapping,
 * and bulk auto-rename button.
 */

import { useState, useEffect, useCallback } from "react";
import {
  getExperimentSplit,
  addOperation,
  type ExperimentSplitResponse,
  type ModelState,
} from "../api/client";

// =============================================================================
// Logging utilities
// =============================================================================

const LOG_PREFIX = "[DatasetInfoPanel]";

function logDebug(message: string, data?: unknown) {
  console.log(`${LOG_PREFIX} ${message}`, data !== undefined ? data : "");
}

function logInfo(message: string, data?: unknown) {
  console.info(`${LOG_PREFIX} ${message}`, data !== undefined ? data : "");
}

function logError(message: string, data?: unknown) {
  console.error(`${LOG_PREFIX} ${message}`, data !== undefined ? data : "");
}

// =============================================================================
// Types
// =============================================================================

type DatasetInfoPanelProps = {
  experimentId: string;
  genomeId: string;
  model: ModelState;
  onOperationChange: () => void;
};

// =============================================================================
// Component
// =============================================================================

export function DatasetInfoPanel({
  experimentId,
  genomeId,
  model,
  onOperationChange,
}: DatasetInfoPanelProps) {
  const [split, setSplit] = useState<ExperimentSplitResponse | null>(null);
  const [collapsed, setCollapsed] = useState(true);
  const [renaming, setRenaming] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!experimentId) return;
    getExperimentSplit(experimentId)
      .then((data) => {
        logDebug("Split data loaded", { dataset: data.dataset_name, features: data.feature_names?.length });
        setSplit(data);
      })
      .catch((err) => {
        logDebug("No split found", { error: err });
        setSplit(null);
      });
  }, [experimentId]);

  const handleAutoRename = useCallback(async () => {
    if (!split?.feature_names || !model) return;

    setRenaming(true);
    setError(null);
    const inputNodes = model.metadata.input_nodes;
    const featureNames = split.feature_names;
    let renamed = 0;

    try {
      for (let i = 0; i < inputNodes.length && i < featureNames.length; i++) {
        const nodeId = inputNodes[i];
        const featureName = featureNames[i];
        const nodeObj = model.nodes.find((n) => n.id === nodeId);

        // Skip if already has a display_name
        if (nodeObj?.display_name) {
          logDebug(`Skipping ${nodeId}: already named ${nodeObj.display_name}`);
          continue;
        }

        // Rename the base node
        logDebug(`Renaming ${nodeId} -> ${featureName}`);
        await addOperation(genomeId, {
          type: "rename_node",
          params: { node_id: nodeId, display_name: featureName },
        });
        renamed++;

        // Also rename any split variants (e.g. "-1_a", "-1_b")
        const splitVariants = model.nodes.filter(
          (n) => n.id !== nodeId && n.id.startsWith(`${nodeId}_`)
        );
        for (const variant of splitVariants) {
          if (variant.display_name) continue;
          const suffix = variant.id.slice(nodeId.length + 1); // e.g. "a", "b"
          const variantName = `${featureName}_${suffix}`;
          logDebug(`Renaming split variant ${variant.id} -> ${variantName}`);
          await addOperation(genomeId, {
            type: "rename_node",
            params: { node_id: variant.id, display_name: variantName },
          });
          renamed++;
        }
      }

      logInfo(`Auto-renamed ${renamed} nodes`);
      onOperationChange();
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Rename failed";
      logError("Auto-rename failed", { error: err });
      setError(msg);
    } finally {
      setRenaming(false);
    }
  }, [split, model, genomeId, onOperationChange]);

  if (!split) {
    return null;
  }

  const inputNodes = model.metadata.input_nodes;
  const featureNames = split.feature_names;
  const featureTypes = split.feature_types;
  const featureDescriptions = split.feature_descriptions;

  // Check if any input nodes lack a display_name
  const hasUnnamedInputs = inputNodes.some((nodeId, i) => {
    const nodeObj = model.nodes.find((n) => n.id === nodeId);
    return !nodeObj?.display_name && featureNames && i < featureNames.length;
  });

  return (
    <div className="dataset-info-panel">
      <div
        className="dataset-info-header"
        onClick={() => setCollapsed(!collapsed)}
        style={{ cursor: "pointer" }}
      >
        <span className="collapse-toggle">{collapsed ? "\u25b6" : "\u25bc"}</span>
        <h3 style={{ margin: 0, fontSize: "0.85rem" }}>
          Dataset: {split.dataset_name}
        </h3>
      </div>

      {!collapsed && (
        <div className="dataset-info-body">
          {/* Summary */}
          <div className="dataset-summary">
            {split.num_classes && split.num_classes > 0 && (
              <span>Classification: {split.num_classes} classes</span>
            )}
            <span>
              {split.num_samples ?? "?"} samples, {split.num_features ?? "?"} features
            </span>
          </div>

          {/* Feature table */}
          {featureNames && featureNames.length > 0 && (
            <div className="feature-table-container">
              <table className="feature-table">
                <thead>
                  <tr>
                    <th>Feature</th>
                    <th>Type</th>
                    <th>Node</th>
                    <th>Name</th>
                  </tr>
                </thead>
                <tbody>
                  {featureNames.map((name, i) => {
                    const nodeId = i < inputNodes.length ? inputNodes[i] : null;
                    const nodeObj = nodeId
                      ? model.nodes.find((n) => n.id === nodeId)
                      : null;
                    const type = featureTypes?.[name] ?? null;
                    return (
                      <tr key={name}>
                        <td title={featureDescriptions?.[name] ?? undefined}>
                          {name}
                        </td>
                        <td className="type-cell">{type ?? "-"}</td>
                        <td className="node-cell">{nodeId ?? "-"}</td>
                        <td className="name-cell">
                          {nodeObj?.display_name ?? (
                            <span style={{ color: "#9ca3af" }}>-</span>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}

          {/* Target info */}
          {split.target_name && (
            <div className="target-info">
              <strong>Target:</strong> {split.target_name}
              {split.target_description && (
                <span style={{ color: "#9ca3af" }}> - {split.target_description}</span>
              )}
            </div>
          )}

          {/* Auto-rename button */}
          {featureNames && hasUnnamedInputs && (
            <button
              className="op-btn primary"
              onClick={handleAutoRename}
              disabled={renaming}
              style={{ marginTop: "8px", width: "100%" }}
            >
              {renaming ? "Renaming..." : "Auto-rename all inputs"}
            </button>
          )}

          {error && (
            <p style={{ color: "#f87171", fontSize: "11px", marginTop: "4px" }}>
              {error}
            </p>
          )}
        </div>
      )}
    </div>
  );
}
