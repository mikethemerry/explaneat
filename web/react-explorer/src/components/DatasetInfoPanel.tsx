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
// Column mapping utilities
// =============================================================================

/**
 * Map input node IDs to dataset column indices using base-node deduplication.
 * Split variants (e.g., -2_a, -2_b from splitting -2) share the same
 * dataset column as their base node.
 *
 * Returns both directions:
 *  - nodeToCol: node_id -> dataset column index
 *  - colToNodes: dataset column index -> list of node_ids
 */
function buildColumnMapping(inputNodes: string[]): {
  nodeToCol: Map<string, number>;
  colToNodes: Map<number, string[]>;
} {
  const nodeToCol = new Map<string, number>();
  const colToNodes = new Map<number, string[]>();
  const seenBases = new Map<string, number>();
  let col = 0;

  for (const nid of inputNodes) {
    const base = nid.replace(/_[a-z]$/, "");
    if (!seenBases.has(base)) {
      seenBases.set(base, col);
      colToNodes.set(col, []);
      col++;
    }
    const c = seenBases.get(base)!;
    nodeToCol.set(nid, c);
    colToNodes.get(c)!.push(nid);
  }

  return { nodeToCol, colToNodes };
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
    const { colToNodes } = buildColumnMapping(inputNodes);
    let renamed = 0;

    try {
      for (let col = 0; col < featureNames.length; col++) {
        const nodes = colToNodes.get(col);
        if (!nodes || nodes.length === 0) continue;
        const featureName = featureNames[col];

        for (const nodeId of nodes) {
          const nodeObj = model.nodes.find((n) => n.id === nodeId);
          if (!nodeObj) continue;
          if (nodeObj.display_name) {
            logDebug(`Skipping ${nodeId}: already named ${nodeObj.display_name}`);
            continue;
          }

          // Use feature name directly for unsplit nodes, add suffix for variants
          let displayName: string;
          if (nodes.length === 1) {
            displayName = featureName;
          } else {
            const match = nodeId.match(/_([a-z])$/);
            displayName = match ? `${featureName}_${match[1]}` : featureName;
          }

          logDebug(`Renaming ${nodeId} -> ${displayName}`);
          await addOperation(genomeId, {
            type: "rename_node",
            params: { node_id: nodeId, display_name: displayName },
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
  const { colToNodes } = buildColumnMapping(inputNodes);

  // Check if any input nodes lack a display_name
  const hasUnnamedInputs = featureNames
    ? Array.from({ length: featureNames.length }).some((_, col) => {
        const nodes = colToNodes.get(col) ?? [];
        return nodes.some((nid) => {
          const nodeObj = model.nodes.find((n) => n.id === nid);
          return nodeObj && !nodeObj.display_name;
        });
      })
    : false;

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
                    const nodes = colToNodes.get(i) ?? [];
                    const type = featureTypes?.[name] ?? null;
                    const displayNames = nodes
                      .map((nid) => model.nodes.find((n) => n.id === nid)?.display_name)
                      .filter(Boolean);
                    return (
                      <tr key={name}>
                        <td title={featureDescriptions?.[name] ?? undefined}>
                          {name}
                        </td>
                        <td className="type-cell">{type ?? "-"}</td>
                        <td className="node-cell">
                          {nodes.length > 0 ? nodes.join(", ") : "-"}
                        </td>
                        <td className="name-cell">
                          {displayNames.length > 0 ? (
                            displayNames.join(", ")
                          ) : (
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
