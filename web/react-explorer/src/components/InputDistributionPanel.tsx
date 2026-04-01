import { useCallback, useEffect, useState } from "react";
import {
  computeInputDistribution,
  getExperimentSplit,
  type InputDistributionResponse,
  type ModelState,
} from "../api/client";
import { VizCanvas } from "./VizCanvas";

type InputDistributionPanelProps = {
  genomeId: string;
  experimentId: string;
  selectedInputNodes: string[];
  model: ModelState;
};

type Stats = {
  mean: number;
  std: number;
  min: number;
  max: number;
  median: number;
  count?: number;
};

export function InputDistributionPanel({
  genomeId,
  experimentId,
  selectedInputNodes,
  model,
}: InputDistributionPanelProps) {
  const [splitId, setSplitId] = useState<string | null>(null);
  const [featureNames, setFeatureNames] = useState<string[] | null>(null);
  const [distData, setDistData] = useState<InputDistributionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch experiment split on mount
  useEffect(() => {
    let cancelled = false;
    getExperimentSplit(experimentId)
      .then((split) => {
        if (!cancelled) {
          setSplitId(split.split_id);
          setFeatureNames(split.feature_names);
        }
      })
      .catch(() => {
        if (!cancelled) setError("No dataset split linked to experiment");
      });
    return () => { cancelled = true; };
  }, [experimentId]);

  // Map node IDs to dataset column indices.
  // Split variants (e.g. "-20_a", "-20_b") share the same dataset column
  // as their base node ("-20"), so we deduplicate by base ID.
  const getFeatureIndex = useCallback(
    (nodeId: string): number => {
      const baseId = nodeId.replace(/_[a-z]$/, "");
      const seen = new Set<string>();
      let colIdx = 0;
      for (const nid of model.metadata.input_nodes) {
        const base = nid.replace(/_[a-z]$/, "");
        if (!seen.has(base)) {
          if (base === baseId) return colIdx;
          seen.add(base);
          colIdx++;
        }
      }
      return -1;
    },
    [model.metadata.input_nodes],
  );

  // Auto-fetch distribution when selection changes
  useEffect(() => {
    if (!splitId || selectedInputNodes.length === 0) {
      setDistData(null);
      return;
    }

    const indices = selectedInputNodes.map(getFeatureIndex).filter((i) => i >= 0);
    if (indices.length === 0) {
      setDistData(null);
      return;
    }

    // Deduplicate indices (split nodes may map to same feature)
    const uniqueIndices = [...new Set(indices)].slice(0, 2);

    let cancelled = false;
    setLoading(true);
    setError(null);

    computeInputDistribution(genomeId, {
      dataset_split_id: splitId,
      feature_indices: uniqueIndices,
    })
      .then((resp) => {
        if (!cancelled) setDistData(resp);
      })
      .catch((err) => {
        if (!cancelled)
          setError(err instanceof Error ? err.message : "Failed to load distribution");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => { cancelled = true; };
  }, [splitId, selectedInputNodes, genomeId, getFeatureIndex]);

  // Render feature names for selected nodes
  const selectedNames = selectedInputNodes.map((nodeId) => {
    const idx = getFeatureIndex(nodeId);
    if (idx >= 0 && featureNames && idx < featureNames.length) {
      return featureNames[idx];
    }
    return nodeId;
  });

  return (
    <div className="input-distribution-panel">
      <h3>Input Distribution</h3>
      <p className="selected-features">
        {selectedNames.join(", ")}
      </p>

      {error && <div className="error-message">{error}</div>}
      {loading && <div className="loading">Loading distribution...</div>}

      {distData && !loading && (
        <>
          <VizCanvas vizType={distData.viz_type} data={distData.data} />
          <div className="distribution-stats">
            {distData.viz_type === "histogram" && (
              <StatsTable
                stats={(distData.data as Record<string, unknown>).stats as Stats}
              />
            )}
            {distData.viz_type === "scatter2d" && (
              <div className="scatter-stats">
                <div>
                  <strong>{distData.feature_names[0]}</strong>
                  <StatsTable
                    stats={(distData.data as Record<string, unknown>).x_stats as Stats}
                  />
                </div>
                <div>
                  <strong>{distData.feature_names[1]}</strong>
                  <StatsTable
                    stats={(distData.data as Record<string, unknown>).y_stats as Stats}
                  />
                </div>
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}

function StatsTable({ stats }: { stats: Stats }) {
  if (!stats) return null;
  const fmt = (v: number) => {
    if (Number.isInteger(v)) return v.toString();
    return v.toFixed(3);
  };
  return (
    <table className="stats-table">
      <tbody>
        <tr><td>Mean</td><td>{fmt(stats.mean)}</td></tr>
        <tr><td>Std</td><td>{fmt(stats.std)}</td></tr>
        <tr><td>Min</td><td>{fmt(stats.min)}</td></tr>
        <tr><td>Median</td><td>{fmt(stats.median)}</td></tr>
        <tr><td>Max</td><td>{fmt(stats.max)}</td></tr>
        {stats.count !== undefined && (
          <tr><td>Count</td><td>{stats.count}</td></tr>
        )}
      </tbody>
    </table>
  );
}
