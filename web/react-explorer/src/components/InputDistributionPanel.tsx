import { useCallback, useEffect, useState } from "react";
import {
  computeInputDistribution,
  getExperimentSplit,
  type InputDistributionResponse,
  type ModelState,
} from "../api/client";
import { VizCanvas } from "./VizCanvas";
import { useFeatureLabels, withLabel } from "../featureLabels";
import type { FeaturePin, PinInput } from "../api/features";
import { PinFeatureModal, type PinModalDraft } from "./PinFeatureModal";

type InputDistributionPanelProps = {
  genomeId: string;
  experimentId: string;
  selectedInputNodes: string[];
  model: ModelState;
  experimentName?: string;
  recordLabel?: string | null;
  onPinned?: (pin: FeaturePin) => void;
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
  experimentName,
  recordLabel,
  onPinned,
}: InputDistributionPanelProps) {
  const [pinDraft, setPinDraft] = useState<PinModalDraft | null>(null);
  const [pinnedMsg, setPinnedMsg] = useState<string | null>(null);
  const [splitId, setSplitId] = useState<string | null>(null);
  const [featureNames, setFeatureNames] = useState<string[] | null>(null);
  const [distData, setDistData] = useState<InputDistributionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fl = useFeatureLabels();

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
      const baseId = nodeId.replace(/_(?:[a-z]|[A-Z][a-z])$/, "");
      const seen = new Set<string>();
      let colIdx = 0;
      for (const nid of model.metadata.input_nodes) {
        const base = nid.replace(/_(?:[a-z]|[A-Z][a-z])$/, "");
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

  const openPin = () => {
    const inputs: PinInput[] = selectedInputNodes.map((nodeId, i) => {
      const f = fl.forNode(nodeId);
      const raw = f?.raw ?? selectedNames[i];
      return { node_id: nodeId, raw, label: f?.decoded ? f.label : raw, index: getFeatureIndex(nodeId) };
    });
    const names = inputs.map((x) => (x.label && x.label !== x.raw ? `${x.raw} · ${x.label}` : x.raw));
    setPinDraft({
      genome_id: genomeId,
      experiment_id: experimentId,
      experiment_name: experimentName ?? null,
      record_label: recordLabel ?? null,
      scope: { kind: "input", node_id: selectedInputNodes[0] },
      inputs,
      viz: {
        type: "input_distribution",
        params: { feature_indices: inputs.map((x) => x.index) },
        input_nodes: selectedInputNodes,
        description: `Input distribution of ${names.join(" vs ")}`,
      },
      also_views: [],
      tags: inputs.map((x) => (x.raw ?? "").toLowerCase()).filter(Boolean),
    });
  };

  return (
    <div className="input-distribution-panel">
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: "8px" }}>
        <h3>Input Distribution</h3>
        <button className="op-btn pin-btn" style={{ fontSize: "11px" }} onClick={openPin} title="Pin this input as a feature of interest">
          Pin this input
        </button>
      </div>
      {pinnedMsg && <div className="pin-saved">{pinnedMsg}</div>}
      {pinDraft && (
        <PinFeatureModal
          draft={pinDraft}
          onClose={() => setPinDraft(null)}
          onSaved={(pin) => {
            setPinDraft(null);
            setPinnedMsg(`Pinned ${pin.id}: see Features.`);
            window.setTimeout(() => setPinnedMsg(null), 6000);
            onPinned?.(pin);
          }}
        />
      )}
      <div className="selected-features">
        {selectedInputNodes.map((nodeId, i) => {
          const f = fl.forNode(nodeId);
          const node = model.nodes.find((n) => n.id === nodeId);
          return (
            <div key={nodeId} className="feature-card" data-testid="feature-card">
              <div className="feature-card-title">
                <code>{f?.raw ?? selectedNames[i]}</code>
                {f?.decoded && f.label !== f.raw && <span className="feature-meaning"> · {f.label}</span>}
              </div>
              <table className="stats-table">
                <tbody>
                  <tr><td>Node</td><td>{nodeId}{node?.display_name ? ` (shown as ${node.display_name})` : ""}</td></tr>
                  <tr><td>Feature name</td><td><code>{f?.raw ?? selectedNames[i]}</code></td></tr>
                  <tr><td>Readable label</td><td>{f ? (f.decoded ? f.full_label : "no codebook meaning (raw name)") : "—"}</td></tr>
                  {f?.one_means && <tr><td>Value 1 means</td><td>{f.one_means}</td></tr>}
                  {f?.unit && <tr><td>Unit</td><td>{f.unit}</td></tr>}
                </tbody>
              </table>
            </div>
          );
        })}
      </div>

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
                  <strong>{withLabel(distData.feature_names[0], fl.forName(distData.feature_names[0]) ?? fl.forNode(selectedInputNodes[0]))}</strong>
                  <StatsTable
                    stats={(distData.data as Record<string, unknown>).x_stats as Stats}
                  />
                </div>
                <div>
                  <strong>{withLabel(distData.feature_names[1], fl.forName(distData.feature_names[1]) ?? fl.forNode(selectedInputNodes[1]))}</strong>
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
