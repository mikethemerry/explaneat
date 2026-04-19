import { useState, useCallback, useRef, useMemo, useEffect } from "react";
import {
  computeVizData,
  computeShap,
  computePerformance,
  saveSnapshot,
  type VizDataRequest,
  type VizDataResponse,
  type AnnotationSummary,
  type PerformanceResponse,
} from "../api/client";
import { FormulaDisplay } from "./FormulaDisplay";
import { DatasetSelector } from "./DatasetSelector";
import { VizCanvas } from "./VizCanvas";
import { EvidenceGallery } from "./EvidenceGallery";

type EvidencePanelProps = {
  genomeId: string;
  experimentId: string;
  annotation: AnnotationSummary;
  isWholeModel?: boolean;
  isNodeLevel?: boolean;
  nodeId?: string;
  width?: number;
  onResize?: (width: number) => void;
  onEdgeInfluence?: (data: Record<string, any> | null) => void;
};

const VIZ_TYPE_LABELS: Record<string, string> = {
  line: "Line Plot",
  heatmap: "Heatmap",
  partial_dependence: "Partial Dep.",
  pca_scatter: "PCA Scatter",
  sensitivity: "Sensitivity",
  ice: "ICE Plot",
  feature_output_scatter: "Feature vs Output",
  output_distribution: "Output Dist.",
  shap: "SHAP Values",
  activation_profile: "Activation Profile",
  regime_map: "Regime Map",
  edge_influence: "Edge Influence",
};

const VIZ_TYPE_EXPLANATIONS: Record<string, { title: string; what: string; how: string; look_for: string }> = {
  line: {
    title: "Line Plot (Sweep)",
    what: "Sweeps one input dimension across its range while holding all others at their median values. Shows the annotation/node's output as a function of that single input. Scatter dots show actual data points.",
    how: "The smooth line is the model's response curve. Scatter dots show real data. If correctness coloring is on, green = correct prediction, red = incorrect.",
    look_for: "Monotonic trends (the node simply passes through the input), sharp kinks (ReLU boundaries switching on/off), flat regions (the node is saturated or clamped). Wide vertical spread in scatter dots at any x-value means other features matter more than this one.",
  },
  heatmap: {
    title: "Heatmap (2D Sweep)",
    what: "Sweeps two input dimensions simultaneously across a grid, holding all others at median. Color shows the annotation/node's output value.",
    how: "Bright regions = high output, dark regions = low output. Contour-like boundaries show where the function changes behavior.",
    look_for: "Axis-aligned boundaries (the function depends mainly on one input), diagonal boundaries (interaction between the two inputs), or flat regions (neither input matters much in that area).",
  },
  partial_dependence: {
    title: "Partial Dependence",
    what: "Shows the average effect of one (or two) features on the output, marginalizing over all other features. Unlike a sweep (which fixes others at median), this averages over the actual distribution of other features.",
    how: "The curve shows the expected output as a function of the selected feature. Steeper slopes = stronger feature effect.",
    look_for: "Compare with the line plot for the same feature. If they differ significantly, interactions with other features matter. Flat partial dependence means the feature has little average effect.",
  },
  pca_scatter: {
    title: "PCA Scatter",
    what: "Projects the annotation's entry activations into 2D using PCA and colors points by exit activation. Reduces high-dimensional input space to its two most important directions.",
    how: "Each dot is a data point. Position = PCA projection of inputs to this annotation. Color = the annotation's output value. Explained variance tells you how much information the 2D view captures.",
    look_for: "Clear color gradients (the annotation's output is well-explained by the principal components), clusters (distinct operating regimes), or random-looking color (the output depends on dimensions PCA didn't capture).",
  },
  sensitivity: {
    title: "Sensitivity Analysis",
    what: "Measures how much each input affects the output by computing the variance of the output when each input is perturbed independently.",
    how: "Bar height = sensitivity score. Higher bars = the output is more sensitive to that input. Inputs are ranked by importance.",
    look_for: "Which inputs dominate. If one bar is much taller than the rest, that input drives the node's behavior. Near-zero bars indicate inputs that barely affect this node.",
  },
  ice: {
    title: "ICE (Individual Conditional Expectation)",
    what: "Like a line plot, but shows one sweep curve per data point instead of a single sweep at median. The bold line is the partial dependence (average of all ICE curves).",
    how: "Each thin line shows how one individual's prediction changes as you vary the selected feature. The bold line averages across all individuals.",
    look_for: "Parallel lines mean the feature effect is consistent across individuals. Crossing lines reveal interactions: the feature's effect depends on other feature values. Clusters of lines suggest distinct subgroups.",
  },
  feature_output_scatter: {
    title: "Feature vs Output Scatter",
    what: "Plots one input feature (x-axis) against the annotation/node's output (y-axis) for all data points, with a partial dependence curve overlay.",
    how: "Each dot is a data point. The smooth curve shows the average trend. Vertical spread at any x-value shows how much other features contribute.",
    look_for: "Tight clustering around the curve (this feature largely determines the output) vs wide spread (other features matter more). The curve shape reveals the relationship type: linear, step-like (ReLU boundary), or nonlinear.",
  },
  output_distribution: {
    title: "Output Distribution",
    what: "Histogram of the annotation/node's output values across the dataset.",
    how: "Shows how the output values are distributed. X-axis = output value, Y-axis = count of data points.",
    look_for: "Bimodal distributions (the node acts as a binary switch), spike at zero (ReLU clamping, the node is mostly dead), or wide uniform spread (the node uses its full range).",
  },
  shap: {
    title: "SHAP Values",
    what: "Shapley Additive Explanations — decomposes each prediction into per-feature contributions. Shows how much each input feature pushes the output up or down from the average.",
    how: "Bar height = mean absolute SHAP value (average importance). Positive SHAP = pushes output higher, negative = pushes lower.",
    look_for: "The ranking tells you which features the model relies on most. Compare with sensitivity analysis — they measure importance differently (SHAP accounts for interactions, sensitivity doesn't).",
  },
  activation_profile: {
    title: "Activation Profile",
    what: "Histogram of a single node's activation values across the entire dataset. Shows how the node behaves in practice — is it mostly active, mostly dead, or somewhere in between?",
    how: "X-axis = activation value, Y-axis = count. The red dashed line marks zero (the ReLU boundary). Stats show: Active (fraction > 0), Dead (fraction exactly 0), mean, std, range.",
    look_for: "Spike at zero = the node is frequently clamped (ReLU dead zone). High 'Dead' percentage (>90%) suggests a near-dead node that barely contributes. Bimodal distribution = the node acts as a switch. Activation rate tells you what fraction of the dataset this node is 'on' for.",
  },
  regime_map: {
    title: "Regime Map",
    what: "For a subgraph (annotation or node ancestors), identifies distinct operating regimes by tracking which ReLU nodes are on vs off across the dataset. Each unique on/off pattern = a regime.",
    how: "Table rows = regimes, sorted by frequency. Green dots = ReLU on, grey dots = ReLU off. Count = how many data points fall in this regime. Accuracy = classification accuracy within the regime.",
    look_for: "Few dominant regimes (2-3 covering >90% of data) means the subgraph has simple, interpretable modes. Many regimes with small counts suggest complex, data-dependent behavior. Low accuracy in a regime = the model struggles with that subpopulation. Within each regime, all ReLUs are fixed, so the subgraph is purely linear — you can reason about its behavior algebraically.",
  },
  edge_influence: {
    title: "Edge Influence",
    what: "Measures how much each connection actually contributes in practice by computing the variance of (weight x source activation) across the dataset. High influence = the edge differentiates predictions. Low influence = the edge is vestigial.",
    how: "Bar length = influence (variance). Blue = positive mean contribution (pushes output up), red = negative (pushes output down). 'Show on graph' overlays this as edge thickness and color on the network diagram.",
    look_for: "Edges with near-zero influence are effectively dead — the weight exists but the source node doesn't vary enough (or the weight is too small) to matter. A few high-influence edges dominating tells you where the real computation happens. Compare with edge weights: a large weight with low influence means the source node is constant; a small weight with high influence means the source node varies wildly.",
  },
};

const ALL_VIZ_TYPES = ["line", "heatmap", "partial_dependence", "pca_scatter", "sensitivity", "ice", "feature_output_scatter", "output_distribution", "shap", "activation_profile", "regime_map", "edge_influence"];

// Which viz types need which dimension selectors
const NEEDS_INPUT_DIM: Record<string, "single" | "pair" | false> = {
  line: "single",
  heatmap: "pair",
  partial_dependence: "single",
  pca_scatter: false,
  sensitivity: false,
  ice: "single",
  feature_output_scatter: "single",
  output_distribution: false,
  shap: false,
};

const NEEDS_OUTPUT_DIM: Record<string, boolean> = {
  line: true,
  heatmap: true,
  partial_dependence: true,
  pca_scatter: true,
  sensitivity: false, // shows all outputs
  ice: true,
  feature_output_scatter: true,
  output_distribution: true,
  shap: false,
};

/* ── VizSlot: independent visualization with its own controls ── */

type VizSlotProps = {
  genomeId: string;
  annotation: AnnotationSummary;
  splitId: string;
  splitChoice: "train" | "test" | "val" | "both";
  sampleFraction: number;
  isWholeModel: boolean;
  isNodeLevel: boolean;
  nodeId?: string;
  nIn: number;
  nOut: number;
  defaultVizType: string;
  onGalleryRefresh: () => void;
  viewMode: "network" | "source";
  onEdgeInfluence?: (data: Record<string, any> | null) => void;
};

function VizSlot({
  genomeId, annotation, splitId, splitChoice, sampleFraction,
  isWholeModel, isNodeLevel, nodeId,
  nIn, nOut, defaultVizType, onGalleryRefresh, viewMode, onEdgeInfluence,
}: VizSlotProps) {
  const [vizData, setVizData] = useState<VizDataResponse | null>(null);
  const [vizType, setVizType] = useState<string>(defaultVizType);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [snapshotNarrative, setSnapshotNarrative] = useState("");
  const svgRef = useRef<SVGSVGElement | null>(null);

  const [inputDim, setInputDim] = useState(0);
  const [inputDims, setInputDims] = useState<[number, number]>([0, 1]);
  const [outputDim, setOutputDim] = useState(0);
  const [showOnGraph, setShowOnGraph] = useState(false);
  const [showExplain, setShowExplain] = useState(false);

  const inputOptions = useMemo(() => Array.from({ length: nIn }, (_, i) => i), [nIn]);
  const outputOptions = useMemo(() => Array.from({ length: nOut }, (_, i) => i), [nOut]);

  const buildVizParams = useCallback((): Record<string, unknown> => {
    const params: Record<string, unknown> = {};
    const inputMode = NEEDS_INPUT_DIM[vizType];
    if (inputMode === "single") {
      params.input_dim = inputDim;
      if (vizType === "partial_dependence") {
        params.vary_dims = [inputDim];
      }
    } else if (inputMode === "pair") {
      params.input_dims = inputDims;
    }
    if (NEEDS_OUTPUT_DIM[vizType]) {
      params.output_dim = outputDim;
    }
    return params;
  }, [vizType, inputDim, inputDims, outputDim]);

  const doComputeViz = useCallback(async (forceRecompute = false) => {
    if (!splitId) return;
    setLoading(true);
    setError(null);

    try {
      if (vizType === "shap") {
        const result = await computeShap(genomeId, {
          dataset_split_id: splitId,
          annotation_id: isWholeModel || isNodeLevel ? undefined : annotation.id,
          node_id: isNodeLevel ? nodeId : undefined,
          split: splitChoice,
          max_samples: 100,
          force_recompute: forceRecompute || undefined,
        });
        setVizData({
          viz_type: "shap",
          data: {
            feature_names: result.feature_names,
            mean_abs_shap: result.mean_abs_shap,
            base_value: result.base_value,
            outputs: result.outputs,
          },
          dimensionality: [result.feature_names.length, 1],
          suggested_viz_types: [],
        });
      } else {
        const result = await computeVizData(genomeId, {
          annotation_id: isWholeModel ? undefined : (isNodeLevel ? undefined : annotation.id),
          node_id: isNodeLevel ? nodeId : undefined,
          dataset_split_id: splitId,
          viz_type: vizType as VizDataRequest["viz_type"],
          params: buildVizParams(),
          split: splitChoice,
          sample_fraction: sampleFraction,
          max_samples: 1000,
          view: viewMode,
        });
        setVizData(result);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to compute visualization");
    } finally {
      setLoading(false);
    }
  }, [genomeId, annotation.id, splitId, vizType, splitChoice, sampleFraction, buildVizParams, isWholeModel, isNodeLevel, nodeId, viewMode]);

  const handleComputeViz = useCallback(() => doComputeViz(false), [doComputeViz]);
  const handleRecomputeViz = useCallback(() => doComputeViz(true), [doComputeViz]);

  // Auto-compute viz when parameters change (skip SHAP — it's expensive)
  const autoComputeInitRef = useRef(false);
  useEffect(() => {
    if (!splitId || loading || vizType === "shap") return;
    if (!autoComputeInitRef.current) {
      autoComputeInitRef.current = true;
      doComputeViz(false);
      return;
    }
    doComputeViz(false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [splitId, vizType, inputDim, inputDims[0], inputDims[1], outputDim, splitChoice, viewMode]);

  const handleSnapshot = useCallback(async () => {
    if (!svgRef.current) return;
    const serializer = new XMLSerializer();
    const svgString = serializer.serializeToString(svgRef.current);
    const svgBase64 = btoa(svgString);
    try {
      await saveSnapshot(
        genomeId, annotation.id,
        { viz_type: vizType, split_id: splitId, split_choice: splitChoice, sample_fraction: sampleFraction, ...buildVizParams() },
        svgBase64, snapshotNarrative,
      );
      setSnapshotNarrative("");
      onGalleryRefresh();
    } catch (err) {
      console.error("Failed to save snapshot:", err);
    }
  }, [genomeId, annotation.id, vizType, splitId, splitChoice, sampleFraction, snapshotNarrative, buildVizParams, onGalleryRefresh]);

  const handleSvgRef = useCallback((svg: SVGSVGElement | null) => {
    svgRef.current = svg;
  }, []);

  const suggestedTypes = vizData?.suggested_viz_types || [];
  const inputMode = NEEDS_INPUT_DIM[vizType];
  const showOutputDim = NEEDS_OUTPUT_DIM[vizType] && nOut > 1;

  return (
    <div className="viz-slot">
      <div className="evidence-section">
        <div className="viz-type-selector">
          <label className="selector-label">Visualization</label>
          <div className="viz-type-buttons">
            {(suggestedTypes.length > 0 ? [...suggestedTypes, "shap"] : ALL_VIZ_TYPES).map((vt) => (
              <button
                key={vt}
                className={`viz-type-btn ${vizType === vt ? "active" : ""}`}
                onClick={() => setVizType(vt)}
              >
                {VIZ_TYPE_LABELS[vt] || vt}
              </button>
            ))}
          </div>
        </div>

        <div className="dim-selectors">
          {inputMode === "single" && nIn > 1 && (
            <div className="dim-selector">
              <label className="dim-label">Input dim</label>
              <select className="dim-select" value={inputDim} onChange={(e) => setInputDim(Number(e.target.value))}>
                {inputOptions.map((i) => (
                  <option key={i} value={i}>{vizData?.entry_names?.[i] ?? `x_${i} (${annotation.entry_nodes[i]})`}</option>
                ))}
              </select>
              {nIn > 1 && <span className="dim-hint">others fixed at median</span>}
            </div>
          )}
          {inputMode === "pair" && nIn >= 2 && (
            <div className="dim-selector">
              <label className="dim-label">Input dims</label>
              <div className="dim-pair">
                <select className="dim-select" value={inputDims[0]} onChange={(e) => setInputDims([Number(e.target.value), inputDims[1]])}>
                  {inputOptions.map((i) => (
                    <option key={i} value={i}>{vizData?.entry_names?.[i] ?? `x_${i}`}</option>
                  ))}
                </select>
                <span className="dim-sep">vs</span>
                <select className="dim-select" value={inputDims[1]} onChange={(e) => setInputDims([inputDims[0], Number(e.target.value)])}>
                  {inputOptions.filter((i) => i !== inputDims[0]).map((i) => (
                    <option key={i} value={i}>{vizData?.entry_names?.[i] ?? `x_${i}`}</option>
                  ))}
                </select>
              </div>
              {nIn > 2 && <span className="dim-hint">others fixed at median</span>}
            </div>
          )}
          {showOutputDim && (
            <div className="dim-selector">
              <label className="dim-label">Output dim</label>
              <select className="dim-select" value={outputDim} onChange={(e) => setOutputDim(Number(e.target.value))}>
                {outputOptions.map((i) => (
                  <option key={i} value={i}>{vizData?.exit_names?.[i] ?? `y_${i} (${annotation.exit_nodes[i]})`}</option>
                ))}
              </select>
            </div>
          )}
        </div>

        <div className="compute-actions">
          <button className="op-btn primary" onClick={handleComputeViz} disabled={loading}>
            {loading ? "Computing..." : "Compute"}
          </button>
          {vizType === "shap" && vizData?.viz_type === "shap" && (
            <button className="op-btn primary" onClick={handleRecomputeViz} disabled={loading}>
              Recompute
            </button>
          )}
          {vizType === "edge_influence" && onEdgeInfluence && vizData && (
            <button
              className={`op-btn ${showOnGraph ? "primary" : ""}`}
              onClick={() => {
                const next = !showOnGraph;
                setShowOnGraph(next);
                if (next && vizData?.data) {
                  // Transform edge data into the Record<string, ...> format
                  const edges = (vizData.data as any).edges as Array<{
                    from: string; to: string; influence: number;
                    mean_contribution: number; normalized_influence: number;
                  }>;
                  if (edges) {
                    const record: Record<string, any> = {};
                    for (const e of edges) {
                      record[`${e.from}->${e.to}`] = {
                        influence: e.influence,
                        mean_contribution: e.mean_contribution,
                        normalized_influence: e.normalized_influence,
                      };
                    }
                    onEdgeInfluence(record);
                  }
                } else {
                  onEdgeInfluence(null);
                }
              }}
              style={{ fontSize: "11px" }}
            >
              {showOnGraph ? "Hide on graph" : "Show on graph"}
            </button>
          )}
          {VIZ_TYPE_EXPLANATIONS[vizType] && (
            <button
              className="op-btn"
              onClick={() => setShowExplain(true)}
              style={{ fontSize: "11px" }}
              title="Explain this visualization"
            >
              ? Explain
            </button>
          )}
        </div>
      </div>

      {showExplain && VIZ_TYPE_EXPLANATIONS[vizType] && (
        <div
          style={{
            position: "fixed", top: 0, left: 0, right: 0, bottom: 0,
            background: "rgba(0,0,0,0.4)", zIndex: 1000,
            display: "flex", alignItems: "center", justifyContent: "center",
          }}
          onClick={() => setShowExplain(false)}
        >
          <div
            style={{
              background: "white", borderRadius: "0.5rem", padding: "1.5rem",
              maxWidth: "550px", width: "90%", maxHeight: "80vh", overflowY: "auto",
              boxShadow: "0 8px 32px rgba(0,0,0,0.2)",
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1rem" }}>
              <h3 style={{ margin: 0, fontSize: "1.1rem" }}>{VIZ_TYPE_EXPLANATIONS[vizType].title}</h3>
              <button
                onClick={() => setShowExplain(false)}
                style={{ background: "none", border: "none", cursor: "pointer", fontSize: "1.2rem", color: "#6b7280" }}
              >
                x
              </button>
            </div>
            <div style={{ fontSize: "0.9rem", lineHeight: 1.6, color: "#374151" }}>
              <p style={{ marginTop: 0 }}><strong>What it shows:</strong> {VIZ_TYPE_EXPLANATIONS[vizType].what}</p>
              <p><strong>How to read it:</strong> {VIZ_TYPE_EXPLANATIONS[vizType].how}</p>
              <p style={{ marginBottom: 0 }}><strong>What to look for:</strong> {VIZ_TYPE_EXPLANATIONS[vizType].look_for}</p>
            </div>
          </div>
        </div>
      )}

      {error && <div className="error-message">{error}</div>}

      {vizData && (
        <div className="evidence-section">
          <VizCanvas
            vizType={vizData.viz_type}
            data={vizData.data}
            onSvgRef={handleSvgRef}
            correctness={vizData.correctness ?? undefined}
            classNames={vizData.class_names ?? undefined}
          />

          {!isWholeModel && !isNodeLevel && (
            <div className="snapshot-controls">
              <input
                type="text"
                className="text-input"
                placeholder="Add narrative for snapshot..."
                value={snapshotNarrative}
                onChange={(e) => setSnapshotNarrative(e.target.value)}
              />
              <button className="op-btn" onClick={handleSnapshot}>
                Save Snapshot
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/* ── EvidencePanel: shared header/dataset/perf + two VizSlots ── */

export function EvidencePanel({ genomeId, experimentId, annotation, isWholeModel = false, isNodeLevel = false, nodeId, width, onResize, onEdgeInfluence }: EvidencePanelProps) {
  const [splitId, setSplitId] = useState<string | null>(null);
  const [splitChoice, setSplitChoice] = useState<"train" | "test" | "val" | "both">("both");
  const [sampleFraction, setSampleFraction] = useState(0.1);
  const [galleryKey, setGalleryKey] = useState(0);
  const [viewMode, setViewMode] = useState<"network" | "source">("network");

  // Whole-model performance
  const [perfData, setPerfData] = useState<PerformanceResponse | null>(null);
  const [perfLoading, setPerfLoading] = useState(false);

  const nIn = annotation.entry_nodes.length;
  const nOut = annotation.exit_nodes.length;

  const handleSplitSelected = useCallback(
    (newSplitId: string, choice: "train" | "test" | "val" | "both") => {
      setSplitId(newSplitId);
      setSplitChoice(choice);
    },
    [],
  );

  const handleGalleryRefresh = useCallback(() => setGalleryKey((k) => k + 1), []);

  // Resize drag handling
  const handleResizeMouseDown = useCallback(
    (e: React.MouseEvent) => {
      if (!onResize) return;
      e.preventDefault();
      const startX = e.clientX;
      const startWidth = width ?? 400;

      document.body.style.userSelect = "none";
      document.body.style.cursor = "col-resize";

      function handleMouseMove(moveEvent: MouseEvent) {
        const delta = startX - moveEvent.clientX;
        const newWidth = Math.min(900, Math.max(300, startWidth + delta));
        onResize!(newWidth);
      }

      function handleMouseUp() {
        document.body.style.userSelect = "";
        document.body.style.cursor = "";
        document.removeEventListener("mousemove", handleMouseMove);
        document.removeEventListener("mouseup", handleMouseUp);
      }

      document.addEventListener("mousemove", handleMouseMove);
      document.addEventListener("mouseup", handleMouseUp);
    },
    [onResize, width],
  );

  // Fetch performance data for whole-model mode
  useEffect(() => {
    if (!isWholeModel || !splitId) return;
    let cancelled = false;
    setPerfLoading(true);
    computePerformance(genomeId, {
      dataset_split_id: splitId,
      split: splitChoice,
    })
      .then((result) => { if (!cancelled) setPerfData(result); })
      .catch(() => { if (!cancelled) setPerfData(null); })
      .finally(() => { if (!cancelled) setPerfLoading(false); });
    return () => { cancelled = true; };
  }, [isWholeModel, genomeId, splitId, splitChoice]);

  return (
    <div className="evidence-panel">
      {onResize && (
        <div
          className="evidence-panel-resize-handle"
          onMouseDown={handleResizeMouseDown}
        />
      )}
      <div className="evidence-panel-header">
        <h3>Evidence: {annotation.name || annotation.id.slice(0, 8)}</h3>
        <span className="evidence-dim">
          {nIn} &rarr; {nOut}
        </span>
      </div>

      {!isWholeModel && (
        <FormulaDisplay
          genomeId={genomeId}
          annotationId={isNodeLevel ? undefined : annotation.id}
          nodeId={isNodeLevel ? nodeId : undefined}
        />
      )}

      <div className="evidence-section">
        <DatasetSelector
          experimentId={experimentId}
          onSplitSelected={handleSplitSelected}
          onSampleFractionChange={setSampleFraction}
          sampleFraction={sampleFraction}
        />
        <div className="radio-group" style={{ marginTop: "8px" }}>
          <label className="radio-label">
            <input type="radio" checked={viewMode === "network"} onChange={() => setViewMode("network")} />
            Network
          </label>
          <label className="radio-label">
            <input type="radio" checked={viewMode === "source"} onChange={() => setViewMode("source")} />
            Source
          </label>
        </div>
      </div>

      {isWholeModel && splitId && (
        <div className="evidence-section">
          <div style={{ fontSize: "13px" }}>
            <div style={{ fontWeight: 600, marginBottom: "6px" }}>Performance</div>
            {perfLoading ? (
              <div style={{ color: "#64748b", fontStyle: "italic" }}>Computing metrics...</div>
            ) : perfData ? (
              <table style={{ width: "100%", fontSize: "12px", borderCollapse: "collapse" }}>
                <tbody>
                  <tr>
                    <td style={{ padding: "2px 6px", color: "#64748b" }}>MSE</td>
                    <td style={{ padding: "2px 6px" }}>{perfData.mse.toFixed(5)}</td>
                    <td style={{ padding: "2px 6px", color: "#64748b" }}>RMSE</td>
                    <td style={{ padding: "2px 6px" }}>{perfData.rmse.toFixed(5)}</td>
                  </tr>
                  <tr>
                    <td style={{ padding: "2px 6px", color: "#64748b" }}>MAE</td>
                    <td style={{ padding: "2px 6px" }}>{perfData.mae.toFixed(5)}</td>
                    <td style={{ padding: "2px 6px", color: "#64748b" }}>Samples</td>
                    <td style={{ padding: "2px 6px" }}>{perfData.n_samples}</td>
                  </tr>
                  {perfData.accuracy !== null && (
                    <>
                      <tr><td colSpan={4} style={{ padding: "6px 6px 2px", fontWeight: 500, borderTop: "1px solid #e2e8f0" }}>Classification</td></tr>
                      <tr>
                        <td style={{ padding: "2px 6px", color: "#64748b" }}>Accuracy</td>
                        <td style={{ padding: "2px 6px" }}>{(perfData.accuracy * 100).toFixed(1)}%</td>
                        {perfData.balanced_accuracy !== null && (
                          <>
                            <td style={{ padding: "2px 6px", color: "#64748b" }}>Bal. Acc</td>
                            <td style={{ padding: "2px 6px" }}>{(perfData.balanced_accuracy! * 100).toFixed(1)}%</td>
                          </>
                        )}
                      </tr>
                      <tr>
                        {perfData.auc_roc !== null && (
                          <>
                            <td style={{ padding: "2px 6px", color: "#64748b" }}>AUC-ROC</td>
                            <td style={{ padding: "2px 6px" }}>{(perfData.auc_roc! * 100).toFixed(1)}%</td>
                          </>
                        )}
                        {perfData.f1 !== null && (
                          <>
                            <td style={{ padding: "2px 6px", color: "#64748b" }}>F1</td>
                            <td style={{ padding: "2px 6px" }}>{(perfData.f1! * 100).toFixed(1)}%</td>
                          </>
                        )}
                      </tr>
                      <tr>
                        {perfData.precision !== null && (
                          <>
                            <td style={{ padding: "2px 6px", color: "#64748b" }}>Precision</td>
                            <td style={{ padding: "2px 6px" }}>{(perfData.precision! * 100).toFixed(1)}%</td>
                          </>
                        )}
                        {perfData.recall !== null && (
                          <>
                            <td style={{ padding: "2px 6px", color: "#64748b" }}>Recall</td>
                            <td style={{ padding: "2px 6px" }}>{(perfData.recall! * 100).toFixed(1)}%</td>
                          </>
                        )}
                      </tr>
                      <tr>
                        {perfData.log_loss !== null && (
                          <>
                            <td style={{ padding: "2px 6px", color: "#64748b" }}>Log Loss</td>
                            <td style={{ padding: "2px 6px" }}>{perfData.log_loss!.toFixed(5)}</td>
                          </>
                        )}
                        {perfData.brier_score !== null && (
                          <>
                            <td style={{ padding: "2px 6px", color: "#64748b" }}>Brier</td>
                            <td style={{ padding: "2px 6px" }}>{perfData.brier_score!.toFixed(5)}</td>
                          </>
                        )}
                      </tr>
                    </>
                  )}
                </tbody>
              </table>
            ) : null}
          </div>
        </div>
      )}

      {splitId && (
        <>
          <VizSlot
            genomeId={genomeId}
            annotation={annotation}
            splitId={splitId}
            splitChoice={splitChoice}
            sampleFraction={sampleFraction}
            isWholeModel={isWholeModel}
            isNodeLevel={isNodeLevel}
            nodeId={nodeId}
            nIn={nIn}
            nOut={nOut}
            defaultVizType="line"
            onGalleryRefresh={handleGalleryRefresh}
            viewMode={viewMode}
            onEdgeInfluence={onEdgeInfluence}
          />
          <div className="viz-slot-divider" />
          <VizSlot
            genomeId={genomeId}
            annotation={annotation}
            splitId={splitId}
            splitChoice={splitChoice}
            sampleFraction={sampleFraction}
            isWholeModel={isWholeModel}
            isNodeLevel={isNodeLevel}
            nodeId={nodeId}
            nIn={nIn}
            nOut={nOut}
            defaultVizType="shap"
            onGalleryRefresh={handleGalleryRefresh}
            viewMode={viewMode}
            onEdgeInfluence={onEdgeInfluence}
          />
        </>
      )}

      {!isWholeModel && !isNodeLevel && (
        <EvidenceGallery
          key={galleryKey}
          genomeId={genomeId}
          annotationId={annotation.id}
        />
      )}
    </div>
  );
}
