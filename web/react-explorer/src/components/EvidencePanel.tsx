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
};

const ALL_VIZ_TYPES = ["line", "heatmap", "partial_dependence", "pca_scatter", "sensitivity", "ice", "feature_output_scatter", "output_distribution", "shap"];

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
};

function VizSlot({
  genomeId, annotation, splitId, splitChoice, sampleFraction,
  isWholeModel, isNodeLevel, nodeId,
  nIn, nOut, defaultVizType, onGalleryRefresh,
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
        });
        setVizData(result);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to compute visualization");
    } finally {
      setLoading(false);
    }
  }, [genomeId, annotation.id, splitId, vizType, splitChoice, sampleFraction, buildVizParams, isWholeModel, isNodeLevel, nodeId]);

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
  }, [splitId, vizType, inputDim, inputDims[0], inputDims[1], outputDim, splitChoice]);

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
        </div>
      </div>

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

export function EvidencePanel({ genomeId, experimentId, annotation, isWholeModel = false, isNodeLevel = false, nodeId }: EvidencePanelProps) {
  const [splitId, setSplitId] = useState<string | null>(null);
  const [splitChoice, setSplitChoice] = useState<"train" | "test" | "val" | "both">("both");
  const [sampleFraction, setSampleFraction] = useState(0.1);
  const [galleryKey, setGalleryKey] = useState(0);

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
