import { useState, useCallback, useRef, useMemo } from "react";
import {
  computeVizData,
  computeShap,
  saveSnapshot,
  type VizDataRequest,
  type VizDataResponse,
  type AnnotationSummary,
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

export function EvidencePanel({ genomeId, experimentId, annotation, isWholeModel = false, isNodeLevel = false, nodeId }: EvidencePanelProps) {
  const [vizData, setVizData] = useState<VizDataResponse | null>(null);
  const [vizType, setVizType] = useState<string>("line");
  const [splitId, setSplitId] = useState<string | null>(null);
  const [splitChoice, setSplitChoice] = useState<"train" | "test" | "both">("both");
  const [sampleFraction, setSampleFraction] = useState(0.1);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [snapshotNarrative, setSnapshotNarrative] = useState("");
  const [galleryKey, setGalleryKey] = useState(0);
  const svgRef = useRef<SVGSVGElement | null>(null);

  // Dimension selection state
  const [inputDim, setInputDim] = useState(0);
  const [inputDims, setInputDims] = useState<[number, number]>([0, 1]);
  const [outputDim, setOutputDim] = useState(0);

  const nIn = annotation.entry_nodes.length;
  const nOut = annotation.exit_nodes.length;

  // Input dimension options
  const inputOptions = useMemo(
    () => Array.from({ length: nIn }, (_, i) => i),
    [nIn],
  );
  const outputOptions = useMemo(
    () => Array.from({ length: nOut }, (_, i) => i),
    [nOut],
  );

  const handleSplitSelected = useCallback(
    (newSplitId: string, choice: "train" | "test" | "both") => {
      setSplitId(newSplitId);
      setSplitChoice(choice);
    },
    [],
  );

  // Build viz params based on current viz type and dimension selections
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

  const handleSnapshot = useCallback(async () => {
    if (!svgRef.current) return;

    const svgElement = svgRef.current;
    const serializer = new XMLSerializer();
    const svgString = serializer.serializeToString(svgElement);
    const svgBase64 = btoa(svgString);

    try {
      await saveSnapshot(
        genomeId,
        annotation.id,
        {
          viz_type: vizType,
          split_id: splitId,
          split_choice: splitChoice,
          sample_fraction: sampleFraction,
          ...buildVizParams(),
        },
        svgBase64,
        snapshotNarrative,
      );
      setSnapshotNarrative("");
      setGalleryKey((k) => k + 1);
    } catch (err) {
      console.error("Failed to save snapshot:", err);
    }
  }, [genomeId, annotation.id, vizType, splitId, splitChoice, sampleFraction, snapshotNarrative, buildVizParams]);

  const handleSvgRef = useCallback((svg: SVGSVGElement | null) => {
    svgRef.current = svg;
  }, []);

  const suggestedTypes = vizData?.suggested_viz_types || [];
  const inputMode = NEEDS_INPUT_DIM[vizType];
  const showOutputDim = NEEDS_OUTPUT_DIM[vizType] && nOut > 1;

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

      {splitId && (
        <div className="evidence-section">
          <div className="viz-type-selector">
            <label className="selector-label">Visualization</label>
            <div className="viz-type-buttons">
              {(suggestedTypes.length > 0
                ? [...suggestedTypes, "shap"]
                : ["line", "heatmap", "partial_dependence", "pca_scatter", "sensitivity", "ice", "feature_output_scatter", "output_distribution", "shap"]
              ).map((vt) => (
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

          {/* Dimension selectors */}
          <div className="dim-selectors">
            {inputMode === "single" && nIn > 1 && (
              <div className="dim-selector">
                <label className="dim-label">Input dim</label>
                <select
                  className="dim-select"
                  value={inputDim}
                  onChange={(e) => setInputDim(Number(e.target.value))}
                >
                  {inputOptions.map((i) => (
                    <option key={i} value={i}>
                      x_{i} ({annotation.entry_nodes[i]})
                    </option>
                  ))}
                </select>
                {nIn > 1 && (
                  <span className="dim-hint">others fixed at median</span>
                )}
              </div>
            )}
            {inputMode === "pair" && nIn >= 2 && (
              <div className="dim-selector">
                <label className="dim-label">Input dims</label>
                <div className="dim-pair">
                  <select
                    className="dim-select"
                    value={inputDims[0]}
                    onChange={(e) =>
                      setInputDims([Number(e.target.value), inputDims[1]])
                    }
                  >
                    {inputOptions.map((i) => (
                      <option key={i} value={i}>
                        x_{i}
                      </option>
                    ))}
                  </select>
                  <span className="dim-sep">vs</span>
                  <select
                    className="dim-select"
                    value={inputDims[1]}
                    onChange={(e) =>
                      setInputDims([inputDims[0], Number(e.target.value)])
                    }
                  >
                    {inputOptions.filter((i) => i !== inputDims[0]).map((i) => (
                      <option key={i} value={i}>
                        x_{i}
                      </option>
                    ))}
                  </select>
                </div>
                {nIn > 2 && (
                  <span className="dim-hint">others fixed at median</span>
                )}
              </div>
            )}
            {showOutputDim && (
              <div className="dim-selector">
                <label className="dim-label">Output dim</label>
                <select
                  className="dim-select"
                  value={outputDim}
                  onChange={(e) => setOutputDim(Number(e.target.value))}
                >
                  {outputOptions.map((i) => (
                    <option key={i} value={i}>
                      y_{i} ({annotation.exit_nodes[i]})
                    </option>
                  ))}
                </select>
              </div>
            )}
          </div>

          <div className="compute-actions">
            <button
              className="op-btn primary"
              onClick={handleComputeViz}
              disabled={loading}
            >
              {loading ? "Computing..." : "Compute"}
            </button>
            {vizType === "shap" && vizData?.viz_type === "shap" && (
              <button
                className="op-btn primary"
                onClick={handleRecomputeViz}
                disabled={loading}
              >
                Recompute
              </button>
            )}
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
