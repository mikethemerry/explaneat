import { useState, useCallback, useEffect, useRef } from "react";
import {
  startRetrain,
  getRetrainStatus,
  applyRetrain,
  cancelRetrain,
  computePerformance,
  type RetrainStatusResponse,
  type PerformanceResponse,
} from "../api/client";
import { DatasetSelector } from "./DatasetSelector";

type RetrainPanelProps = {
  genomeId: string;
  experimentId: string;
  onOperationChange: () => void;
};

function MetricRow({ label, before, after, format, higherIsBetter }: {
  label: string;
  before: number | null | undefined;
  after: number | null | undefined;
  format: (v: number) => string;
  higherIsBetter: boolean;
}) {
  if (before === null || before === undefined) return null;
  return (
    <tr>
      <td style={{ padding: "2px 6px", color: "#64748b" }}>{label}</td>
      <td style={{ padding: "2px 6px" }}>{format(before)}</td>
      {after !== null && after !== undefined ? (
        <td style={{ padding: "2px 6px" }}>
          {format(after)}
          <span style={{
            color: (higherIsBetter ? after > before : after < before) ? "#16a34a" : after === before ? "#64748b" : "#dc2626",
            marginLeft: "4px",
            fontSize: "11px",
          }}>
            ({(higherIsBetter ? after > before : after < before) ? "" : after === before ? "" : "+"}{format(after - before)})
          </span>
        </td>
      ) : (
        <td style={{ padding: "2px 6px", color: "#cbd5e1" }}>--</td>
      )}
    </tr>
  );
}

function PerfComparisonTable({ before, after }: {
  before: PerformanceResponse;
  after: PerformanceResponse | null;
}) {
  const fmtDec = (v: number) => v.toFixed(5);
  const fmtPct = (v: number) => `${(v * 100).toFixed(1)}%`;
  const isClassification = before.accuracy !== null;

  return (
    <table style={{ width: "100%", fontSize: "12px", borderCollapse: "collapse" }}>
      <thead>
        <tr style={{ borderBottom: "1px solid #e2e8f0" }}>
          <th style={{ padding: "2px 6px", textAlign: "left", fontWeight: 500 }}>Metric</th>
          <th style={{ padding: "2px 6px", textAlign: "left", fontWeight: 500 }}>Before</th>
          <th style={{ padding: "2px 6px", textAlign: "left", fontWeight: 500 }}>After</th>
        </tr>
      </thead>
      <tbody>
        <MetricRow label="MSE" before={before.mse} after={after?.mse} format={fmtDec} higherIsBetter={false} />
        <MetricRow label="RMSE" before={before.rmse} after={after?.rmse} format={fmtDec} higherIsBetter={false} />
        <MetricRow label="MAE" before={before.mae} after={after?.mae} format={fmtDec} higherIsBetter={false} />
        {isClassification && (
          <>
            <tr><td colSpan={3} style={{ padding: "4px 6px 2px", fontWeight: 500, borderTop: "1px solid #e2e8f0" }}>Classification</td></tr>
            <MetricRow label="Accuracy" before={before.accuracy} after={after?.accuracy} format={fmtPct} higherIsBetter={true} />
            <MetricRow label="Bal. Acc" before={before.balanced_accuracy} after={after?.balanced_accuracy} format={fmtPct} higherIsBetter={true} />
            <MetricRow label="AUC-ROC" before={before.auc_roc} after={after?.auc_roc} format={fmtPct} higherIsBetter={true} />
            <MetricRow label="F1" before={before.f1} after={after?.f1} format={fmtPct} higherIsBetter={true} />
            <MetricRow label="Precision" before={before.precision} after={after?.precision} format={fmtPct} higherIsBetter={true} />
            <MetricRow label="Recall" before={before.recall} after={after?.recall} format={fmtPct} higherIsBetter={true} />
            <MetricRow label="Log Loss" before={before.log_loss} after={after?.log_loss} format={fmtDec} higherIsBetter={false} />
            <MetricRow label="Brier" before={before.brier_score} after={after?.brier_score} format={fmtDec} higherIsBetter={false} />
          </>
        )}
      </tbody>
    </table>
  );
}

export function RetrainPanel({ genomeId, experimentId, onOperationChange }: RetrainPanelProps) {
  // Config
  const [splitId, setSplitId] = useState<string | null>(null);
  const [splitChoice, setSplitChoice] = useState<"train" | "test" | "val" | "both">("train");
  const [sampleFraction, setSampleFraction] = useState(1.0);
  const [nEpochs, setNEpochs] = useState(50);
  const [learningRate, setLearningRate] = useState(0.01);
  const [freezeAnnotations, setFreezeAnnotations] = useState(false);

  // Job state
  const [jobId, setJobId] = useState<string | null>(null);
  const [jobStatus, setJobStatus] = useState<RetrainStatusResponse | null>(null);
  const [polling, setPolling] = useState(false);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Result state
  const [applying, setApplying] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Before/after performance
  const [beforePerf, setBeforePerf] = useState<PerformanceResponse | null>(null);
  const [afterPerf, setAfterPerf] = useState<PerformanceResponse | null>(null);

  // Cleanup polling on unmount
  useEffect(() => {
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  const startPolling = useCallback((jid: string) => {
    setPolling(true);
    const interval = setInterval(async () => {
      try {
        const status = await getRetrainStatus(genomeId, jid);
        setJobStatus(status);
        if (status.status === "completed" || status.status === "failed" || status.status === "cancelled") {
          clearInterval(interval);
          pollRef.current = null;
          setPolling(false);
        }
      } catch {
        clearInterval(interval);
        pollRef.current = null;
        setPolling(false);
      }
    }, 500);
    pollRef.current = interval;
  }, [genomeId]);

  const handleStart = useCallback(async () => {
    if (!splitId) return;
    setError(null);
    setJobStatus(null);
    setBeforePerf(null);
    setAfterPerf(null);

    try {
      // Compute before-performance
      const perf = await computePerformance(genomeId, {
        dataset_split_id: splitId,
        split: splitChoice,
      });
      setBeforePerf(perf);

      const resp = await startRetrain(genomeId, {
        dataset_split_id: splitId,
        split: splitChoice,
        n_epochs: nEpochs,
        learning_rate: learningRate,
        freeze_annotations: freezeAnnotations,
      });
      setJobId(resp.job_id);
      startPolling(resp.job_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to start training");
    }
  }, [genomeId, splitId, splitChoice, nEpochs, learningRate, freezeAnnotations, startPolling]);

  const handleCancel = useCallback(async () => {
    if (!jobId) return;
    try {
      await cancelRetrain(genomeId, jobId);
    } catch {
      // Ignore cancel errors
    }
  }, [genomeId, jobId]);

  const handleApply = useCallback(async () => {
    if (!jobId) return;
    setApplying(true);
    setError(null);
    try {
      await applyRetrain(genomeId, jobId);

      // Compute after-performance
      if (splitId) {
        try {
          const perf = await computePerformance(genomeId, {
            dataset_split_id: splitId,
            split: splitChoice,
          });
          setAfterPerf(perf);
        } catch {
          // Non-critical
        }
      }

      onOperationChange();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to apply weights");
    } finally {
      setApplying(false);
    }
  }, [genomeId, jobId, splitId, splitChoice, onOperationChange]);

  const handleDiscard = useCallback(() => {
    setJobId(null);
    setJobStatus(null);
    setBeforePerf(null);
    setAfterPerf(null);
    setError(null);
  }, []);

  const isRunning = jobStatus?.status === "running" || jobStatus?.status === "pending";
  const isCompleted = jobStatus?.status === "completed";
  const isFailed = jobStatus?.status === "failed";
  const progress = jobStatus ? (jobStatus.current_epoch / Math.max(jobStatus.total_epochs, 1)) * 100 : 0;

  return (
    <div style={{ padding: "12px", fontSize: "13px" }}>
      <h3 style={{ margin: "0 0 12px", fontSize: "14px" }}>Retrain Model</h3>

      {/* Dataset selection */}
      {!jobId && (
        <>
          <DatasetSelector
            experimentId={experimentId}
            onSplitSelected={(id: string, choice: "train" | "test" | "val" | "both") => {
              setSplitId(id);
              setSplitChoice(choice);
            }}
            onSampleFractionChange={setSampleFraction}
            sampleFraction={sampleFraction}
          />

          <div style={{ display: "flex", gap: "8px", marginTop: "8px" }}>
            <div style={{ flex: 1 }}>
              <label style={{ display: "block", marginBottom: "4px", fontWeight: 500 }}>
                Epochs
              </label>
              <input
                type="number"
                value={nEpochs}
                onChange={(e) => setNEpochs(Math.max(1, parseInt(e.target.value) || 1))}
                min={1}
                max={1000}
                style={{ width: "100%", padding: "4px", fontSize: "12px" }}
              />
            </div>
            <div style={{ flex: 1 }}>
              <label style={{ display: "block", marginBottom: "4px", fontWeight: 500 }}>
                Learning Rate
              </label>
              <input
                type="number"
                value={learningRate}
                onChange={(e) => setLearningRate(parseFloat(e.target.value) || 0.01)}
                step={0.001}
                min={0.0001}
                max={1}
                style={{ width: "100%", padding: "4px", fontSize: "12px" }}
              />
            </div>
          </div>

          <label style={{ display: "flex", alignItems: "center", gap: "6px", marginTop: "8px" }}>
            <input
              type="checkbox"
              checked={freezeAnnotations}
              onChange={(e) => setFreezeAnnotations(e.target.checked)}
            />
            <span>Freeze annotated subgraphs</span>
          </label>

          <button
            onClick={handleStart}
            disabled={!splitId}
            style={{
              marginTop: "12px",
              width: "100%",
              padding: "8px",
              background: splitId ? "#2563eb" : "#94a3b8",
              color: "white",
              border: "none",
              borderRadius: "4px",
              cursor: splitId ? "pointer" : "not-allowed",
              fontWeight: 600,
            }}
          >
            Start Training
          </button>
        </>
      )}

      {/* Progress */}
      {jobId && jobStatus && (
        <div>
          <div style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            marginBottom: "8px",
          }}>
            <span style={{ fontWeight: 500 }}>
              {isRunning ? "Training..." : isCompleted ? "Training Complete" : isFailed ? "Training Failed" : jobStatus.status}
            </span>
            <span style={{ fontSize: "11px", color: "#64748b" }}>
              Epoch {jobStatus.current_epoch}/{jobStatus.total_epochs}
            </span>
          </div>

          {/* Progress bar */}
          <div style={{
            width: "100%",
            height: "6px",
            background: "#e2e8f0",
            borderRadius: "3px",
            overflow: "hidden",
            marginBottom: "8px",
          }}>
            <div style={{
              width: `${progress}%`,
              height: "100%",
              background: isFailed ? "#ef4444" : isCompleted ? "#22c55e" : "#2563eb",
              transition: "width 0.3s",
            }} />
          </div>

          {/* Loss display */}
          {jobStatus.metrics.loss.length > 0 && (
            <div style={{
              display: "flex",
              gap: "16px",
              marginBottom: "8px",
              fontSize: "12px",
            }}>
              <div>
                <span style={{ color: "#64748b" }}>Loss: </span>
                <span style={{ fontWeight: 500 }}>
                  {jobStatus.metrics.loss[jobStatus.metrics.loss.length - 1].toFixed(6)}
                </span>
              </div>
              {jobStatus.metrics.val_loss.length > 0 && (
                <div>
                  <span style={{ color: "#64748b" }}>Val Loss: </span>
                  <span style={{ fontWeight: 500 }}>
                    {jobStatus.metrics.val_loss[jobStatus.metrics.val_loss.length - 1].toFixed(6)}
                  </span>
                </div>
              )}
            </div>
          )}

          {/* Mini loss curve (text-based) */}
          {jobStatus.metrics.loss.length > 1 && (
            <div style={{
              background: "#f8fafc",
              padding: "6px 8px",
              borderRadius: "4px",
              marginBottom: "8px",
              fontSize: "11px",
              fontFamily: "monospace",
              maxHeight: "80px",
              overflowY: "auto",
            }}>
              {jobStatus.metrics.loss
                .filter((_, i) => i % Math.max(1, Math.floor(jobStatus.metrics.loss.length / 10)) === 0 || i === jobStatus.metrics.loss.length - 1)
                .map((loss, i, arr) => {
                  const epochIdx = jobStatus.metrics.loss.indexOf(loss);
                  const valLoss = jobStatus.metrics.val_loss[epochIdx];
                  return (
                    <div key={i}>
                      E{epochIdx + 1}: loss={loss.toFixed(5)}
                      {valLoss !== undefined ? ` val={valLoss.toFixed(5)}` : ""}
                    </div>
                  );
                })}
            </div>
          )}

          {/* Error message */}
          {isFailed && jobStatus.error && (
            <div style={{
              background: "#fef2f2",
              color: "#dc2626",
              padding: "8px",
              borderRadius: "4px",
              marginBottom: "8px",
              fontSize: "12px",
            }}>
              {jobStatus.error}
            </div>
          )}

          {/* Before/After comparison */}
          {beforePerf && (
            <div style={{
              background: "#f0f9ff",
              padding: "8px",
              borderRadius: "4px",
              marginBottom: "8px",
              fontSize: "12px",
            }}>
              <div style={{ fontWeight: 500, marginBottom: "4px" }}>Performance</div>
              <PerfComparisonTable before={beforePerf} after={afterPerf} />
            </div>
          )}

          {/* Action buttons */}
          <div style={{ display: "flex", gap: "8px", marginTop: "8px" }}>
            {isRunning && (
              <button
                onClick={handleCancel}
                style={{
                  flex: 1,
                  padding: "8px",
                  background: "#ef4444",
                  color: "white",
                  border: "none",
                  borderRadius: "4px",
                  cursor: "pointer",
                }}
              >
                Cancel
              </button>
            )}
            {isCompleted && (
              <>
                <button
                  onClick={handleApply}
                  disabled={applying}
                  style={{
                    flex: 1,
                    padding: "8px",
                    background: "#22c55e",
                    color: "white",
                    border: "none",
                    borderRadius: "4px",
                    cursor: "pointer",
                    fontWeight: 600,
                  }}
                >
                  {applying ? "Applying..." : "Apply Weights"}
                </button>
                <button
                  onClick={handleDiscard}
                  style={{
                    flex: 1,
                    padding: "8px",
                    background: "#94a3b8",
                    color: "white",
                    border: "none",
                    borderRadius: "4px",
                    cursor: "pointer",
                  }}
                >
                  Discard
                </button>
              </>
            )}
            {(isFailed || jobStatus.status === "cancelled") && (
              <button
                onClick={handleDiscard}
                style={{
                  flex: 1,
                  padding: "8px",
                  background: "#94a3b8",
                  color: "white",
                  border: "none",
                  borderRadius: "4px",
                  cursor: "pointer",
                }}
              >
                Dismiss
              </button>
            )}
          </div>
        </div>
      )}

      {/* Error display */}
      {error && (
        <div style={{
          marginTop: "8px",
          background: "#fef2f2",
          color: "#dc2626",
          padding: "8px",
          borderRadius: "4px",
          fontSize: "12px",
        }}>
          {error}
        </div>
      )}
    </div>
  );
}
