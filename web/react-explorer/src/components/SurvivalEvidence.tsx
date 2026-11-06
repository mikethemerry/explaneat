/**
 * Whole-model evidence for a Cox survival model: Harrell's and Uno's C, td-AUC,
 * IBS, Cox loss, calibration at a horizon, and hazard-ratio readings of the
 * terms at the output (direct edges, hinges, annotation exits).
 */
import { useEffect, useRef, useState } from "react";
import * as Plot from "@observablehq/plot";
import { getHazardRatios, type HazardRatios, type SurvivalPerf } from "../api/portal";

const f = (v: unknown, d = 3) => (typeof v === "number" && Number.isFinite(v) ? v.toFixed(d) : "—");

function CalibrationPlot({ cal }: { cal: NonNullable<SurvivalPerf["calibration"]> }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!ref.current) return;
    const pts = (cal.groups || []).map((g) => ({ p: g.mean_predicted, o: g.observed_km }));
    const max = Math.max(0.05, ...pts.map((d) => Math.max(d.p, d.o)));
    const plot = Plot.plot({
      width: 300,
      height: 220,
      x: { label: "Predicted risk", domain: [0, max] },
      y: { label: "Observed (KM)", domain: [0, max] },
      marks: [
        Plot.line([[0, 0], [max, max]], { stroke: "#9ca3af", strokeDasharray: "4 3" }),
        Plot.dot(pts, { x: "p", y: "o", fill: "#2563eb", r: 4, tip: true }),
      ],
    });
    ref.current.replaceChildren(plot);
  }, [cal]);
  return <div ref={ref} />;
}

export function SurvivalEvidence({ genomeId, perf }: { genomeId: string; perf: SurvivalPerf }) {
  const [hr, setHr] = useState<HazardRatios | null>(null);
  const [hrErr, setHrErr] = useState<string | null>(null);
  useEffect(() => {
    getHazardRatios(genomeId).then(setHr).catch((e) => setHrErr(e instanceof Error ? e.message : String(e)));
  }, [genomeId]);
  const unit = perf.time_unit ? ` ${perf.time_unit}` : "";
  const td = perf.td_auc;
  return (
    <div style={{ fontSize: "12px" }}>
      <div style={{ fontWeight: 600, marginBottom: 6 }}>Survival performance</div>
      <table className="tools-table">
        <tbody>
          <tr><td>Harrell's C</td><td><strong>{f(perf.harrell_c)}</strong></td></tr>
          <tr><td>Uno's C</td><td>{f(perf.uno_c)}</td></tr>
          {td && (
            <tr>
              <td>td-AUC (mean {f(td.mean_auc)})</td>
              <td>{(td.times || []).map((t, i) => `${f(t, 2)}${unit}: ${f(td.auc?.[i])}`).join(", ")}</td>
            </tr>
          )}
          <tr><td>Integrated Brier score</td><td>{f(perf.ibs)}</td></tr>
          <tr><td>Cox loss (null)</td><td>{f(perf.cox_loss)} ({f(perf.null_cox_loss)})</td></tr>
          <tr><td>Rows / events / censored</td><td>{perf.n_test} / {perf.events_test} / {f(perf.censored_fraction, 2)}</td></tr>
          {perf.log_risk && <tr><td>HR p95 vs p5 of g(x)</td><td>{f(perf.log_risk.hr_p95_vs_p5, 2)}</td></tr>}
        </tbody>
      </table>
      {perf.calibration && (
        <>
          <div style={{ fontWeight: 600, margin: "8px 0 4px" }}>
            Calibration at {f(perf.calibration.horizon, 2)}{unit}: E/O {f(perf.calibration.expected_over_observed, 2)},
            max group |Δ| {f(perf.calibration.max_abs_group_diff, 3)}
          </div>
          <CalibrationPlot cal={perf.calibration} />
        </>
      )}
      <p className="hint">{perf.ph_caveat}</p>

      <div style={{ fontWeight: 600, margin: "10px 0 4px" }}>Hazard-ratio readings</div>
      {hrErr && <p className="error-text">{hrErr}</p>}
      {hr && (
        <>
          <table className="tools-table">
            <thead><tr><th>Term at the output</th><th>Reading</th><th title="HR p95 vs p5 of the term on train">HR p95/p5</th></tr></thead>
            <tbody>
              {hr.terms.map((t, i) => (
                <tr key={i}>
                  <td>{t.feature || t.node}</td>
                  <td>
                    {t.hr_per_sd != null
                      ? `HR ${f(t.hr_per_sd, 3)} per SD${t.hr_per_unit != null ? `, ${f(t.hr_per_unit, 4)} per unit` : ""}`
                      : t.reading}
                  </td>
                  <td>{f(t.spread?.hr_p95_vs_p5, 2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {hr.annotation_terms.length > 0 && (
            <table className="tools-table" style={{ marginTop: 6 }}>
              <thead><tr><th>Annotation exit</th><th>w→out</th><th>log-HR SD</th><th>HR p75/median</th><th>HR p95/p5</th></tr></thead>
              <tbody>
                {hr.annotation_terms.map((t, i) => (
                  <tr key={i}>
                    <td>{t.annotation} ({t.exit})</td>
                    <td>{f(t.output_weight, 3)}</td>
                    <td>{f(t.spread.log_hr_sd, 3)}</td>
                    <td>{f(t.spread.hr_p75_vs_median, 2)}</td>
                    <td>{f(t.spread.hr_p95_vs_p5, 2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          <p className="hint">{hr.note}</p>
        </>
      )}
    </div>
  );
}
