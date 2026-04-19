import { useEffect, useRef, useCallback } from "react";
import * as Plot from "@observablehq/plot";
import * as d3 from "d3";

/** Compact tick format: avoids long decimals, uses SI suffixes for large values. */
const compactTick = (v: number) => {
  if (v === 0) return "0";
  const abs = Math.abs(v);
  if (abs >= 1e6) return d3.format(".2s")(v);
  if (abs >= 1000) return d3.format(".3s")(v);
  if (abs >= 1) return d3.format(".3g")(v);
  if (abs >= 0.01) return d3.format(".2g")(v);
  return d3.format(".2e")(v);
};

/** Standard axis config: limited ticks with compact formatting on a linear scale. */
const niceAxis = (label: string, ticks = 6) => ({
  label,
  type: "linear" as const,
  ticks,
  tickFormat: compactTick,
});

/** Axis config for categorical (band/point) scales — no tick limiting needed. */
const catAxis = (label: string) => ({
  label,
});

type VizCanvasProps = {
  vizType: string;
  data: Record<string, unknown>;
  onSvgRef?: (svg: SVGSVGElement | null) => void;
  correctness?: boolean[];
  classNames?: string[];
};

export function VizCanvas({ vizType, data, onSvgRef, correctness, classNames }: VizCanvasProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement | null>(null);

  useEffect(() => {
    if (!containerRef.current || !data) return;

    // Clear previous
    containerRef.current.innerHTML = "";

    let plot: SVGSVGElement | HTMLElement | null = null;

    try {
      if (vizType === "line") {
        plot = renderLinePlot(data, correctness);
      } else if (vizType === "heatmap") {
        plot = renderHeatmap(data, correctness);
      } else if (vizType === "partial_dependence") {
        const pdType = (data as Record<string, unknown>).type;
        if (pdType === "1d") {
          plot = renderLinePlot(data, correctness);
        } else {
          plot = renderHeatmap(data, correctness);
        }
      } else if (vizType === "pca_scatter") {
        plot = renderPCAScatter(data, correctness);
      } else if (vizType === "sensitivity") {
        plot = renderSensitivity(data);
      } else if (vizType === "histogram") {
        plot = renderHistogram(data);
      } else if (vizType === "scatter2d") {
        plot = renderScatter2D(data);
      } else if (vizType === "shap") {
        plot = renderShapBar(data);
      } else if (vizType === "ice") {
        plot = renderICEPlot(data);
      } else if (vizType === "feature_output_scatter") {
        plot = renderFeatureOutputScatter(data, correctness);
      } else if (vizType === "output_distribution") {
        plot = renderHistogram(data);
      } else if (vizType === "activation_profile") {
        plot = renderActivationProfile(data);
      } else if (vizType === "regime_map") {
        plot = renderRegimeMap(data);
      } else if (vizType === "edge_influence") {
        plot = renderEdgeInfluence(data);
      }
    } catch (err) {
      console.error("Viz rendering error:", err);
      containerRef.current.textContent = `Rendering error: ${err}`;
      return;
    }

    if (plot) {
      containerRef.current.appendChild(plot);
      // Find SVG element
      const svg =
        plot instanceof SVGSVGElement
          ? plot
          : plot.querySelector("svg");
      svgRef.current = svg as SVGSVGElement | null;
      if (onSvgRef) onSvgRef(svgRef.current);
    }
  }, [vizType, data, onSvgRef, correctness, classNames]);

  return <div className="viz-canvas" ref={containerRef} />;
}

function renderLinePlot(data: Record<string, unknown>, correctness?: boolean[]): SVGSVGElement | HTMLElement {
  const gridX = data.grid_x as number[];
  const gridY = data.grid_y as number[];
  const scatterX = data.scatter_x as number[] | undefined;
  const scatterY = data.scatter_y as number[] | undefined;
  const xLabel = (data.x_label as string) || "x";
  const yLabel = (data.y_label as string) || "y";

  const lineData = gridX.map((x, i) => ({ x, y: gridY[i] }));
  const hasCorrectness = correctness && scatterX && correctness.length === scatterX.length;
  const scatterData = scatterX
    ? scatterX.map((x, i) => ({
        x,
        y: scatterY![i],
        ...(hasCorrectness ? { result: correctness![i] ? "Correct" : "Incorrect" } : {}),
      }))
    : [];

  return Plot.plot({
    width: 500,
    height: 350,
    x: niceAxis(xLabel),
    y: niceAxis(yLabel),
    ...(hasCorrectness
      ? { color: { domain: ["Correct", "Incorrect"], range: ["#22c55e", "#ef4444"], legend: true } }
      : {}),
    marks: [
      Plot.line(lineData, { x: "x", y: "y", stroke: "#2563eb", strokeWidth: 2 }),
      ...(scatterData.length > 0
        ? [
            Plot.dot(scatterData, {
              x: "x",
              y: "y",
              fill: hasCorrectness ? "result" : "#f97316",
              fillOpacity: 0.5,
              r: 3,
            }),
          ]
        : []),
    ],
  });
}

function renderHeatmap(data: Record<string, unknown>, correctness?: boolean[]): SVGSVGElement | HTMLElement {
  const xRange = data.x_range as number[];
  const yRange = data.y_range as number[];
  const zGrid = data.z_grid as number[][];
  const xLabel = (data.x_label as string) || "x";
  const yLabel = (data.y_label as string) || "y";

  // Compute bin half-widths for rect rendering on a linear scale
  const dx = xRange.length > 1 ? (xRange[1] - xRange[0]) / 2 : 0.5;
  const dy = yRange.length > 1 ? (yRange[1] - yRange[0]) / 2 : 0.5;

  // Flatten grid to rect data with explicit boundaries
  const heatData: { x1: number; x2: number; y1: number; y2: number; z: number }[] = [];
  for (let j = 0; j < yRange.length; j++) {
    for (let i = 0; i < xRange.length; i++) {
      heatData.push({
        x1: xRange[i] - dx, x2: xRange[i] + dx,
        y1: yRange[j] - dy, y2: yRange[j] + dy,
        z: zGrid[j][i],
      });
    }
  }

  // Scatter overlay
  const scatterX = data.scatter_x as number[] | undefined;
  const scatterY = data.scatter_y as number[] | undefined;
  const hasCorrectness = correctness && scatterX && correctness.length === scatterX.length;
  const scatterData = scatterX
    ? scatterX.map((x, i) => ({
        x,
        y: scatterY![i],
        ...(hasCorrectness ? { result: correctness![i] ? "Correct" : "Incorrect" } : {}),
      }))
    : [];

  return Plot.plot({
    width: 500,
    height: 400,
    color: { scheme: "YlOrRd", legend: true },
    x: niceAxis(xLabel, 8),
    y: niceAxis(yLabel, 8),
    marks: [
      Plot.rect(heatData, {
        x1: "x1", x2: "x2",
        y1: "y1", y2: "y2",
        fill: "z",
        inset: 0,
      }),
      ...(scatterData.length > 0
        ? [
            Plot.dot(scatterData, {
              x: "x",
              y: "y",
              stroke: hasCorrectness ? undefined : "white",
              strokeWidth: hasCorrectness ? undefined : 1,
              fill: hasCorrectness
                ? (d: { result: string }) => d.result === "Correct" ? "#22c55e" : "#ef4444"
                : "black",
              r: 2,
            }),
          ]
        : []),
    ],
  });
}

function renderPCAScatter(data: Record<string, unknown>, correctness?: boolean[]): SVGSVGElement | HTMLElement {
  const pcaX = data.pca_x as number[];
  const pcaY = data.pca_y as number[];
  const colorValues = data.color_values as number[];
  const colorLabel = (data.color_label as string) || "output";
  const explained = data.explained_variance as number[];

  const hasCorrectness = correctness && correctness.length === pcaX.length;

  const points = pcaX.map((x, i) => ({
    x,
    y: pcaY[i],
    color: colorValues[i],
    ...(hasCorrectness ? { result: correctness![i] ? "Correct" : "Incorrect" } : {}),
  }));

  return Plot.plot({
    width: 500,
    height: 400,
    color: hasCorrectness
      ? { domain: ["Correct", "Incorrect"], range: ["#22c55e", "#ef4444"], legend: true }
      : { scheme: "Viridis", legend: true, label: colorLabel },
    x: niceAxis(`PC1 (${(explained[0] * 100).toFixed(1)}%)`),
    y: niceAxis(`PC2 (${((explained[1] || 0) * 100).toFixed(1)}%)`),
    marks: [
      Plot.dot(points, {
        x: "x",
        y: "y",
        fill: hasCorrectness ? "result" : "color",
        r: 3,
        fillOpacity: 0.7,
      }),
    ],
  });
}

function renderSensitivity(data: Record<string, unknown>): SVGSVGElement | HTMLElement {
  const inputLabels = data.input_labels as string[];
  const sensitivities = data.sensitivities as Record<string, number[]>;

  // Flatten for multi-output
  const barData: { input: string; output: string; sensitivity: number }[] = [];
  for (const [outKey, scores] of Object.entries(sensitivities)) {
    for (let i = 0; i < inputLabels.length; i++) {
      barData.push({
        input: inputLabels[i],
        output: outKey,
        sensitivity: scores[i],
      });
    }
  }

  return Plot.plot({
    width: 500,
    height: 300,
    x: catAxis("Input"),
    y: niceAxis("Sensitivity"),
    color: { legend: Object.keys(sensitivities).length > 1 },
    marks: [
      Plot.barY(barData, {
        x: "input",
        y: "sensitivity",
        fill: "output",
        tip: true,
      }),
      Plot.ruleY([0]),
    ],
  });
}

function renderHistogram(data: Record<string, unknown>): SVGSVGElement | HTMLElement {
  const binEdges = data.bin_edges as number[];
  const counts = data.counts as number[];
  const xLabel = (data.x_label as string) || "value";

  const bars = counts.map((count, i) => ({
    x0: binEdges[i],
    x1: binEdges[i + 1],
    count,
  }));

  return Plot.plot({
    width: 500,
    height: 350,
    x: niceAxis(xLabel),
    y: niceAxis("Count"),
    marks: [
      Plot.rectY(bars, {
        x1: "x0",
        x2: "x1",
        y: "count",
        fill: "#2563eb",
        fillOpacity: 0.8,
        tip: true,
      }),
      Plot.ruleY([0]),
    ],
  });
}

function renderActivationProfile(data: Record<string, unknown>): SVGSVGElement | HTMLElement {
  const binEdges = data.bin_edges as number[];
  const counts = data.counts as number[];
  const stats = data.stats as Record<string, number>;

  // Color the zero/dead bin differently from active bins
  const bars = counts.map((count, i) => ({
    x0: binEdges[i],
    x1: binEdges[i + 1],
    count,
    zone: binEdges[i + 1] <= 0 ? "dead" : "active",
  }));

  const container = document.createElement("div");

  // Stats summary
  const statsDiv = document.createElement("div");
  statsDiv.style.cssText = "display:flex;gap:1rem;flex-wrap:wrap;margin-bottom:0.75rem;font-size:0.8rem;color:#374151;";
  const pct = (v: number) => `${(v * 100).toFixed(1)}%`;
  const statItems = [
    `Active: ${pct(stats.activation_rate)}`,
    `Dead: ${pct(stats.zero_fraction)}`,
    `Mean: ${stats.mean.toFixed(4)}`,
    `Std: ${stats.std.toFixed(4)}`,
    `Range: [${stats.min.toFixed(3)}, ${stats.max.toFixed(3)}]`,
    `N: ${stats.count}`,
  ];
  statsDiv.innerHTML = statItems.map(s => `<span style="background:#f3f4f6;padding:0.2rem 0.5rem;border-radius:0.25rem">${s}</span>`).join("");
  container.appendChild(statsDiv);

  // Histogram with dead zone highlighted
  const svg = Plot.plot({
    width: 500,
    height: 300,
    x: niceAxis("Activation value"),
    y: niceAxis("Count"),
    color: { domain: ["dead", "active"], range: ["#9ca3af", "#2563eb"] },
    marks: [
      Plot.rectY(bars, {
        x1: "x0",
        x2: "x1",
        y: "count",
        fill: "zone",
        fillOpacity: 0.8,
        tip: true,
      }),
      // Zero boundary line
      Plot.ruleX([0], { stroke: "#ef4444", strokeWidth: 1.5 }),
      Plot.ruleY([0]),
    ],
  });
  container.appendChild(svg);

  return container;
}

function renderScatter2D(data: Record<string, unknown>): SVGSVGElement | HTMLElement {
  const xValues = data.x_values as number[];
  const yValues = data.y_values as number[];
  const xLabel = (data.x_label as string) || "x";
  const yLabel = (data.y_label as string) || "y";

  const points = xValues.map((x, i) => ({ x, y: yValues[i] }));

  return Plot.plot({
    width: 500,
    height: 400,
    x: niceAxis(xLabel),
    y: niceAxis(yLabel),
    marks: [
      Plot.dot(points, {
        x: "x",
        y: "y",
        fill: "#2563eb",
        fillOpacity: 0.5,
        r: 3,
      }),
    ],
  });
}

function renderICEPlot(data: Record<string, unknown>): SVGSVGElement | HTMLElement {
  const gridX = data.grid_x as number[];
  const iceCurves = data.ice_curves as number[][];
  const pdCurve = data.pd_curve as number[];
  const xLabel = (data.x_label as string) || "x";
  const yLabel = (data.y_label as string) || "y";

  // Build individual ICE curve data
  const iceData: { x: number; y: number; sample: number }[] = [];
  for (let s = 0; s < iceCurves.length; s++) {
    for (let i = 0; i < gridX.length; i++) {
      iceData.push({ x: gridX[i], y: iceCurves[s][i], sample: s });
    }
  }

  // PD average overlay
  const pdData = gridX.map((x, i) => ({ x, y: pdCurve[i] }));

  return Plot.plot({
    width: 500,
    height: 350,
    x: niceAxis(xLabel),
    y: niceAxis(yLabel),
    marks: [
      Plot.line(iceData, {
        x: "x",
        y: "y",
        z: "sample",
        stroke: "#9ca3af",
        strokeOpacity: 0.3,
        strokeWidth: 1,
      }),
      Plot.line(pdData, {
        x: "x",
        y: "y",
        stroke: "#2563eb",
        strokeWidth: 2.5,
      }),
    ],
  });
}

function renderFeatureOutputScatter(data: Record<string, unknown>, correctness?: boolean[]): SVGSVGElement | HTMLElement {
  const scatterX = data.scatter_x as number[];
  const scatterY = data.scatter_y as number[];
  const pdX = data.pd_x as number[];
  const pdY = data.pd_y as number[];
  const xLabel = (data.x_label as string) || "x";
  const yLabel = (data.y_label as string) || "y";

  const hasCorrectness = correctness && correctness.length === scatterX.length;
  const scatterData = scatterX.map((x, i) => ({
    x,
    y: scatterY[i],
    ...(hasCorrectness ? { result: correctness![i] ? "Correct" : "Incorrect" } : {}),
  }));
  const pdData = pdX.map((x, i) => ({ x, y: pdY[i] }));

  return Plot.plot({
    width: 500,
    height: 350,
    x: niceAxis(xLabel),
    y: niceAxis(yLabel),
    ...(hasCorrectness
      ? { color: { domain: ["Correct", "Incorrect"], range: ["#22c55e", "#ef4444"], legend: true } }
      : {}),
    marks: [
      Plot.dot(scatterData, {
        x: "x",
        y: "y",
        fill: hasCorrectness ? "result" : "#f97316",
        fillOpacity: 0.5,
        r: 3,
      }),
      Plot.line(pdData, {
        x: "x",
        y: "y",
        stroke: "#2563eb",
        strokeWidth: 2,
      }),
    ],
  });
}

function renderShapBarSingle(
  featureNames: string[],
  meanAbsShap: number[],
  title?: string,
): SVGSVGElement | HTMLElement {
  const barData = featureNames
    .map((name, i) => ({ feature: name, importance: meanAbsShap[i] }))
    .sort((a, b) => b.importance - a.importance);

  return Plot.plot({
    width: 500,
    height: Math.max(200, barData.length * 30 + 60),
    marginLeft: 120,
    x: niceAxis(title ? `Mean |SHAP| \u2192 ${title}` : "Mean |SHAP value|"),
    y: { label: null, domain: barData.map((d) => d.feature) },
    marks: [
      Plot.barX(barData, {
        x: "importance",
        y: "feature",
        fill: "#7C3AED",
        tip: true,
      }),
      Plot.ruleX([0]),
    ],
  });
}

function renderRegimeMap(data: Record<string, unknown>): HTMLElement {
  const regimes = data.regimes as Array<{
    pattern: Record<string, boolean>;
    count: number;
    fraction: number;
    mean_prediction: number;
    accuracy: number;
    class_distribution: Record<string, number>;
  }>;
  const reluNodes = data.relu_nodes as string[];
  const reluNodeLabels = (data.relu_node_labels || {}) as Record<string, string>;

  const container = document.createElement("div");
  container.style.overflowX = "auto";

  const table = document.createElement("table");
  table.className = "experiment-table";
  table.style.fontSize = "12px";
  table.style.width = "100%";

  // Header
  const thead = document.createElement("thead");
  const headerRow = document.createElement("tr");
  for (const nodeId of reluNodes) {
    const th = document.createElement("th");
    th.textContent = reluNodeLabels[nodeId] || nodeId;
    th.style.padding = "4px 6px";
    th.style.textAlign = "center";
    headerRow.appendChild(th);
  }
  for (const label of ["Count", "Fraction", "Accuracy", "Mean Pred"]) {
    const th = document.createElement("th");
    th.textContent = label;
    th.style.padding = "4px 6px";
    th.style.textAlign = "right";
    headerRow.appendChild(th);
  }
  thead.appendChild(headerRow);
  table.appendChild(thead);

  // Body
  const tbody = document.createElement("tbody");
  for (const regime of regimes) {
    const row = document.createElement("tr");

    for (const nodeId of reluNodes) {
      const td = document.createElement("td");
      td.style.padding = "4px 6px";
      td.style.textAlign = "center";
      const on = regime.pattern[nodeId];
      const dot = document.createElement("span");
      dot.style.color = on ? "#22c55e" : "#d1d5db";
      dot.style.fontSize = "1.2rem";
      dot.textContent = "\u25CF"; // filled circle
      td.appendChild(dot);
      row.appendChild(td);
    }

    // Count
    const countTd = document.createElement("td");
    countTd.style.padding = "4px 6px";
    countTd.style.textAlign = "right";
    countTd.textContent = String(regime.count);
    row.appendChild(countTd);

    // Fraction
    const fracTd = document.createElement("td");
    fracTd.style.padding = "4px 6px";
    fracTd.style.textAlign = "right";
    fracTd.textContent = `${(regime.fraction * 100).toFixed(1)}%`;
    row.appendChild(fracTd);

    // Accuracy
    const accTd = document.createElement("td");
    accTd.style.padding = "4px 6px";
    accTd.style.textAlign = "right";
    const accPct = (regime.accuracy * 100).toFixed(1) + "%";
    accTd.textContent = accPct;
    if (regime.accuracy > 0.8) {
      accTd.style.color = "#22c55e";
    } else if (regime.accuracy > 0.6) {
      accTd.style.color = "#eab308";
    } else {
      accTd.style.color = "#ef4444";
    }
    accTd.style.fontWeight = "600";
    row.appendChild(accTd);

    // Mean Pred
    const predTd = document.createElement("td");
    predTd.style.padding = "4px 6px";
    predTd.style.textAlign = "right";
    predTd.textContent = regime.mean_prediction.toFixed(3);
    row.appendChild(predTd);

    tbody.appendChild(row);
  }
  table.appendChild(tbody);
  container.appendChild(table);
  return container;
}

function renderEdgeInfluence(data: Record<string, unknown>): SVGSVGElement | HTMLElement {
  const edges = (data.edges as Array<{
    from: string;
    to: string;
    weight: number;
    influence: number;
    mean_contribution: number;
    normalized_influence: number;
  }>).slice(0, 20);

  const barData = edges.map((e: any) => ({
    edge: `${e.from_label || e.from} \u2192 ${e.to_label || e.to}`,
    influence: e.influence,
    sign: e.mean_contribution >= 0 ? "positive" : "negative",
    weight: e.weight,
  }));

  return Plot.plot({
    width: 500,
    height: Math.max(200, barData.length * 28 + 60),
    marginLeft: 100,
    x: niceAxis("Influence"),
    y: { label: null, domain: barData.map((d) => d.edge) },
    color: { domain: ["positive", "negative"], range: ["#3b82f6", "#ef4444"] },
    marks: [
      Plot.barX(barData, {
        x: "influence",
        y: "edge",
        fill: "sign",
        tip: true,
      }),
      Plot.text(barData, {
        x: "influence",
        y: "edge",
        text: (d: { weight: number }) => `w=${d.weight.toFixed(2)}`,
        dx: 4,
        textAnchor: "start",
        fontSize: 10,
        fill: "#555",
      }),
      Plot.ruleX([0]),
    ],
  });
}

function renderShapBar(data: Record<string, unknown>): SVGSVGElement | HTMLElement {
  const featureNames = data.feature_names as string[];
  const meanAbsShap = data.mean_abs_shap as number[];
  const outputs = data.outputs as { output_name: string; mean_abs_shap: number[]; base_value: number }[] | undefined;

  if (!outputs || outputs.length <= 1) {
    return renderShapBarSingle(featureNames, meanAbsShap);
  }

  // Multi-output: render aggregate + per-output charts
  const container = document.createElement("div");
  container.style.display = "flex";
  container.style.flexDirection = "column";
  container.style.gap = "16px";

  const aggLabel = document.createElement("h4");
  aggLabel.textContent = "Aggregate (mean across outputs)";
  aggLabel.style.margin = "0";
  aggLabel.style.fontSize = "13px";
  aggLabel.style.color = "#666";
  container.appendChild(aggLabel);
  container.appendChild(renderShapBarSingle(featureNames, meanAbsShap));

  for (const out of outputs) {
    const label = document.createElement("h4");
    label.textContent = out.output_name;
    label.style.margin = "0";
    label.style.fontSize = "13px";
    label.style.color = "#666";
    container.appendChild(label);
    container.appendChild(
      renderShapBarSingle(featureNames, out.mean_abs_shap, out.output_name),
    );
  }

  return container;
}
