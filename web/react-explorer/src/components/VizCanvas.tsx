import { useEffect, useRef, useCallback } from "react";
import * as Plot from "@observablehq/plot";
import * as d3 from "d3";

type VizCanvasProps = {
  vizType: string;
  data: Record<string, unknown>;
  onSvgRef?: (svg: SVGSVGElement | null) => void;
};

export function VizCanvas({ vizType, data, onSvgRef }: VizCanvasProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement | null>(null);

  useEffect(() => {
    if (!containerRef.current || !data) return;

    // Clear previous
    containerRef.current.innerHTML = "";

    let plot: SVGSVGElement | HTMLElement | null = null;

    try {
      if (vizType === "line") {
        plot = renderLinePlot(data);
      } else if (vizType === "heatmap") {
        plot = renderHeatmap(data);
      } else if (vizType === "partial_dependence") {
        const pdType = (data as Record<string, unknown>).type;
        if (pdType === "1d") {
          plot = renderLinePlot(data);
        } else {
          plot = renderHeatmap(data);
        }
      } else if (vizType === "pca_scatter") {
        plot = renderPCAScatter(data);
      } else if (vizType === "sensitivity") {
        plot = renderSensitivity(data);
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
  }, [vizType, data, onSvgRef]);

  return <div className="viz-canvas" ref={containerRef} />;
}

function renderLinePlot(data: Record<string, unknown>): SVGSVGElement | HTMLElement {
  const gridX = data.grid_x as number[];
  const gridY = data.grid_y as number[];
  const scatterX = data.scatter_x as number[] | undefined;
  const scatterY = data.scatter_y as number[] | undefined;
  const xLabel = (data.x_label as string) || "x";
  const yLabel = (data.y_label as string) || "y";

  const lineData = gridX.map((x, i) => ({ x, y: gridY[i] }));
  const scatterData = scatterX
    ? scatterX.map((x, i) => ({ x, y: scatterY![i] }))
    : [];

  return Plot.plot({
    width: 500,
    height: 350,
    x: { label: xLabel },
    y: { label: yLabel },
    marks: [
      Plot.line(lineData, { x: "x", y: "y", stroke: "#2563eb", strokeWidth: 2 }),
      ...(scatterData.length > 0
        ? [
            Plot.dot(scatterData, {
              x: "x",
              y: "y",
              fill: "#f97316",
              fillOpacity: 0.5,
              r: 3,
            }),
          ]
        : []),
    ],
  });
}

function renderHeatmap(data: Record<string, unknown>): SVGSVGElement | HTMLElement {
  const xRange = data.x_range as number[];
  const yRange = data.y_range as number[];
  const zGrid = data.z_grid as number[][];
  const xLabel = (data.x_label as string) || "x";
  const yLabel = (data.y_label as string) || "y";

  // Flatten grid to data points
  const heatData: { x: number; y: number; z: number }[] = [];
  for (let j = 0; j < yRange.length; j++) {
    for (let i = 0; i < xRange.length; i++) {
      heatData.push({ x: xRange[i], y: yRange[j], z: zGrid[j][i] });
    }
  }

  // Scatter overlay
  const scatterX = data.scatter_x as number[] | undefined;
  const scatterY = data.scatter_y as number[] | undefined;
  const scatterData = scatterX
    ? scatterX.map((x, i) => ({ x, y: scatterY![i] }))
    : [];

  const dx = xRange.length > 1 ? xRange[1] - xRange[0] : 1;
  const dy = yRange.length > 1 ? yRange[1] - yRange[0] : 1;

  return Plot.plot({
    width: 500,
    height: 400,
    color: { scheme: "YlOrRd", legend: true },
    x: { label: xLabel },
    y: { label: yLabel },
    marks: [
      Plot.cell(heatData, {
        x: "x",
        y: "y",
        fill: "z",
        inset: 0,
      }),
      ...(scatterData.length > 0
        ? [
            Plot.dot(scatterData, {
              x: "x",
              y: "y",
              stroke: "white",
              strokeWidth: 1,
              fill: "black",
              r: 2,
            }),
          ]
        : []),
    ],
  });
}

function renderPCAScatter(data: Record<string, unknown>): SVGSVGElement | HTMLElement {
  const pcaX = data.pca_x as number[];
  const pcaY = data.pca_y as number[];
  const colorValues = data.color_values as number[];
  const colorLabel = (data.color_label as string) || "output";
  const explained = data.explained_variance as number[];

  const points = pcaX.map((x, i) => ({
    x,
    y: pcaY[i],
    color: colorValues[i],
  }));

  return Plot.plot({
    width: 500,
    height: 400,
    color: { scheme: "Viridis", legend: true, label: colorLabel },
    x: { label: `PC1 (${(explained[0] * 100).toFixed(1)}%)` },
    y: { label: `PC2 (${((explained[1] || 0) * 100).toFixed(1)}%)` },
    marks: [
      Plot.dot(points, {
        x: "x",
        y: "y",
        fill: "color",
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
    x: { label: "Input" },
    y: { label: "Sensitivity" },
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
