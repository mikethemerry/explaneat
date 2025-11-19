import { useEffect, useMemo, useRef, useState } from "react";
import { DataSet, Network, type NodeOptions, type EdgeOptions } from "vis-network/standalone";
import type { ExplorerData } from "../types";

type GraphCanvasProps = {
  data: ExplorerData;
  filters: {
    showDirectConnections: boolean;
    activeAnnotations: Set<string>;
  };
};

export function GraphCanvas({ data, filters }: GraphCanvasProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const networkRef = useRef<Network | null>(null);
  const nodesRef = useRef(new DataSet<NodeOptions>([]));
  const edgesRef = useRef(new DataSet<EdgeOptions>([]));
  const [selectedInfo, setSelectedInfo] = useState({ nodes: [] as number[], edges: [] as string[] });

  const annotationColorMap = useMemo(() => {
    const palette = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A", "#98D8C8", "#F7DC6F", "#BB8FCE"];
    return Object.fromEntries(
      data.annotations.map((ann, index) => [ann.id, palette[index % palette.length]]),
    );
  }, [data.annotations]);

  // Initialize network
  useEffect(() => {
    if (!containerRef.current) return;

    const nodes = data.nodes.map((node) => ({
      id: node.id,
      label: node.label,
      color: node.color,
      title: buildNodeTitle(node),
      physics: false,
      x: node.position?.x,
      y: node.position?.y,
      fixed: node.position ? { x: false, y: false } : undefined,
    }));

    const edges = data.edges.map((edge) => ({
      id: edge.id,
      from: edge.from,
      to: edge.to,
      arrows: "to",
      color: edge.color,
      title: buildEdgeTitle(edge),
      width: 1 + Math.abs(edge.weight) * 2,
      smooth: { enabled: false },
    }));

    nodesRef.current = new DataSet(nodes);
    edgesRef.current = new DataSet(edges);

    networkRef.current = new Network(
      containerRef.current,
      {
        nodes: nodesRef.current,
        edges: edgesRef.current,
      },
      {
        physics: {
          enabled: false,
        },
        interaction: {
          hover: true,
          multiselect: true,
        },
      },
    );

    networkRef.current.on("select", (params) => {
      setSelectedInfo({
        nodes: (params.nodes as number[]) ?? [],
        edges: (params.edges as string[]) ?? [],
      });
    });

    return () => {
      networkRef.current?.destroy();
      networkRef.current = null;
    };
  }, [data]);

  // Apply filters
  useEffect(() => {
    const nodes = nodesRef.current;
    const edges = edgesRef.current;
    if (!nodes || !edges) return;

    const activeAnnotations = filters.activeAnnotations;

    data.nodes.forEach((node) => {
      const annotatedVisible =
        node.annotationIds.length === 0 ||
        node.annotationIds.some((annId) => activeAnnotations.has(annId));
      const showNode =
        annotatedVisible && (filters.showDirectConnections || !node.isDirectConnection);

      nodes.update({
        id: node.id,
        hidden: !showNode,
      });
    });

    data.edges.forEach((edge) => {
      const annotatedVisible =
        edge.annotationIds.length === 0 ||
        edge.annotationIds.some((annId) => activeAnnotations.has(annId));
      const showEdge =
        annotatedVisible && (filters.showDirectConnections || !edge.isDirectConnection);

      edges.update({
        id: edge.id,
        hidden: !showEdge,
      });
    });
  }, [data.nodes, data.edges, filters]);

  return (
    <section className="graph-panel">
      <div className="graph-toolbar">
        <div>
          <strong>Genome:</strong> {data.metadata.genomeId}
        </div>
        <div>
          <strong>Selected nodes:</strong> {selectedInfo.nodes.join(", ") || "None"}
        </div>
        <div>
          <strong>Selected edges:</strong> {selectedInfo.edges.length}
        </div>
      </div>
      <div className="graph-container" ref={containerRef} />
    </section>
  );
}

function buildNodeTitle(node: ExplorerData["nodes"][number]) {
  const annotationInfo =
    node.annotationIds.length > 0 ? `Annotations: ${node.annotationIds.join(", ")}` : "";
  return [`Node ${node.id}`, `Type: ${node.type}`, annotationInfo].filter(Boolean).join("<br/>");
}

function buildEdgeTitle(edge: ExplorerData["edges"][number]) {
  const annotationInfo =
    edge.annotationIds.length > 0 ? `Annotations: ${edge.annotationIds.join(", ")}` : "";
  return [`${edge.from} → ${edge.to}`, `Weight: ${edge.weight.toFixed(3)}`, annotationInfo]
    .filter(Boolean)
    .join("<br/>");
}

