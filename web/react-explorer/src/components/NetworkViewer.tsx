import { useEffect, useRef, useCallback } from "react";
import { DataSet, Network, type Options } from "vis-network/standalone";
import type { ModelState } from "../api/client";

type NetworkViewerProps = {
  model: ModelState;
  selectedNodes: Set<string>;
  onNodeSelect: (nodeIds: string[]) => void;
};

const NODE_COLORS: Record<string, string> = {
  input: "#4CAF50",
  output: "#2196F3",
  hidden: "#9E9E9E",
  identity: "#FF9800",
};

export function NetworkViewer({
  model,
  selectedNodes,
  onNodeSelect,
}: NetworkViewerProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const networkRef = useRef<Network | null>(null);
  const nodesDataRef = useRef(new DataSet<{ id: string; label: string; color: string; title: string }>());
  const edgesDataRef = useRef(new DataSet<{ id: string; from: string; to: string; color: string; width: number; title: string }>());

  // Build network data from model
  const buildNetworkData = useCallback(() => {
    const nodes = model.nodes.map((node) => ({
      id: node.id,
      label: node.id,
      color: NODE_COLORS[node.type] || NODE_COLORS.hidden,
      title: buildNodeTooltip(node),
    }));

    const edges = model.connections
      .filter((conn) => conn.enabled)
      .map((conn) => ({
        id: `${conn.from}->${conn.to}`,
        from: conn.from,
        to: conn.to,
        color: conn.weight >= 0 ? "#4CAF50" : "#F44336",
        width: 1 + Math.abs(conn.weight) * 2,
        title: `Weight: ${conn.weight.toFixed(3)}`,
      }));

    return { nodes, edges };
  }, [model]);

  // Initialize network
  useEffect(() => {
    if (!containerRef.current) return;

    const { nodes, edges } = buildNetworkData();

    nodesDataRef.current = new DataSet(nodes);
    edgesDataRef.current = new DataSet(edges);

    const options: Options = {
      physics: {
        enabled: true,
        hierarchicalRepulsion: {
          centralGravity: 0.0,
          springLength: 100,
          springConstant: 0.01,
          nodeDistance: 120,
          damping: 0.09,
        },
        solver: "hierarchicalRepulsion",
      },
      layout: {
        hierarchical: {
          enabled: true,
          direction: "LR",
          sortMethod: "directed",
          levelSeparation: 150,
          nodeSpacing: 80,
        },
      },
      interaction: {
        hover: true,
        multiselect: true,
        selectConnectedEdges: false,
      },
      edges: {
        arrows: { to: { enabled: true, scaleFactor: 0.5 } },
        smooth: { enabled: true, type: "cubicBezier" },
      },
      nodes: {
        shape: "dot",
        size: 16,
        font: { size: 12, color: "#333" },
      },
    };

    networkRef.current = new Network(
      containerRef.current,
      { nodes: nodesDataRef.current, edges: edgesDataRef.current },
      options,
    );

    networkRef.current.on("selectNode", (params) => {
      onNodeSelect(params.nodes as string[]);
    });

    networkRef.current.on("deselectNode", () => {
      onNodeSelect([]);
    });

    // Fit to screen after stabilization
    networkRef.current.once("stabilizationIterationsDone", () => {
      networkRef.current?.fit({ animation: { duration: 500 } });
    });

    return () => {
      networkRef.current?.destroy();
      networkRef.current = null;
    };
  }, [model, buildNetworkData, onNodeSelect]);

  // Update selection highlight
  useEffect(() => {
    if (!networkRef.current) return;

    // Update node colors based on selection
    const updates = model.nodes.map((node) => ({
      id: node.id,
      borderWidth: selectedNodes.has(node.id) ? 3 : 1,
      borderWidthSelected: 3,
    }));

    nodesDataRef.current.update(updates);
  }, [selectedNodes, model.nodes]);

  return (
    <div className="network-viewer">
      <div className="network-toolbar">
        <span>
          <strong>Nodes:</strong> {model.nodes.length}
        </span>
        <span>
          <strong>Connections:</strong>{" "}
          {model.connections.filter((c) => c.enabled).length}
        </span>
        <span>
          <strong>Selected:</strong> {selectedNodes.size}
        </span>
        {model.metadata.is_original ? (
          <span className="badge original">Original</span>
        ) : (
          <span className="badge modified">Modified</span>
        )}
      </div>
      <div className="network-container" ref={containerRef} />
      <div className="network-legend">
        <span className="legend-item">
          <span className="legend-color" style={{ backgroundColor: NODE_COLORS.input }} />
          Input
        </span>
        <span className="legend-item">
          <span className="legend-color" style={{ backgroundColor: NODE_COLORS.hidden }} />
          Hidden
        </span>
        <span className="legend-item">
          <span className="legend-color" style={{ backgroundColor: NODE_COLORS.output }} />
          Output
        </span>
        <span className="legend-item">
          <span className="legend-color" style={{ backgroundColor: NODE_COLORS.identity }} />
          Identity
        </span>
      </div>
    </div>
  );
}

function buildNodeTooltip(node: ModelState["nodes"][number]): string {
  const lines = [
    `Node ${node.id}`,
    `Type: ${node.type}`,
  ];
  if (node.bias !== null) lines.push(`Bias: ${node.bias.toFixed(3)}`);
  if (node.activation) lines.push(`Activation: ${node.activation}`);
  return lines.join("\n");
}
