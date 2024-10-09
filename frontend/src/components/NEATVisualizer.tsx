import React, { useCallback, useEffect, useState } from "react";
import ReactFlow, {
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  addEdge,
  Connection,
  Edge,
  Node,
  Position,
  NodeDragHandler,
} from "reactflow";
import dagre from "@dagrejs/dagre";
import "reactflow/dist/style.css";

export interface NEATModel {
  id: number;
  model_name: string;
  dataset: string;
  version: string;
  created_at: string;
  updated_at: string;
  parsed_model?: {
    nodes: Array<{
      id: string;
      type: string;
      position: { x: number; y: number };
      data: { label: string };
    }>;
    edges: Array<{
      id: string;
      source: string;
      target: string;
      label: string;
      type: string;
    }>;
  };
  raw_data: string;
}

interface NEATVisualizerProps {
  model: NEATModel;
  height?: string;
}

const nodeWidth = 172;
const nodeHeight = 36;

const getLayoutedElements = (
  nodes: Node[],
  edges: Edge[],
  direction = "TB"
) => {
  console.log("Laying out elements with direction:", direction);
  const dagreGraph = new dagre.graphlib.Graph();
  dagreGraph.setDefaultEdgeLabel(() => ({}));

  const isHorizontal = direction === "LR";
  dagreGraph.setGraph({ rankdir: direction });

  nodes.forEach((node) => {
    dagreGraph.setNode(node.id, { width: nodeWidth, height: nodeHeight });
  });

  edges.forEach((edge) => {
    dagreGraph.setEdge(edge.source, edge.target);
  });

  dagre.layout(dagreGraph);

  return nodes.map((node) => {
    const nodeWithPosition = dagreGraph.node(node.id);
    node.targetPosition = isHorizontal ? Position.Left : Position.Top;
    node.sourcePosition = isHorizontal ? Position.Right : Position.Bottom;

    // We are shifting the dagre node position (anchor=center center) to the top left
    // so it matches the React Flow node anchor point (top left).
    node.position = {
      x: nodeWithPosition.x - nodeWidth / 2,
      y: nodeWithPosition.y - nodeHeight / 2,
    };

    console.log(`Node ${node.id} positioned at:`, node.position);
    return node;
  });
};

const NEATVisualizer: React.FC<NEATVisualizerProps> = ({
  model,
  height = "500px",
}) => {
  console.log("Rendering NEATVisualizer with model:", model);
  const [isInitialLayout, setIsInitialLayout] = useState(true);

  const initialNodes: Node[] = React.useMemo(() => {
    if (!model.parsed_model?.nodes) {
      console.log("No nodes found in parsed_model");
      return [];
    }
    console.log("Initializing nodes:", model.parsed_model.nodes);
    return model.parsed_model.nodes.map((node) => ({
      ...node,
      position: node.position || {
        x: Math.random() * 500,
        y: Math.random() * 500,
      },
      draggable: true,
    }));
  }, [model.parsed_model?.nodes]);

  const initialEdges: Edge[] = React.useMemo(() => {
    if (!model.parsed_model?.edges) {
      console.log("No edges found in parsed_model");
      return [];
    }
    console.log("Initializing edges:", model.parsed_model.edges);
    return model.parsed_model.edges.map((edge) => ({
      ...edge,
      animated: true,
    }));
  }, [model.parsed_model?.edges]);

  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);

  useEffect(() => {
    if (isInitialLayout) {
      console.log("Applying initial layout to nodes and edges");
      const layoutedNodes = getLayoutedElements(nodes, edges);
      setNodes([...layoutedNodes]);
      setIsInitialLayout(false);
    }
  }, [isInitialLayout, nodes, edges, setNodes]);

  const onConnect = useCallback(
    (params: Connection) => {
      console.log("New connection:", params);
      setEdges((eds) => addEdge(params, eds));
    },
    [setEdges]
  );

  const onNodeDragStop: NodeDragHandler = useCallback((event, node) => {
    console.log("Node dragged:", node);
    // You can add additional logic here if needed
  }, []);

  if (!model.parsed_model) {
    console.log("No parsed model data available");
    return <div>No parsed model data available.</div>;
  }

  return (
    <div style={{ width: "100%", height }}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onNodeDragStop={onNodeDragStop}
        fitView
      >
        <Background />
        <MiniMap />
        <Controls />
      </ReactFlow>
    </div>
  );
};

export default NEATVisualizer;
