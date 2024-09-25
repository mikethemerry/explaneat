import React, { useCallback } from "react";
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
} from "reactflow";
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
}

const NEATVisualizer: React.FC<NEATVisualizerProps> = ({ model }) => {
  console.log(model);
  const initialNodes: Node[] = React.useMemo(() => {
    if (!model.parsed_model?.nodes) return [];
    return model.parsed_model.nodes.map((node) => ({
      ...node,
      position: node.position || {
        x: Math.random() * 500,
        y: Math.random() * 500,
      },
    }));
  }, [model.parsed_model?.nodes]);

  const initialEdges: Edge[] = React.useMemo(() => {
    if (!model.parsed_model?.edges) return [];
    return model.parsed_model.edges.map((edge) => ({
      ...edge,
      animated: true,
    }));
  }, [model.parsed_model?.edges]);

  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);

  const onConnect = useCallback(
    (params: Connection) => setEdges((eds) => addEdge(params, eds)),
    [setEdges]
  );

  if (!model.parsed_model) {
    return <div>No parsed model data available.</div>;
  }

  return (
    <div style={{ width: "100%", height: "500px" }}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
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
