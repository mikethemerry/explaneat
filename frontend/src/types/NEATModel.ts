export enum NodeType {
  INPUT = "input",
  OUTPUT = "output",
  HIDDEN = "default",
}
export interface Node {
  id: string;
  type: NodeType;
  position: { x: number; y: number };
  bias: number;
  activation: string;
}

export interface Edge {
  id: string;
  source: string;
  target: string;
  weight: number;
  type: string;
}

export interface NEATModel {
  id: number;
  model_name: string;
  dataset: string;
  version: string;
  created_at: string;
  updated_at: string;
  parsed_model: {
    nodes: Array<Node>;
    edges: Array<Edge>;
  };
  raw_data: string;
}
