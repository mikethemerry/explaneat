export interface NEATModel {
  id: number;
  model_name: string;
  dataset: string;
  version: string;
  created_at: string;
  updated_at: string;
  parsed_model: {
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
