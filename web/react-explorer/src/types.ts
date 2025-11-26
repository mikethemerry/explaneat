export type Annotation = {
  id: string;
  name?: string | null;
  hypothesis?: string | null;
  nodes: number[];
  edges: [number, number][];
};

export type NodePayload = {
  id: number;
  label: string;
  type: "input" | "hidden" | "output";
  depth: number;
  color: string;
  annotationIds: string[];
  isDirectConnection: boolean;
  position?: { x: number; y: number };
};

export type EdgePayload = {
  id: string;
  from: number;
  to: number;
  weight: number;
  color: string;
  annotationIds: string[];
  isDirectConnection: boolean;
  isSkip: boolean;
};

export type ExplorerData = {
  metadata: {
    genomeId: string;
    generatedAt: string;
    schemaVersion: number;
    layout?: {
      type: string;
      dimensions?: { width: number; height: number };
    };
  };
  nodes: NodePayload[];
  edges: EdgePayload[];
  annotations: Annotation[];
};




