
type Node = {}

type Genome = {
  nodes: Array<Node>,
  edges: Array<Edge>
}
// Create a sum function
export function sum(a: number, b: number): number {
    return a + b
  }
  // Parse Node
export function parseNode(nodeString: string): Node {
      const key = parseInt(nodeString.match(/DefaultNodeGene\(key=(\d+),/)[1]);
      const bias = parseFloat(nodeString.match(/bias=(-?\d+\.\d+),/)[1]);
      const response = parseFloat(nodeString.match(/response=(\d+\.\d+),/)[1]);
      const activation = nodeString.match(/activation=(\w+),/)[1];
      const aggregation = nodeString.match(/aggregation=(\w+)\)/)[1];

      return { key, data: { bias, response, activation, aggregation } };
  }



  // Parse Edge
export  function parseEdge(edgeString: string): Edge {
      const keyMatch = edgeString.match(/\(key=\((-?\d+), (-?\d+)\),/);
      const key = [keyMatch[1], keyMatch[2]];
      const source = key[0];
      const target = key[1];
      const weight = parseFloat(edgeString.match(/weight=(-?\d+\.\d+),/)[1]);
      const enabled = edgeString.match(/enabled=(\w+)\)/)[1] === 'True';

      return { id: `e${source}-${target}`, source: source, target: target, weight: weight, enabled: enabled };
};
  

export function parseGenome(genome: string): Genome {
    const lines = genome.split('\n');
    const nodes = [];
    const edges = [];

    let isNodeSection = false;
    let isEdgeSection = false;

    for (const line of lines) {
        if (line.startsWith('Nodes:')) {
            isNodeSection = true;
            continue;
        }
        if (line.startsWith('Connections:')) {
            isNodeSection = false;
            isEdgeSection = true;
            continue;
        }
        if (isNodeSection) {
            const node = parseNode(line);
            nodes.push(node);
        }
        if (isEdgeSection) {
            const edge = parseEdge(line);
            edges.push(edge);
        }
    }

    return { nodes, edges };
}