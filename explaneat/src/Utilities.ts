
import { Node, Edge } from 'reactflow';
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
    const node: Node = { id: `node${key}`, type: 'default', data: { label: `Node ${key}` }, position: { x: Math.random() * 500, y: Math.random() * 500 } };
    return node;
  }



  // Parse Edge
export  function parseEdge(edgeString: string): Edge {
      const keyMatch = edgeString.match(/\(key=\((-?\d+), (-?\d+)\),/);
      const key = [keyMatch[1], keyMatch[2]];
      const source = key[0];
      const target = key[1];
      const weight = parseFloat(edgeString.match(/weight=(-?\d+\.\d+),/)[1]);
      const enabled = edgeString.match(/enabled=(\w+)\)/)[1] === 'True';

      const edge: Edge = { id: `edge${source}-${target}`, source: `node${source}`, target: `node${target}`, animated: enabled, label: `weight: ${weight}` };
      return edge;
};
  

export function parseGenome(genome: string): Genome {
    const lines = genome.split('\n');
    const nodes: Node[] = [];
    const edges: Edge[] = [];

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

    edges.forEach(edge => {
        if (parseInt(edge.source.replace('node', '')) < 0) {
            // Check if node with the same id already exists
            if (!nodes.some(node => node.id === edge.source)) {
                const node: Node = { id: edge.source, type: 'input', data: { label: `Node ${edge.source.replace('node', '')}` }, position: { x: Math.random() * 500, y: Math.random() * 500 }, bias: 0, response: 0 };
                nodes.push(node);
            }
        }
    });

    return { nodes, edges };
}