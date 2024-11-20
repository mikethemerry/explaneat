import { NEATModel, Node, Edge, NodeType } from "../types/NEATModel";

export class NEAT {
  private nodes: Map<string, Node>;
  private edges: Map<string, Edge>;
  private nodeValues: Map<string, number>;
  private sorted: string[];

  constructor(model: NEATModel) {
    this.nodes = new Map(
      model.parsed_model.nodes.map((node) => [node.id, node])
    );
    this.edges = new Map(
      model.parsed_model.edges.map((edge) => [edge.id, edge])
    );
    this.nodeValues = new Map();
    this.sorted = this.topologicalSort();
  }

  private getActivationFunction(node: Node): (x: number) => number {
    switch (node.type) {
      case NodeType.HIDDEN: // Relu
        return (x: number) => Math.max(0, x);
      case NodeType.OUTPUT: // Sigmoid
        return (x: number) => 1 / (1 + Math.exp(-x));
      case NodeType.INPUT: // Linear
        return (x: number) => x;
      default:
        return (x: number) => x;
    }
  }

  private topologicalSort(): string[] {
    const visited = new Set<string>();
    const temp = new Set<string>();
    const order: string[] = [];

    // Get input nodes first
    const inputNodes = Array.from(this.nodes.values())
      .filter((node) => node.type === NodeType.INPUT)
      .map((node) => node.id);

    // Helper function for depth-first search
    const visit = (nodeId: string) => {
      if (temp.has(nodeId)) {
        throw new Error("Neural network has cycles");
      }
      if (visited.has(nodeId)) {
        return;
      }

      temp.add(nodeId);

      // Find all edges where this node is the source
      Array.from(this.edges.values())
        .filter((edge) => edge.source === nodeId)
        .forEach((edge) => visit(edge.target));

      temp.delete(nodeId);
      visited.add(nodeId);
      order.unshift(nodeId);
    };

    // Start with input nodes
    inputNodes.forEach((nodeId) => {
      if (!visited.has(nodeId)) {
        visit(nodeId);
      }
    });
    return order;
    // return order.reverse();
  }

  public forward(inputs: number[]): number[] {
    // Clear previous values
    this.nodeValues.clear();

    // Set input values
    const inputNodes = Array.from(this.nodes.values()).filter(
      (node) => node.type === NodeType.INPUT
    );

    if (inputs.length !== inputNodes.length) {
      throw new Error(
        `Expected ${inputNodes.length} inputs, but got ${inputs.length}`
      );
    }

    // Initialize input nodes
    inputNodes.forEach((node, index) => {
      this.nodeValues.set(node.id, inputs[index]);
    });

    // Process nodes in topological order
    this.sorted.forEach((nodeId) => {
      const node = this.nodes.get(nodeId)!;

      // Skip input nodes as they're already processed
      if (node.type === NodeType.INPUT) {
        return;
      }

      // Calculate incoming connections
      let sum = node.bias;

      // Find all incoming edges
      Array.from(this.edges.values())
        .filter((edge) => edge.target === nodeId)
        .forEach((edge) => {
          const sourceValue = this.nodeValues.get(edge.source);
          if (sourceValue === undefined) {
            throw new Error(
              `No value calculated for source node ${edge.source}`
            );
          }
          sum += sourceValue * edge.weight;
        });

      // Apply activation function
      const activationFn = this.getActivationFunction(node);
      this.nodeValues.set(nodeId, activationFn(sum));
    });

    // Return output values
    const outputNodes = Array.from(this.nodes.values())
      .filter((node) => node.type === NodeType.OUTPUT)
      .sort((a, b) => a.id.localeCompare(b.id));

    return outputNodes.map((node) => this.nodeValues.get(node.id) ?? 0);
  }

  // Helper method to get intermediate values for debugging
  public getNodeValues(): Map<string, number> {
    return new Map(this.nodeValues);
  }
}
