import React, { useEffect, useState, useMemo } from "react";
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  ZAxis,
  CartesianGrid,
  Tooltip,
} from "recharts";
import { Panel } from "reactflow";
import { NEAT } from "../propneat/NEAT";
import { NEATModel } from "../types/NEATModel";

const NeatVisualizationPanel = React.memo(
  ({ neatModel }: { neatModel: NEATModel }) => {
    // Create NEAT instance
    const neat = useMemo(() => new NEAT(neatModel), [neatModel]);

    // Get input nodes
    const inputNodes = useMemo(
      () =>
        neatModel.parsed_model.nodes
          .filter((node) => node.type === "input")
          .map((node) => ({ id: node.id, label: node.id })),
      [neatModel]
    );

    // Get all nodes for watch node selection
    const allNodes = useMemo(
      () =>
        neatModel.parsed_model.nodes.map((node) => ({
          id: node.id,
          label: node.id,
          type: node.type,
        })),
      [neatModel]
    );

    // State for selected input dimensions and watch node
    const [selectedInputs, setSelectedInputs] = useState<[string, string]>([
      inputNodes[0]?.id || "",
      inputNodes[1]?.id || "",
    ]);
    const [watchNode, setWatchNode] = useState<string>("");

    // Generate data points for visualization
    const generateDataPoints = () => {
      const points = [];
      const allInputs = new Array(inputNodes.length).fill(0);

      // Find indices of selected inputs
      const idx1 = inputNodes.findIndex(
        (node) => node.id === selectedInputs[0]
      );
      const idx2 = inputNodes.findIndex(
        (node) => node.id === selectedInputs[1]
      );

      // Increased resolution for smoother visualization
      for (let x = -1; x <= 1; x += 0.025) {
        for (let y = -1; y <= 1; y += 0.025) {
          const inputs = [...allInputs];
          inputs[idx1] = x;
          inputs[idx2] = y;
          const output = neat.forward(inputs, watchNode || undefined)[0];

          // Create a more contrasting color scheme
          const hue = output * 240; // Maps 0-1 to blue(240) to red(0)
          const saturation = 100;
          const lightness = 50;
          points.push({
            x: x.toFixed(3),
            y: y.toFixed(3),
            z: output,
            fill: `hsl(${hue}, ${saturation}%, ${lightness}%)`,
          });
        }
      }
      return points;
    };

    const [data, setData] = useState<
      { x: string; y: string; z: number; fill: string }[]
    >([]);

    useEffect(() => {
      setData(generateDataPoints());
    }, [selectedInputs, watchNode]);

    return (
      <Panel
        position="top-right"
        className="bg-white rounded-lg shadow-lg w-[450px]"
      >
        <div className="p-4">
          <h3 className="text-lg font-semibold mb-4">
            NEAT Output Visualization
          </h3>
          <div className="mb-4 flex gap-4">
            <div>
              <label className="block text-sm font-medium mb-1">
                X-Axis Input:
              </label>
              <select
                value={selectedInputs[0]}
                onChange={(e) =>
                  setSelectedInputs([e.target.value, selectedInputs[1]])
                }
                className="border rounded p-1 text-sm"
              >
                {inputNodes.map((node) => (
                  <option key={node.id} value={node.id}>
                    {node.label}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium mb-1">
                Y-Axis Input:
              </label>
              <select
                value={selectedInputs[1]}
                onChange={(e) =>
                  setSelectedInputs([selectedInputs[0], e.target.value])
                }
                className="border rounded p-1 text-sm"
              >
                {inputNodes.map((node) => (
                  <option key={node.id} value={node.id}>
                    {node.label}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium mb-1">
                Watch Node:
              </label>
              <select
                value={watchNode}
                onChange={(e) => setWatchNode(e.target.value)}
                className="border rounded p-1 text-sm"
              >
                <option value="">Output Node</option>
                {allNodes.map((node) => (
                  <option key={node.id} value={node.id}>
                    {node.label} ({node.type})
                  </option>
                ))}
              </select>
            </div>
          </div>
          <ScatterChart
            width={400}
            height={400}
            margin={{ top: 20, right: 20, bottom: 20, left: 20 }}
          >
            <CartesianGrid />
            <XAxis
              type="number"
              dataKey="x"
              name={inputNodes.find((n) => n.id === selectedInputs[0])?.label}
              domain={[-1, 1]}
              label={{
                value: inputNodes.find((n) => n.id === selectedInputs[0])
                  ?.label,
                position: "bottom",
              }}
            />
            <YAxis
              type="number"
              dataKey="y"
              name={inputNodes.find((n) => n.id === selectedInputs[1])?.label}
              domain={[-1, 1]}
              label={{
                value: inputNodes.find((n) => n.id === selectedInputs[1])
                  ?.label,
                angle: -90,
                position: "left",
              }}
            />
            <ZAxis type="number" dataKey="z" range={[0, 400]} name="Output" />
            <Tooltip
              cursor={{ strokeDasharray: "3 3" }}
              content={({ payload }) => {
                if (payload && payload.length) {
                  const { x, y, z } = payload[0].payload;
                  return (
                    <div className="bg-white p-2 border rounded shadow">
                      <p className="text-sm">
                        {
                          inputNodes.find((n) => n.id === selectedInputs[0])
                            ?.label
                        }
                        : {x}
                      </p>
                      <p className="text-sm">
                        {
                          inputNodes.find((n) => n.id === selectedInputs[1])
                            ?.label
                        }
                        : {y}
                      </p>
                      <p className="text-sm">
                        {watchNode ? `Node ${watchNode}` : "Output"}:{" "}
                        {Number(z).toFixed(4)}
                      </p>
                    </div>
                  );
                }
                return null;
              }}
            />
            <Scatter
              data={data}
              fill="#8884d8"
              shape={(props: any) => (
                <circle {...props} fill={props.payload.fill} r={5} />
              )}
            />
          </ScatterChart>
        </div>
      </Panel>
    );
  }
);

export default NeatVisualizationPanel;
