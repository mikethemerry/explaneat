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

const NeatVisualizationPanel = ({ neatModel }: { neatModel: NEATModel }) => {
  // Create NEAT instance
  const neat = useMemo(() => new NEAT(neatModel), [neatModel]);

  // Generate data points for visualization
  const generateDataPoints = () => {
    const points = [];
    const n_inputs =
      neatModel.parsed_model.nodes.filter((node) => node.type === "input")
        .length - 2;
    let default_input = new Array(n_inputs).fill(0);

    for (let x = 0; x <= 1; x += 0.1) {
      for (let y = 0; y <= 1; y += 0.1) {
        const input = [x, y, ...default_input];
        const output = neat.forward(input)[0];
        points.push({
          x: x.toFixed(2),
          y: y.toFixed(2),
          z: output,
          fill: `rgb(${Math.floor(output * 255)}, ${Math.floor(
            output * 255
          )}, 255)`,
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
  }, [neatModel]);

  return (
    <Panel
      position="top-right"
      className="bg-white rounded-lg shadow-lg w-[450px]"
    >
      <div className="p-4">
        <h3 className="text-lg font-semibold mb-4">
          NEAT Output Visualization
        </h3>
        <ScatterChart
          width={400}
          height={400}
          margin={{ top: 20, right: 20, bottom: 20, left: 20 }}
        >
          <CartesianGrid />
          <XAxis
            type="number"
            dataKey="x"
            name="Input 1"
            domain={[0, 1]}
            label={{ value: "Input 1", position: "bottom" }}
          />
          <YAxis
            type="number"
            dataKey="y"
            name="Input 2"
            domain={[0, 1]}
            label={{ value: "Input 2", angle: -90, position: "left" }}
          />
          <ZAxis type="number" dataKey="z" range={[0, 400]} name="Output" />
          <Tooltip
            cursor={{ strokeDasharray: "3 3" }}
            content={({ payload }) => {
              if (payload && payload.length) {
                const { x, y, z } = payload[0].payload;
                return (
                  <div className="bg-white p-2 border rounded shadow">
                    <p className="text-sm">Input 1: {x}</p>
                    <p className="text-sm">Input 2: {y}</p>
                    <p className="text-sm">Output: {Number(z).toFixed(4)}</p>
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
              <circle {...props} fill={props.payload.fill} r={4} />
            )}
          />
        </ScatterChart>
      </div>
    </Panel>
  );
};

export default NeatVisualizationPanel;
