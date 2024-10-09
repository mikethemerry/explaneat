import React, { useState, useEffect, useMemo } from "react";
import axios from "axios";
import { useParams, useNavigate } from "react-router-dom";
import { NEATModel } from "../types/NEATModel";

interface ModelFormProps {
  mode: "create" | "edit";
}

const ModelForm: React.FC<ModelFormProps> = ({ mode }) => {
  const [model, setModel] = useState<Partial<NEATModel>>({
    model_name: "",
    dataset: "",
    version: "",
    raw_data: "",
    parsed_model: {
      nodes: [],
      edges: [],
    },
  });
  const [reactflowJson, setReactflowJson] = useState<{
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
  } | null>(null);
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();

  useEffect(() => {
    if (mode === "edit" && id) {
      const fetchModel = async () => {
        try {
          const response = await axios.get(
            `http://127.0.0.1:5000/api/models/${id}`
          );
          setModel(response.data);
        } catch (error) {
          console.error("Error fetching model:", error);
        }
      };
      fetchModel();
    }
  }, [mode, id]);

  const generateReactflowJson = useMemo(() => {
    if (!model.raw_data) return null;

    const rawData = model.raw_data as string;

    // Extract nodes and connections
    const nodesData = Array.from(
      rawData.matchAll(/(\d+)\s+DefaultNodeGene.*bias=([-\d.]+)/g)
    );
    const connectionsData = Array.from(
      rawData.matchAll(
        /DefaultConnectionGene\(key=\(([-\d]+),\s*([-\d]+)\),\s*weight=([-\d.]+),\s*enabled=(True|False)\)/g
      )
    );

    // Ensure all source nodes exist in nodesData, adding them if they don't
    const existingNodeIds = new Set(nodesData.map(([, nodeId]) => nodeId));
    const validConnectionsData = connectionsData.filter(
      ([, source, target]) => {
        if (!existingNodeIds.has(source)) {
          console.warn(
            `Source node ${source} not found in nodesData. Adding it.`
          );
          nodesData.push(["", source, "0"]); // Adding with default bias of 0
          existingNodeIds.add(source);
        }
        if (!existingNodeIds.has(target)) {
          console.warn(
            `Target node ${target} not found in nodesData. Adding it.`
          );
          nodesData.push(["", target, "0"]); // Adding with default bias of 0
          existingNodeIds.add(target);
        }
        return true;
      }
    );

    // If any nodes were added, log a message
    if (nodesData.length > existingNodeIds.size) {
      console.warn(
        `${
          nodesData.length - existingNodeIds.size
        } node(s) were added to ensure all connections are valid.`
      );
    }

    // Prepare nodes for ReactFlow
    const nodes = nodesData.map(([, nodeId, bias]) => {
      let nodeType = "default";
      if (parseInt(nodeId) < 0) nodeType = "input";
      else if (parseInt(nodeId) === 0) nodeType = "output";

      return {
        id: nodeId,
        type: nodeType,
        position: { x: 0, y: 0 }, // You may want to implement a layout algorithm here
        data: { label: `Node ${nodeId}\nBias: ${bias}` },
      };
    });

    // Prepare edges for ReactFlow
    const edges = connectionsData
      .filter(([, , , , enabled]) => enabled === "True")
      .map(([, source, target, weight]) => ({
        id: `e${source}-${target}`,
        source,
        target,
        label: `Weight: ${weight}`,
        type: "smoothstep",
      }));

    // Combine nodes and edges into the final ReactFlow-compatible JSON
    return { nodes, edges };
  }, [model.raw_data]);

  useEffect(() => {
    setReactflowJson(generateReactflowJson);
  }, [generateReactflowJson]);

  const handleChange = (
    event: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>
  ) => {
    const { name, value } = event.target;
    setModel((prevModel) => ({ ...prevModel, [name]: value }));
  };
  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    try {
      const modelToSubmit = {
        ...model,
        parsed_model: JSON.stringify(reactflowJson),
      };
      if (mode === "create") {
        await axios.post("http://127.0.0.1:5000/api/models", modelToSubmit);
      } else {
        await axios.put(
          `http://127.0.0.1:5000/api/models/${id}`,
          modelToSubmit
        );
      }
      navigate("/");
    } catch (error) {
      console.error("Error saving model:", error);
    }
  };

  return (
    <div className="bg-white shadow overflow-hidden sm:rounded-lg">
      <form onSubmit={handleSubmit}>
        <div className="px-4 py-5 sm:p-6">
          <div className="grid grid-cols-6 gap-6">
            <div className="col-span-6 sm:col-span-3">
              <label
                htmlFor="model_name"
                className="block text-sm font-medium text-neutral-700"
              >
                Model Name
              </label>
              <input
                type="text"
                name="model_name"
                id="model_name"
                value={model.model_name}
                onChange={handleChange}
                className="mt-1 focus:ring-primary-500 focus:border-primary-500 block w-full shadow-sm sm:text-sm border-neutral-300 rounded-md"
                required
              />
            </div>
            <div className="col-span-6 sm:col-span-3">
              <label
                htmlFor="dataset"
                className="block text-sm font-medium text-neutral-700"
              >
                Dataset
              </label>
              <input
                type="text"
                name="dataset"
                id="dataset"
                value={model.dataset}
                onChange={handleChange}
                className="mt-1 focus:ring-primary-500 focus:border-primary-500 block w-full shadow-sm sm:text-sm border-neutral-300 rounded-md"
                required
              />
            </div>
            <div className="col-span-6 sm:col-span-3">
              <label
                htmlFor="version"
                className="block text-sm font-medium text-neutral-700"
              >
                Version
              </label>
              <input
                type="text"
                name="version"
                id="version"
                value={model.version}
                onChange={handleChange}
                className="mt-1 focus:ring-primary-500 focus:border-primary-500 block w-full shadow-sm sm:text-sm border-neutral-300 rounded-md"
                required
              />
            </div>
            <div className="col-span-6">
              <label
                htmlFor="raw_data"
                className="block text-sm font-medium text-neutral-700"
              >
                Raw Data
              </label>
              <textarea
                name="raw_data"
                id="raw_data"
                value={model.raw_data as string}
                onChange={handleChange}
                rows={4}
                className="mt-1 focus:ring-primary-500 focus:border-primary-500 block w-full shadow-sm sm:text-sm border-neutral-300 rounded-md"
                required
              />
            </div>
            {reactflowJson && (
              <div className="col-span-6">
                <label className="block text-sm font-medium text-neutral-700">
                  Parsed Model Preview
                </label>
                <pre className="mt-1 block w-full rounded-md border-neutral-300 shadow-sm bg-neutral-50 p-2 overflow-auto max-h-60">
                  {JSON.stringify(reactflowJson, null, 2)}
                </pre>
              </div>
            )}
          </div>
        </div>
        <div className="px-4 py-3 bg-neutral-50 text-right sm:px-6">
          <button
            type="submit"
            className="inline-flex justify-center py-2 px-4 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-primary-600 hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500"
          >
            {mode === "create" ? "Create Model" : "Update Model"}
          </button>
        </div>
      </form>
    </div>
  );
};

export default ModelForm;
