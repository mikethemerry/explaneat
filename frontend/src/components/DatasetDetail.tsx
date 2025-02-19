import React from "react";
import { useParams } from "react-router-dom";

interface Dataset {
  id: number;
  name: string;
  source: string;
  version: string;
  n_samples: number;
  n_features: number;
  task_type: string;
  n_classes: number | null;
  description: string | null;
  paper_url: string | null;
  data_url: string;
  columns: Array<{
    name: string;
    data_type: string;
    is_target: boolean;
  }>;
  splits: Array<{
    split_type: string;
    seed: number;
  }>;
}

export const DatasetDetail: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const [dataset, setDataset] = React.useState<Dataset | null>(null);
  const [loading, setLoading] = React.useState(true);

  React.useEffect(() => {
    fetch(`http://127.0.0.1:5000/api/datasets/${id}`)
      .then((res) => res.json())
      .then((data) => {
        setDataset(data);
        setLoading(false);
      });
  }, [id]);

  if (loading) return <div>Loading dataset details...</div>;
  if (!dataset) return <div>Dataset not found</div>;

  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold mb-6">{dataset.name}</h1>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        <div className="space-y-4">
          <h2 className="text-xl font-semibold">Dataset Information</h2>
          <div className="space-y-2">
            <p>
              <span className="font-medium">Source:</span> {dataset.source}
            </p>
            <p>
              <span className="font-medium">Version:</span> {dataset.version}
            </p>
            <p>
              <span className="font-medium">Samples:</span> {dataset.n_samples}
            </p>
            <p>
              <span className="font-medium">Features:</span>{" "}
              {dataset.n_features}
            </p>
            <p>
              <span className="font-medium">Task:</span> {dataset.task_type}
            </p>
            {dataset.n_classes && (
              <p>
                <span className="font-medium">Classes:</span>{" "}
                {dataset.n_classes}
              </p>
            )}
          </div>
        </div>

        <div className="space-y-4">
          <h2 className="text-xl font-semibold">Available Splits</h2>
          <div className="grid grid-cols-2 gap-4">
            {dataset.splits.map((split, index) => (
              <div key={index} className="p-3 border rounded">
                <p className="font-medium">{split.split_type}</p>
                <p className="text-sm">Seed: {split.seed}</p>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="mt-8">
        <h2 className="text-xl font-semibold mb-4">Features</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {dataset.columns.map((column, index) => (
            <div key={index} className="p-3 border rounded">
              <p className="font-medium">{column.name}</p>
              <p className="text-sm">Type: {column.data_type}</p>
              {column.is_target && (
                <span className="text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded">
                  Target
                </span>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
