import React from "react";
import { Link } from "react-router-dom";

interface Dataset {
  id: number;
  name: string;
  source: string;
  n_samples: number;
  n_features: number;
  task_type: string;
  n_classes: number | null;
}

export const DatasetList: React.FC = () => {
  const [datasets, setDatasets] = React.useState<Dataset[]>([]);
  const [loading, setLoading] = React.useState(true);

  React.useEffect(() => {
    fetch("http://127.0.0.1:5000/api/datasets")
      .then((res) => res.json())
      .then((data) => {
        setDatasets(data);
        setLoading(false);
      });
  }, []);

  if (loading) return <div>Loading datasets...</div>;

  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-2xl font-bold mb-6">Available Datasets</h1>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {datasets.map((dataset) => (
          <Link
            key={dataset.id}
            to={`/datasets/${dataset.id}`}
            className="p-4 border rounded-lg hover:shadow-lg transition-shadow"
          >
            <h2 className="text-xl font-semibold">{dataset.name}</h2>
            <div className="mt-2 text-gray-600">
              <p>Source: {dataset.source}</p>
              <p>Samples: {dataset.n_samples}</p>
              <p>Features: {dataset.n_features}</p>
              <p>Type: {dataset.task_type}</p>
              {dataset.n_classes && <p>Classes: {dataset.n_classes}</p>}
            </div>
          </Link>
        ))}
      </div>
    </div>
  );
};
