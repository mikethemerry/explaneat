import React from "react";
import { useParams, useSearchParams } from "react-router-dom";

interface DatasetData {
  features: number[][];
  targets: number[][];
  feature_columns: string[];
  target_column: string;
}

export const DatasetViewer: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const [searchParams] = useSearchParams();
  const splitType = searchParams.get("split");

  const [data, setData] = React.useState<DatasetData | null>(null);
  const [loading, setLoading] = React.useState(true);
  const [currentPage, setCurrentPage] = React.useState(1);
  const rowsPerPage = 50;

  React.useEffect(() => {
    const url = splitType
      ? `http://127.0.0.1:5000/api/datasets/${id}/data?split_type=${splitType}`
      : `http://127.0.0.1:5000/api/datasets/${id}/data`;

    fetch(url)
      .then((res) => res.json())
      .then((data) => {
        setData(data);
        setLoading(false);
      });
  }, [id, splitType]);

  if (loading) return <div>Loading dataset data...</div>;
  if (!data) return <div>No data available</div>;

  const startIdx = (currentPage - 1) * rowsPerPage;
  const endIdx = startIdx + rowsPerPage;
  const totalPages = Math.ceil(data.features.length / rowsPerPage);

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="mb-4 flex justify-between items-center">
        <h2 className="text-xl font-bold">
          Dataset View {splitType && `(${splitType} split)`}
        </h2>
        <div className="flex gap-2">
          <button
            onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
            disabled={currentPage === 1}
            className="px-3 py-1 border rounded disabled:opacity-50"
          >
            Previous
          </button>
          <span className="px-3 py-1">
            Page {currentPage} of {totalPages}
          </span>
          <button
            onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
            disabled={currentPage === totalPages}
            className="px-3 py-1 border rounded disabled:opacity-50"
          >
            Next
          </button>
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Row
              </th>
              {data.feature_columns.map((col) => (
                <th
                  key={col}
                  className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                >
                  {col}
                </th>
              ))}
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                {data.target_column}
              </th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {data.features.slice(startIdx, endIdx).map((row, idx) => (
              <tr key={startIdx + idx}>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                  {startIdx + idx + 1}
                </td>
                {row.map((val, colIdx) => (
                  <td
                    key={colIdx}
                    className="px-6 py-4 whitespace-nowrap text-sm text-gray-900"
                  >
                    {val.toFixed(4)}
                  </td>
                ))}
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                  {data.targets[startIdx + idx][0]}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};
