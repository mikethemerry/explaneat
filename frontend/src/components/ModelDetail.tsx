import React, { useState, useEffect } from "react";
import axios from "axios";
import { useParams, useNavigate } from "react-router-dom";
import { NEATModel } from "../types/NEATModel";
import NEATVisualizer from "./NEATVisualizer";

const ModelDetail: React.FC = () => {
  const [model, setModel] = useState<NEATModel | null>(null);
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();

  useEffect(() => {
    const fetchModel = async () => {
      try {
        const response = await axios.get(
          `http://127.0.0.1:5000/api/models/${id}`
        );
        console.log(response.data);
        // Parse the JSON of the parsed_model
        // if (response.data.parsed_model) {
        // try {
        // response.data.parsed_model = JSON.parse(response.data.parsed_model);
        // } catch (parseError) {
        // console.error("Error parsing parsed_model JSON:", parseError);
        // If parsing fails, set parsed_model to null or an empty object
        // response.data.parsed_model = null;
        // }
        // }
        setModel(response.data);
      } catch (error) {
        console.error("Error fetching model:", error);
      }
    };

    fetchModel();
  }, [id]);

  const handleDelete = async () => {
    if (window.confirm("Are you sure you want to delete this model?")) {
      try {
        await axios.delete(`http://127.0.0.1:5000/api/models/${id}`);
        navigate("/");
      } catch (error) {
        console.error("Error deleting model:", error);
      }
    }
  };

  if (!model) return <div className="text-center">Loading...</div>;

  return (
    <div className="space-y-8">
      <div className="bg-white shadow overflow-hidden sm:rounded-lg">
        <div className="px-4 py-5 sm:px-6">
          <h3 className="text-lg leading-6 font-medium text-neutral-900">
            {model.model_name}
          </h3>
          <p className="mt-1 max-w-2xl text-sm text-neutral-500">
            Model details and properties
          </p>
        </div>
        <div className="border-t border-neutral-200 px-4 py-5 sm:p-0">
          <dl className="sm:divide-y sm:divide-neutral-200">
            <div className="py-4 sm:py-5 sm:grid sm:grid-cols-3 sm:gap-4 sm:px-6">
              <dt className="text-sm font-medium text-neutral-500">Dataset</dt>
              <dd className="mt-1 text-sm text-neutral-900 sm:mt-0 sm:col-span-2">
                {model.dataset}
              </dd>
            </div>
            <div className="py-4 sm:py-5 sm:grid sm:grid-cols-3 sm:gap-4 sm:px-6">
              <dt className="text-sm font-medium text-neutral-500">Version</dt>
              <dd className="mt-1 text-sm text-neutral-900 sm:mt-0 sm:col-span-2">
                {model.version}
              </dd>
            </div>
            <div className="py-4 sm:py-5 sm:grid sm:grid-cols-3 sm:gap-4 sm:px-6">
              <dt className="text-sm font-medium text-neutral-500">Created</dt>
              <dd className="mt-1 text-sm text-neutral-900 sm:mt-0 sm:col-span-2">
                {new Date(model.created_at).toLocaleString()}
              </dd>
            </div>
            <div className="py-4 sm:py-5 sm:grid sm:grid-cols-3 sm:gap-4 sm:px-6">
              <dt className="text-sm font-medium text-neutral-500">Updated</dt>
              <dd className="mt-1 text-sm text-neutral-900 sm:mt-0 sm:col-span-2">
                {new Date(model.updated_at).toLocaleString()}
              </dd>
            </div>
          </dl>
        </div>
      </div>

      <div className="bg-white shadow overflow-hidden sm:rounded-lg">
        <div className="px-4 py-5 sm:px-6">
          <h3 className="text-lg leading-6 font-medium text-neutral-900">
            Model Visualization
          </h3>
        </div>
        <div className="border-t border-neutral-200 px-4 py-5 sm:p-0">
          <NEATVisualizer model={model} />
        </div>
      </div>

      <div className="bg-white shadow overflow-hidden sm:rounded-lg">
        <div className="px-4 py-5 sm:px-6">
          <h3 className="text-lg leading-6 font-medium text-neutral-900">
            Model Data
          </h3>
        </div>
        <div className="border-t border-neutral-200 px-4 py-5 sm:p-0">
          <div className="px-4 py-5 sm:px-6">
            <pre className="bg-neutral-50 p-2 rounded overflow-x-auto">
              {JSON.stringify(model.parsed_model, null, 2)}
            </pre>
          </div>
        </div>
      </div>

      <div className="flex justify-end space-x-4">
        <button
          onClick={() => navigate(`/edit/${id}`)}
          className="inline-flex justify-center py-2 px-4 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-primary-600 hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500"
        >
          Edit
        </button>
        <button
          onClick={handleDelete}
          className="inline-flex justify-center py-2 px-4 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-red-600 hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500"
        >
          Delete
        </button>
      </div>
    </div>
  );
};

export default ModelDetail;
