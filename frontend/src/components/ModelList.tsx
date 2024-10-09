import React, { useState, useEffect } from "react";
import axios from "axios";
import { Link } from "react-router-dom";
import { NEATModel } from "../types/NEATModel";

const ModelList: React.FC = () => {
  const [models, setModels] = useState<NEATModel[]>([]);

  useEffect(() => {
    const fetchModels = async () => {
      try {
        const response = await axios.get("http://127.0.0.1:5000/api/models", {
          withCredentials: true,
          headers: {
            "Content-Type": "application/json",
          },
        });
        setModels(response.data);
      } catch (error) {
        console.error("Error fetching models:", error);
      }
    };

    fetchModels();
  }, []);

  return (
    <div className="bg-white shadow overflow-hidden sm:rounded-md">
      <ul className="divide-y divide-neutral-200">
        {models.map((model) => (
          <li key={model.id}>
            <Link
              to={`/model/${model.id}`}
              className="block hover:bg-neutral-50"
            >
              <div className="px-4 py-4 sm:px-6">
                <div className="flex items-center justify-between">
                  <p className="text-sm font-medium text-primary-600 truncate">
                    {model.model_name}
                  </p>
                  <div className="ml-2 flex-shrink-0 flex">
                    <p className="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-green-100 text-green-800">
                      {model.version}
                    </p>
                  </div>
                </div>
                <div className="mt-2 sm:flex sm:justify-between">
                  <div className="sm:flex">
                    <p className="flex items-center text-sm text-neutral-500">
                      Dataset: {model.dataset}
                    </p>
                  </div>
                  <div className="mt-2 flex items-center text-sm text-neutral-500 sm:mt-0">
                    <p>
                      Created: {new Date(model.created_at).toLocaleDateString()}
                    </p>
                  </div>
                </div>
              </div>
            </Link>
          </li>
        ))}
      </ul>
    </div>
  );
};

export default ModelList;
