import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import { installStaticFetch } from "./api/staticData";
import { loadAppConfig } from "./api/appConfig";
import "./styles.css";

installStaticFetch();
loadAppConfig();

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
