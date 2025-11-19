import { Explorer } from "./components/Explorer";
import type { ExplorerData } from "./types";

function readEmbeddedData(): ExplorerData | null {
  const script = document.getElementById("explorer-data");
  if (!script) {
    console.warn("Explorer data script tag not found");
    return null;
  }

  try {
    return JSON.parse(script.textContent || "") as ExplorerData;
  } catch (error) {
    console.error("Failed to parse explorer data", error);
    return null;
  }
}

export default function App() {
  const data = readEmbeddedData();

  if (!data) {
    return (
      <div className="app-shell">
        <div className="app-panel">
          <h1>ExplaNEAT React Explorer</h1>
          <p>Unable to load embedded graph payload.</p>
        </div>
      </div>
    );
  }

  return <Explorer data={data} />;
}

