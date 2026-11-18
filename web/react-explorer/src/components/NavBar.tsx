import { useAppConfig } from "../api/appConfig";

export type NavTab = "experiments" | "datasets" | "templates" | "review" | "portal" | "features";

type NavBarProps = {
  activeTab: NavTab;
  onTabChange: (tab: NavTab) => void;
};

export function NavBar({ activeTab, onTabChange }: NavBarProps) {
  const cfg = useAppConfig();
  const allTabs = [
    { key: "experiments" as const, label: "Experiments" },
    { key: "portal" as const, label: "Run experiments" },
    { key: "datasets" as const, label: "Datasets" },
    { key: "templates" as const, label: "Templates" },
    { key: "review" as const, label: "Review explanations" },
    { key: "features" as const, label: "Features" },
  ];
  // read only: no running experiments; the static appendix has only the networks and their reviews
  const tabs = allTabs.filter((t) =>
    cfg.static ? t.key === "experiments" || t.key === "review" : !(cfg.read_only && t.key === "portal"),
  );

  return (
    <nav
      style={{
        display: "flex",
        alignItems: "center",
        gap: "2rem",
        padding: "0 2rem",
        background: "white",
        borderBottom: "1px solid #e5e7eb",
        height: "48px",
      }}
    >
      <span
        style={{
          fontWeight: 700,
          fontSize: "1rem",
          color: "#111827",
          marginRight: "1rem",
        }}
      >
        ExplaNEAT
      </span>
      {cfg.static && (
        <a href="../" style={{ fontSize: "0.85rem", color: "#6b7280" }} title="The appendix's landing page">
          ← Appendix
        </a>
      )}
      {tabs.map((tab) => (
        <button
          key={tab.key}
          onClick={() => onTabChange(tab.key)}
          style={{
            background: "none",
            border: "none",
            borderBottom:
              activeTab === tab.key ? "2px solid #2563eb" : "2px solid transparent",
            padding: "0.75rem 0.25rem",
            marginBottom: "-1px",
            fontWeight: activeTab === tab.key ? 600 : 400,
            color: activeTab === tab.key ? "#2563eb" : "#6b7280",
            cursor: "pointer",
            fontSize: "0.9rem",
          }}
        >
          {tab.label}
        </button>
      ))}
    </nav>
  );
}
