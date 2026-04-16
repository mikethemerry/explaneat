type NavBarProps = {
  activeTab: "experiments" | "datasets" | "templates";
  onTabChange: (tab: "experiments" | "datasets" | "templates") => void;
};

export function NavBar({ activeTab, onTabChange }: NavBarProps) {
  const tabs = [
    { key: "experiments" as const, label: "Experiments" },
    { key: "datasets" as const, label: "Datasets" },
    { key: "templates" as const, label: "Templates" },
  ];

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
