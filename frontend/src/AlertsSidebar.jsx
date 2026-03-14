import React, { useState } from "react";

export default function AlertsSidebar({ alerts }) {
  const [isQuickActionsOpen, setIsQuickActionsOpen] = useState(true);

  const handleResolveAll = () => {
    alert("Resolving all filtered alerts...");
  };

  const handleExportCSV = () => {
    if (alerts.length === 0) {
      alert("No alerts to export");
      return;
    }

    const headers = ["Source", "Organization", "Device", "Alert Type", "Severity", "Timestamp", "Status"];
    const csvContent = [
      headers.join(","),
      ...alerts.map((alert) =>
        [
          alert.source || "N/A",
          alert.organization || "N/A",
          alert.device || "N/A",
          alert.alert_type || "N/A",
          alert.severity || "N/A",
          alert.timestamp || "N/A",
          alert.status || "N/A",
        ]
          .map((field) => `"${String(field).replace(/"/g, '""')}"`)
          .join(",")
      ),
    ].join("\n");

    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const link = document.createElement("a");
    const url = URL.createObjectURL(blob);
    link.setAttribute("href", url);
    link.setAttribute("download", `alerts_${new Date().toISOString().split("T")[0]}.csv`);
    link.style.visibility = "hidden";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="alerts-sidebar">
      {/* Quick Actions */}
      <div className="filter-section quick-actions-section">
        <button
          className="section-header"
          onClick={() => setIsQuickActionsOpen((prev) => !prev)}
        >
          <span>⚡ Quick Actions</span>
          <span className="toggle-icon">{isQuickActionsOpen ? "−" : "+"}</span>
        </button>
        {isQuickActionsOpen && (
          <div className="quick-actions">
            <button className="action-btn resolve-btn" onClick={handleResolveAll}>
              ✔ Resolve All
            </button>
            <button className="action-btn export-btn" onClick={handleExportCSV}>
              📥 Export CSV
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
