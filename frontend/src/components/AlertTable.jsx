import React from "react";

export default function AlertTable({ alerts }) {
  const rows = Array.isArray(alerts) ? alerts : [];

  return (
    <div className="table-wrapper">
      <h2>Alerts</h2>
      <table className="alerts-table">
        <thead>
          <tr>
            <th>Source</th>
            <th>Organization</th>
            <th>Device</th>
            <th>Alert Type</th>
            <th>Severity</th>
            <th>Timestamp</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row, idx) => (
            <tr key={idx}>
              <td>{row.source || "-"}</td>
              <td>{row.organization || "-"}</td>
              <td>{row.device || "-"}</td>
              <td>{row.alert_type || row.alertType || row.alert || "-"}</td>
              <td>{row.severity || "-"}</td>
              <td>{row.timestamp || "-"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
