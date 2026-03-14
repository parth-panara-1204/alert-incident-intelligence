import React from "react";
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer, Legend } from "recharts";

const SEVERITY_COLORS = {
  critical: "#B23A2B",
  warning: "#D97706",
  minor: "#C19A6B",
  normal: "#6B8E23",
  default: "#8B5E3C",
};

function getSeverityKey(key) {
  const normalized = String(key || "").trim().toLowerCase();
  if (normalized.includes("crit")) return "critical";
  if (normalized.includes("warn")) return "warning";
  if (normalized.includes("minor")) return "minor";
  if (normalized.includes("normal") || normalized === "info") return "normal";
  return "default";
}

export default function SeverityChart({ data }) {
  const chartData = Object.entries(data || {}).reduce((acc, [severity, count]) => {
    const key = getSeverityKey(severity);
    const existing = acc.find((d) => d.severity === key);
    if (existing) {
      existing.count += count;
    } else {
      acc.push({ severity: key, count });
    }
    return acc;
  }, []);

  return (
    <div className="chart">
      <h2>Alerts by Severity</h2>
      <ResponsiveContainer width="100%" height={250}>
        <PieChart>
          <Pie
            data={chartData}
            dataKey="count"
            nameKey="severity"
            cx="50%"
            cy="50%"
            outerRadius={90}
            innerRadius={40}
            label={({ percent }) => `${(percent * 100).toFixed(0)}%`}
          >
            {chartData.map((entry, idx) => (
              <Cell
                key={`cell-${idx}`}
                fill={SEVERITY_COLORS[entry.severity] || SEVERITY_COLORS.default}
              />
            ))}
          </Pie>
          <Tooltip formatter={(value) => [value, "alerts"]} />
          <Legend verticalAlign="bottom" height={36} />
        </PieChart>
      </ResponsiveContainer>
    </div>
  );
}
