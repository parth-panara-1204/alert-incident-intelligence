import React, { useMemo } from "react";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from "recharts";

function formatDate(date) {
  const d = new Date(date);
  if (Number.isNaN(d.getTime())) return null;
  return d.toISOString().slice(0, 10);
}

export default function IncidentTrendChart({ alerts }) {
  const data = useMemo(() => {
    if (!Array.isArray(alerts)) return [];

    const counts = alerts.reduce((acc, alert) => {
      const dateKey = formatDate(alert.timestamp);
      if (!dateKey) return acc;
      acc[dateKey] = (acc[dateKey] || 0) + 1;
      return acc;
    }, {});

    return Object.entries(counts)
      .map(([date, count]) => ({ date, count }))
      .sort((a, b) => (a.date < b.date ? -1 : 1));
  }, [alerts]);

  return (
    <div className="chart">
      <h2>Incident Trend</h2>
      <ResponsiveContainer width="100%" height={250}>
        <LineChart data={data} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.08)" />
          <XAxis dataKey="date" tick={{ fill: "var(--muted)" }} />
          <YAxis allowDecimals={false} tick={{ fill: "var(--muted)" }} />
          <Tooltip />
          <Line type="monotone" dataKey="count" stroke="var(--primary)" strokeWidth={3} dot={{ r: 2 }} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
