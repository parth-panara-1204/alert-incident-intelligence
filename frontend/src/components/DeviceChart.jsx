import React from "react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from "recharts";

export default function DeviceChart({ data }) {
  const chartData = Object.entries(data || {}).map(([device, count]) => ({
    device,
    count,
  }));

  return (
    <div className="chart">
      <h2>Alerts by Device</h2>
      <ResponsiveContainer width="100%" height={250}>
        <BarChart data={chartData} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.08)" />
          <XAxis dataKey="device" tick={{ fill: "var(--muted)" }} />
          <YAxis allowDecimals={false} tick={{ fill: "var(--muted)" }} />
          <Tooltip />
          <Bar dataKey="count" fill="var(--secondary)" radius={[6, 6, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
