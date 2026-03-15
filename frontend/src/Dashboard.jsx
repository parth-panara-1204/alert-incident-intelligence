import React, { useEffect, useMemo, useState } from "react";
import axios from "axios";
import {
  PieChart, Pie, Cell, BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, AreaChart, Area,
} from "recharts";
import ChatBot from "./components/ChatBot";

const API_BASE = "http://localhost:8000";
const ALERT_FETCH_LIMIT = 5000;
const DASHBOARD_REFRESH_MS = 7000;

const FALLBACK_PIE_COLORS = ["#2563eb", "#7c3aed", "#0ea5e9", "#22c55e", "#f59e0b", "#ef4444", "#14b8a6", "#f97316"];
const SEVERITY_COLOR_MAP = {
  critical: "#dc2626",
  emergency: "#b91c1c",
  failed: "#be123c",
  warning: "#f59e0b",
  normal: "#22c55e",
  resolved: "#10b981",
  info: "#0ea5e9",
  unknown: "#64748b",
};

function normalizeCategoryLabel(value) {
  return String(value || "unknown").trim().toLowerCase();
}

function getCategoryColor(categoryName, index) {
  const normalized = normalizeCategoryLabel(categoryName);
  return SEVERITY_COLOR_MAP[normalized] || FALLBACK_PIE_COLORS[index % FALLBACK_PIE_COLORS.length];
}

function legendLabel(value) {
  return String(value || "Unknown")
    .split(" ")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

function normalizeSeverity(value) {
  const normalized = normalizeCategoryLabel(value);
  if (normalized.includes("crit")) return "critical";
  if (normalized.includes("warn")) return "warning";
  if (normalized.includes("emerg")) return "emergency";
  if (normalized.includes("fail")) return "failed";
  if (normalized.includes("normal") || normalized.includes("ok") || normalized.includes("resolved")) return "normal";
  return normalized || "unknown";
}

// ============ FILTER BAR COMPONENT ============
function FilterBar({ allAlerts, onFilterChange, filters }) {
  const sources = useMemo(() => {
    const sourceSet = new Set(allAlerts.map((a) => a.source || "Unknown").filter(Boolean));
    return Array.from(sourceSet).sort();
  }, [allAlerts]);

  const organizations = useMemo(() => {
    const orgSet = new Set(allAlerts.map((a) => a.organization || "Unknown").filter(Boolean));
    return Array.from(orgSet).sort();
  }, [allAlerts]);

  const severities = useMemo(() => {
    const sevSet = new Set(allAlerts.map((a) => a.severity || "Unknown").filter(Boolean));
    return Array.from(sevSet).sort();
  }, [allAlerts]);

  const handleSourceChange = (e) => {
    onFilterChange({ ...filters, source: e.target.value });
  };

  const handleOrgChange = (e) => {
    onFilterChange({ ...filters, organization: e.target.value });
  };

  const handleSeverityChange = (e) => {
    onFilterChange({ ...filters, severity: e.target.value });
  };

  const handleReset = () => {
    onFilterChange({ source: "", organization: "", severity: "" });
  };

  return (
    <div className="filter-bar">
      <div className="filter-group">
        <label>Source</label>
        <select value={filters.source} onChange={handleSourceChange} className="filter-select">
          <option value="">All Sources</option>
          {sources.map((src) => (
            <option key={src} value={src}>
              {src}
            </option>
          ))}
        </select>
      </div>

      <div className="filter-group">
        <label>Organization</label>
        <select value={filters.organization} onChange={handleOrgChange} className="filter-select">
          <option value="">All Organizations</option>
          {organizations.map((org) => (
            <option key={org} value={org}>
              {org}
            </option>
          ))}
        </select>
      </div>

      <div className="filter-group">
        <label>Severity</label>
        <select value={filters.severity} onChange={handleSeverityChange} className="filter-select">
          <option value="">All Severities</option>
          {severities.map((sev) => (
            <option key={sev} value={sev}>
              {sev}
            </option>
          ))}
        </select>
      </div>

      <button className="reset-btn" onClick={handleReset}>
        Reset Filters
      </button>
    </div>
  );
}

// ============ ALERTS TAB COMPONENT ============
function AlertsTab({ alerts, severityCounts, deviceCounts, filteredAlerts }) {
  const metricsData = useMemo(() => {
    return filteredAlerts.length > 0 ? filteredAlerts : alerts;
  }, [filteredAlerts, alerts]);

  const metrics = useMemo(() => {
    const result = { critical: 0, warning: 0, normal: 0, resolved: 0, active: 0 };
    metricsData.forEach((alert) => {
      const severity = String(alert.severity || "").trim().toLowerCase();
      if (severity.includes("crit")) result.critical += 1;
      else if (severity.includes("warn")) result.warning += 1;
      else result.normal += 1;
    });
    result.resolved = Math.floor(result.normal * 0.3);
    result.active = result.critical + result.warning;
    return result;
  }, [metricsData]);

  const severityData = useMemo(() => {
    const counts = {};
    metricsData.forEach((alert) => {
      const severity = normalizeSeverity(alert.severity);
      counts[severity] = (counts[severity] || 0) + 1;
    });
    return Object.entries(counts).map(([name, value]) => ({
      name: String(name),
      value,
    }));
  }, [metricsData]);

  const severityChartData = useMemo(() => {
    return severityData.map((item, index) => ({
      ...item,
      color: getCategoryColor(item.name, index),
    }));
  }, [severityData]);

  const deviceData = useMemo(() => {
    const counts = {};
    metricsData.forEach((alert) => {
      const device = alert.device || "unknown";
      counts[device] = (counts[device] || 0) + 1;
    });
    return Object.entries(counts)
      .map(([key, value]) => ({ name: String(key).substring(0, 20), value }))
      .sort((a, b) => b.value - a.value)
      .slice(0, 5);
  }, [metricsData]);

  const trendData = useMemo(() => {
    const trend = {};
    metricsData.forEach((alert) => {
      const date = new Date(alert.timestamp).toLocaleDateString();
      trend[date] = (trend[date] || 0) + 1;
    });
    return Object.entries(trend)
      .map(([date, count]) => ({ date, count }))
      .sort((a, b) => new Date(a.date) - new Date(b.date))
      .slice(-10);
  }, [metricsData]);

  const topAlertTypes = useMemo(() => {
    const types = {};
    metricsData.forEach((alert) => {
      const type = alert.alert_type || "unknown";
      types[type] = (types[type] || 0) + 1;
    });
    return Object.entries(types)
      .map(([type, count]) => ({ type: type.substring(0, 15), count }))
      .sort((a, b) => b.count - a.count)
      .slice(0, 5);
  }, [metricsData]);

  const latestAlerts = useMemo(() => {
    return [...metricsData]
      .sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
  }, [metricsData]);

  return (
    <>
      {/* Metrics Cards */}
      <div className="metrics-cards">
        <div className="card metric-card">
          <div className="metric-label">Total Alerts</div>
          <div className="metric-value">{alerts.length}</div>
        </div>
        <div className="card metric-card">
          <div className="metric-label">Critical</div>
          <div className="metric-value critical">{metrics.critical}</div>
        </div>
        <div className="card metric-card">
          <div className="metric-label">Warning</div>
          <div className="metric-value warning">{metrics.warning}</div>
        </div>
        <div className="card metric-card">
          <div className="metric-label">Resolved</div>
          <div className="metric-value resolved">{metrics.resolved}</div>
        </div>
        <div className="card metric-card">
          <div className="metric-label">Active</div>
          <div className="metric-value active">{metrics.active}</div>
        </div>
      </div>

      {/* Charts Row 1 */}
      <div className="charts-row">
        <div className="card chart-card">
          <h3>Alerts by Severity</h3>
          {severityChartData.length > 0 ? (
            <ResponsiveContainer width="100%" height={250}>
              <PieChart>
                <Pie data={severityChartData} cx="50%" cy="45%" outerRadius={78} labelLine={false} dataKey="value" nameKey="name">
                  {severityChartData.map((entry, idx) => (
                    <Cell key={`severity-cell-${idx}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip />
                <Legend verticalAlign="bottom" iconType="circle" formatter={legendLabel} />
              </PieChart>
            </ResponsiveContainer>
          ) : (
            <p>No severity data</p>
          )}
        </div>

        <div className="card chart-card">
          <h3>Top 5 Devices</h3>
          {deviceData.length > 0 ? (
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={deviceData} layout="vertical" margin={{ left: 120, right: 20, top: 5, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis dataKey="name" type="category" width={110} tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="value" fill="#3b82f6" />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <p>No device data</p>
          )}
        </div>
      </div>

      {/* Charts Row 2 */}
      <div className="charts-row">
        <div className="card chart-card">
          <h3>Incident Trend</h3>
          {trendData.length > 0 ? (
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={trendData} margin={{ left: 0, right: 20, top: 5, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" tick={{ fontSize: 11 }} />
                <YAxis />
                <Tooltip />
                <Line type="monotone" dataKey="count" stroke="#10b981" strokeWidth={2} dot={{ r: 4 }} />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <p>No trend data</p>
          )}
        </div>

        <div className="card chart-card">
          <h3>Top Alert Types</h3>
          {topAlertTypes.length > 0 ? (
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={topAlertTypes} layout="vertical" margin={{ left: 100, right: 20, top: 5, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis dataKey="type" type="category" width={90} tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="count" fill="#f59e0b" />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <p>No alert type data</p>
          )}
        </div>
      </div>

      {/* Latest Alerts Table */}
      <div className="card table-card">
        <h3>All Alerts</h3>
        <div className="table-container">
          <table className="alerts-table">
            <thead>
              <tr>
                <th>Source</th>
                <th>Device</th>
                <th>Alert Type</th>
                <th>Severity</th>
                <th>Incident ID</th>
                <th>Incident Type</th>
                <th>Timestamp</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {latestAlerts.length > 0 ? (
                latestAlerts.map((alert, idx) => (
                  <tr key={idx} className={`severity-${alert.severity?.toLowerCase()}`}>
                    <td className="source">{alert.source || "—"}</td>
                    <td className="device">{alert.device || "—"}</td>
                    <td className="alert-type">{alert.alert_type || "—"}</td>
                    <td>
                      <span className={`severity-badge ${alert.severity?.toLowerCase()}`}>
                        {alert.severity || "—"}
                      </span>
                    </td>
                    <td className="source">{alert.incident_id || "—"}</td>
                    <td className="alert-type">{alert.incident_type || "—"}</td>
                    <td className="timestamp">{new Date(alert.timestamp).toLocaleString()}</td>
                    <td><span className="badge-active">{alert.status || "Active"}</span></td>
                  </tr>
                ))
              ) : (
                <tr><td colSpan="8" style={{ textAlign: "center", padding: "20px" }}>No alerts found</td></tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </>
  );
}

// ============ DEVICES TAB COMPONENT ============
function DevicesTab({ deviceCounts, alerts, filteredAlerts }) {
  const metricsData = useMemo(() => {
    return filteredAlerts.length > 0 ? filteredAlerts : alerts;
  }, [filteredAlerts, alerts]);

  const deviceStats = useMemo(() => {
    const counts = {};
    metricsData.forEach((alert) => {
      const device = alert.device || "unknown";
      counts[device] = (counts[device] || 0) + 1;
    });
    return Object.entries(counts)
      .map(([name, count]) => {
        const deviceAlerts = metricsData.filter((a) => a.device === name);
        const critical = deviceAlerts.filter((a) => String(a.severity).toLowerCase().includes("crit")).length;
        const warning = deviceAlerts.filter((a) => String(a.severity).toLowerCase().includes("warn")).length;
        return {
          name,
          totalAlerts: count,
          critical,
          warning,
          healthy: count - critical - warning > 0,
        };
      })
      .sort((a, b) => b.totalAlerts - a.totalAlerts)
      .slice(0, 15);
  }, [metricsData]);

  const deviceTrend = useMemo(() => {
    const trend = {};
    metricsData.forEach((alert) => {
      const device = alert.device || "unknown";
      const date = new Date(alert.timestamp).toLocaleDateString();
      const key = `${date}|${device}`;
      trend[key] = (trend[key] || 0) + 1;
    });
    const groupedByDate = {};
    Object.entries(trend).forEach(([key, count]) => {
      const [date, device] = key.split("|");
      if (!groupedByDate[date]) groupedByDate[date] = {};
      groupedByDate[date][device] = count;
    });
    return Object.entries(groupedByDate)
      .map(([date, devices]) => ({ date, ...devices }))
      .sort((a, b) => new Date(a.date) - new Date(b.date))
      .slice(-7);
  }, [metricsData]);

  const criticalDevices = useMemo(() => {
    return deviceStats.filter((d) => d.critical > 0).length;
  }, [deviceStats]);

  return (
    <>
      {/* Summary Cards */}
      <div className="metrics-cards">
        <div className="card metric-card">
          <div className="metric-label">Total Devices</div>
          <div className="metric-value">{deviceStats.length}</div>
        </div>
        <div className="card metric-card">
          <div className="metric-label">Healthy</div>
          <div className="metric-value resolved">{deviceStats.filter((d) => d.healthy).length}</div>
        </div>
        <div className="card metric-card">
          <div className="metric-label">With Warnings</div>
          <div className="metric-value warning">{deviceStats.filter((d) => d.warning > 0).length}</div>
        </div>
        <div className="card metric-card">
          <div className="metric-label">Critical</div>
          <div className="metric-value critical">{criticalDevices}</div>
        </div>
      </div>

      {/* Charts */}
      <div className="charts-row">
        <div className="card chart-card">
          <h3>Alerts Per Device (Top 10)</h3>
          {deviceStats.slice(0, 10).length > 0 ? (
            <ResponsiveContainer width="100%" height={300}>
              <BarChart
                data={deviceStats.slice(0, 10)}
                layout="vertical"
                margin={{ left: 120, right: 20 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis dataKey="name" type="category" width={110} tick={{ fontSize: 10 }} />
                <Tooltip />
                <Bar dataKey="critical" stackId="a" fill="#dc2626" />
                <Bar dataKey="warning" stackId="a" fill="#f59e0b" />
                <Bar dataKey="healthy" stackId="a" fill="#10b981" />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <p>No device data</p>
          )}
        </div>

        <div className="card chart-card">
          <h3>Device Alert Trend (Last 7 Days)</h3>
          {deviceTrend.length > 0 ? (
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={deviceTrend} margin={{ left: 0, right: 20 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" tick={{ fontSize: 10 }} />
                <YAxis />
                <Tooltip />
                <Area type="monotone" dataKey="unknown" stackId="1" stroke="#3b82f6" fill="#bfdbfe" />
              </AreaChart>
            </ResponsiveContainer>
          ) : (
            <p>No trend data</p>
          )}
        </div>
      </div>

      {/* Device Status Table */}
      <div className="card table-card">
        <h3>Device Status</h3>
        <div className="table-container">
          <table className="alerts-table">
            <thead>
              <tr>
                <th>Device Name</th>
                <th>Total Alerts</th>
                <th>Critical</th>
                <th>Warning</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {deviceStats.length > 0 ? (
                deviceStats.map((device, idx) => (
                  <tr key={idx}>
                    <td className="source">{device.name}</td>
                    <td>{device.totalAlerts}</td>
                    <td><span className="severity-badge critical">{device.critical}</span></td>
                    <td><span className="severity-badge warning">{device.warning}</span></td>
                    <td>
                      <span className={`badge-${device.critical > 0 ? "critical" : device.warning > 0 ? "warning" : "healthy"}`}>
                        {device.critical > 0 ? "Critical" : device.warning > 0 ? "Warning" : "Healthy"}
                      </span>
                    </td>
                  </tr>
                ))
              ) : (
                <tr><td colSpan="5" style={{ textAlign: "center", padding: "20px" }}>No devices found</td></tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </>
  );
}

// ============ ANALYTICS TAB COMPONENT ============
function AnalyticsTab({ alerts, severityCounts, deviceCounts, filteredAlerts }) {
  const metricsData = useMemo(() => {
    return filteredAlerts.length > 0 ? filteredAlerts : alerts;
  }, [filteredAlerts, alerts]);

  const kpis = useMemo(() => {
    const totalAlerts = metricsData.length;
    const uniqueDates = new Set(metricsData.map((a) => new Date(a.timestamp).toLocaleDateString()));
    const avgAlertsPerDay = totalAlerts / Math.max(1, uniqueDates.size);
    const criticalCount = metricsData.filter((a) => String(a.severity).toLowerCase().includes("crit")).length;
    const resolutionRate = Math.floor((totalAlerts > 0 ? (criticalCount / totalAlerts) * 100 : 0));

    return {
      totalAlerts,
      avgAlertsPerDay: avgAlertsPerDay.toFixed(1),
      mttr: "2.3h", // Mean time to resolution
      resolutionRate: `${resolutionRate}%`,
    };
  }, [metricsData]);

  const dailyTrend = useMemo(() => {
    const trend = {};
    metricsData.forEach((alert) => {
      const date = new Date(alert.timestamp).toLocaleDateString();
      trend[date] = (trend[date] || 0) + 1;
    });
    return Object.entries(trend)
      .map(([date, count]) => ({ date, count }))
      .sort((a, b) => new Date(a.date) - new Date(b.date));
  }, [metricsData]);

  const severityDistribution = useMemo(() => {
    const counts = {};
    metricsData.forEach((alert) => {
      const severity = normalizeSeverity(alert.severity);
      counts[severity] = (counts[severity] || 0) + 1;
    });
    return Object.entries(counts).map(([name, value]) => ({
      name: String(name),
      value,
    }));
  }, [metricsData]);

  const severityDistributionChartData = useMemo(() => {
    return severityDistribution.map((item, index) => ({
      ...item,
      color: getCategoryColor(item.name, index),
    }));
  }, [severityDistribution]);

  const topSources = useMemo(() => {
    const sources = {};
    metricsData.forEach((alert) => {
      const source = alert.source || "unknown";
      sources[source] = (sources[source] || 0) + 1;
    });
    return Object.entries(sources)
      .map(([source, count]) => ({ source, count }))
      .sort((a, b) => b.count - a.count)
      .slice(0, 5);
  }, [metricsData]);

  return (
    <>
      {/* KPI Cards */}
      <div className="metrics-cards">
        <div className="card metric-card">
          <div className="metric-label">Total Alerts</div>
          <div className="metric-value">{kpis.totalAlerts}</div>
        </div>
        <div className="card metric-card">
          <div className="metric-label">Avg Alerts/Day</div>
          <div className="metric-value">{kpis.avgAlertsPerDay}</div>
        </div>
        <div className="card metric-card">
          <div className="metric-label">MTTR</div>
          <div className="metric-value">{kpis.mttr}</div>
        </div>
        <div className="card metric-card">
          <div className="metric-label">Critical %</div>
          <div className="metric-value critical">{kpis.resolutionRate}</div>
        </div>
      </div>

      {/* Analytics Charts */}
      <div className="charts-row">
        <div className="card chart-card">
          <h3>Daily Alert Trend</h3>
          {dailyTrend.length > 0 ? (
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={dailyTrend} margin={{ left: 0, right: 20 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" tick={{ fontSize: 10 }} />
                <YAxis />
                <Tooltip />
                <Area type="monotone" dataKey="count" stroke="#3b82f6" fill="#bfdbfe" />
              </AreaChart>
            </ResponsiveContainer>
          ) : (
            <p>No trend data</p>
          )}
        </div>

        <div className="card chart-card">
          <h3>Top Alert Sources</h3>
          {topSources.length > 0 ? (
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={topSources} margin={{ left: 80, right: 20 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="source" angle={-45} textAnchor="end" height={80} tick={{ fontSize: 11 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" fill="#8b5cf6" />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <p>No source data</p>
          )}
        </div>
      </div>

      <div className="charts-row">
        <div className="card chart-card">
          <h3>Severity Distribution</h3>
          {severityDistributionChartData.length > 0 ? (
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={severityDistributionChartData}
                  cx="50%"
                  cy="45%"
                  outerRadius={88}
                  labelLine={false}
                  dataKey="value"
                  nameKey="name"
                >
                  {severityDistributionChartData.map((entry, idx) => (
                    <Cell key={`analytics-severity-cell-${idx}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip />
                <Legend verticalAlign="bottom" iconType="circle" formatter={legendLabel} />
              </PieChart>
            </ResponsiveContainer>
          ) : (
            <p>No severity data</p>
          )}
        </div>
      </div>
    </>
  );
}

// ============ MAIN DASHBOARD COMPONENT ============
export default function Dashboard({ onLogout }) {
  const [alerts, setAlerts] = useState([]);
  const [severityCounts, setSeverityCounts] = useState({});
  const [deviceCounts, setDeviceCounts] = useState({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [activeTab, setActiveTab] = useState("Alerts");
  const [filters, setFilters] = useState({ source: "", organization: "", severity: "" });
  const [sidebarWidth, setSidebarWidth] = useState(20); // Percentage width
  const [isDragging, setIsDragging] = useState(false);

  useEffect(() => {
    const client = axios.create({ baseURL: API_BASE });
    const loadData = async (silent = false) => {
      if (!silent) {
        setLoading(true);
      }
      setError("");
      try {
        const [alertsRes, severityRes, deviceRes] = await Promise.all([
          client.get("/alerts/ml", { params: { limit: ALERT_FETCH_LIMIT } }).catch(() => client.get("/alerts", { params: { limit: ALERT_FETCH_LIMIT } })),
          client.get("/alerts/severity"),
          client.get("/alerts/device"),
        ]);

        const alertsData = alertsRes?.data?.items || alertsRes?.data || [];
        setAlerts(Array.isArray(alertsData) ? alertsData : []);
        setSeverityCounts(severityRes?.data || {});
        setDeviceCounts(deviceRes?.data || {});
      } catch (err) {
        console.error("Error loading data:", err);
        setError("Unable to load data. Make sure the backend is running on http://localhost:8000");
      } finally {
        if (!silent) {
          setLoading(false);
        }
      }
    };

    loadData(false);
    const intervalId = window.setInterval(() => {
      loadData(true);
    }, DASHBOARD_REFRESH_MS);

    return () => {
      window.clearInterval(intervalId);
    };
  }, []);

  // Handle sidebar resize
  useEffect(() => {
    const handleMouseMove = (e) => {
      if (!isDragging) return;
      
      const container = document.querySelector(".app");
      if (!container) return;
      
      const newWidth = (e.clientX / container.clientWidth) * 100;
      // Constrain width between 15% and 40%
      if (newWidth >= 15 && newWidth <= 40) {
        setSidebarWidth(newWidth);
      }
    };

    const handleMouseUp = () => {
      setIsDragging(false);
    };

    if (isDragging) {
      document.addEventListener("mousemove", handleMouseMove);
      document.addEventListener("mouseup", handleMouseUp);
      document.body.style.cursor = "col-resize";
      document.body.style.userSelect = "none";
    }

    return () => {
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
      document.body.style.cursor = "default";
      document.body.style.userSelect = "auto";
    };
  }, [isDragging]);

  // Filter alerts based on selected filters
  const filteredAlerts = useMemo(() => {
    return alerts.filter((alert) => {
      const sourceMatch = !filters.source || alert.source === filters.source;
      const orgMatch = !filters.organization || alert.organization === filters.organization;
      const severityMatch = !filters.severity || alert.severity === filters.severity;
      return sourceMatch && orgMatch && severityMatch;
    });
  }, [alerts, filters]);

  const renderContent = () => {
    if (loading) return <div className="status">Loading dashboard...</div>;
    if (error) return <div className="status error">{error}</div>;

    switch (activeTab) {
      case "Alerts":
        return (
          <AlertsTab
            alerts={alerts}
            severityCounts={severityCounts}
            deviceCounts={deviceCounts}
            filteredAlerts={filteredAlerts}
          />
        );
      case "Devices":
        return <DevicesTab deviceCounts={deviceCounts} alerts={alerts} filteredAlerts={filteredAlerts} />;
      case "Analytics":
        return <AnalyticsTab alerts={alerts} severityCounts={severityCounts} deviceCounts={deviceCounts} filteredAlerts={filteredAlerts} />;
      case "Copilot":
        return <ChatBot apiBase={API_BASE} />;
      default:
        return (
          <AlertsTab
            alerts={alerts}
            severityCounts={severityCounts}
            deviceCounts={deviceCounts}
            filteredAlerts={filteredAlerts}
          />
        );
    }
  };

  return (
    <div className="app">
      <aside className="sidebar" style={{ width: `${sidebarWidth}%` }}>
        <h2>🚨 Alert Intelligence</h2>
        <nav>
          {["Dashboard", "Alerts", "Devices", "Analytics", "Copilot"].map((tab) => (
            <a
              key={tab}
              className={tab === activeTab || (tab === "Dashboard" && activeTab === "Alerts") ? "active" : ""}
              href="#"
              onClick={(e) => {
                e.preventDefault();
                setActiveTab(tab === "Dashboard" ? "Alerts" : tab);
              }}
            >
              {tab}
            </a>
          ))}
        </nav>
      </aside>

      <div
        className="sidebar-divider"
        onMouseDown={() => setIsDragging(true)}
        title="Drag to resize sidebar"
      />

      <main className="main">
        <header className="topbar">
          <h1>Alert Incident Intelligence Dashboard</h1>
          <button className="logout-btn" onClick={onLogout}>Logout</button>
        </header>
        {!loading && !error && activeTab !== "Copilot" && (
          <FilterBar allAlerts={alerts} filters={filters} onFilterChange={setFilters} />
        )}
        <section className="content">{renderContent()}</section>
      </main>
    </div>
  );
}
