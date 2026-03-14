import React, { useEffect, useMemo, useState } from "react";
import axios from "axios";
import AlertTable from "./components/AlertTable";
import SeverityChart from "./components/SeverityChart";
import DeviceChart from "./components/DeviceChart";
import IncidentTrendChart from "./components/IncidentTrendChart";

const MENU_ITEMS = ["Dashboard", "Alerts", "Devices", "Analytics"];

export default function Dashboard({ onLogout }) {
  const [alerts, setAlerts] = useState([]);
  const [severityCounts, setSeverityCounts] = useState({});
  const [deviceCounts, setDeviceCounts] = useState({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [activeMenu, setActiveMenu] = useState("Dashboard");

  useEffect(() => {
    const client = axios.create({
      baseURL: import.meta.env.VITE_API_BASE || "",
    });

    const loadData = async () => {
      setLoading(true);
      setError("");
      try {
        const [alertsRes, severityRes, deviceRes] = await Promise.all([
          client.get("/alerts", { params: { limit: 500 } }),
          client.get("/alerts/severity"),
          client.get("/alerts/device"),
        ]);

        const alertsData = alertsRes?.data?.items || alertsRes?.data || [];
        setAlerts(Array.isArray(alertsData) ? alertsData : []);
        setSeverityCounts(severityRes?.data || {});
        setDeviceCounts(deviceRes?.data || {});
      } catch (err) {
        console.error(err);
        setError("Unable to load data. Make sure the backend is running and reachable.");
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, []);

  const totalAlerts = alerts.length;

  const metrics = useMemo(() => {
    const normalized = {
      critical: 0,
      warning: 0,
      normal: 0,
    };

    Object.entries(severityCounts).forEach(([key, value]) => {
      const label = String(key || "").trim().toLowerCase();
      if (label.includes("crit")) normalized.critical += value;
      else if (label.includes("warn")) normalized.warning += value;
      else if (label.includes("norm") || label === "info") normalized.normal += value;
      else normalized.normal += value;
    });

    return normalized;
  }, [severityCounts]);

  const renderMain = () => {
    if (loading) {
      return <div className="status">Loading...</div>;
    }

    if (error) {
      return <div className="status error">{error}</div>;
    }

    return (
      <>
        <div className="widgets">
          <div className="widget">
            <div className="widget-title">Total Alerts</div>
            <div className="widget-value">{totalAlerts}</div>
          </div>

          <div className="widget">
            <div className="widget-title">Critical Alerts</div>
            <div className="widget-value">{metrics.critical}</div>
          </div>

          <div className="widget">
            <div className="widget-title">Warning Alerts</div>
            <div className="widget-value">{metrics.warning}</div>
          </div>

          <div className="widget">
            <div className="widget-title">Resolved Incidents</div>
            <div className="widget-value">{metrics.normal}</div>
          </div>
        </div>

        <div className="charts">
          <SeverityChart data={severityCounts} />
          <DeviceChart data={deviceCounts} />
          <IncidentTrendChart alerts={alerts} />
        </div>

        <AlertTable alerts={alerts} />
      </>
    );
  };

  return (
    <div className="app">
      <aside className="sidebar">
        <h2>Alert Intelligence</h2>
        <nav>
          {MENU_ITEMS.map((item) => (
            <a
              key={item}
              className={item === activeMenu ? "active" : ""}
              href="#"
              onClick={(event) => {
                event.preventDefault();
                setActiveMenu(item);
              }}
            >
              {item}
            </a>
          ))}
        </nav>
      </aside>

      <main className="main">
        <header className="topbar">
          <h1>Alert Incident Intelligence Dashboard</h1>
          <button onClick={onLogout}>Logout</button>
        </header>

        <section className="content">{renderMain()}</section>
      </main>
    </div>
  );
}
