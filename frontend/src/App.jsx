import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Settings } from "lucide-react";
import AlertBanner from "./components/AlertBanner.jsx";
import WebcamFeed from "./components/WebcamFeed.jsx";
import MetricsPanel from "./components/MetricsPanel.jsx";
import StatusIndicators from "./components/StatusIndicators.jsx";
import SessionStats from "./components/SessionStats.jsx";
import SettingsPanel from "./components/SettingsPanel.jsx";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

const initialPrediction = {
  ear: 0,
  mar: 0,
  head_pitch: 0,
  head_yaw: 0,
  head_roll: 0,
  cnn_prob: 0,
  fusion_score: 0,
  smoothed_score: 0,
  alert: false,
  drowsiness_level: "LOW",
  eye_closed: false,
  yawning: false,
  head_nodding: false,
  model_alert: false,
  rule_alert: false,
  alert_sources: [],
  head_calibrating: false,
  landmarks_detected: false,
  fps: 0,
};

function formatDuration(totalSeconds) {
  const hrs = Math.floor(totalSeconds / 3600).toString().padStart(2, "0");
  const mins = Math.floor((totalSeconds % 3600) / 60).toString().padStart(2, "0");
  const secs = Math.floor(totalSeconds % 60).toString().padStart(2, "0");
  return `${hrs}:${mins}:${secs}`;
}

export default function App() {
  const startedAt = useRef(Date.now());
  const [prediction, setPrediction] = useState(initialPrediction);
  const [history, setHistory] = useState([]);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [soundEnabled, setSoundEnabled] = useState(true);
  const [config, setConfig] = useState({
    ear_threshold: 0.25,
    mar_threshold: 0.6,
    pitch_threshold: 25,
    alert_threshold: 0.25,
  });
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    fetch(`${API_BASE}/api/config`)
      .then((res) => res.json())
      .then((data) => {
        setConfig({
          ear_threshold: data.ear_threshold ?? 0.25,
          mar_threshold: data.mar_threshold ?? 0.6,
          pitch_threshold: data.pitch_threshold ?? 25,
          alert_threshold: data.alert_threshold ?? 0.25,
        });
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    const id = window.setInterval(() => {
      setElapsed(Math.floor((Date.now() - startedAt.current) / 1000));
    }, 1000);
    return () => window.clearInterval(id);
  }, []);

  const handlePrediction = useCallback((next) => {
    setPrediction((previous) => ({ ...previous, ...next }));
    setHistory((previous) => {
      const sample = {
        time: Date.now(),
        score: next.smoothed_score ?? next.fusion_score ?? 0,
        alert: Boolean(next.alert),
      };
      return [...previous, sample].filter((point) => Date.now() - point.time <= 60000);
    });
  }, []);

  const session = useMemo(() => {
    const alertEdges = history.reduce((count, point, index) => {
      const previous = history[index - 1];
      return count + (point.alert && !previous?.alert ? 1 : 0);
    }, 0);
    const drowsySamples = history.filter((point) => point.alert).length;
    const drowsyPercent = history.length ? (drowsySamples / history.length) * 100 : 0;
    const hours = Math.max(elapsed / 3600, 1 / 3600);
    return {
      duration: formatDuration(elapsed),
      alerts: alertEdges,
      drowsyPercent,
      alertsPerHour: alertEdges / hours,
    };
  }, [elapsed, history]);

  return (
    <main className={`app ${prediction.alert ? "is-alerting" : ""}`}>
      <AlertBanner active={prediction.alert} prediction={prediction} soundEnabled={soundEnabled} />
      <header className="topbar">
        <div>
          <p className="kicker">REAL-TIME DRIVER MONITORING</p>
          <h1>DrowsyGuard</h1>
        </div>
        <div className="topbar-actions">
          <span className="session-chip">Session: {session.duration}</span>
          <button
            className="icon-button"
            type="button"
            aria-label="Open settings"
            title="Settings"
            onClick={() => setSettingsOpen(true)}
          >
            <Settings size={20} />
          </button>
        </div>
      </header>

      <section className="dashboard-grid">
        <WebcamFeed apiBase={API_BASE} prediction={prediction} onPrediction={handlePrediction} />
        <aside className="side-stack">
          <MetricsPanel prediction={prediction} config={config} />
          <StatusIndicators prediction={prediction} />
        </aside>
      </section>

      <SessionStats session={session} history={history} />
      <SettingsPanel
        open={settingsOpen}
        config={config}
        soundEnabled={soundEnabled}
        apiBase={API_BASE}
        onClose={() => setSettingsOpen(false)}
        onSoundChange={setSoundEnabled}
        onConfigChange={setConfig}
      />
    </main>
  );
}
