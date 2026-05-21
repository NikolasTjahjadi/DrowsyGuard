import { X } from "lucide-react";

function Slider({ label, value, min, max, step, onChange }) {
  return (
    <label className="slider-row">
      <span>
        {label}
        <strong>{Number(value).toFixed(2)}</strong>
      </span>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(event) => onChange(Number(event.target.value))} />
    </label>
  );
}

export default function SettingsPanel({ open, config, soundEnabled, apiBase, onClose, onSoundChange, onConfigChange }) {
  const updateConfig = async (patch) => {
    const next = { ...config, ...patch };
    onConfigChange(next);
    await fetch(`${apiBase}/api/config`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(patch),
    }).catch(() => {});
  };

  const recalibrateHead = async () => {
    const res = await fetch(`${apiBase}/api/calibrate/head`, { method: "POST" }).catch(() => null);
    if (!res?.ok) return;
    const data = await res.json();
    onConfigChange({
      ...config,
      ear_threshold: data.ear_threshold ?? config.ear_threshold,
      mar_threshold: data.mar_threshold ?? config.mar_threshold,
      pitch_threshold: data.pitch_threshold ?? config.pitch_threshold,
      alert_threshold: data.alert_threshold ?? config.alert_threshold,
    });
  };

  return (
    <>
      <div className={`scrim ${open ? "visible" : ""}`} onClick={onClose} />
      <aside className={`settings-drawer ${open ? "open" : ""}`}>
        <div className="drawer-head">
          <h2>Settings</h2>
          <button className="icon-button" type="button" aria-label="Close settings" title="Close" onClick={onClose}>
            <X size={20} />
          </button>
        </div>
        <Slider label="EAR threshold" value={config.ear_threshold} min={0.15} max={0.35} step={0.01} onChange={(value) => updateConfig({ ear_threshold: value })} />
        <Slider label="MAR threshold" value={config.mar_threshold} min={0.4} max={0.8} step={0.01} onChange={(value) => updateConfig({ mar_threshold: value })} />
        <Slider label="Head nod threshold" value={config.pitch_threshold} min={15} max={45} step={1} onChange={(value) => updateConfig({ pitch_threshold: value })} />
        <Slider label="Alert threshold" value={config.alert_threshold} min={0.25} max={0.8} step={0.01} onChange={(value) => updateConfig({ alert_threshold: value })} />
        <label className="toggle-row">
          <span>Sound alerts</span>
          <input type="checkbox" checked={soundEnabled} onChange={(event) => onSoundChange(event.target.checked)} />
        </label>
        <button className="drawer-button" type="button" onClick={recalibrateHead}>
          Recalibrate head pose
        </button>
      </aside>
    </>
  );
}
