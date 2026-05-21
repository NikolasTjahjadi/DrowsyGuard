function clamp(value, min = 0, max = 1) {
  return Math.min(max, Math.max(min, value));
}

function MetricBar({ label, value, min, max, dangerZone, unit = "" }) {
  const pct = clamp((value - min) / (max - min));
  const danger = dangerZone(value);
  return (
    <div className="metric-row">
      <div className="metric-label">
        <span>{label}</span>
        <strong className={danger ? "danger-text" : ""}>
          {value.toFixed(label === "FPS" ? 1 : 3)}
          {unit}
        </strong>
      </div>
      <div className="bar-track">
        <div className={`bar-fill ${danger ? "danger" : ""}`} style={{ width: `${pct * 100}%` }} />
      </div>
    </div>
  );
}

export default function MetricsPanel({ prediction, config }) {
  const fusion = Number(prediction.smoothed_score ?? prediction.fusion_score ?? 0);
  const level = prediction.drowsiness_level ?? "LOW";
  const fusionClass = fusion > config.alert_threshold ? "danger" : fusion > config.alert_threshold * 0.65 ? "warn" : "safe";

  return (
    <section className="metrics-panel">
      <div className="fusion-readout">
        <span>FUSION SCORE</span>
        <em className={`level-chip ${level.toLowerCase()}`}>{level}</em>
        <strong className={fusionClass}>{fusion.toFixed(2)}</strong>
        <div className="bar-track large">
          <div className={`bar-fill ${fusionClass}`} style={{ width: `${clamp(fusion) * 100}%` }} />
        </div>
        {prediction.rule_alert && <p className="rule-note">Rule alert active. Model score unchanged.</p>}
      </div>
      <MetricBar label="EAR" value={Number(prediction.ear ?? 0)} min={0.1} max={0.45} dangerZone={(v) => v < config.ear_threshold} />
      <MetricBar label="MAR" value={Number(prediction.mar ?? 0)} min={0.2} max={0.9} dangerZone={(v) => v > config.mar_threshold} />
      <MetricBar label="PITCH" value={Math.abs(Number(prediction.head_pitch ?? 0))} min={0} max={45} dangerZone={(v) => v > config.pitch_threshold} unit=" deg" />
      <MetricBar label="CNN" value={Number(prediction.cnn_prob ?? 0)} min={0} max={1} dangerZone={(v) => v > config.alert_threshold} />
      <MetricBar label="FPS" value={Number(prediction.fps ?? 0)} min={0} max={30} dangerZone={() => false} />
    </section>
  );
}
