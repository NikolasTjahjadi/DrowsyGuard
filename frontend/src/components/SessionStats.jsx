import { Area, AreaChart, ResponsiveContainer, Tooltip, YAxis } from "recharts";

export default function SessionStats({ session, history }) {
  const data = history.map((point) => ({
    time: new Date(point.time).toLocaleTimeString(),
    score: Number(point.score.toFixed(3)),
  }));

  return (
    <section className="session-panel">
      <div className="stat-strip">
        <div>
          <span>ALERTS</span>
          <strong>{session.alerts}</strong>
        </div>
        <div>
          <span>DROWSY TIME</span>
          <strong>{session.drowsyPercent.toFixed(1)}%</strong>
        </div>
        <div>
          <span>ALERTS/HOUR</span>
          <strong>{session.alertsPerHour.toFixed(1)}</strong>
        </div>
      </div>
      <div className="sparkline">
        <ResponsiveContainer width="100%" height={118}>
          <AreaChart data={data}>
            <defs>
              <linearGradient id="scoreFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#f59e0b" stopOpacity={0.55} />
                <stop offset="100%" stopColor="#f59e0b" stopOpacity={0.04} />
              </linearGradient>
            </defs>
            <YAxis domain={[0, 1]} hide />
            <Tooltip contentStyle={{ background: "#111118", border: "1px solid #2b2b35", color: "#f7f7fb" }} />
            <Area type="monotone" dataKey="score" stroke="#f59e0b" strokeWidth={2} fill="url(#scoreFill)" isAnimationActive={false} />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}
