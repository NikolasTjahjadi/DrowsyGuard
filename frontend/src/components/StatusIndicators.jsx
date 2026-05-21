import { Eye, Ruler, Wind } from "lucide-react";

export default function StatusIndicators({ prediction }) {
  const items = [
    { label: "EYE CLOSED", active: prediction.eye_closed, className: "red", Icon: Eye },
    { label: "YAWNING", active: prediction.yawning, className: "orange", Icon: Wind },
    { label: "HEAD NOD", active: prediction.head_nodding, className: "yellow", Icon: Ruler },
  ];

  return (
    <section className="status-panel">
      {items.map(({ label, active, className, Icon }) => (
        <div key={label} className={`status-pill ${className} ${active ? "active" : ""}`}>
          <Icon size={17} />
          {label}
        </div>
      ))}
    </section>
  );
}
