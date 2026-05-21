import { useEffect, useRef } from "react";
import { AlertTriangle } from "lucide-react";

function alertMessage(prediction) {
  const level = prediction?.drowsiness_level ?? "HIGH";
  const sources = prediction?.alert_sources ?? [];
  if (sources.includes("EYE_CLOSED")) return `${level} ALERT - Eyes closed`;
  if (sources.includes("HEAD_NOD")) return `${level} ALERT - Head nod detected`;
  if (sources.includes("YAWNING")) return `${level} ALERT - Yawning detected`;
  if (sources.includes("MODEL")) return `${level} ALERT - Drowsiness detected`;
  return `${level} ALERT - Please take a break`;
}

export default function AlertBanner({ active, prediction, soundEnabled }) {
  const audioRef = useRef(null);

  useEffect(() => {
    if (!active || !soundEnabled) return;
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    if (!AudioContext) return;
    const ctx = audioRef.current ?? new AudioContext();
    audioRef.current = ctx;
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = "square";
    osc.frequency.setValueAtTime(880, ctx.currentTime);
    gain.gain.setValueAtTime(0.0001, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.12, ctx.currentTime + 0.02);
    gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + 0.35);
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start();
    osc.stop(ctx.currentTime + 0.38);
  }, [active, soundEnabled]);

  return (
    <div className={`alert-banner ${active ? "visible" : ""}`}>
      <AlertTriangle size={22} />
      {alertMessage(prediction)}
    </div>
  );
}
