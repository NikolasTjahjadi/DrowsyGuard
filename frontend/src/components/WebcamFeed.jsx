import { useEffect, useRef, useState } from "react";
import { Camera, WifiOff } from "lucide-react";

export default function WebcamFeed({ apiBase, prediction, onPrediction }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const overlayRef = useRef(null);
  const inflight = useRef(false);
  const [cameraState, setCameraState] = useState("starting");
  const [apiState, setApiState] = useState("idle");

  useEffect(() => {
    let stream;
    navigator.mediaDevices
      .getUserMedia({ video: { width: 1280, height: 720, facingMode: "user" }, audio: false })
      .then((mediaStream) => {
        stream = mediaStream;
        videoRef.current.srcObject = mediaStream;
        setCameraState("ready");
      })
      .catch(() => setCameraState("blocked"));
    return () => stream?.getTracks().forEach((track) => track.stop());
  }, []);

  useEffect(() => {
    let frameId;
    const canvas = canvasRef.current;
    const video = videoRef.current;

    const sendFrame = () => {
      if (video?.readyState >= 2 && !inflight.current) {
        inflight.current = true;
        canvas.width = video.videoWidth || 640;
        canvas.height = video.videoHeight || 360;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        canvas.toBlob(
          async (blob) => {
            if (!blob) {
              inflight.current = false;
              frameId = requestAnimationFrame(sendFrame);
              return;
            }
            const formData = new FormData();
            formData.append("frame", blob, "frame.jpg");
            try {
              const res = await fetch(`${apiBase}/api/predict`, { method: "POST", body: formData });
              if (!res.ok) throw new Error("predict failed");
              onPrediction(await res.json());
              setApiState("online");
            } catch {
              setApiState("offline");
            } finally {
              inflight.current = false;
              frameId = requestAnimationFrame(sendFrame);
            }
          },
          "image/jpeg",
          0.76,
        );
      } else {
        frameId = requestAnimationFrame(sendFrame);
      }
    };

    frameId = requestAnimationFrame(sendFrame);
    return () => cancelAnimationFrame(frameId);
  }, [apiBase, onPrediction]);

  useEffect(() => {
    const canvas = overlayRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;
    const rect = video.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.lineWidth = 2;
    ctx.font = "12px JetBrains Mono, monospace";

    if (prediction.landmarks_detected && prediction.face_box) {
      const { x, y, w, h } = prediction.face_box;
      ctx.strokeStyle = prediction.alert ? "#ef4444" : "#22c55e";
      ctx.shadowColor = ctx.strokeStyle;
      ctx.shadowBlur = 10;
      ctx.strokeRect(x * canvas.width, y * canvas.height, w * canvas.width, h * canvas.height);
      ctx.shadowBlur = 0;
      ctx.fillStyle = "rgba(10,10,15,0.72)";
      ctx.fillRect(x * canvas.width, y * canvas.height - 27, 156, 22);
      ctx.fillStyle = prediction.eye_closed ? "#ef4444" : "#22c55e";
      ctx.fillText(`EAR ${Number(prediction.ear ?? 0).toFixed(3)}`, x * canvas.width + 8, y * canvas.height - 11);

      ctx.fillStyle = "#f59e0b";
      for (const point of prediction.eye_landmarks ?? []) {
        ctx.beginPath();
        ctx.arc(point.x * canvas.width, point.y * canvas.height, 2.5, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }, [prediction]);

  return (
    <section className="feed-panel">
      <div className="feed-status">
        <span className={`live-dot ${apiState}`} />
        {apiState === "offline" ? "API OFFLINE" : "LIVE CAMERA"}
      </div>
      <video ref={videoRef} autoPlay playsInline muted />
      <canvas ref={overlayRef} className="overlay-canvas" />
      <canvas ref={canvasRef} className="hidden-canvas" />
      {cameraState === "blocked" && (
        <div className="feed-warning">
          <Camera size={26} />
          Camera permission required
        </div>
      )}
      {apiState === "offline" && (
        <div className="feed-warning lower">
          <WifiOff size={22} />
          Backend unavailable
        </div>
      )}
      {!prediction.landmarks_detected && cameraState === "ready" && (
        <div className="no-face">NO FACE DETECTED</div>
      )}
      {prediction.head_calibrating && prediction.landmarks_detected && (
        <div className="no-face">CALIBRATING HEAD POSE</div>
      )}
      {prediction.alert && <div className="alert-flash" />}
    </section>
  );
}
