# DrowsyGuard

Real-time driver fatigue detection using MediaPipe facial landmark extraction, geometric feature analysis (EAR, MAR, head pose via PnP), MobileNetV2 CNN inference, Logistic Regression fusion, and temporal smoothing.

## What It Detects

DrowsyGuard monitors the driver's face from a webcam and estimates:

- **Eye closure** — Eye Aspect Ratio (EAR), threshold < 0.15 sustained ≥3 frames
- **Yawning** — Mouth Aspect Ratio (MAR), threshold > 0.60
- **Head nodding** — calibrated pitch deviation > 25° from per-session neutral baseline
- **CNN drowsiness probability** — MobileNetV2 (`drowsyguard_best.h5`)
- **Fusion score** — Logistic Regression meta-learner (`drowsyguard_lr_fusion.pkl`) combining CNN probability, inverted EAR, MAR, and head pitch
- **PERCLOS** — percentage of eye closure over a 30-second rolling window, used as a severity indicator only
- **Alert levels** — `LOW`, `MEDIUM`, `HIGH`, `CRITICAL`

### Alert Logic

The final alert is determined by the union of two independent signals:
- `model_alert` — smoothed LR fusion score > 0.25 (window=5 frames)
- `rule_alert` — sustained eye closure (EAR < 0.15 for ≥3 consecutive frames), yawning, or head nodding

| Level | Condition |
|-------|-----------|
| `LOW` | Fusion score below alert threshold, no physical signals |
| `MEDIUM` | Yawning detected, or PERCLOS ≥ 15%, or fusion score approaching threshold |
| `HIGH` | Eyes closed, head nodding, or model alert triggered |
| `CRITICAL` | Model alert AND at least one strong physical signal (eye closed, head nodding, or PERCLOS ≥ 25%) |

The rule-based layer uses deliberately stricter thresholds than the experimental baseline (EAR 0.15 vs 0.25, with 3-frame persistence) and serves as a fail-safe that complements rather than replaces the learned model.

## Repository Structure

```text
DrowsyGuard/
├── backend/
│   ├── main.py                          # FastAPI app, API routes
│   ├── inference.py                     # DrowsyGuardInference class (full pipeline)
│   ├── smoother.py                      # TemporalSmoother (window=5, threshold=0.25)
│   ├── requirements.txt
│   └── models/
│       └── face_landmarker.task         # MediaPipe FaceLandmarker task asset
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   └── styles.css
│   ├── package.json
│   └── vite.config.js
├── modeling_result/
│   ├── drowsyguard_best.h5              # Fine-tuned MobileNetV2 CNN
│   ├── drowsyguard_lr_fusion.pkl        # Trained LR meta-learner + scalers
│   └── drowsyguard_inference_config.json  # Scaler params, feature order, smoother config
└── docker-compose.yml
```

## Model Artifacts

The backend loads artifacts from `modeling_result/`:

- `drowsyguard_best.h5` — MobileNetV2 fine-tuned on NTHU-DDD (128×128 input, sigmoid output)
- `drowsyguard_lr_fusion.pkl` — LR meta-learner with coefficients: CNN_prob +6.503, EAR_inv +4.512, MAR_norm −3.516, Pitch_norm −0.541
- `drowsyguard_inference_config.json` — MinMax scaler params, feature order, temporal smoother settings

MediaPipe FaceLandmarker task asset:

```text
backend/models/face_landmarker.task
```

## Setup

### Option 1 — Docker (recommended)

```bash
docker compose up
```

Frontend: http://localhost:3000  
Backend: http://localhost:8000

### Option 2 — Manual

**Backend** (Python 3.10 or 3.11 required; TensorFlow may fail on Python 3.13)

```bash
cd backend
python -m venv .venv

# Windows
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn main:app --reload --host 0.0.0.0 --port 8000

# macOS / Linux
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Frontend**

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:3000 and allow camera permission.

## API

### `GET /api/health`

```json
{ "status": "ok", "model_loaded": true }
```

### `GET /api/config`

Returns runtime thresholds and raw inference config.

### `POST /api/config`

Updates runtime thresholds without retraining or modifying model files.

```json
{
  "ear_threshold": 0.15,
  "mar_threshold": 0.60,
  "pitch_threshold": 25.0,
  "alert_threshold": 0.25
}
```

Accepted ranges: EAR 0.15–0.35, MAR 0.40–0.80, Pitch 15–45°, Alert 0.25–0.80.

### `POST /api/calibrate/head`

Resets per-session head pitch calibration. Look directly at the camera for 1–2 seconds after calling. The system collects 18 frames to establish your neutral head position baseline before head nodding detection activates.

### `POST /api/predict`

Accepts a multipart JPEG webcam frame:

```text
frame=<jpeg webcam frame>
```

Returns:

```json
{
  "ear": 0.21,
  "mar": 0.45,
  "head_pitch": 12.3,
  "raw_head_pitch": -8.1,
  "head_pitch_baseline": -20.4,
  "head_calibrating": false,
  "head_yaw": -3.1,
  "head_roll": 1.2,
  "cnn_prob": 0.87,
  "fusion_score": 0.74,
  "smoothed_score": 0.68,
  "model_alert": true,
  "rule_alert": true,
  "alert_sources": ["MODEL", "EYE_CLOSED"],
  "alert": true,
  "drowsiness_level": "CRITICAL",
  "eye_closed": true,
  "eye_closed_sustained": true,
  "yawning": false,
  "head_nodding": false,
  "landmarks_detected": true,
  "perclos": 0.23,
  "face_box": { "x": 0.21, "y": 0.08, "w": 0.54, "h": 0.72 },
  "fps": 14.2
}
```

## Notes

- The frontend sends a new frame only after the previous prediction returns, naturally throttling to backend speed.
- Head pitch calibration collects 18 non-drowsy frames at session start. Call `POST /api/calibrate/head` to reset if your posture or camera angle changes.
- PERCLOS is computed over a 30-second rolling window and used only to determine drowsiness severity level — it does not directly trigger alerts.
- The backend serves the frontend static build from `frontend/dist/` when deployed as a single container.
