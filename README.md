# DrowsyGuard

Real-time driver fatigue detection using facial landmarks, head pose, CNN inference, logistic-regression fusion, and temporal smoothing.

This repository now contains both the trained model artifacts and a fullstack webcam dashboard.

## What It Detects

DrowsyGuard monitors the driver's face from a webcam and estimates:

- Eye closure using Eye Aspect Ratio (EAR)
- Yawning using Mouth Aspect Ratio (MAR)
- Head nodding using calibrated head-pose pitch
- CNN drowsiness probability from `drowsyguard_best.h5`
- Final model fusion score from `drowsyguard_lr_fusion.pkl`
- Alert levels: `LOW`, `MEDIUM`, `HIGH`, `CRITICAL`

The app keeps the model score honest: `fusion_score` remains the output of the trained CNN + LR model. Runtime rule alerts can still trigger sound/visual warnings when strong physical signals appear, such as eyes closed or head nodding.

## Repository Structure

```text
DrowsyGuard/
├── backend/
│   ├── main.py
│   ├── inference.py
│   ├── smoother.py
│   ├── requirements.txt
│   └── models/
│       └── face_landmarker.task
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   └── styles.css
│   ├── package.json
│   └── vite.config.js
├── modeling_result/
│   ├── drowsyguard_best.h5
│   ├── drowsyguard_lr_fusion.pkl
│   └── drowsyguard_inference_config.json
└── docker-compose.yml
```

## Model Artifacts

The backend loads the existing artifacts from `modeling_result/`:

- `drowsyguard_best.h5` - fine-tuned MobileNetV2 CNN
- `drowsyguard_lr_fusion.pkl` - Logistic Regression meta-learner
- `drowsyguard_inference_config.json` - scaler and smoother config

MediaPipe FaceLandmarker uses the task asset at:

```text
backend/models/face_landmarker.task
```

## Backend Setup

Use Python 3.10 or 3.11. TensorFlow on Windows may fail on Python 3.13.

```powershell
cd backend
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Health check:

```text
http://localhost:8000/api/health
```

## Frontend Setup

```powershell
cd frontend
npm.cmd install
npm.cmd run dev
```

Open:

```text
http://localhost:3000
```

Allow camera permission. The webcam is used to monitor the driver's face, not to detect cars.

## API

### `GET /api/health`

```json
{
  "status": "ok",
  "model_loaded": true
}
```

### `GET /api/config`

Returns runtime thresholds and raw inference config.

### `POST /api/config`

Updates app-side thresholds at runtime:

```json
{
  "ear_threshold": 0.25,
  "mar_threshold": 0.6,
  "pitch_threshold": 25,
  "alert_threshold": 0.25
}
```

These thresholds do not retrain or modify the model files.

### `POST /api/calibrate/head`

Resets neutral head-pose calibration. Face the camera normally for 1-2 seconds after calling it.

### `POST /api/predict`

Accepts multipart form data:

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
  "head_yaw": -3.1,
  "head_roll": 1.2,
  "cnn_prob": 0.87,
  "fusion_score": 0.74,
  "smoothed_score": 0.68,
  "model_alert": true,
  "rule_alert": true,
  "alert": true,
  "alert_sources": ["MODEL", "EYE_CLOSED"],
  "drowsiness_level": "CRITICAL",
  "eye_closed": true,
  "yawning": false,
  "head_nodding": false,
  "landmarks_detected": true,
  "fps": 28.5
}
```

## Alert Logic

- `MEDIUM`: yawning, elevated PERCLOS, or warning-level fusion score
- `HIGH`: eyes closed, head nodding, or model alert
- `CRITICAL`: model alert plus a strong physical signal such as eyes closed, head nodding, or high PERCLOS

## Notes

- The backend uses MediaPipe Tasks API `FaceLandmarker` in `IMAGE` mode.
- The frontend sends a new frame only after the previous prediction returns, so it naturally throttles to backend speed.
- Head nod detection is calibrated against the user's neutral head pose to reduce false positives from webcam angle.
