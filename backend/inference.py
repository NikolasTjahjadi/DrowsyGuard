import json
import pickle
import time
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tensorflow as tf
from mediapipe import Image, ImageFormat
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from smoother import TemporalSmoother


LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
MOUTH_TOP = 13
MOUTH_BOTTOM = 14
MOUTH_TOP_INNER = 81
MOUTH_BOTTOM_INNER = 311

PNP_LANDMARKS = [1, 152, 33, 263, 61, 291]
FACE_3D_MODEL = np.array(
    [
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0),
    ],
    dtype=np.float64,
)


class DrowsyGuardInference:
    def __init__(self, model_dir: str | Path) -> None:
        self.model_dir = Path(model_dir)
        self.cnn_model: Any | None = None
        self.lr_meta: Any | None = None
        self.config: dict[str, Any] = {}
        self.face_landmarker: Any | None = None
        self.smoother = TemporalSmoother()
        self.perclos_samples: deque[tuple[float, bool]] = deque()
        self.pitch_calibration_samples: deque[float] = deque(maxlen=45)
        self.pitch_baseline: float | None = None
        self.ear_threshold = 0.25
        self.mar_threshold = 0.60
        self.pitch_threshold = 25.0
        self.alert_threshold = 0.25

    def load(self) -> None:
        config_path = self.model_dir / "drowsyguard_inference_config.json"
        with config_path.open("r", encoding="utf-8") as f:
            self.config = json.load(f)

        fusion_cfg = self.config.get("fusion", {})
        smoother_cfg = self.config.get("temporal_smoother", {})
        self.alert_threshold = float(
            smoother_cfg.get("alert_threshold", fusion_cfg.get("alert_threshold", 0.25))
        )
        self.smoother = TemporalSmoother(
            window_size=int(smoother_cfg.get("window_size", 5)),
            alert_threshold=self.alert_threshold,
        )

        thresholds = self.config.get("thresholds", {})
        self.ear_threshold = float(thresholds.get("ear", self.ear_threshold))
        self.mar_threshold = float(thresholds.get("mar", self.mar_threshold))
        self.pitch_threshold = float(thresholds.get("pitch", self.pitch_threshold))

        self.cnn_model = tf.keras.models.load_model(
            self.model_dir / "drowsyguard_best.h5", compile=False
        )
        with (self.model_dir / "drowsyguard_lr_fusion.pkl").open("rb") as f:
            self.lr_meta = pickle.load(f)

        task_path = Path(__file__).resolve().parent / "models" / "face_landmarker.task"
        if not task_path.exists():
            raise FileNotFoundError(
                "Missing face_landmarker.task. Download the MediaPipe task model into backend/models."
            )

        base_options = python.BaseOptions(model_asset_path=str(task_path))
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self.face_landmarker = vision.FaceLandmarker.create_from_options(options)

    @property
    def model_loaded(self) -> bool:
        return self.cnn_model is not None and self.lr_meta is not None and self.face_landmarker is not None

    def get_runtime_config(self) -> dict[str, Any]:
        return {
            "ear_threshold": self.ear_threshold,
            "mar_threshold": self.mar_threshold,
            "pitch_threshold": self.pitch_threshold,
            "head_pitch_baseline": self.pitch_baseline,
            "alert_threshold": self.alert_threshold,
            "raw_config": self.config,
        }

    def update_thresholds(
        self,
        ear_threshold: float | None = None,
        mar_threshold: float | None = None,
        pitch_threshold: float | None = None,
        alert_threshold: float | None = None,
    ) -> dict[str, Any]:
        if ear_threshold is not None:
            self.ear_threshold = float(ear_threshold)
        if mar_threshold is not None:
            self.mar_threshold = float(mar_threshold)
        if pitch_threshold is not None:
            self.pitch_threshold = float(pitch_threshold)
        if alert_threshold is not None:
            self.alert_threshold = float(alert_threshold)
            self.smoother.update_threshold(self.alert_threshold)
        return self.get_runtime_config()

    def reset_head_calibration(self) -> dict[str, Any]:
        self.pitch_calibration_samples.clear()
        self.pitch_baseline = None
        return self.get_runtime_config()

    def predict_bytes(self, image_bytes: bytes) -> dict[str, Any]:
        started = time.perf_counter()
        np_bytes = np.frombuffer(image_bytes, np.uint8)
        frame_bgr = cv2.imdecode(np_bytes, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            raise ValueError("Unable to decode frame as an image.")
        result = self.predict_frame(frame_bgr)
        elapsed = max(time.perf_counter() - started, 1e-6)
        result["fps"] = round(1.0 / elapsed, 2)
        return result

    def predict_frame(self, frame_bgr: np.ndarray) -> dict[str, Any]:
        if not self.model_loaded:
            raise RuntimeError("DrowsyGuard models are not loaded.")

        height, width = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=frame_rgb)
        detection = self.face_landmarker.detect(mp_image)

        if not detection.face_landmarks:
            return {
                "landmarks_detected": False,
                "alert": False,
                "drowsiness_level": "NO_FACE",
                "fps": 0.0,
            }

        landmarks = detection.face_landmarks[0]
        points = np.array([(lm.x * width, lm.y * height, lm.z) for lm in landmarks], dtype=np.float64)
        xy_points = points[:, :2]

        ear = self._eye_aspect_ratio(xy_points)
        mar = self._mouth_aspect_ratio(xy_points)
        raw_pitch, yaw, roll = self._head_pose(xy_points, width, height)
        face_box = self._face_box(xy_points, width, height)
        cnn_prob = self._cnn_probability(frame_rgb, face_box, width, height)
        fusion_score = self._fusion_score(cnn_prob, ear, mar, raw_pitch)
        smooth = self.smoother.step(fusion_score)

        eye_closed = ear < self.ear_threshold
        yawning = mar > self.mar_threshold
        relative_pitch, head_calibrating = self._relative_pitch(raw_pitch, eye_closed, yawning)
        head_nodding = (not head_calibrating) and abs(relative_pitch) > self.pitch_threshold
        perclos = self._update_perclos(eye_closed)
        model_alert = bool(smooth["alert"])
        rule_alert = bool(eye_closed or yawning or head_nodding)
        alert = bool(model_alert or rule_alert)
        alert_sources = self._alert_sources(model_alert, eye_closed, yawning, head_nodding)

        return {
            "ear": round(float(ear), 4),
            "mar": round(float(mar), 4),
            "head_pitch": round(float(relative_pitch), 2),
            "raw_head_pitch": round(float(raw_pitch), 2),
            "head_pitch_baseline": None if self.pitch_baseline is None else round(float(self.pitch_baseline), 2),
            "head_calibrating": bool(head_calibrating),
            "head_yaw": round(float(yaw), 2),
            "head_roll": round(float(roll), 2),
            "cnn_prob": round(float(cnn_prob), 4),
            "fusion_score": round(float(fusion_score), 4),
            "smoothed_score": round(float(smooth["smoothed"]), 4),
            "model_alert": model_alert,
            "rule_alert": rule_alert,
            "alert_sources": alert_sources,
            "alert": alert,
            "drowsiness_level": self._level(smooth["smoothed"], model_alert, eye_closed, yawning, head_nodding, perclos),
            "eye_closed": bool(eye_closed),
            "yawning": bool(yawning),
            "head_nodding": bool(head_nodding),
            "landmarks_detected": True,
            "perclos": round(float(perclos), 4),
            "face_box": face_box,
            "eye_landmarks": self._normalized_eye_landmarks(xy_points, width, height),
        }

    def _eye_aspect_ratio(self, xy_points: np.ndarray) -> float:
        def ear_for(indices: list[int]) -> float:
            p1, p2, p3, p4, p5, p6 = [xy_points[i] for i in indices]
            vertical = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
            horizontal = 2.0 * np.linalg.norm(p1 - p4)
            return float(vertical / horizontal) if horizontal else 0.0

        return (ear_for(LEFT_EYE) + ear_for(RIGHT_EYE)) / 2.0

    def _mouth_aspect_ratio(self, xy_points: np.ndarray) -> float:
        horizontal = np.linalg.norm(xy_points[MOUTH_LEFT] - xy_points[MOUTH_RIGHT])
        vertical_outer = np.linalg.norm(xy_points[MOUTH_TOP] - xy_points[MOUTH_BOTTOM])
        vertical_inner = np.linalg.norm(xy_points[MOUTH_TOP_INNER] - xy_points[MOUTH_BOTTOM_INNER])
        vertical = (vertical_outer + vertical_inner) / 2.0
        return float(vertical / horizontal) if horizontal else 0.0

    def _head_pose(self, xy_points: np.ndarray, width: int, height: int) -> tuple[float, float, float]:
        image_points = np.array([xy_points[i] for i in PNP_LANDMARKS], dtype=np.float64)
        focal_length = float(width)
        camera_matrix = np.array(
            [[focal_length, 0.0, width / 2.0], [0.0, focal_length, height / 2.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)
        ok, rvec, _ = cv2.solvePnP(
            FACE_3D_MODEL,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return 0.0, 0.0, 0.0
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)
        pitch, yaw, roll = [self._wrap_angle(float(angle)) for angle in angles]
        return pitch, yaw, roll

    def _wrap_angle(self, angle: float) -> float:
        return ((angle + 180.0) % 360.0) - 180.0

    def _face_box(self, xy_points: np.ndarray, width: int, height: int) -> dict[str, float]:
        min_xy = np.maximum(np.min(xy_points, axis=0), [0, 0])
        max_xy = np.minimum(np.max(xy_points, axis=0), [width, height])
        pad_x = (max_xy[0] - min_xy[0]) * 0.18
        pad_y = (max_xy[1] - min_xy[1]) * 0.22
        x1 = max(0.0, min_xy[0] - pad_x)
        y1 = max(0.0, min_xy[1] - pad_y)
        x2 = min(float(width), max_xy[0] + pad_x)
        y2 = min(float(height), max_xy[1] + pad_y)
        return {"x": x1 / width, "y": y1 / height, "w": (x2 - x1) / width, "h": (y2 - y1) / height}

    def _cnn_probability(self, frame_rgb: np.ndarray, face_box: dict[str, float], width: int, height: int) -> float:
        x1 = int(face_box["x"] * width)
        y1 = int(face_box["y"] * height)
        x2 = int((face_box["x"] + face_box["w"]) * width)
        y2 = int((face_box["y"] + face_box["h"]) * height)
        crop = frame_rgb[max(0, y1) : min(height, y2), max(0, x1) : min(width, x2)]
        if crop.size == 0:
            crop = frame_rgb
        resized = cv2.resize(crop, (128, 128), interpolation=cv2.INTER_AREA)
        batch = np.expand_dims(resized.astype(np.float32) / 255.0, axis=0)
        prediction = self.cnn_model.predict(batch, verbose=0)
        return float(np.ravel(prediction)[0])

    def _fusion_score(self, cnn_prob: float, ear: float, mar: float, pitch: float) -> float:
        features = {
            "cnn_prob": float(cnn_prob),
            "ear_norm": self._normalize("ear", ear),
            "ear_inv_norm": self._normalize("ear", ear, invert=True),
            "mar_norm": self._normalize("mar", mar),
            "pitch_norm": self._normalize("pitch", pitch),
            "head_pitch_norm": self._normalize("pitch", pitch),
        }
        order = self.config.get("fusion", {}).get(
            "feature_order", ["cnn_prob", "ear_inv_norm", "mar_norm", "pitch_norm"]
        )
        vector = np.array([[features.get(name, 0.0) for name in order]], dtype=np.float64)
        return float(self.lr_meta.predict_proba(vector)[0, 1])

    def _normalize(self, key: str, value: float, invert: bool = False) -> float:
        scaler = self.config.get("scaler", {})
        if f"{key}_mean" in scaler and f"{key}_std" in scaler:
            std = float(scaler.get(f"{key}_std") or 1.0)
            normalized = (float(value) - float(scaler[f"{key}_mean"])) / std
            return -normalized if invert else normalized

        cfg = self.config.get("fusion", {}).get("scalers", {}).get(key, {})
        min_value = float(cfg.get("min", 0.0))
        max_value = float(cfg.get("max", 1.0))
        denom = max(max_value - min_value, 1e-9)
        normalized = (float(value) - min_value) / denom
        normalized = float(np.clip(normalized, 0.0, 1.0))
        return 1.0 - normalized if invert else normalized

    def _update_perclos(self, eye_closed: bool) -> float:
        now = time.monotonic()
        self.perclos_samples.append((now, bool(eye_closed)))
        while self.perclos_samples and now - self.perclos_samples[0][0] > 30.0:
            self.perclos_samples.popleft()
        if not self.perclos_samples:
            return 0.0
        closed = sum(1 for _, closed_now in self.perclos_samples if closed_now)
        return closed / len(self.perclos_samples)

    def _relative_pitch(self, raw_pitch: float, eye_closed: bool, yawning: bool) -> tuple[float, bool]:
        if self.pitch_baseline is None and not eye_closed and not yawning:
            self.pitch_calibration_samples.append(float(raw_pitch))
            if len(self.pitch_calibration_samples) >= 18:
                self.pitch_baseline = float(np.median(self.pitch_calibration_samples))

        if self.pitch_baseline is None:
            return 0.0, True

        return self._wrap_angle(float(raw_pitch - self.pitch_baseline)), False

    def _normalized_eye_landmarks(self, xy_points: np.ndarray, width: int, height: int) -> list[dict[str, float]]:
        indices = LEFT_EYE + RIGHT_EYE
        return [{"x": float(xy_points[i][0] / width), "y": float(xy_points[i][1] / height)} for i in indices]

    def _alert_sources(
        self,
        model_alert: bool,
        eye_closed: bool,
        yawning: bool,
        head_nodding: bool,
    ) -> list[str]:
        sources = []
        if model_alert:
            sources.append("MODEL")
        if eye_closed:
            sources.append("EYE_CLOSED")
        if yawning:
            sources.append("YAWNING")
        if head_nodding:
            sources.append("HEAD_NOD")
        return sources

    def _level(
        self,
        score: float,
        model_alert: bool,
        eye_closed: bool,
        yawning: bool,
        head_nodding: bool,
        perclos: float,
    ) -> str:
        if model_alert and (eye_closed or head_nodding or perclos >= 0.25):
            return "CRITICAL"
        if eye_closed or head_nodding or model_alert or score >= self.alert_threshold:
            return "HIGH"
        if yawning or perclos >= 0.15:
            return "MEDIUM"
        if score >= self.alert_threshold * 0.65:
            return "MEDIUM"
        return "LOW"
