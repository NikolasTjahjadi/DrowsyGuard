from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from inference import DrowsyGuardInference


MODEL_DIR = Path(__file__).resolve().parents[1] / "modeling_result"
detector = DrowsyGuardInference(MODEL_DIR)


@asynccontextmanager
async def lifespan(app: FastAPI):
    detector.load()
    yield


app = FastAPI(title="DrowsyGuard API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_origin_regex=r"https://.*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ConfigUpdate(BaseModel):
    ear_threshold: float | None = Field(default=None, ge=0.15, le=0.35)
    mar_threshold: float | None = Field(default=None, ge=0.40, le=0.80)
    pitch_threshold: float | None = Field(default=None, ge=15.0, le=45.0)
    alert_threshold: float | None = Field(default=None, ge=0.25, le=0.80)


@app.get("/api/health")
async def health() -> dict:
    return {"status": "ok", "model_loaded": detector.model_loaded}


@app.get("/api/config")
async def get_config() -> dict:
    return detector.get_runtime_config()


@app.post("/api/config")
async def update_config(payload: ConfigUpdate) -> dict:
    return detector.update_thresholds(
        ear_threshold=payload.ear_threshold,
        mar_threshold=payload.mar_threshold,
        pitch_threshold=payload.pitch_threshold,
        alert_threshold=payload.alert_threshold,
    )


@app.post("/api/calibrate/head")
async def calibrate_head() -> dict:
    return detector.reset_head_calibration()


@app.post("/api/predict")
async def predict(frame: UploadFile = File(...)) -> dict:
    try:
        image_bytes = await frame.read()
        return detector.predict_bytes(image_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc
