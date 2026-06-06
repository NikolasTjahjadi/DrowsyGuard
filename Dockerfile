FROM python:3.11-slim

# System deps buat OpenCV (libgl) + MediaPipe + build frontend
RUN apt-get update && apt-get install -y --no-install-recommends     libgl1-mesa-glx     libglib2.0-0     nodejs     npm     && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy semua project files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r backend/requirements.txt

# Build frontend — VITE_API_BASE= biar pake relative path (/api/predict)
WORKDIR /app/frontend
RUN VITE_API_BASE= npm run build

WORKDIR /app

# HF Spaces default port
EXPOSE 7860

CMD [uvicorn, backend.main:app, --host, 0.0.0.0, --port, 7860]
