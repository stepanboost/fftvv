#!/bin/bash
# Дневная запись: камеры → ID → data/<день>/. Запускается launchd, перезапускается при падении.
cd "$(dirname "$0")/.."
set -a; source .env; set +a
export OPENCV_FFMPEG_CAPTURE_OPTIONS="rtsp_transport;tcp"
exec .venv/bin/python -m salon_analytics.pipeline.live --config config/salon.yaml \
  --nvr "$NVR_HOST" --data data --models models --device mps
