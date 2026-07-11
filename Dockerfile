# White Lights — one container serving the pages + the /ws/live WebSocket.
# Works locally and free on Hugging Face Spaces (Docker SDK, CPU basic).
FROM python:3.11-slim

# System libraries OpenCV (cv2) needs at import time.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces runs the container as uid 1000 — set up a writable home.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    YOLO_CONFIG_DIR=/home/user/.config/Ultralytics
WORKDIR /home/user/app

# Install the app + pose runtime (torch/opencv) + API server.
COPY --chown=user . .
RUN pip install --no-cache-dir --user -e ".[cv,api]"

# Bake the pose weights into the image so there is no download at runtime.
RUN python -c "from ultralytics import YOLO; YOLO('yolo11n-pose.pt')"

EXPOSE 7860
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
