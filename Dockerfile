FROM python:3.9-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app:/app/src \
    REFINED_DATA_DIR=/data/refined \
    MODEL_NAME=wikipedia_model \
    ENTITY_SET=wikipedia \
    USE_PRECOMPUTED_DESCRIPTIONS=true \
    REFINED_DEVICE=cpu

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-inference.txt ./
RUN pip install --no-cache-dir torch==2.2.2+cpu --extra-index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements-inference.txt

COPY src/ ./src/
COPY server/ ./server/

RUN useradd --create-home appuser \
    && mkdir -p /data/refined \
    && chown -R appuser:appuser /data /app

USER appuser

EXPOSE 8000

CMD ["uvicorn", "server.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
