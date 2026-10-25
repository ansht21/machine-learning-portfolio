# Hardened multi-stage build for the retention gateway.
# Stage 1 compiles native deps (chromadb / sentence-transformers wheels),
# stage 2 runs as a non-root user with only the runtime pieces.
FROM python:3.10-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /wheels
COPY requirements-api.txt .
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements-api.txt

FROM python:3.10-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -u 8888 -m appuser

WORKDIR /app
COPY --from=builder /wheels /wheels
COPY requirements-api.txt .
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements-api.txt \
    && rm -rf /wheels

RUN mkdir -p /app/models /app/hf_cache && chown -R appuser:appuser /app
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser app/ ./app/
COPY --chown=appuser:appuser models/ ./models/
COPY --chown=appuser:appuser data/ ./data/

ENV HF_HOME=/app/hf_cache \
    PYTHONUNBUFFERED=1
USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
