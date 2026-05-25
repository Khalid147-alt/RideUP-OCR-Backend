# syntax=docker/dockerfile:1.6
#
# RideUP OCR Backend — production image for HuggingFace Spaces.
#
# Build:   docker build -t rideup-ocr-backend .
# Run:     docker run -p 7860:7860 -e OPENAI_API_KEY=sk-... rideup-ocr-backend

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=7860

WORKDIR /app

# System deps — curl for HEALTHCHECK, libjpeg/zlib for Pillow runtime.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl libjpeg62-turbo zlib1g \
 && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first to maximise layer caching.
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Copy application source.
COPY . .

# Create a non-root user (HuggingFace Spaces convention).
RUN useradd --create-home --uid 1000 appuser \
 && chown -R appuser:appuser /app
USER appuser

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD curl -fsS http://localhost:7860/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
