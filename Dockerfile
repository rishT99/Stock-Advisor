# ---------- Base ----------
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # yfinance/requests behave better with a desktop UA set in your code already,
    # but keeping TZ + certs sane helps inside slim images:
    TZ=UTC

# System deps kept minimal; add more only if you truly need them
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates tzdata curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---------- Python deps (cache-friendly) ----------
# If you have a requirements.txt, keep it in repo root (recommended).
COPY requirements.txt /app/requirements.txt

# Upgrade pip and install wheels
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# ---------- App code ----------
# Copy the rest of your repo (kept after deps for better layer caching)
COPY . /app

# Create a non-root user for better security
RUN useradd -m -u 10001 appuser && chown -R appuser /app
USER appuser

# Render will provide $PORT; default to 8000 for local runs
ENV PORT=8000

EXPOSE 8000

# ---------- Start ----------
# IMPORTANT: "app:app" points to `app.py`'s FastAPI instance named `app`
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT} --forwarded-allow-ips='*' --proxy-headers"]
