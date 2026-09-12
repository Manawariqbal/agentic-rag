FROM python:3.11-slim

# ============================================================
# Environment
# ============================================================

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

# ============================================================
# System dependencies
# ============================================================

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libpq-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# ============================================================
# Application directory
# ============================================================

WORKDIR /app

# ============================================================
# Python dependencies
# ============================================================

COPY requirements.txt .

RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# ============================================================
# Application
# ============================================================

COPY app ./app
COPY documents ./documents

# ============================================================
# Runtime
# ============================================================

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]