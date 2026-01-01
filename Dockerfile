FROM python:3.11-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl \
    libgl1 libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

RUN python -m venv /app/.venv
COPY requirements.txt ./
RUN /app/.venv/bin/pip install --no-cache-dir -U pip setuptools wheel
RUN /app/.venv/bin/pip install --no-cache-dir -r requirements.txt

FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/.venv /app/.venv
COPY . .

ENV PATH="/app/.venv/bin:$PATH"

CMD ["python", "subject_extractor.py"]
