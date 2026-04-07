# OcuSmart Distance API — FastAPI + uvicorn
FROM python:3.12-slim-bookworm

WORKDIR /app

# OpenCV headless wheels occasionally need a few runtime libs on Debian slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

# Hosts like Railway/Render set PORT; default 8000 for local docker run
CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
