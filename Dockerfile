FROM python:3.11-slim

WORKDIR /app

# System libs required by OpenCV headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
# Strip Windows CRLF — otherwise the kernel reports "no such file or directory"
# when exec'ing the shebang (looks for /bin/sh\r). See Render troubleshooting deploys.
RUN sed -i 's/\r$//' /app/docker-entrypoint.sh && chmod +x /app/docker-entrypoint.sh

EXPOSE 8000
# Invoke via sh so a bad shebang line cannot break startup after sed fix above.
ENTRYPOINT ["/bin/sh", "/app/docker-entrypoint.sh"]
# Many hosts set PORT via env; default 8000 when unset.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
