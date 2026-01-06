FROM python:3.12-slim

# Avoid buffering so logs show up immediately
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first (better Docker layer caching)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY app.py /app/app.py
COPY templates /app/templates

# Create data dir inside container (we’ll mount a volume to persist)
RUN mkdir -p /app/data

# Flask host/port defaults (compose will override if desired)
ENV FLASK_HOST=0.0.0.0
ENV FLASK_PORT=8080

EXPOSE 8080

CMD ["python", "app.py"]
