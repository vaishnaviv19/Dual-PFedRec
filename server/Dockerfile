FROM python:3.12-slim

WORKDIR /app

# Install system dependencies (minimal for slim image)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install PyTorch CPU version (Python 3.12 compatible)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu --no-cache-dir

# Copy server code
COPY server/app.py ./app.py
COPY server/aggregator.py ./aggregator.py
COPY server/config.py ./config.py

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "app:app"]