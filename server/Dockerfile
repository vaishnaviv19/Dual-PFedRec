# client/Dockerfile - PFedRec Client
# Identical to server for consistency

FROM python:3.12-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install PyTorch CPU version (pre-built wheel)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu --no-cache-dir

# Copy client code
COPY client/ ./client/
COPY utils/ ./utils/
COPY data/ ./data/
COPY config.yaml .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port (for client API endpoints)
EXPOSE 8001

# Run client
CMD ["python", "-u", "client/main.py"]