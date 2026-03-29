# server/Dockerfile - PFedRec Server
# Python 3.12 compatible, pre-built PyTorch wheels

FROM python:3.12-slim-bookworm

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies (minimal, no gcc needed for pre-built wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install PyTorch CPU version (pre-built wheel - NO compilation needed)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu --no-cache-dir

# Copy server code
COPY server/ ./server/
COPY utils/ ./utils/
COPY config.yaml .

# Create non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run with gunicorn (production WSGI server)
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "4", \
     "--timeout", "120", "--access-logfile", "-", \
     "server.main:app"]