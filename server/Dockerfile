# server/Dockerfile - PFedRec Server
# Python 3.12 compatible, pre-built PyTorch wheels

FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install Python dependencies (torch is already available in base image)
COPY requirements.txt .
RUN pip install --no-cache-dir --retries 20 --timeout 120 -r requirements.txt

# Copy server code
COPY server/ ./server/
COPY utils/ ./utils/
COPY config.yaml .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Health check endpoint (without curl dependency)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=5)" || exit 1

# Expose port
EXPOSE 8000

# Run server with gunicorn
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--workers", "4", "--timeout", "120", "--access-logfile", "-", "server.main:app"]