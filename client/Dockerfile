FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install Python dependencies (torch is already available in base image)
COPY requirements.txt .
RUN pip install --no-cache-dir --retries 20 --timeout 120 -r requirements.txt

# Copy client code
COPY client/ ./client/
COPY utils/ ./utils/
COPY data/ ./data/
COPY config.yaml .

# Create non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8001

# Run client
CMD ["python", "-u", "client/main.py"]