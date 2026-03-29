FROM python:3.12-slim

WORKDIR /app

# Install build dependencies for compiling C extensions if needed
RUN apt-get update && apt-get install -y gcc g++ libffi-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy config, models, data, and client source
COPY config.yaml .
COPY src/ ./src/

EXPOSE 8001

ENV PYTHONPATH=/app
CMD ["uvicorn", "src.client.main:app", "--host", "0.0.0.0", "--port", "8001"]
