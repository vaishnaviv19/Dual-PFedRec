FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install PyTorch CPU version (Python 3.12 compatible)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu --no-cache-dir

# Copy client code
COPY client/app.py ./app.py
COPY client/model.py ./model.py
COPY client/config.py ./config.py

CMD ["python", "-u", "app.py"]