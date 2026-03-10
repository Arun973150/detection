FROM python:3.11-slim

WORKDIR /app

# System dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev]"

# Copy source code
COPY config/ config/
COPY src/ src/

# Create model weights directory
RUN mkdir -p model_weights

EXPOSE 8000 7860

# Default: run FastAPI
CMD ["uvicorn", "detection.app:app", "--host", "0.0.0.0", "--port", "8000"]
