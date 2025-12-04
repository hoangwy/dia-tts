# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:12.6.0-base-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    CUDA_HOME=/usr/local/cuda

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3-pip \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Clone the dia repository
RUN git clone https://github.com/nari-labs/dia.git .

# Create and activate virtual environment
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip
RUN pip install --upgrade pip

# Install PyTorch with CUDA support first
RUN pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126

# Install triton for Linux
RUN pip install triton==3.2.0

# Install the dia package and its dependencies
RUN pip install -e .

# Install FastAPI dependencies
RUN pip install fastapi==0.115.6 uvicorn[standard]==0.34.0 python-multipart==0.0.20

# Expose port for Gradio app (7860) and FastAPI (8000)
EXPOSE 7860
EXPOSE 8000

# Copy API server code
COPY api.py .

# Default command to run the simple example
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
