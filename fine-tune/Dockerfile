# Dockerfile for fine-tuning using ROCm PyTorch 2.0.1 (Python 3.10)
FROM rocm/pytorch:rocm5.7_ubuntu22.04_py3.10_pytorch_2.0.1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    cmake \
    cargo \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip/setuptools/wheel first to allow binary wheel resolution, then install Python dependencies
RUN pip install --upgrade pip setuptools wheel

# Install Python dependencies (do NOT reinstall torch from pip; base image includes ROCm PyTorch)
RUN pip install --no-cache-dir \
    transformers==4.33.3 \
    sentencepiece==0.1.99 \
    protobuf==4.25.1 \
    accelerate==0.25.0 \
    datasets==2.16.1 \
    scikit-learn==1.4.0

# Copy the fine-tuning script
COPY fine_tune.py .

# Create necessary directories
RUN mkdir -p /models /data

# Expose model/data mounts
VOLUME ["/models", "/data"]

# Run the fine-tuning script
CMD ["python", "fine_tune.py"]
