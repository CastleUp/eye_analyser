# Use NVIDIA CUDA base image
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
# Install torch with CUDA support explicitly
RUN pip3 install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121
# Install the rest
RUN pip3 install --no-cache-dir -r requirements.txt
# For Linux/Docker, we switch back to onnxruntime-gpu
RUN pip3 uninstall -y onnxruntime-directml && pip3 install onnxruntime-gpu

# Copy the rest of the project
COPY . .

# Default command (can be overridden)
CMD ["python3", "compare.py"]
