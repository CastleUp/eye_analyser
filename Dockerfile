# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies for OpenCV, MediaPipe, and InsightFace
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# Note: Using onnxruntime instead of onnxruntime-gpu for broader compatibility in Docker
# unless specific CUDA images are used.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Environment variable to help with GUI (if configured)
ENV DISPLAY=:0

# The application requires a webcam and GUI. 
# Docker containers don't have these by default.
# Instructions on how to run with GUI/Webcam are in README.md
CMD ["python", "main.py"]
