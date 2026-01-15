# GPU Mesher Builder - Docker Image for Cloud Deployments
# Builds _gpumesher.so for Linux-based cloud environments (AWS, Modal, etc.)

FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

# Prevent interactive timezone prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install build dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    git \
    python3.10 \
    python3.10-dev \
    python3-pip \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Install Python build dependencies
RUN pip3 install --no-cache-dir pybind11

# Copy GPU mesher source
WORKDIR /build/gpu_experiments
COPY gpu_experiments/ ./

# Build the module
RUN mkdir -p build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    cmake --build . --target _gpumesher && \
    # Copy to output (replace .pyd with .so for Linux)
    mkdir -p /output && \
    cp Release/_gpumesher.so /output/ || cp _gpumesher.so /output/

# Final stage: minimal runtime image
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3.10 \
    && rm -rf /var/lib/apt/lists/*

# Copy built module
COPY --from=0 /output/_gpumesher.so /app/core/

WORKDIR /app

# Usage: Mount your mesh package and run
# docker run --gpus all -v $(pwd):/app gpu-mesher-build python apps/cli/...
