FROM nvidia/vulkan:1.3-470

LABEL org.opencontainers.image.source=https://github.com/ls-ads/runpod-real-esrgan

# Build-time dependencies
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
RUN apt-get update && apt-get install -y \
    wget \
    unzip \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv for Python management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Download and extract Real-ESRGAN NCNN Vulkan
RUN wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-ubuntu.zip && \
    unzip realesrgan-ncnn-vulkan-20220424-ubuntu.zip && \
    rm realesrgan-ncnn-vulkan-20220424-ubuntu.zip && \
    chmod +x realesrgan-ncnn-vulkan

# Set environment variable for binary path
ENV REAL_ESRGAN_BIN=/app/realesrgan-ncnn-vulkan

# NVIDIA Driver Capabilities
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all

# Copy dependency files
COPY pyproject.toml uv.lock .python-version ./

# Install Python using uv
RUN uv sync --no-dev --no-cache

# Copy source code
COPY src/*.py ./

# Start the handler
CMD [ "uv", "run", "python", "-u", "handler.py" ]