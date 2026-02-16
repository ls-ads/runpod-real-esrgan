FROM runpod/base:1.0.3-cuda1300-ubuntu2404

# Build-time dependencies
RUN apt-get update && apt-get install -y \
    wget \
    unzip \
    libvulkan1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set environment variable for binary path
ENV REAL_ESRGAN_BIN=/app/realesrgan-ncnn-vulkan

# Set working directory
WORKDIR /app

# Download and extract Real-ESRGAN NCNN Vulkan
# Using the version from the official readme: v0.2.5.0
RUN wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-ubuntu.zip && \
    unzip realesrgan-ncnn-vulkan-20220424-ubuntu.zip && \
    rm realesrgan-ncnn-vulkan-20220424-ubuntu.zip && \
    chmod +x realesrgan-ncnn-vulkan

# Copy source code from src directory
COPY src/*.py ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Start the handler
CMD [ "python", "-u", "/app/handler.py" ]
