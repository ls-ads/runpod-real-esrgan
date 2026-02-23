FROM nvcr.io/nvidia/tensorrt:26.01-py3

WORKDIR /app

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the compiled Go binary for the C++ TensorRT backend from the official release image
COPY --from=ghcr.io/ls-ads/real-esrgan-serve/cli:v0.1.0 /app/real-esrgan-serve /app/real-esrgan-serve

# Copy the RunPod python handler
COPY handler.py .

CMD ["python3", "-u", "handler.py"]
