# Runpod Real-ESRGAN Worker

A high-performance Runpod serverless worker for image upscaling using the NCNN Vulkan implementation of Real-ESRGAN.

This project packages the official [Real-ESRGAN-ncnn-vulkan](https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan) binary into a Runpod-compatible Docker image, providing a highly scalable and cost-effective way to deploy state-of-the-art super-resolution.

## Features

- **High Performance**: Optimized for GPU acceleration via Vulkan (no full CUDA stack required).
- **Flexible Input**: Supports both `image_url` and `image_base64`.
- **Advanced Controls**: Fully supports NCNN binary parameters:
    - **TTA Mode**: Test-time augmentation for higher quality.
    - **Custom Scales**: Upscale by 2x, 3x, or 4x.
    - **Model Selection**: Choose between `realesrgan-x4plus`, `realesrgan-x4plus-anime`, `realesr-animevideov3`, and more.
    - **Thread Tuning**: Customize `load:proc:save` thread counts for performance optimization.
    - **Tiling**: Adjustable `tile_size` for processing large images on limited GPU memory.
- **Developer Friendly**: Built with `uv` for dependency management and `pytest` for unit testing.

## Project Structure

- `src/`: Core implementation logic.
    - `handler.py`: Runpod worker entry point.
    - `utils.py`: Image processing and utility functions.
- `test/`: Project tests.
- `Dockerfile`: Production-ready Docker configuration.
- `test_input.json`: Local testing configuration.

## Local Development

Manage dependencies and run locally using [uv](https://astral.sh/uv).

### Setup
Dependencies are managed automatically by `uv`.

### Running Unit Tests
```bash
uv run pytest
```

### Simulating a Runpod Job
1. Configure your input in `test_input.json`.
2. (Optional) Set the path to your local Real-ESRGAN binary:
   ```bash
   export REAL_ESRGAN_BIN=/path/to/your/binary
   ```
3. Run the handler:
   ```bash
   uv run python src/handler.py --rp_server_api
   ```

## Deployment

### 1. Build & Push Image
```bash
docker build -t your-registry/runpod-real-esrgan:latest .
docker push your-registry/runpod-real-esrgan:latest
```

### 2. Configure Runpod
- Create a new **Serverless Endpoint**.
- Use the image built in the previous step.
- Ensure the template has appropriate GPU resources.

### 3. API Usage
Send a job with the following input schema:

```json
{
  "input": {
    "image_url": "https://example.com/image.jpg",
    "model_name": "realesrgan-x4plus",
    "scale": 4,
    "tta": true,
    "format": "webp"
  }
}
```
