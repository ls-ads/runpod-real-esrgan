# RunPod Real-ESRGAN TensorRT Worker

A highly-optimized Python-based RunPod Serverless worker for deploying a **TensorRT (C/C++)** implementation of Real-ESRGAN. Optimized specifically for the `realesrgan-x4plus` model.

## Internal Architecture

This serverless worker natively connects to RunPod's job queue using the official **RunPod Python SDK** while avoiding Cold-Start VRAM latency. 

Dynamically at inference initialization before accepting jobs, the `handler.py` script performs:

1. **Auto-Detects Hardware**: Queries `nvidia-smi` and the installed `libnvinfer` (TensorRT) engine compatibility version.
2. **Auto-Downloads Engine**: Pulls the mathematically exact serialized `.engine` cache file from `ls-ads/real-esrgan-serve` matching the specific GPU architecture (e.g., `sm86`, `sm89`, `sm90`).
3. **Warms Up VRAM**: Launches the internal C++ API wrapper daemon to keep the 4x model constantly hot inside GPU memory.
4. **Proxies Serverless Invocations**: Your RunPod `POST /run` JSON calls are parsed by the Python SDK and proxied locally into the pre-warmed TensorRT daemon.

## Installation Requirements

Since this worker container auto-builds from the official TensorRT and CUDA Toolkit images, you simply need Docker to build it. 

```bash
docker build -t runpod-real-esrgan-worker .
```

## RunPod Serverless Payload

Your `input` JSON payload sent to the RunPod API must conform to the following schema:

```json
{
  "input": {
    "image_url": "https://example.com/input.jpg",
    "image_base64": "...", 
    "output_format": "jpg" 
  }
}
```

*Note: You must provide EITHER `image_url` OR `image_base64`. `output_format` is optional (defaults to jpg).*

*Note: The upscale ratio is fixed to 4x, as this is optimized specifically for `realesrgan-x4plus`.*

*Note: The maximum input image dimensions supported by this worker are 1280x1280. Images exceeding this limit will be rejected to prevent out-of-memory errors.*

### Output Payload

The worker will return the processed image encoded in base64 within the `output` wrapper:

```json
{
  "output": {
    "image_base64": "...",
    "model": "realesrgan-x4plus",
    "input_resolution": "1280x720",
    "output_resolution": "5120x2880",
    "output_format": "jpg"
  }
}
```

## Acknowledgements

This project is a RunPod serverless wrapper over the incredible engineering of others:

- **real-esrgan-serve**: [ls-ads/real-esrgan-serve](https://github.com/ls-ads/real-esrgan-serve)
- **Original Project**: [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
- **Author**: [xinntao](https://github.com/xinntao)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
