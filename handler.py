import runpod
from runpod import RunPodLogger
from pydantic import BaseModel, ValidationError, model_validator
from typing import Optional, Literal
import os
import subprocess
import requests
import time
import base64
import sys
from io import BytesIO
from PIL import Image

log = RunPodLogger()

class InputPayload(BaseModel):
    image_url: Optional[str] = None
    image_base64: Optional[str] = None
    output_format: Literal["png", "jpg"] = "jpg"

    @model_validator(mode='after')
    def check_image_provided(self):
        if not self.image_url and not self.image_base64:
            raise ValueError("No input image provided. Please provide image_url or image_base64.")
        return self

def detect_trt_version():
    try:
        # dpkg-query -W -f='${Version}' libnvinfer-bin 2>/dev/null || dpkg -l | awk '/^ii  libnvinfer[0-9]+/ {print $3}' | head -n 1
        cmd = "dpkg-query -W -f='${Version}' libnvinfer-bin 2>/dev/null || dpkg -l | awk '/^ii  libnvinfer[0-9]+/ {print $3}' | head -n 1"
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        out = res.stdout.strip()
        if not out:
            return "unknown"
        parts = out.split('.')
        if len(parts) >= 2:
            return f"{parts[0]}.{parts[1]}"
        return "unknown"
    except Exception as e:
        log.error(f"TRT detect error: {e}")
        return "unknown"

def detect_gpu_arch():
    try:
        # nvidia-smi --query-gpu=compute_cap --format=csv,noheader
        res = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"], capture_output=True, text=True)
        cap = res.stdout.strip().split('\n')[0].strip()
        return "sm" + cap.replace('.', '')
    except Exception as e:
        log.error(f"GPU detect error: {e}")
        return "unknown"

def download_engine(trt_ver, arch):
    os.makedirs("/workspace", exist_ok=True)
    engine_name = f"realesrgan-x4plus-{arch}-trt{trt_ver}.engine"
    engine_url = f"https://github.com/ls-ads/real-esrgan-serve/releases/download/v0.1.0/{engine_name}"
    engine_path = f"/workspace/{engine_name}"
    
    if os.path.exists(engine_path):
        log.info(f"Engine {engine_name} already exists.")
        return engine_path

    log.info(f"Downloading engine from {engine_url}...")
    try:
        r = requests.get(engine_url, stream=True)
        r.raise_for_status()
        with open(engine_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        log.info("Engine downloaded successfully.")
        return engine_path
    except Exception as e:
        log.error(f"Failed to download engine: {e}")
        sys.exit(1)

def wait_for_server(port=8080):
    url = f"http://localhost:{port}/health"
    for _ in range(30):
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                return True
        except:
            pass
        time.sleep(1)
    return False

def initialize_worker():
    """Load the machine learning model via the C++ TensorRT daemon."""
    log.info("Initializing RunPod Real-ESRGAN TensorRT Worker (Python SDK)...")

    trt_ver = detect_trt_version()
    arch = detect_gpu_arch()
    log.info(f"Detected TRT: {trt_ver}, Architecture: {arch}")

    engine_path = download_engine(trt_ver, arch)

    log.info("Starting real-esrgan-serve HTTP server...")
    serve_cmd = ["/app/real-esrgan-serve", "server", "start", "-p", "8080", "--engine", engine_path, "--gpu-id", "0"]
    serve_proc = subprocess.Popen(serve_cmd, stdout=sys.stdout, stderr=sys.stderr)

    if not wait_for_server(8080):
        log.error("Failed to wait for real-esrgan-serve to become healthy.")
        sys.exit(1)

    log.info("real-esrgan-serve is warmed up and ready!")
    return serve_proc

def fetch_image(image_url, image_base64):
    """Preprocess and fetch the image."""
    log.debug("Fetching and decoding image...")
    if image_base64:
        if ";base64," in image_base64:
            image_base64 = image_base64.split(";base64,")[1]
        img_bytes = base64.b64decode(image_base64)
    else:
        r = requests.get(image_url)
        r.raise_for_status()
        img_bytes = r.content
    return img_bytes

def upscale_image(img_bytes, out_format):
    """Proxy image to the TensorRT daemon for upscaling."""
    log.debug("Upscaling image via TensorRT daemon...")
    if not out_format.startswith('.'):
        out_format = f".{out_format}"

    files = {'image': ('input.jpg', img_bytes, 'application/octet-stream')}
    url = f"http://localhost:8080/upscale?ext={out_format}"
    
    r = requests.post(url, files=files)
    r.raise_for_status()
    out_bytes = r.content
    
    content_type = r.headers.get("Content-Type", "")
    ret_format = out_format[1:]
    if "image/jpeg" in content_type:
        ret_format = "jpg"
    elif "image/png" in content_type:
        ret_format = "png"
        
    b64_out = base64.b64encode(out_bytes).decode("utf-8")
    return b64_out, ret_format

# Load model globally so it is warmed up only once per container
initialize_worker()

def handler(job):
    try:
        # Input validation
        try:
            payload = InputPayload.model_validate(job.get('input', {}))
        except ValidationError as e:
            log.error(f"Validation error: {e}")
            return {"error": str(e)}
        
        image_url = payload.image_url
        image_base64 = payload.image_base64
        out_format = payload.output_format

        # Fetch/decode image
        img_bytes = fetch_image(image_url, image_base64)

        # Get original resolution
        with Image.open(BytesIO(img_bytes)) as img:
            input_width, input_height = img.size
            if input_width > 1280 or input_height > 1280:
                raise ValueError(f"Image dimensions ({input_width}x{input_height}) exceed the maximum allowed size of 1280x1280.")

        # Upscale image
        b64_out, ret_format = upscale_image(img_bytes, out_format)

        # Calculate output resolution (realesrgan-x4plus is fixed 4x)
        output_width = input_width * 4
        output_height = input_height * 4

        log.info("Upscaling completed successfully")

        return {
            "image_base64": b64_out,
            "model": "realesrgan-x4plus",
            "input_resolution": f"{input_width}x{input_height}",
            "output_resolution": f"{output_width}x{output_height}",
            "output_format": ret_format
        }

    except Exception as e:
        log.error(f"An error occurred: {str(e)}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
