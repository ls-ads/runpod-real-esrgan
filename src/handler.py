import runpod
from runpod import RunPodLogger
import subprocess
import os
import uuid
from utils import download_image, decode_base64_image, encode_image_base64, cleanup

log = RunPodLogger()

def handler(job):
    job_input = job["input"]
    
    image_url = job_input.get("image_url")
    image_base64 = job_input.get("image_base64")
    model_name = job_input.get("model_name", "realesrgan-x4plus")
    scale = job_input.get("scale", 4)
    
    # New options from README
    tta = job_input.get("tta", False)
    output_format = job_input.get("format", "png") # jpg/png/webp
    threads = job_input.get("threads") # load:proc:save, e.g. "2:2:2"
    gpu_id = job_input.get("gpu_id") # e.g. "0"
    tile_size = job_input.get("tile_size")

    log.info(f"Starting job {job.get('id', 'local')} with model: {model_name}, scale: {scale}")

    if not image_url and not image_base64:
        log.error("Job failed: No input image provided.")
        return {"error": "No input image provided. Please provide image_url or image_base64."}

    input_path = None
    output_path = f"/tmp/{uuid.uuid4()}.{output_format}"
    
    try:
        # 1. Prepare input image
        if image_url:
            input_path = download_image(image_url)
        else:
            input_path = decode_base64_image(image_base64)
        
        # 2. Run Real-ESRGAN NCNN Vulkan binary
        # Use environment variable for the binary path, fallback to local file
        binary_path = os.environ.get("REAL_ESRGAN_BIN", "/app/realesrgan-ncnn-vulkan")
        if not os.path.exists(binary_path):
            local_fallback = "./realesrgan-ncnn-vulkan"
            if os.path.exists(local_fallback):
                binary_path = local_fallback
        
        log.debug(f"Using binary at: {binary_path}")

        cmd = [
            binary_path,
            "-i", input_path,
            "-o", output_path,
            "-n", model_name,
            "-s", str(scale),
            "-f", output_format
        ]
        
        if tta:
            cmd.append("-x")
        if tile_size:
            cmd.extend(["-t", str(tile_size)])
        if threads:
            cmd.extend(["-j", str(threads)])
        if gpu_id is not None:
            cmd.extend(["-g", str(gpu_id)])

        log.info(f"Executing upscaler command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            log.error(f"Real-ESRGAN binary failed with return code {result.returncode}")
            log.error(f"Binary Stderr: {result.stderr}")
            
            # Diagnostic: Check shared libraries
            try:
                ldd_result = subprocess.run(["ldd", binary_path], capture_output=True, text=True)
                log.info(f"Diagnostic - ldd output:\n{ldd_result.stdout}")
            except Exception as le:
                log.error(f"Diagnostic - Failed to run ldd: {str(le)}")

            # Diagnostic: Check for models directory (binary expects it next to itself)
            bin_dir = os.path.dirname(os.path.abspath(binary_path))
            models_dir = os.path.join(bin_dir, "models")
            log.info(f"Diagnostic - Checking models dir at {models_dir}: exists={os.path.exists(models_dir)}")
            if os.path.exists(bin_dir):
                log.info(f"Diagnostic - bin directory contents: {os.listdir(bin_dir)}")

            # Diagnostic: ICD check
            icd_paths = ["/usr/share/vulkan/icd.d/", "/etc/vulkan/icd.d/"]
            for path in icd_paths:
                if os.path.exists(path):
                    log.info(f"Diagnostic - ICD directory {path} contents: {os.listdir(path)}")
                else:
                    log.info(f"Diagnostic - ICD directory {path} does not exist")

            # Diagnostic: Environment audit
            log.info(f"Diagnostic - LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH')}")
            log.info(f"Diagnostic - NVIDIA_DRIVER_CAPABILITIES: {os.environ.get('NVIDIA_DRIVER_CAPABILITIES')}")
            log.info(f"Diagnostic - NVIDIA_VISIBLE_DEVICES: {os.environ.get('NVIDIA_VISIBLE_DEVICES')}")
            
            return {
                "error": "Real-ESRGAN failed",
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode
            }
        
        # 3. Process output
        log.info("Process completed successfully, encoding output...")
        output_base64 = encode_image_base64(output_path)
        
        return {
            "image_base64": output_base64,
            "model": model_name,
            "scale": scale,
            "format": output_format
        }

    except Exception as e:
        log.error(f"An unexpected error occurred: {str(e)}")
        return {"error": str(e)}
    
    finally:
        # 4. Cleanup
        log.debug("Entering cleanup phase...")
        paths_to_cleanup = [output_path]
        if input_path:
            paths_to_cleanup.append(input_path)
        cleanup(paths_to_cleanup)

runpod.serverless.start({"handler": handler})
