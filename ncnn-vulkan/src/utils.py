import os
import requests
import uuid
import base64
from runpod import RunPodLogger
from PIL import Image
import io

log = RunPodLogger()

def download_image(url):
    log.info(f"Downloading image from URL: {url}")
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    
    file_extension = url.split('.')[-1].split('?')[0].lower()
    if file_extension not in ['jpg', 'jpeg', 'png', 'webp']:
        log.warn(f"Unsupported file extension '{file_extension}' detected. Defaulting to 'jpg'.")
        file_extension = 'jpg'
        
    temp_filename = f"/tmp/{uuid.uuid4()}.{file_extension}"
    with open(temp_filename, 'wb') as f:
        f.write(response.content)
    log.debug(f"Image downloaded to: {temp_filename}")
    return temp_filename

def decode_base64_image(base64_str):
    log.info("Decoding base64 image...")
    if ',' in base64_str:
        base64_str = base64_str.split(',')[1]
    
    img_data = base64.b64decode(base64_str)
    img = Image.open(io.BytesIO(img_data))
    
    file_extension = img.format.lower() if img.format else 'png'
    temp_filename = f"/tmp/{uuid.uuid4()}.{file_extension}"
    img.save(temp_filename)
    log.debug(f"Base64 image decoded and saved to: {temp_filename}")
    return temp_filename

def encode_image_base64(file_path):
    log.debug(f"Encoding image to base64: {file_path}")
    with open(file_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def upload_to_s3(file_path):
    # This is a placeholder for S3 upload logic if needed
    pass

def cleanup(file_paths):
    for path in file_paths:
        if os.path.exists(path):
            log.debug(f"Cleaning up temporary file: {path}")
            os.remove(path)
