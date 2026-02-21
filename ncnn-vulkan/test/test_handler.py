import sys
import os
from unittest.mock import MagicMock, patch

# Add src to sys.path to allow imports from handler and utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Mock runpod before importing handler
mock_runpod = MagicMock()
sys.modules["runpod"] = mock_runpod

import handler

@patch('subprocess.run')
@patch('handler.download_image')
@patch('handler.decode_base64_image')
@patch('handler.encode_image_base64')
@patch('handler.cleanup')
def test_handler_mock(mock_cleanup, mock_encode, mock_decode, mock_download, mock_run):
    # Setup mocks
    mock_run.return_value.returncode = 0
    mock_run.return_value.stdout = "Success"
    mock_run.return_value.stderr = ""
    
    mock_download.return_value = "/tmp/mock_input.jpg"
    mock_decode.return_value = "/tmp/mock_input.jpg"
    mock_encode.return_value = "mock_base64_data"

    # Create a mock output file (handler expects it to exist to encode it)
    mock_output_path = "/tmp/mock_output.png"
    with open(mock_output_path, "w") as f:
        f.write("mock data")

    job = {
        "id": "test_job_id",
        "input": {
            "image_url": "http://example.com/test.jpg",
            "model_name": "realesrgan-x4plus",
            "scale": 4,
            "tta": True,
            "format": "webp",
            "threads": "2:2:2",
            "gpu_id": 0
        }
    }

    response = handler.handler(job)

    # Verify that the command was built correctly with all options
    args = mock_run.call_args[0][0]
    binary_path = os.environ.get("REAL_ESRGAN_BIN", "/app/realesrgan-ncnn-vulkan")
    assert binary_path in args
    assert "-x" in args  # TTA
    assert "-f" in args  # format
    assert "webp" in args
    assert "-j" in args  # threads
    assert "2:2:2" in args
    assert "-g" in args  # gpu_id
    assert "0" in args

    assert response["image_base64"] == "mock_base64_data"
    assert response["format"] == "webp"
    
    # Verify cleanup was called
    assert mock_cleanup.called
