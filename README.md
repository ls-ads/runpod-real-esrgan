# RunPod Real-ESRGAN Monorepo

This repository contains multiple high-performance [RunPod](https://www.runpod.io/) serverless worker implementations for [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) image upscaling.

## Implementations

Each directory contains a standalone implementation with its own `Dockerfile` and dependencies.

- [**ncnn-vulkan/**](./ncnn-vulkan): High-performance NCNN implementation that runs on Vulkan. Optimized for efficiency and doesn't require a full CUDA stack.
- [**pytorch/**](./pytorch): (Work in Progress) Standard PyTorch implementation using the original weights.
- [**onnx/**](./onnx): (Work in Progress) Optimized ONNX Runtime implementation with TensorRT and CUDA execution providers.
- [**tensor-rt/**](./tensor-rt): (Work in Progress) Direct TensorRT implementation for maximum performance on NVIDIA GPUs.

## Getting Started

To get started with a specific implementation, navigate to its directory and follow the instructions in its `README.md`.

For example, to use the NCNN Vulkan implementation:

```bash
cd ncnn-vulkan
# Follow instructions in ncnn-vulkan/README.md
```

## Credits

This project is a wrapper around the hard work of the following projects:

- [**Real-ESRGAN-ncnn-vulkan**](https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan): The ncnn implementation of Real-ESRGAN.
- [**Real-ESRGAN**](https://github.com/xinntao/Real-ESRGAN): The original training and implementation project for Real-ESRGAN.

Special thanks to [xinntao](https://github.com/xinntao) and all contributors for these incredible projects.

## Contributing

This is a monorepo setup to experiment with different deployment patterns for upscaling models. Contributions of new engines or optimizations for existing ones are welcome!
