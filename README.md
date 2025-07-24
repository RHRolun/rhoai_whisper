# RHOAI Whisper - Speech Recognition Serving Solutions

This repository provides multiple approaches to serve OpenAI's Whisper speech recognition model in production environments. The solutions focus on deploying Whisper models using modern MLOps frameworks like KServe and NVIDIA Triton Inference Server.

## Table of Contents
- [Overview](#overview)
- [Available Solutions](#available-solutions)
- [Quick Start](#quick-start)
- [Detailed Setup](#detailed-setup)
- [Components](#components)
- [Contributing](#contributing)
- [License](#license)

## Overview

This repository demonstrates various methods to deploy and serve Whisper models for automatic speech recognition (ASR). The solutions are designed for Red Hat OpenShift AI (RHOAI) and similar Kubernetes-based MLOps platforms.

The repository includes:
- KServe deployments using Hugging Face pipelines
- KServe deployments using vLLM for high-performance inference
- Triton Inference Server deployments with vLLM backend
- Preprocessing examples for S3 file fetching

## Available Solutions

### 1. KServe with Hugging Face Pipelines
A straightforward approach using KServe's Hugging Face model server with the transformers pipeline API.

**Features:**
- Easy deployment with minimal configuration
- Built-in support for Whisper models
- Automatic batching and preprocessing
- S3 integration for audio file fetching

### 2. KServe with vLLM
High-performance serving using vLLM, which provides:
- PagedAttention for efficient memory management
- Continuous batching for high throughput
- Tensor parallelism for multi-GPU inference
- Optimized for large language models

### 3. Triton with vLLM Backend
Experimental approach using NVIDIA Triton Inference Server with vLLM as the backend:
- Advanced model orchestration capabilities
- Dynamic batching and model pipeline composition
- Multi-framework support
- Note: This approach is still maturing

## Quick Start

### Prerequisites
- Kubernetes cluster with KServe installed
- Storage configured for model artifacts
- Appropriate GPU resources (for GPU acceleration)

### Deployment Steps

1. **Clone the repository:**
```bash
git clone https://github.com/your-org/rhoai_whisper.git
cd rhoai_whisper
```

2. **Choose your deployment method:**
   - For KServe with Hugging Face pipelines: Navigate to `kserve_hf_pipelines/`
   - For KServe with vLLM: Navigate to `kserve_vllm/`

3. **Deploy the InferenceService:**
```bash
kubectl apply -f inferenceservice.yaml
```

4. **Test the deployment:**
```bash
# Get the ingress URL
kubectl get inferenceservice <service-name> -o jsonpath='{.status.url}'

# Make a prediction
curl -v -H "Content-Type: application/json" \
  -d '{"inputs": [{"data": "https://example.com/audio.wav"}]}' \
  <ingress-url>
```

## Detailed Setup

### KServe with Hugging Face Pipelines
Located in `kserve_hf_pipelines/` directory.

**Key Files:**
- `inferenceservice.yaml`: KServe InferenceService configuration
- `servingruntime.yaml`: Custom ServingRuntime for Whisper models
- `example_inference.ipynb`: Jupyter notebook with example usage
- `docker/`: Docker configuration for the model server

### KServe with vLLM
Located in `kserve_vllm/` directory.

**Key Files:**
- `inferenceservice.yaml`: KServe configuration with vLLM predictor
- `servingruntime.yaml`: Custom runtime for vLLM
- `docker/`: Docker configuration optimized for vLLM

### Triton Integration
Located in `triton_vllm/` and `triton_python/` directories.

**Note:** These solutions are experimental and may require additional configuration.

## Components

### ModelCar
The `modelcar` directory contains tools for packaging models into OCI containers for deployment.

### Preprocessing
The repository includes examples of preprocessing pipelines, including:
- S3 file fetching for audio inputs
- Audio format conversion
- Metadata extraction

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.

## References
- [KServe Documentation](https://kserve.github.io/website/)
- [vLLM Documentation](https://vllm.readthedocs.io/)
- [NVIDIA Triton Inference Server](https://github.com/triton-inference-server/server)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
