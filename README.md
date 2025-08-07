# Serving Whisper with RHOAI

This repo, created by the CAI team, contains a few ways to serve Whisper:

## Methods

* [KServe with Hugging Face Pipelines](./kserve_hf_pipelines): A KServe model using Hugging Face pipelines.
* [KServe with vLLM](./kserve_vllm): A KServe model using vLLM.
* [Triton with vLLM](./triton_vllm): A Triton model using vLLM (currently not mature enough).

## Additional Features

This repo also contains an example for how to fetch files from S3 as part of the preprocessing.
