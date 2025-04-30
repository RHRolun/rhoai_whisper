### Triton vLLM Whisper

We are storing our model in a PvC for simplicity (OCI artifact would work well also) and then attaching the model and config files to overwrite the existing ones, giving us good flexibility to iterate on those files.
Example and docs here: https://github.com/triton-inference-server/vllm_backend

Note: vLLM Whisper with Triton seems unstable at the moment, so this example will load the model but does not seem to work during inference.