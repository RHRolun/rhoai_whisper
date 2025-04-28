# rhoai_whisper

This repo contains a few ways to serve Whisper:
1. With a KServe model using hf pipelines
2. With a KServe model using vLLM
3. With Triton using vLLM (currently not mature enough)

It also contains an example for how to fetch files from S3 as part of the preprocessing.