There are two Dockerfiles here:
- Dockerfile.local
- Dockerfile.builder

**Dockerfile.local** shows a simple example where you already have the model downloaded locally.

**Dockerfile.builder** is using the modelcar builder to download the model for you and then package it:
Modelcar builder: https://github.com/redhat-ai-services/modelcar-catalog/tree/main/builder-images/huggingface-modelcar-builder
Docs here: https://developers.redhat.com/articles/2025/01/30/build-and-deploy-modelcar-container-openshift-ai#how_to_build_a_modelcar_container