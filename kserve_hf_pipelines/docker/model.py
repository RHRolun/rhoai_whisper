import os
import io
import base64
import argparse
import json

from typing import Dict, Union
import torch
from transformers import pipeline

from kserve import (
    Model,
    ModelServer,
    model_server,
    InferRequest,
    InferResponse,
)
from kserve.errors import InvalidInput


class AsrModel(Model):
    def __init__(self, name: str):
        super().__init__(name)
        self.model_id = os.environ.get("MODEL_ID", default="/mnt/models")
        self.device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        self.pipeline = None
        self.ready = False
        self.load()

    def load(self):
        print(f"Loading model {self.model_id} on device {self.device}")
        self.pipeline = pipeline("automatic-speech-recognition", model=self.model_id, device=self.device)
        # The ready flag is used by model ready endpoint for readiness probes,
        # set to True when model is loaded successfully without exceptions.
        self.ready = True

    def preprocess(
            self,
            payload: Union[Dict, InferRequest],
            headers: Dict[str, str] = None
    ) -> Union[Dict, InferRequest]:
        if isinstance(payload, Dict) and "instances" in payload:
            headers["request-type"] = "v1"
        elif isinstance(payload, bytes):
            raise InvalidInput("form data not implemented")
        elif isinstance(payload, InferRequest):
            raise InvalidInput("v2 protocol not implemented")
        else:
            raise InvalidInput("invalid payload")

        return payload["instances"][0]

    def predict(
            self,
            payload: Union[Dict, InferRequest],
            headers: Dict[str, str] = None
    ) -> Union[Dict, InferResponse]:
        bytes_data = base64.b64decode(payload["audio"]["b64"])
        transcription = self.pipeline(bytes_data)

        return {
            "predictions": [
                {
                    "model_name": self.model_id,
                    "transcription": transcription
                }
            ]}


parser = argparse.ArgumentParser(parents=[model_server.parser])
args, _ = parser.parse_known_args()

if __name__ == "__main__":
    model = AsrModel(args.model_name)
    model.load()
    ModelServer().start([model])