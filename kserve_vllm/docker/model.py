import os
import io
import base64
import argparse
import json

from typing import Dict, Union
import torch
from vllm import LLM, SamplingParams
import numpy as np

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
        self.model = None
        self.ready = False
        self.load()

    def load(self):
        print(f"Loading model {self.model_id} on device {self.device}")
        self.model = LLM(
            model="/mnt/models",
            max_model_len=448,
            max_num_seqs=400,
            limit_mm_per_prompt={"audio": 1},
        )
        # The ready flag is used by model ready endpoint for readiness probes,
        # set to True when model is loaded successfully without exceptions.
        self.ready = True

    def deserialize_audio(self, audio_b64):
        audio_bytes = base64.b64decode(audio_b64)
        audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
        return audio_array

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
        audio_array, sample_rate = payload["multi_modal_data"]["audio"]
        audio_array = self.deserialize_audio(audio_array)
        payload["multi_modal_data"]["audio"] = (audio_array, sample_rate)


        sampling_params = SamplingParams(
            temperature=0,
            top_p=1.0,
            max_tokens=200,
        )

        outputs = self.model.generate(payload, sampling_params)

        generated_text = outputs[0].outputs[0].text

        return {
            "predictions": [
                {
                    "model_name": self.model_id,
                    "transcription": generated_text
                }
            ]}


parser = argparse.ArgumentParser(parents=[model_server.parser])
args, _ = parser.parse_known_args()

if __name__ == "__main__":
    model = AsrModel(args.model_name)
    ModelServer().start([model])
