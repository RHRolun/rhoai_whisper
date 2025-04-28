import os
import io
import base64
import argparse
import json
import boto3

from typing import Dict, Union
import torch
from vllm import LLM, SamplingParams
import numpy as np
import librosa

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
            # kv_cache_dtype="fp8",
        )
        # The ready flag is used by model ready endpoint for readiness probes,
        # set to True when model is loaded successfully without exceptions.
        self.ready = True

    def deserialize_audio(self, audio_b64):
        audio_bytes = base64.b64decode(audio_b64)
        audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
        return audio_array
    
    def serialize_audio(self, audio_array):
        audio_bytes = audio_array.astype(np.float32).tobytes()
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        return audio_b64

    def download_file_from_s3(self, object_key, download_path):
        session = boto3.session.Session()

        s3_access_key = os.environ.get('AWS_ACCESS_KEY_ID')
        s3_secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
        s3_endpoint = os.environ.get('AWS_S3_ENDPOINT')
        bucket_name = os.environ.get('AWS_S3_BUCKET')

        s3_client = session.client(
            service_name='s3',
            aws_access_key_id=s3_access_key,
            aws_secret_access_key=s3_secret_key,
            endpoint_url=s3_endpoint,
        )

        try:
            s3_client.download_file(bucket_name, object_key, download_path)
            print(f"File downloaded successfully to {download_path}")
        except Exception as e:
            print(f"Error downloading file: {e}")
            raise e


    def preprocess(
            self,
            payload: Union[Dict, InferRequest],
            headers: Dict[str, str] = None
    ) -> Union[Dict, InferRequest]:
        ######## Check request format
        if isinstance(payload, Dict) and "instances" in payload:
            headers["request-type"] = "v1"
        elif isinstance(payload, bytes):
            raise InvalidInput("form data not implemented")
        elif isinstance(payload, InferRequest):
            raise InvalidInput("v2 protocol not implemented")
        else:
            raise InvalidInput("invalid payload")

        ######## Download from S3
        filename = payload["instances"][0]
        print(f"Downloading file {filename} from {os.environ.get('AWS_S3_ENDPOINT')}/{os.environ.get('AWS_S3_BUCKET')}")
        self.download_file_from_s3(filename, f"/tmp/{filename}")

        ######## Load file as numpy array and serialize
        audio, sr = librosa.load(f"/tmp/{filename}", sr=16000)
        serialized_audio = self.serialize_audio(audio)

        return (serialized_audio, sr)

    def predict(
            self,
            payload: Union[Dict, InferRequest],
            headers: Dict[str, str] = None
    ) -> Union[Dict, InferResponse]:

        audio_array, sample_rate = payload
        audio_array = self.deserialize_audio(audio_array)

        model_input = {
            "instances": [
                {
                    "prompt": "<|startoftranscript|>",
                    "multi_modal_data": {
                        "audio": (audio_array, sample_rate),
                    },
                }
            ]
        }

        print(model_input)


        sampling_params = SamplingParams(
            temperature=0,
            top_p=1.0,
            max_tokens=200,
        )

        outputs = self.model.generate(model_input, sampling_params)

        generated_text = outputs[0].outputs[0].text
        # transcription = self.pipeline(bytes_data)

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
