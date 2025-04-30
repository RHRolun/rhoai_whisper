# Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import triton_python_backend_utils as pb_utils
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import os
import pathlib
from typing import Final
import time
import torch

# The path to search for the model and feature extractor. 
MODEL_PATH = os.path.split(__file__)[0]
SAMPLING_RATE: Final[int] = 16000

class TritonPythonModel:
    def initialize(self, args):
        try:
            # load whisper model and processor from the path
            self.processor = WhisperProcessor.from_pretrained(os.path.join(MODEL_PATH, "model_files"), predict_timestamps=True)
            self.model = WhisperForConditionalGeneration.from_pretrained(os.path.join(MODEL_PATH, "model_files"), cache_dir=MODEL_PATH)

            self.model.to("cuda")
        except Exception as e:
            print(f"Error in initializing the model: {e} with path {MODEL_PATH}")
            raise Exception(f">>> Error initializing the model **{e}** on the model path:{MODEL_PATH} <<<")

    def execute(self, requests):
        responses = []
        for request in requests:
            start = time.time()
            # Load task parameter
            task_input = pb_utils.get_input_tensor_by_name(request, "task")
            if task_input is not None:
                task = task_input.as_numpy()[0].item().decode("utf-8")

            # Load language parameter
            language_input = pb_utils.get_input_tensor_by_name(request, "language")
            if language_input is not None:
                language = language_input.as_numpy()[0].item().decode("utf-8")
            
            # pull audio data from the request and convert to numpy array
            inp = pb_utils.get_input_tensor_by_name(request, "audio")
            # transform the audio
            input_audio = np.squeeze(inp.as_numpy())

            # do whisper processing
            input_features = self.processor(
                input_audio, 
                sampling_rate=SAMPLING_RATE, return_tensors="pt").input_features.to("cuda")
            
            predicted_ids = self.model.generate(
                input_features, 
                # TODO configure force_decoder_ids
                forced_decoder_ids = self.processor.get_decoder_prompt_ids(language=language, task=task, no_timestamps=False), 
                return_timestamps=True
            )
            transcription = self.processor.tokenizer.decode(predicted_ids[0], decode_with_timestamps=True)
            end_time = time.time()
            inference_time = end_time - start
            device: str = "cuda" if torch.cuda.is_available() else "cpu"
            output_text = f"Task: {task}. Language: {language}. Audio shape: {input_audio.shape}. Inference time: {inference_time:.2f}s. Device: {device}. Transcription: {transcription}"
            # output_text = f"Task: {task}. Language: {language}. Audio shape: {input_audio.shape}. 2"

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    pb_utils.Tensor(
                        "text", np.array([output_text.encode("utf-8")], dtype=np.object_)
                    )
                ]
            )
            responses.append(inference_response)
            

        return responses
