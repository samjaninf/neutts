import logging
from functools import lru_cache
from pathlib import Path
from contextlib import asynccontextmanager

import numpy as np
import torch
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# --- ADD THIS IMPORT ---
from huggingface_hub import hf_hub_download

from neutts import NeuTTS

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("Resolving local model paths from cache...")

    # 1. Fetch the exact local path to the GGUF backbone
    gguf_path = hf_hub_download(
        repo_id="neuphonic/qwen3-0.2b-9langs-grpo-1750-14-04-26-GGUF",
        filename="qwen3-0.2b-9langs-grpo-1750-14-04-26_Q4_0.gguf",
        local_files_only=True,
    )

    # 2. Fetch the exact local path to the ONNX codec
    onnx_path = hf_hub_download(
        repo_id="neuphonic/neucodec-onnx-decoder",
        filename="model.onnx",  # Exact filename for the codec
        local_files_only=True,
    )

    logging.info("Loading model...")

    # 3. Pass BOTH direct file paths into NeuTTS
    app.state.tts = NeuTTS(
        backbone_repo=gguf_path,
        backbone_device="metal",
        codec_repo=onnx_path,  # <-- Inject the local .onnx path here
        codec_device="cpu",
    )
    logging.info("Model loaded.")
    yield


app = FastAPI(lifespan=lifespan)


class TTSRequest(BaseModel):
    text: str
    ref_codes_path: str
    ref_text: str
    language: str


@lru_cache(maxsize=32)
def _load_ref_codes(path: str):
    return torch.load(path)


@app.post("/generate-streaming")
def generate(request: TTSRequest):
    ref_codes_path = Path(request.ref_codes_path)
    if not ref_codes_path.exists():
        raise FileNotFoundError(f"Pre-encoded reference not found: {ref_codes_path}")
    ref_codes = _load_ref_codes(str(ref_codes_path.resolve()))

    def audio_stream():
        for chunk in app.state.tts.infer_stream(
            request.text, ref_codes, request.ref_text, language=request.language
        ):
            yield (chunk * 32767).astype(np.int16).tobytes()

    return StreamingResponse(audio_stream(), media_type="application/octet-stream")
