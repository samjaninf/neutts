import os
import time

import numpy as np
import pyaudio
import requests
from fire import Fire

SERVER_URL = "http://localhost:50252"


def stream_generated_audio(
    input_text,
    ref_codes_path="samples/jo.pt",
    ref_text="samples/jo.txt",
    language="english",
    server_url=SERVER_URL,
):
    if ref_text and os.path.exists(ref_text):
        with open(ref_text, "r") as f:
            ref_text = f.read().strip()

    data = {
        "text": input_text,
        "ref_codes_path": ref_codes_path,
        "ref_text": ref_text,
        "language": language,
    }

    print(f"Generating [{language}]: {input_text}")
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=24_000, output=True)

    response = requests.post(f"{server_url}/generate-streaming", json=data, stream=True)
    for chunk in response.iter_content(chunk_size=None):
        stream.write(chunk)

    tail_pad = np.zeros(int(0.5 * 24_000), dtype=np.int16)
    stream.write(tail_pad.tobytes(), exception_on_underflow=False)
    time.sleep(0.05)

    stream.stop_stream()
    stream.close()
    p.terminate()


if __name__ == "__main__":
    Fire(stream_generated_audio)
