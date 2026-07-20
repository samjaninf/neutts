import numpy as np
from neutts import NeuTTS2E
import pyaudio
import queue
import threading

from basic_streaming_example import audio_player_thread


def main(input_text, speaker, emotion, backbone, device, seed):
    assert backbone.endswith(
        "gguf"
    ), "Must be a GGUF ckpt as streaming is only currently supported by llama-cpp."

    tts = NeuTTS2E(
        backbone_repo=backbone,
        backbone_device=device,
        codec_repo="neuphonic/neucodec-onnx-decoder",
        seed=seed,
    )

    print(f"Streaming '{emotion}' audio for speaker '{speaker}': {input_text}")
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=tts.sample_rate, output=True)

    audio_queue = queue.Queue()
    player = threading.Thread(target=audio_player_thread, args=(audio_queue, stream))
    player.start()

    for chunk in tts.infer_stream(input_text, speaker=speaker, emotion=emotion):
        audio = (chunk * 32767).astype(np.int16)
        audio_queue.put(audio.tobytes())

    tail_pad = np.zeros(int(0.25 * tts.sample_rate), dtype=np.int16)
    audio_queue.put(tail_pad.tobytes())
    audio_queue.put(None)
    player.join()

    stream.stop_stream()
    stream.close()
    p.terminate()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NeuTTS-2E Streaming Example")
    parser.add_argument(
        "--input_text",
        type=str,
        required=True,
        help="Input text to be converted to speech",
    )
    parser.add_argument(
        "--speaker",
        type=str,
        default="emily",
        help="One of the bundled speakers: emily, paul, sophie, steven",
    )
    parser.add_argument(
        "--emotion",
        type=str,
        default="neutral",
        help="One of: angry, disgusted, fearful, happy, neutral, sad, surprised",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        required=True,
        help="Huggingface repo or local path containing the GGUF backbone checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for the backbone, e.g. cpu or gpu",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed for reproducible generation",
    )
    args = parser.parse_args()
    main(
        input_text=args.input_text,
        speaker=args.speaker,
        emotion=args.emotion,
        backbone=args.backbone,
        device=args.device,
        seed=args.seed,
    )
