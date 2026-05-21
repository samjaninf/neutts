import os
import queue
import threading
import time

import numpy as np
import requests
import sounddevice as sd
import soundfile as sf
from mlx_lm import generate, load
from pynput import keyboard

from stream_audio import stream_generated_audio

ASR_URL = "http://localhost:50250"

WHISPER_LANG = {
    "english": "en",
    "french":  "fr",
    "spanish": "es",
    "german":  "de",
}

# Speaker for each language: (ref_codes_path, ref_text_path)
SPEAKERS = {
    "english": ("samples/jo.pt",       "samples/jo.txt"),
    "french":  ("samples/juliette.pt", "samples/juliette.txt"),
    "spanish": ("samples/mateo.pt",    "samples/mateo.txt"),
    "german":  ("samples/greta.pt",    "samples/greta.txt"),
}

# Key bindings: key → (src_language, tgt_language)
MODES = {
    "z": ("english", "french"),
    "x": ("french",  "english"),
    "c": ("english", "spanish"),
    "v": ("spanish", "english"),
    "b": ("english", "german"),
    "n": ("german",  "english"),
}

mic_index = sd.default.device[0]
mic_sr = int(sd.query_devices(mic_index)["default_samplerate"])

# Use a Queue to pass events to the main thread safely
event_queue = queue.Queue()
held_keys = set()
audio_buffer = []
audio_lock = threading.Lock()
recording = False
start_delay = 0.2

print("Loading translation model...")
llm_model, llm_tokenizer = load("mlx-community/LFM2.5-1.2B-Instruct-8bit")
print("Ready.\n")


def transcribe_audio(audio_data: np.ndarray, sample_rate: int, language: str) -> str:
    sf.write("/tmp/agent_tmp.wav", audio_data, sample_rate)
    with open("/tmp/agent_tmp.wav", "rb") as f:
        response = requests.post(
            f"{ASR_URL}/inference",
            files={"file": ("audio.wav", f, "audio/wav")},
            data={"language": WHISPER_LANG.get(language, "auto")},
        )
    return response.json().get("text", "").strip()


def translate(text: str, src_lang: str, tgt_lang: str) -> str:
    prompt = (
        f"Translate the following {src_lang} text to {tgt_lang}. "
        f"Output only the translation, nothing else.\n\n{text}"
    )
    messages = [{"role": "user", "content": prompt}]
    formatted = llm_tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    return generate(llm_model, llm_tokenizer, prompt=formatted, max_tokens=512, verbose=False).strip()


def audio_callback(indata, frames, time_info, status):
    with audio_lock:
        if recording:
            audio_buffer.append(indata.copy())


def on_press(key):
    try:
        k = key.char
    except AttributeError:
        return
    if k in MODES and k not in held_keys:
        held_keys.add(k)
        event_queue.put(("START", k))


def on_release(key):
    try:
        k = key.char
    except AttributeError:
        return
    if k in held_keys:
        held_keys.discard(k)
        event_queue.put(("STOP", k))


def main():
    global recording, audio_buffer

    print("Multilingual translation agent")
    for k, (src, tgt) in MODES.items():
        print(f"  [{k}] hold → translate {src} → {tgt}")
    print("Ctrl+C to quit.\n")

    # Start the keyboard listener in the background
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    stream = None
    current_mode = None

    # Main Thread Event Loop
    while True:
        try:
            action, mode_key = event_queue.get()
        except KeyboardInterrupt:
            break

        if action == "START":
            # Wait briefly to trim key-click noise
            time.sleep(start_delay)

            # If the user released the key during the delay, abort.
            if mode_key not in held_keys:
                continue

            if stream is not None:
                continue  # Already recording

            current_mode = mode_key
            with audio_lock:
                audio_buffer = []
                recording = True

            stream = sd.InputStream(
                samplerate=mic_sr, channels=1, callback=audio_callback, device=mic_index
            )
            stream.start()
            src, tgt = MODES[mode_key]
            print(f"[{mode_key}] Recording {src} → {tgt}... (release to stop)")

        elif action == "STOP":
            if stream is None or mode_key != current_mode:
                continue

            with audio_lock:
                recording = False
            stream.stop()
            stream.close()
            stream = None
            current_mode = None

            with audio_lock:
                audio_data = np.concatenate(audio_buffer, axis=0) if audio_buffer else None

            if audio_data is None or len(audio_data) < mic_sr * 0.3:
                print("Too short, ignoring.")
                continue

            # Trim trailing button-click noise
            trim = int(0.05 * mic_sr)
            if len(audio_data) > 2 * trim:
                audio_data = audio_data[trim:-trim]
            audio_data = audio_data.squeeze()

            src_lang, tgt_lang = MODES[mode_key]
            ref_codes_path, ref_text_path = SPEAKERS[tgt_lang]

            print("Transcribing...")
            transcript = transcribe_audio(audio_data, mic_sr, src_lang)
            print(f"  [{src_lang}] {transcript}")

            print(f"Translating to {tgt_lang}...")
            translation = translate(transcript, src_lang, tgt_lang)
            print(f"  [{tgt_lang}] {translation}")

            # It is safe to spawn a thread for playback so the user
            # doesn't have to wait for the audio to finish to record again.
            threading.Thread(
                target=stream_generated_audio,
                kwargs={
                    "input_text": translation,
                    "ref_codes_path": ref_codes_path,
                    "ref_text": ref_text_path,
                    "language": tgt_lang,
                },
                daemon=True,
            ).start()


if __name__ == "__main__":
    main()
