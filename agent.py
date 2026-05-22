import os
import queue
import sys
import termios
import threading
import time
from contextlib import contextmanager

import numpy as np
import requests
import sounddevice as sd
import soundfile as sf
from mlx_lm import generate, load
from pynput import keyboard

from stream_audio import stream_generated_audio


@contextmanager
def silent_terminal():
    """Disable terminal echo + line buffering while the agent runs.

    pynput hooks keystrokes at the OS event level, so stdin doesn't need
    to see them — but the terminal still echoes them by default. No-op
    if stdin isn't a TTY.
    """
    if not sys.stdin.isatty():
        yield
        return
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    new = termios.tcgetattr(fd)
    new[3] &= ~(termios.ECHO | termios.ICANON)  # lflags
    try:
        termios.tcsetattr(fd, termios.TCSADRAIN, new)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


ASR_URL = "http://localhost:50250"

WHISPER_LANG = {
    "english":    "en",
    "french":     "fr",
    "spanish":    "es",
    "german":     "de",
    "portuguese": "pt",
    "japanese":   "ja",
    "korean":     "ko",
    "chinese":    "zh",
}

# Speaker for each language: (ref_codes_path, ref_text_path)
SPEAKERS = {
    "english":    ("samples/english/paul.pt",       "samples/english/paul.txt"),
    "french":     ("samples/french/amelie.pt",      "samples/french/amelie.txt"),
    "spanish":    ("samples/spanish/martina.pt",    "samples/spanish/martina.txt"),
    "german":     ("samples/german/carla.pt",       "samples/german/carla.txt"),
    "portuguese": ("samples/portuguese/diogo.pt",   "samples/portuguese/diogo.txt"),
    "japanese":   ("samples/japanese/miwa.pt",      "samples/japanese/miwa.txt"),
    "korean":     ("samples/korean/siwoo.pt",       "samples/korean/siwoo.txt"),
    "chinese":    ("samples/chinese/mei.pt",        "samples/chinese/mei.txt"),
}

# First-letter shortcuts for source/target selection. All 8 are unique.
LANG_KEYS = {
    "e": "english",
    "f": "french",
    "s": "spanish",
    "g": "german",
    "p": "portuguese",
    "j": "japanese",
    "k": "korean",
    "c": "chinese",
}

mic_index = sd.default.device[0]
mic_sr = int(sd.query_devices(mic_index)["default_samplerate"])

# Use a Queue to pass events to the main thread safely
event_queue = queue.Queue()
audio_buffer = []
audio_lock = threading.Lock()
recording = False
start_delay = 0.2

# Listener-thread state (read/written from the listener thread only,
# except `_space_held` which is read from the main thread to decide
# whether the user is still holding SPACE after the start_delay).
_space_held = False
_pressed_lang_keys: set[str] = set()

print("Loading translation model...")
llm_model, llm_tokenizer = load("mlx-community/LFM2.5-1.2B-Instruct-8bit")

# Warmup: forces MLX kernel compilation + faults all weights into resident memory
# so the first real translation isn't several hundred ms slower than steady state.
print("Warming up translation model...")
_warmup_msgs = [{"role": "user", "content": "Translate english to spanish. Output only the translation.\n\nhello"}]
_warmup_prompt = llm_tokenizer.apply_chat_template(_warmup_msgs, add_generation_prompt=True)
generate(llm_model, llm_tokenizer, prompt=_warmup_prompt, max_tokens=16, verbose=False)
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
    global _space_held
    if key == keyboard.Key.space:
        if not _space_held:
            _space_held = True
            event_queue.put(("SPACE_DOWN",))
        return
    try:
        k = key.char
    except AttributeError:
        return
    if not k:
        return
    k = k.lower()
    if k in LANG_KEYS and k not in _pressed_lang_keys:
        _pressed_lang_keys.add(k)
        event_queue.put(("LANG", k))


def on_release(key):
    global _space_held
    if key == keyboard.Key.space:
        _space_held = False
        event_queue.put(("SPACE_UP",))
        return
    try:
        k = key.char
    except AttributeError:
        return
    if not k:
        return
    _pressed_lang_keys.discard(k.lower())


def main():
    global recording, audio_buffer

    keyboard.Listener(on_press=on_press, on_release=on_release).start()

    pending_src: str | None = None
    pair: tuple[str, str] | None = None
    stream = None

    _term = silent_terminal()
    _term.__enter__()
    try:
        while True:
            try:
                event = event_queue.get()
            except KeyboardInterrupt:
                break
            action = event[0]

            if action == "LANG":
                letter = event[1]
                lang = LANG_KEYS[letter]
                if pending_src is None:
                    pending_src = lang
                    pair = None  # invalidate any previous pair
                    print(f"Source: {lang}. Now tap target.")
                else:
                    if lang == pending_src:
                        print(
                            f"Target same as source ({lang}); tap a different language."
                        )
                        continue
                    pair = (pending_src, lang)
                    pending_src = None
                    print(f"Ready: {pair[0]} → {pair[1]}. Hold SPACE to talk.")
                continue

            if action == "SPACE_DOWN":
                if pair is None:
                    print("No pair selected. Tap source then target letters first.")
                    continue
                if stream is not None:
                    continue  # already recording

                # Wait briefly so the SPACE keystroke doesn't get into the recording.
                time.sleep(start_delay)
                if not _space_held:
                    continue  # released during delay — abort

                with audio_lock:
                    audio_buffer = []
                    recording = True
                stream = sd.InputStream(
                samplerate=mic_sr, channels=1, callback=audio_callback, device=mic_index
            )
                stream.start()
                print(f"Recording {pair[0]} → {pair[1]}... (release SPACE to stop)")
                continue

            if action == "SPACE_UP":
                if stream is None or pair is None:
                    continue

                with audio_lock:
                    recording = False
                stream.stop()
                stream.close()
                stream = None

                with audio_lock:
                    audio_data = np.concatenate(audio_buffer, axis=0) if audio_buffer else None

                if audio_data is None or len(audio_data) < mic_sr * 0.3:
                    print("Too short, ignoring.")
                    continue

                trim = int(0.05 * mic_sr)
                if len(audio_data) > 2 * trim:
                    audio_data = audio_data[trim:-trim]
                audio_data = audio_data.squeeze()

                src_lang, tgt_lang = pair
                ref_codes_path, ref_text_path = SPEAKERS[tgt_lang]

                print("Transcribing...")
                transcript = transcribe_audio(audio_data, mic_sr, src_lang)
                print(f"  [{src_lang}] {transcript}")

                print(f"Translating to {tgt_lang}...")
                translation = translate(transcript, src_lang, tgt_lang)
                print(f"  [{tgt_lang}] {translation}")

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
    finally:
        _term.__exit__(None, None, None)


if __name__ == "__main__":
    main()
