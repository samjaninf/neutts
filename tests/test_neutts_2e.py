from importlib import import_module
from types import SimpleNamespace

import pytest
import torch
from transformers import AutoTokenizer

from neutts import NeuTTS, NeuTTS2E

_normalize_text = import_module(NeuTTS.__module__)._normalize_text

EMOTIONAL_BACKBONE = "neuphonic/qwen3-0.2b-en-emotional-grpo-500-16-06-26"

REF_TEXT = "Last. So we just need to make the rest of the connection."
GEN_TEXT = "Hello there, how are you today?"
REF_CODES = [12, 345, 6789]


@pytest.fixture(scope="module")
def bpe_model():
    model = NeuTTS.__new__(NeuTTS)
    model.input_format = "BPE"
    try:
        model.tokenizer = AutoTokenizer.from_pretrained(EMOTIONAL_BACKBONE)
    except Exception as e:
        pytest.skip(f"Emotional checkpoint unavailable: {e}")
    return model


def test_normalize_text():
    assert _normalize_text("“Don’t.”") == '"Don\'t."'


def test_bpe_prompt_neutral(bpe_model):
    ids = bpe_model._apply_chat_template(REF_CODES, REF_TEXT, GEN_TEXT)
    decoded = bpe_model.tokenizer.decode(ids)
    codes_str = "".join(f"<|speech_{i}|>" for i in REF_CODES)
    assert decoded == (
        f"<|TEXT_PROMPT_START|>{REF_TEXT} {GEN_TEXT}<|TEXT_PROMPT_END|>"
        f"<|SPEECH_GENERATION_START|>{codes_str}"
    )


def test_bpe_prompt_emotion(bpe_model):
    tokenizer = bpe_model.tokenizer
    ids = bpe_model._apply_chat_template(REF_CODES, REF_TEXT, GEN_TEXT, emotion="happy")
    codes_str = "".join(f"<|speech_{i}|>" for i in REF_CODES)
    assert tokenizer.decode(ids) == (
        f"<|TEXT_PROMPT_START|>{REF_TEXT}<|HAPPY|>{GEN_TEXT}<|TEXT_PROMPT_END|>"
        f"<|SPEECH_GENERATION_START|>{codes_str}"
    )

    # Ref and gen text must be encoded separately either side of the emotion token.
    emotion_id = tokenizer.convert_tokens_to_ids("<|HAPPY|>")
    idx = ids.index(emotion_id)
    start = ids.index(tokenizer.convert_tokens_to_ids("<|TEXT_PROMPT_START|>"))
    assert ids[start + 1 : idx] == tokenizer.encode(REF_TEXT, add_special_tokens=False)


def test_bpe_prompt_unknown_emotion_raises(bpe_model):
    with pytest.raises(ValueError, match="not in the model vocab"):
        bpe_model._apply_chat_template(REF_CODES, REF_TEXT, GEN_TEXT, emotion="ecstatic")


@pytest.mark.parametrize("emotion", [None, "happy"])
def test_ggml_prompt_matches_torch_prompt(bpe_model, emotion):
    tokenizer = bpe_model.tokenizer
    bpe_model.backbone = SimpleNamespace(
        tokenize=lambda b, add_bos, special: tokenizer.encode(b.decode(), add_special_tokens=False)
    )
    torch_ids = bpe_model._apply_chat_template(REF_CODES, REF_TEXT, GEN_TEXT, emotion=emotion)
    ggml_prompt = bpe_model._ggml_prompt(REF_CODES, REF_TEXT, GEN_TEXT, emotion=emotion)
    assert tokenizer.encode(ggml_prompt) == torch_ids
    assert tokenizer.decode(torch_ids) == ggml_prompt


def test_phoneme_model_rejects_emotion():
    model = NeuTTS.__new__(NeuTTS)
    model.input_format = "phonemes"
    with pytest.raises(ValueError, match="BPE"):
        model._check_emotion("happy")
    assert model._check_emotion(None) is None


def test_supported_emotions_validation():
    model = NeuTTS.__new__(NeuTTS)
    model.input_format = "BPE"
    model._supported_emotions = list(NeuTTS2E.EMOTIONS)
    with pytest.raises(ValueError, match="Supported emotions"):
        model._check_emotion("furious")
    for emotion in NeuTTS2E.EMOTIONS:
        model._check_emotion(emotion)
    assert model._check_emotion("neutral") is None

    model._supported_emotions = None
    assert model._check_emotion("furious") == "furious"  # deferred to the vocab check


def test_speaker_data_complete():
    assert NeuTTS2E.SPEAKERS == ("emily", "paul", "sophie", "steven")
    for name in NeuTTS2E.SPEAKERS:
        assert (NeuTTS2E.SAMPLE_DIR / f"{name}.wav").exists()
        assert (NeuTTS2E.SAMPLE_DIR / f"{name}.txt").read_text().strip()
        codes = torch.load(NeuTTS2E.SAMPLE_DIR / f"{name}.pt")
        assert codes.ndim == 1 and len(codes) > 0


def test_2e_validation():
    model = NeuTTS2E.__new__(NeuTTS2E)
    model._speaker_refs = {}

    with pytest.raises(ValueError, match="Unknown speaker"):
        model._speaker("dave")
    codes, ref_text = model._speaker("emily")
    assert len(codes) > 0 and len(ref_text) > 0
