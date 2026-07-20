import soundfile as sf
from neutts import NeuTTS2E


def main(input_text, speaker, emotion, backbone, device, seed, output_path="output.wav"):
    tts = NeuTTS2E(
        backbone_repo=backbone,
        backbone_device=device,
        seed=seed,
    )

    print(f"Generating '{emotion}' audio for speaker '{speaker}': {input_text}")
    wav = tts.infer(input_text, speaker=speaker, emotion=emotion)

    print(f"Saving output to {output_path}")
    sf.write(output_path, wav, 24000)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NeuTTS-2E Example")
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
        default="neuphonic/qwen3-0.2b-en-emotional-grpo-500-16-06-26",
        help="Huggingface repo or local path containing the backbone checkpoint",
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
    parser.add_argument(
        "--output_path",
        type=str,
        default="output.wav",
        help="Path to save the output audio",
    )
    args = parser.parse_args()
    main(
        input_text=args.input_text,
        speaker=args.speaker,
        emotion=args.emotion,
        backbone=args.backbone,
        device=args.device,
        seed=args.seed,
        output_path=args.output_path,
    )
