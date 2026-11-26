"""Render one standard-rate speech clip per translated cue."""

import argparse
import json
import wave
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory

from ingest import MICROSECONDS_PER_SECOND
from llm import AIConfig, ProviderError, SPEECH_SAMPLE_RATE_HZ, SPEECH_SAMPLE_WIDTH_BYTES, SpeechRequest, call_ai
from translate import read_english_cues


def synthesize(translation_path: Path, output_dir: Path, *, config: AIConfig) -> Path:
    if output_dir.exists():
        raise ValueError(f"Output directory already exists. Choose a new directory: {output_dir}")
    document = read_english_cues(translation_path)
    locale = document.get("target_locale")
    if not isinstance(locale, str) or not locale.strip():
        raise ValueError("Expected a target_locale in the JSON produced by translate.py.")
    for cue in document["cues"]:
        text = cue.get("translated_text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Cue {cue['id']} has no translated text.")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output_dir.parent) as temp_dir:
        artifacts = Path(temp_dir) / "artifacts"
        (artifacts / "clips").mkdir(parents=True)
        clips = []
        voice = None
        for index, cue in enumerate(document["cues"], start=1):
            cue_id = cue["id"]
            try:
                result = call_ai(SpeechRequest(cue["translated_text"].strip(), locale), config=config)
            except (ValueError, ProviderError) as exc:
                raise ValueError(f"Cue {cue_id}: {exc}") from None
            voice = result.voice
            path = f"clips/cue-{index:06d}.wav"
            with wave.open(str(artifacts / path), "wb") as wav:
                wav.setparams((1, SPEECH_SAMPLE_WIDTH_BYTES, SPEECH_SAMPLE_RATE_HZ, 0, "NONE", ""))
                wav.writeframes(result.pcm)
            duration_us = round(Fraction(result.sample_count * MICROSECONDS_PER_SECOND, SPEECH_SAMPLE_RATE_HZ))
            clips.append({
                "cue_id": cue_id,
                "start_us": cue["start_us"],
                "end_us": cue["end_us"],
                "path": path,
                "sample_count": result.sample_count,
                "duration_us": duration_us,
            })

        manifest = {
            "target_locale": locale,
            "tts_model": config.tts_model,
            "voice": voice,
            "sample_rate_hz": SPEECH_SAMPLE_RATE_HZ,
            "channels": 1,
            "sample_format": "pcm_s16le",
            "clips": clips,
        }
        manifest_name = "speech.json"
        (artifacts / manifest_name).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        artifacts.rename(output_dir)
    return output_dir / manifest_name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Translated transcript from translate.py")
    parser.add_argument("--output", required=True, type=Path, help="New directory for speech.json and WAV clips")
    args = parser.parse_args()
    try:
        manifest = synthesize(args.input, args.output, config=AIConfig.from_env())
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Speech synthesis failed: {exc}\n")
    print(manifest)


if __name__ == "__main__":
    main()
