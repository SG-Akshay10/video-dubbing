"""Run the v0 video dubbing workflow from one local video."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import av

from ingest import ingest
from llm import AIConfig, ProviderError
from mix import mix
from synthesize import synthesize
from transcribe import transcribe_to_file
from translate import translate_to_file


class WorkflowError(RuntimeError):
    """A failed workflow stage and its user-facing error."""


def _verify_playback(path):
    with av.open(str(path)) as media:
        if len(media.streams.video) != 1 or len(media.streams.audio) != 1:
            raise ValueError("Expected one picture track and one dubbed audio track.")
        video_frames = audio_frames = audio_samples = 0
        for frame in media.decode(media.streams.video[0], media.streams.audio[0]):
            if frame.is_corrupt:
                raise ValueError("The exported MP4 contains a corrupt decoded frame.")
            if isinstance(frame, av.VideoFrame):
                video_frames += 1
            else:
                audio_frames += 1
                audio_samples += frame.samples
        if not video_frames or not audio_samples:
            raise ValueError("The exported MP4 did not decode both picture and audio.")
    return {"verified": True, "video_frames": video_frames, "audio_frames": audio_frames}


def dub(source_path: Path, target_locale: str, output_root: Path, *, config: AIConfig, audio_stream=None) -> Path:
    missing = [name for name, value in (
        ("AI_STT_MODEL", config.stt_model), ("AI_CHAT_MODEL", config.chat_model), ("AI_TTS_MODEL", config.tts_model),
    ) if not value.strip()]
    if missing:
        raise WorkflowError(f"Configuration failed: set {', '.join(missing)} in .env or the shell environment.")

    run_dir = output_root / config.provider_name
    if run_dir.exists():
        raise WorkflowError(f"Output already exists at {run_dir}. Choose a new output root for this provider.")
    stage = "Output setup"
    try:
        output_root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=output_root) as temporary:
            artifacts = Path(temporary) / "run"
            stage = "Video ingest"
            media_path = ingest(source_path, artifacts / "media", audio_stream)
            stage = "Transcription"
            transcript_path = transcribe_to_file(media_path, artifacts / "transcript.json", config=config)
            stage = "Translation"
            translation_path = translate_to_file(transcript_path, artifacts / "translation.json", target_locale,
                                                 config=config)
            stage = "Speech synthesis"
            speech_path = synthesize(translation_path, artifacts / "speech", config=config)
            stage = "Audio mixing"
            mix_path = mix(media_path, speech_path, artifacts / "mix")
            mix_report = json.loads(mix_path.read_text(encoding="utf-8"))
            video_path = mix_path.parent / mix_report["video"]
            stage = "Playback verification"
            playback = _verify_playback(video_path)

            stage = "Output publication"
            report = {
                "provider": config.provider_name,
                "target_locale": mix_report["target_locale"],
                "status": mix_report["status"],
                "playback": playback,
                "placed_cue_ids": [cue["cue_id"] for cue in mix_report["placements"] if cue["rendered_sample_count"]],
                "artifacts": {name: path.relative_to(artifacts).as_posix() for name, path in (
                    ("media_manifest", media_path), ("transcript", transcript_path), ("translation", translation_path),
                    ("speech_manifest", speech_path), ("mix_report", mix_path), ("video", video_path),
                )},
            }
            (artifacts / "run.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            artifacts.rename(run_dir)
    except (OSError, ValueError, ProviderError, av.FFmpegError) as exc:
        raise WorkflowError(f"{stage} failed: {exc}") from None
    return run_dir / "run.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Local English source video")
    parser.add_argument("--target-locale", required=True, help="Indian target locale, for example ta-IN")
    parser.add_argument("--output", required=True, type=Path, help="Output root for the openai or sarvam run")
    parser.add_argument("--audio-stream", type=int, help="Container index of the selected audio stream")
    args = parser.parse_args()
    try:
        config = AIConfig.from_env()
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Configuration failed: {exc}\n")
    try:
        report_path = dub(args.input, args.target_locale, args.output, config=config, audio_stream=args.audio_stream)
    except WorkflowError as exc:
        parser.exit(1, f"{exc}\n")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    print(report_path.parent / report["artifacts"]["video"])
    print(report_path)
    if report["status"] == "review_required":
        parser.exit(2, "Review required. See the mix report for cue boundary crossings.\n")


if __name__ == "__main__":
    main()
