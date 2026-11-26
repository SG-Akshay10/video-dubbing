"""Produce timestamped English cues from a Task 1 media manifest."""

import argparse
import io
import json
import sys
import wave
from array import array
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

from ingest import MICROSECONDS_PER_SECOND, SAMPLE_RATE_HZ, SAMPLE_WIDTH_BYTES
from llm import AIConfig, ProviderError, TranscriptionRequest, call_ai


UPLOAD_SAMPLES = 28 * SAMPLE_RATE_HZ
BATCH_MAX_SAMPLES = 2 * 60 * 60 * SAMPLE_RATE_HZ
WINDOW_SAMPLES = 8 * SAMPLE_RATE_HZ
FRAME_SAMPLES = SAMPLE_RATE_HZ // 50  # 20 ms
PAUSE_SAMPLES = 15 * FRAME_SAMPLES  # 300 ms
RMS_THRESHOLD = 180  # Roughly -45 dBFS for 16-bit PCM.


@dataclass(frozen=True)
class Cue:
    id: str
    start_us: int
    end_us: int
    text: str
    timestamp_provenance: str
    review_flags: tuple[str, ...]
    speaker_id: str | None = None


def _speech_windows(pcm):
    """Yield bounded sample ranges separated by low-energy pauses."""
    samples = array("h", pcm)
    if sys.byteorder == "big":
        samples.byteswap()
    start = end = None
    for position in range(0, len(samples), FRAME_SAMPLES):
        frame = samples[position:position + FRAME_SAMPLES]
        if sum(sample * sample for sample in frame) < len(frame) * RMS_THRESHOLD**2:
            continue
        frame_end = position + len(frame)
        if start is None:
            start = position
        elif position - end >= PAUSE_SAMPLES or frame_end - start > WINDOW_SAMPLES:
            yield start, end
            start = position
        end = frame_end
    if start is not None:
        yield start, end


def _transcribe_pcm(pcm, config):
    wav = io.BytesIO()
    with wave.open(wav, "wb") as output:
        output.setparams((1, SAMPLE_WIDTH_BYTES, SAMPLE_RATE_HZ, 0, "NONE", ""))
        output.writeframes(pcm)
    return call_ai(TranscriptionRequest(wav.getvalue()), config=config)


def _sample_time_us(sample):
    return round(Fraction(sample * MICROSECONDS_PER_SECOND, SAMPLE_RATE_HZ))


def _chunk_transcript(pcm, config):
    """Yield start/end microseconds relative to the supplied PCM chunk."""
    sample_count = len(pcm) // SAMPLE_WIDTH_BYTES
    chunk_duration_us = _sample_time_us(sample_count)
    result = None
    if config.supports_stt_timestamps:
        result = _transcribe_pcm(pcm, config)
        if result.segments:
            for segment in result.segments:
                yield max(0, segment.start_us), min(chunk_duration_us, segment.end_us), segment.text, "provider", None
            return
        if not result.text:
            return

    for start, end in _speech_windows(pcm):
        if result is not None and start == 0 and end == sample_count:
            text = result.text
        else:
            window = pcm[start * SAMPLE_WIDTH_BYTES:end * SAMPLE_WIDTH_BYTES]
            text = _transcribe_pcm(window, config).text
        if text:
            yield _sample_time_us(start), _sample_time_us(end), text, "speech_window", None


def _pcm_chunks(audio):
    while audio.tell() < audio.getnframes():
        start = audio.tell()
        count = min(UPLOAD_SAMPLES, audio.getnframes() - start)
        pcm = audio.readframes(count)
        if len(pcm) != count * SAMPLE_WIDTH_BYTES:
            raise ValueError("Analysis audio is truncated. Run ingest.py again.")
        yield start, pcm


def _audio_transcript(audio, wav_path, config):
    """Yield timed segments relative to WAV sample zero."""
    if config.uses_batch_stt:
        if audio.getnframes() > BATCH_MAX_SAMPLES:
            raise ValueError("Sarvam Batch transcription accepts up to two hours of audio per file.")
        has_audio = False
        for _, pcm in _pcm_chunks(audio):
            has_audio = has_audio or any(pcm)
        if not has_audio:
            return

        result = call_ai(TranscriptionRequest(wav_path), config=config)
        if result.text and not result.segments:
            raise ProviderError("Sarvam Batch transcription returned text without timestamps.")
        wav_end = _sample_time_us(audio.getnframes())
        for segment in result.segments:
            yield (max(0, segment.start_us), min(wav_end, segment.end_us), segment.text,
                   "provider", segment.speaker_id)
        return

    for start_sample, pcm in _pcm_chunks(audio):
        if any(pcm):
            chunk_start = _sample_time_us(start_sample)
            for start, end, text, provenance, speaker_id in _chunk_transcript(pcm, config):
                yield chunk_start + start, chunk_start + end, text, provenance, speaker_id


def transcribe(manifest_path: Path, *, config: AIConfig) -> list[Cue]:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        analysis = manifest["analysis_audio"]
        wav_path = manifest_path.parent / analysis["path"]
        offset_us = analysis["offset_us"]
        duration_us = manifest["video_duration_us"]
        if type(offset_us) is not int or type(duration_us) is not int or duration_us <= 0:
            raise ValueError
    except (KeyError, TypeError, ValueError):
        raise ValueError("Invalid media manifest. Pass the media.json produced by ingest.py.") from None

    segments = []
    try:
        with wave.open(str(wav_path), "rb") as audio:
            if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) != (
                1, SAMPLE_WIDTH_BYTES, SAMPLE_RATE_HZ,
            ):
                raise ValueError("Analysis audio must be mono 16 kHz PCM16 WAV. Run ingest.py again.")
            for start, end, text, provenance, speaker_id in _audio_transcript(audio, wav_path, config):
                start_us = max(0, offset_us + start)
                end_us = min(duration_us, offset_us + end)
                if start_us < end_us:
                    flags = ()
                    if provenance == "speech_window":
                        flags = ("approximate_timing",)
                    if config.uses_batch_stt and speaker_id is None:
                        flags = ("speaker_unavailable",)
                    segments.append((start_us, end_us, text, provenance, flags, speaker_id))
    except (wave.Error, EOFError):
        raise ValueError("Could not read analysis WAV. Run ingest.py again.") from None

    segments.sort(key=lambda segment: segment[:2])
    return [
        Cue(f"cue-{index:06d}", *segment) for index, segment in enumerate(segments, start=1)
    ]


def transcribe_to_file(manifest_path: Path, output_path: Path, *, config: AIConfig) -> Path:
    if output_path.exists():
        raise ValueError("Output already exists. Choose a new transcript file.")
    cues = [asdict(cue) for cue in transcribe(manifest_path, config=config)]
    if not config.uses_batch_stt:
        for cue in cues:
            del cue["speaker_id"]
    document = {"source_language": "en", "cues": cues}
    with output_path.open("x", encoding="utf-8") as output:
        output.write(json.dumps(document, indent=2, ensure_ascii=False) + "\n")
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Media manifest from ingest.py")
    parser.add_argument("--output", required=True, type=Path, help="New English transcript JSON file")
    args = parser.parse_args()
    try:
        transcribe_to_file(args.input, args.output, config=AIConfig.from_env())
    except ProviderError as exc:
        parser.exit(1, f"{exc}\n")
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Transcription failed: {exc}\n")
    print(args.output)


if __name__ == "__main__":
    main()
