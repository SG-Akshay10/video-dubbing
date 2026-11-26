"""Mux standard-rate dubbed clips onto the original picture timeline."""

import argparse
import json
import wave
from array import array
from dataclasses import dataclass
from fractions import Fraction
from heapq import merge
from math import lcm
from pathlib import Path
from tempfile import TemporaryDirectory

import av

from ingest import MICROSECONDS_PER_SECOND


@dataclass(frozen=True)
class Clip:
    cue_id: str
    start_us: int
    end_us: int
    path: Path
    sample_count: int


def _read_manifest(path):
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        raise ValueError(f"Invalid manifest JSON in {path.name}.") from None
    if not isinstance(document, dict):
        raise ValueError(f"Expected a JSON object in {path.name}.")
    return document


def _load_clips(speech, speech_path, duration_us, rate):
    if not isinstance(speech.get("clips"), list):
        raise ValueError("Expected a clips array in the speech manifest.")
    clips = []
    ids = set()
    for index, entry in enumerate(speech["clips"], start=1):
        cue_id = entry.get("cue_id") if isinstance(entry, dict) else None
        if not isinstance(cue_id, str) or not cue_id.strip():
            raise ValueError(f"Speech clip {index} has no cue ID.")
        if cue_id in ids:
            raise ValueError(f"Duplicate speech clip for cue {cue_id}.")
        ids.add(cue_id)
        start, end = entry.get("start_us"), entry.get("end_us")
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= duration_us:
            raise ValueError(f"Cue {cue_id} has an invalid source slot.")
        path = entry.get("path")
        if not isinstance(path, str) or not path or Path(path).is_absolute():
            raise ValueError(f"Cue {cue_id} has no manifest-relative clip path.")
        count = entry.get("sample_count")
        if type(count) is not int or count <= 0 or _to_us(count, rate) != entry.get("duration_us"):
            raise ValueError(f"Cue {cue_id} has an invalid clip sample count or duration.")
        clips.append(Clip(cue_id, start, end, speech_path.parent / path, count))
    return sorted(clips, key=lambda clip: clip.start_us)


def _to_us(samples, rate):
    return round(Fraction(samples * MICROSECONDS_PER_SECOND, rate))


def _read_clip(clip, rate):
    try:
        with wave.open(str(clip.path), "rb") as wav:
            if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, rate):
                raise ValueError(f"Cue {clip.cue_id} must be a mono {rate} Hz PCM16 WAV.")
            if wav.getnframes() != clip.sample_count:
                raise ValueError(f"Cue {clip.cue_id} WAV sample count differs from the speech manifest.")
            pcm = wav.readframes(clip.sample_count)
    except (OSError, EOFError, wave.Error) as exc:
        raise ValueError(f"Could not read WAV for cue {clip.cue_id}: {clip.path}") from exc
    if len(pcm) != clip.sample_count * 2:
        raise ValueError(f"Cue {clip.cue_id} has truncated WAV audio.")
    return array("h", pcm)


def _mix_clips(clips, rate, picture_samples):
    mixed = array("i", [0]) * picture_samples
    starts = [round(Fraction(clip.start_us * rate, MICROSECONDS_PER_SECOND)) for clip in clips]
    placements, overruns = [], []
    for index, (clip, start) in enumerate(zip(clips, starts, strict=True)):
        samples = _read_clip(clip, rate)
        clip_end = start + len(samples)
        rendered_end = min(clip_end, picture_samples)
        for offset in range(rendered_end - start):
            mixed[start + offset] += samples[offset]
        placements.append({
            "cue_id": clip.cue_id,
            "start_us": clip.start_us,
            "end_us": clip.end_us,
            "start_sample": start,
            "sample_count": len(samples),
            "rendered_sample_count": rendered_end - start,
        })

        if index + 1 < len(clips) and clip_end > starts[index + 1]:
            overruns.append({
                "cue_id": clip.cue_id, "type": "next_cue", "next_cue_id": clips[index + 1].cue_id,
                "overrun_us": _to_us(clip_end - starts[index + 1], rate),
            })
        if clip_end > picture_samples:
            overruns.append({
                "cue_id": clip.cue_id, "type": "video_end",
                "overrun_us": _to_us(clip_end - picture_samples, rate),
            })

    peak = max(max(mixed), -min(mixed))
    gain = 32767 / peak if peak > 32767 else 1.0
    pcm = array("h", (round(sample * gain) for sample in mixed))
    return pcm, placements, overruns, peak, gain


def _video_packets(source, video, output_video):
    for packet in source.demux(video):
        if packet.size:
            packet.stream = output_video
            yield packet


def _audio_packets(pcm, rate, start, stream):
    time_base = Fraction(1, rate)
    for position in range(0, len(pcm), 1024):
        samples = pcm[position:position + 1024]
        frame = av.AudioFrame(format="s16", layout="mono", samples=len(samples))
        frame.sample_rate = rate
        frame.time_base = time_base
        frame.pts = start + position
        frame.planes[0].update(samples.tobytes())
        yield from stream.encode(frame)
    yield from stream.encode(None)


def _packet_time(packet):
    timestamp = packet.dts if packet.dts is not None else packet.pts
    if timestamp is None or not packet.time_base:
        raise ValueError("Source video has a packet without usable timing.")
    return timestamp * packet.time_base


def _mux_video(source, video, picture_start, pcm, rate, output_path):
    options = {"movflags": "+faststart", "avoid_negative_ts": "disabled"}
    timescale = lcm(video.time_base.denominator, rate)
    options["movie_timescale"] = str(timescale if timescale < 2**31 else MICROSECONDS_PER_SECOND)
    with av.open(str(output_path), "w", format="mp4", options=options) as output:
        output_video = output.add_stream_from_template(video)
        output_video.time_base = video.time_base
        audio = output.add_stream("aac", rate=rate)
        audio.layout = "mono"
        audio.bit_rate = 96_000
        audio_start = round(picture_start * rate)
        packets = merge(_video_packets(source, video, output_video), _audio_packets(pcm, rate, audio_start, audio),
                        key=_packet_time)
        for packet in packets:
            output.mux(packet)


def mix(media_path: Path, speech_path: Path, output_dir: Path) -> Path:
    if output_dir.exists():
        raise ValueError(f"Output directory already exists. Choose a new directory: {output_dir}")
    media = _read_manifest(media_path)
    speech = _read_manifest(speech_path)
    source_path, duration_us = media.get("source_video"), media.get("video_duration_us")
    if not isinstance(source_path, str) or not source_path or type(duration_us) is not int or duration_us <= 0:
        raise ValueError("Expected a source video path and duration in the media manifest.")
    rate, locale = speech.get("sample_rate_hz"), speech.get("target_locale")
    if (type(rate) is not int or rate <= 0 or speech.get("channels") != 1
            or speech.get("sample_format") != "pcm_s16le"):
        raise ValueError("Expected mono PCM16 audio in the speech manifest.")
    if not isinstance(locale, str) or not locale.strip():
        raise ValueError("Expected a target locale in the speech manifest.")
    clips = _load_clips(speech, speech_path, duration_us, rate)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output_dir.parent) as temp_dir, av.open(source_path) as source:
        video = source.streams.best("video")
        if video is None or video.start_time is None or not video.duration or not video.time_base:
            raise ValueError("Source video has no usable picture start or duration. Run ingest.py again.")
        picture_start = video.start_time * video.time_base
        picture_duration = video.duration * video.time_base
        if round(picture_duration * MICROSECONDS_PER_SECOND) != duration_us:
            raise ValueError("Source picture duration differs from the media manifest. Run ingest.py again.")
        picture_samples = round(picture_duration * rate)
        if picture_samples < 1:
            raise ValueError("Source picture duration is shorter than one output audio sample.")
        pcm, placements, overruns, peak, gain = _mix_clips(clips, rate, picture_samples)

        artifacts = Path(temp_dir) / "artifacts"
        artifacts.mkdir()
        video_name = "preview.mp4" if overruns else "dubbed.mp4"
        output_path = artifacts / video_name
        _mux_video(source, video, picture_start, pcm, rate, output_path)
        with av.open(str(output_path)) as rendered:
            picture = rendered.streams.video[0]
            if (len(rendered.streams.audio) != 1 or picture.start_time is None or picture.duration is None
                    or picture.start_time * picture.time_base != picture_start
                    or picture.duration * picture.time_base != picture_duration):
                raise ValueError("Export did not preserve the source picture timeline and one replacement audio track.")

        report = {
            "target_locale": locale,
            "video": video_name,
            "status": "review_required" if overruns else "complete",
            "video_duration_us": duration_us,
            "background_source": "none",
            "sample_rate_hz": rate,
            "mixed_sample_count": picture_samples,
            "pcm_peak_before_gain": peak,
            "mix_gain": gain,
            "placements": placements,
            "overruns": overruns,
        }
        (artifacts / "mix.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        artifacts.rename(output_dir)
    return output_dir / "mix.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--media", required=True, type=Path, help="Media manifest from ingest.py")
    parser.add_argument("--speech", required=True, type=Path, help="Speech manifest from synthesize.py")
    parser.add_argument("--output", required=True, type=Path, help="New directory for the MP4 and mix.json")
    args = parser.parse_args()
    try:
        report_path = mix(args.media, args.speech, args.output)
    except (OSError, ValueError, av.FFmpegError) as exc:
        parser.exit(1, f"Mix failed: {exc}\n")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    print(report_path.parent / report["video"])
    print(report_path)
    if report["overruns"]:
        print(f"Review required. See {len(report['overruns'])} boundary crossings in the mix report.")


if __name__ == "__main__":
    main()
