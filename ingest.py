"""Extract transcription audio and record its position on the source timeline."""

import argparse
import json
import wave
from pathlib import Path
from tempfile import TemporaryDirectory

import av


SAMPLE_RATE_HZ = 16_000
SAMPLE_WIDTH_BYTES = 2
MICROSECONDS_PER_SECOND = 1_000_000


def select_audio_stream(container, stream_index):
    streams = container.streams.audio
    if not streams:
        raise ValueError("No audio stream found. Supply a video with an audio track.")
    if stream_index is None and len(streams) == 1:
        return streams[0]
    for stream in streams:
        if stream.index == stream_index:
            return stream
    choices = ", ".join(str(stream.index) for stream in streams)
    raise ValueError(f"Select an audio stream with --audio-stream. Available indices: {choices}.")


def extract_audio(container, stream, wav_path):
    resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE_HZ)
    first_frame_time = None
    sample_count = 0
    with wave.open(str(wav_path), "wb") as output:
        output.setparams((1, SAMPLE_WIDTH_BYTES, SAMPLE_RATE_HZ, 0, "NONE", ""))
        try:
            for frame in container.decode(stream):
                if first_frame_time is None:
                    if frame.pts is None or frame.time_base is None:
                        raise ValueError(
                            f"Audio stream {stream.index} has no usable start timestamp. "
                            "Re-export the source or select another audio stream."
                        )
                    first_frame_time = frame.pts * frame.time_base
                for converted in resampler.resample(frame):
                    size = converted.samples * SAMPLE_WIDTH_BYTES
                    output.writeframesraw(memoryview(converted.planes[0])[:size])
                    sample_count += converted.samples
            for converted in resampler.resample(None):
                size = converted.samples * SAMPLE_WIDTH_BYTES
                output.writeframesraw(memoryview(converted.planes[0])[:size])
                sample_count += converted.samples
        except av.FFmpegError as exc:
            raise ValueError(
                f"Could not decode audio stream {stream.index}. "
                f"Re-export the source or select another audio stream. ({exc})"
            ) from exc

    if sample_count == 0:
        raise ValueError(
            f"Audio stream {stream.index} yielded no samples. "
            "Re-export the source or select another audio stream."
        )
    return first_frame_time, sample_count


def ingest(source_path, output_dir, audio_stream_index=None):
    source_path = source_path.resolve()
    if not source_path.is_file():
        raise ValueError(f"Source file does not exist or is not a regular file: {source_path}")
    if output_dir.exists():
        raise ValueError(f"Output directory already exists. Choose a new directory: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    try:
        container = av.open(str(source_path), mode="r")
    except av.FFmpegError as exc:
        raise ValueError(f"Could not open {source_path} as media. Check or re-export the file. ({exc})") from exc

    with container, TemporaryDirectory(dir=output_dir.parent) as temp_dir:
        video = container.streams.best("video")
        if video is None:
            raise ValueError("No video stream found. Supply a video file with a picture track.")
        audio = select_audio_stream(container, audio_stream_index)
        if video.start_time is None or video.time_base is None or not video.duration:
            raise ValueError("No usable video start or duration. Re-export the video with valid timing.")
        video_start = video.start_time * video.time_base
        duration_us = round(video.duration * video.time_base * MICROSECONDS_PER_SECOND)

        artifacts = Path(temp_dir) / "artifacts"
        artifacts.mkdir()
        wav_name = "analysis.wav"
        audio_start, sample_count = extract_audio(container, audio, artifacts / wav_name)
        wav_offset_us = round((audio_start - video_start) * MICROSECONDS_PER_SECOND)
        stream_offset_us = wav_offset_us
        if audio.start_time is not None and audio.time_base is not None:
            stream_start = audio.start_time * audio.time_base
            stream_offset_us = round((stream_start - video_start) * MICROSECONDS_PER_SECOND)

        manifest = {
            "source_video": str(source_path),
            "video_duration_us": duration_us,
            "audio_stream": {"index": audio.index, "offset_us": stream_offset_us},
            "analysis_audio": {
                "path": wav_name,
                "offset_us": wav_offset_us,
                "sample_rate_hz": SAMPLE_RATE_HZ,
                "channels": 1,
                "sample_format": "pcm_s16le",
                "sample_count": sample_count,
            },
        }
        (artifacts / "media.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        artifacts.rename(output_dir)
    return output_dir / "media.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", dest="source", required=True, type=Path, help="Local source video")
    parser.add_argument(
        "--output", dest="output_dir", required=True, type=Path,
        help="New directory for media.json and analysis.wav",
    )
    parser.add_argument("--audio-stream", type=int, help="Container index of the selected audio stream")
    args = parser.parse_args()
    try:
        manifest_path = ingest(args.source, args.output_dir, args.audio_stream)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Ingest failed: {exc}\n")
    print(manifest_path)


if __name__ == "__main__":
    main()
