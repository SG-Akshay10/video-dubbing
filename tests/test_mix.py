import json
import math
import unittest
import wave
from array import array
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory

import av

from ingest import ingest
from mix import mix


RATE = 24_000


def tone(frequency, count):
    return array("h", (round(10_000 * math.sin(math.tau * frequency * i / RATE)) for i in range(count)))


def make_video(path):
    with av.open(str(path), "w") as output:
        video = output.add_stream("mpeg4", rate=10)
        video.width = video.height = 32
        video.pix_fmt = "yuv420p"
        audio = output.add_stream("aac", rate=RATE)
        audio.layout = "mono"
        for pts in (2, 3, 4, 6, *range(7, 22)):
            frame = av.VideoFrame(32, 32, "yuv420p")
            for plane in frame.planes:
                plane.update(bytes(plane.buffer_size))
            frame.time_base = Fraction(1, 10)
            frame.pts = pts
            output.mux(video.encode(frame))
        output.mux(video.encode(None))
        original = tone(880, 2 * RATE)
        frame = av.AudioFrame(format="s16", layout="mono", samples=len(original))
        frame.sample_rate = RATE
        frame.time_base = Fraction(1, RATE)
        frame.pts = 0
        frame.planes[0].update(original.tobytes())
        output.mux(audio.encode(frame))
        output.mux(audio.encode(None))


def picture(path):
    with av.open(str(path)) as media:
        video = media.streams.video[0]
        metadata = (video.codec_context.name, video.start_time * video.time_base, video.duration * video.time_base)
        timestamps = [frame.pts * frame.time_base for frame in media.decode(video)]
    return metadata, timestamps


def decoded_rms(path, start, end):
    square_sum = count = 0
    with av.open(str(path)) as media:
        audio, = media.streams.audio
        assert audio.codec_context.name == "aac"
        resampler = av.AudioResampler(format="s16", layout="mono", rate=RATE)
        for frame in media.decode(audio):
            for decoded in resampler.resample(frame):
                position = round(decoded.pts * decoded.time_base * RATE)
                low, high = max(0, start - position), min(decoded.samples, end - position)
                if low < high:
                    pcm = array("h", bytes(memoryview(decoded.planes[0])[low * 2:high * 2]))
                    count += len(pcm)
                    square_sum += sum(sample * sample for sample in pcm)
    assert count == end - start
    return math.sqrt(square_sum / count)


class MixTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.video = self.root / "source.mp4"
        make_video(self.video)
        self.media = ingest(self.video, self.root / "media")
        self.duration = json.loads(self.media.read_text())["video_duration_us"]
        self.speech = self.root / "speech.json"

    def write_clips(self, first_duration, second_duration):
        clips = []
        for index, start, length, hz in ((1, 250_000, first_duration, 440), (2, 1_000_000, second_duration, 660)):
            samples = round(Fraction(length * RATE, 1_000_000))
            name = f"clip{index}.wav"
            with wave.open(str(self.root / name), "wb") as audio:
                audio.setparams((1, 2, RATE, 0, "NONE", ""))
                audio.writeframes(tone(hz, samples).tobytes())
            clips.append({
                "cue_id": f"cue-{index}", "start_us": start, "end_us": start + 250_000,
                "path": name, "duration_us": length, "sample_count": samples,
            })
        self.speech.write_text(json.dumps({
            "target_locale": "ta-IN", "sample_rate_hz": RATE, "channels": 1,
            "sample_format": "pcm_s16le", "clips": clips,
        }))

    def test_fitting_clips_export_with_unchanged_variable_picture_timing(self):
        self.write_clips(250_000, 400_000)
        output = self.root / "clean"
        report = json.loads(mix(self.media, self.speech, output).read_text())
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["overruns"], [])
        result = output / "dubbed.mp4"
        self.assertEqual(picture(self.video), picture(result))
        (_, picture_start, _), timestamps = picture(result)
        self.assertGreater(picture_start, 0)
        self.assertGreater(len({b - a for a, b in zip(timestamps, timestamps[1:])}), 1)
        origin = round(picture_start * RATE)
        for start, end, audible in ((0.12, 0.20, False), (0.30, 0.33, True), (0.45, 0.49, True),
                                    (0.58, 0.85, False), (1.03, 1.08, True), (1.34, 1.38, True),
                                    (1.55, 1.70, False)):
            with self.subTest(start=start, end=end):
                rms = decoded_rms(result, origin + round(start * RATE), origin + round(end * RATE))
                if audible:
                    self.assertGreater(rms, 5000)
                else:
                    self.assertLess(rms, 100)
        self.assertEqual([row["rendered_sample_count"] for row in report["placements"]], [6000, 9600])

    def test_boundary_crossings_produce_a_labeled_preview(self):
        self.write_clips(900_000, self.duration - 800_000)
        output = self.root / "overrun"
        report = json.loads(mix(self.media, self.speech, output).read_text())
        self.assertEqual(report["status"], "review_required")
        self.assertEqual(report["overruns"], [
            {"cue_id": "cue-1", "type": "next_cue", "next_cue_id": "cue-2", "overrun_us": 150_000},
            {"cue_id": "cue-2", "type": "video_end", "overrun_us": 200_000},
        ])
        self.assertEqual(report["placements"][0]["rendered_sample_count"], 21_600)
        last = report["placements"][1]
        self.assertEqual(last["start_sample"] + last["rendered_sample_count"], report["mixed_sample_count"])
        self.assertEqual(picture(self.video), picture(output / "preview.mp4"))
        self.assertFalse((output / "dubbed.mp4").exists())
