import base64
import io
import json
import unittest
import wave
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from llm import AIConfig
from synthesize import synthesize
from test_transcription import Response


def wav_bytes(pcm):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setparams((1, 2, 24_000, 0, "NONE", ""))
        wav.writeframes(pcm)
    return buffer.getvalue()


class SpeechTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.input = self.root / "translation.json"
        self.cues = [
            {"id": "cue-a", "start_us": 100, "end_us": 1_000_000,
             "text": "Hello.", "translated_text": "வணக்கம்.", "speaker_id": "0"},
            {"id": "cue-b", "start_us": 1_000_000, "end_us": 2_000_000,
             "text": "Thank you.", "translated_text": "நன்றி.", "speaker_id": "1"},
        ]
        self.write_translation("ta-IN")
        self.sarvam = AIConfig("test-key", "https://api.sarvam.ai", tts_model="bulbul:v4-flash")
        self.openai = AIConfig("test-key", "https://gateway.OpenAI.example/v1", tts_model="gpt-4o-mini-tts")
        self.samples = (b"\xe8\x03" * 4800, b"\x19\xfc" * 6000)

    def write_translation(self, locale):
        self.input.write_text(json.dumps({"source_language": "en", "target_locale": locale, "cues": self.cues}))

    @patch("llm.build_opener")
    def test_both_providers_write_measured_clips_with_one_voice(self, opener):
        for name, config, voice, endpoint in (
            ("openai", self.openai, "alloy", "https://gateway.OpenAI.example/v1/audio/speech"),
            ("sarvam", self.sarvam, "gokul_ta_narration", "https://api.sarvam.ai/text-to-speech"),
        ):
            with self.subTest(provider=name):
                connection = opener.return_value.open
                connection.reset_mock()
                if name == "openai":
                    connection.side_effect = [Response(pcm, "audio/pcm") for pcm in self.samples]
                else:
                    connection.side_effect = [
                        Response({"audios": [base64.b64encode(wav_bytes(pcm)).decode()]}) for pcm in self.samples
                    ]
                output = self.root / name
                manifest = json.loads(synthesize(self.input, output, config=config).read_text())
                self.assertEqual(manifest["voice"], voice)
                self.assertEqual(manifest["target_locale"], "ta-IN")
                self.assertEqual(manifest["tts_model"], config.tts_model)
                self.assertEqual(len(manifest["clips"]), 2)
                for cue, clip, pcm, duration, call in zip(
                    self.cues, manifest["clips"], self.samples, (200_000, 250_000),
                    connection.call_args_list, strict=True,
                ):
                    self.assertEqual((clip["cue_id"], clip["start_us"], clip["end_us"]),
                                     (cue["id"], cue["start_us"], cue["end_us"]))
                    self.assertEqual(clip["duration_us"], duration)
                    self.assertEqual(clip["sample_count"], len(pcm) // 2)
                    with wave.open(str(output / clip["path"]), "rb") as wav:
                        self.assertEqual((wav.getnchannels(), wav.getsampwidth(), wav.getframerate()), (1, 2, 24_000))
                        self.assertEqual(wav.readframes(wav.getnframes()), pcm)
                    request = call.args[0]
                    self.assertEqual(request.full_url, endpoint)
                    payload = json.loads(request.data)
                    if name == "openai":
                        self.assertEqual(request.get_header("Authorization"), "Bearer test-key")
                        self.assertEqual((payload["input"], payload["voice"], payload["speed"], payload["response_format"]),
                                         (cue["translated_text"], voice, 1.0, "pcm"))
                    else:
                        self.assertEqual(request.get_header("Api-subscription-key"), "test-key")
                        self.assertEqual((payload["text"], payload["speaker"], payload["pace"], payload["language_code"]),
                                         (cue["translated_text"], voice, 1.0, "ta-IN"))

    @patch("llm.build_opener")
    def test_truncated_second_clip_identifies_the_cue_and_leaves_no_output(self, opener):
        opener.return_value.open.side_effect = [
            Response({"audios": [base64.b64encode(wav_bytes(self.samples[0])).decode()]}),
            Response({"audios": [base64.b64encode(wav_bytes(self.samples[1])[:-6]).decode()]}),
        ]
        output = self.root / "failed"
        with self.assertRaisesRegex(ValueError, r"Cue cue-b: .* returned a truncated WAV"):
            synthesize(self.input, output, config=self.sarvam)
        self.assertFalse(output.exists())

    @patch("llm.build_opener")
    def test_unsupported_locale_identifies_the_voice_and_cue(self, opener):
        self.write_translation("as-IN")
        with self.assertRaisesRegex(ValueError, r"Cue cue-a: Sarvam speech synthesis does not support as-IN with shubh_enhi_ads"):
            synthesize(self.input, self.root / "unsupported", config=self.sarvam)
        opener.assert_not_called()
