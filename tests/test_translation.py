import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from llm import AIConfig
from test_transcription import Response
from translate import translate


class TranslationTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.source_path = Path(directory.name) / "transcript.json"
        self.cues = [
            {"id": "cue-001", "start_us": 0, "end_us": 2_000_000,
             "text": "Meena paid 25 rupees.", "speaker_id": "0", "review_flags": []},
            {"id": "cue-002", "start_us": 2_200_000, "end_us": 3_400_000,
             "text": "Don't send it yet.", "speaker_id": "1", "review_flags": ["approximate_timing"]},
        ]
        self.source_path.write_text(json.dumps({"source_language": "en", "cues": self.cues}), encoding="utf-8")
        self.sarvam = AIConfig("test-key", "https://api.sarvam.ai", chat_model="sarvam-105b")

    @staticmethod
    def completion(translations):
        content = json.dumps({"translations": translations}, ensure_ascii=False)
        return Response({"choices": [{"message": {"content": content}, "finish_reason": "stop"}]})

    @patch("llm.build_opener")
    def test_both_chat_paths_preserve_source_cues_and_target_locale(self, opener):
        lines = {"cue-001": "மீனா 25 ரூபாய் செலுத்தினார்.", "cue-002": "இன்னும் அதை அனுப்பாதே."}
        for config, endpoint, header in (
            (AIConfig("test-key", "https://gateway.OpenAI.example/v1", chat_model="openai-test"),
             "https://gateway.OpenAI.example/v1/chat/completions", ("Authorization", "Bearer test-key")),
            (self.sarvam, "https://api.sarvam.ai/v1/chat/completions", ("Api-subscription-key", "test-key")),
        ):
            with self.subTest(endpoint=endpoint):
                opener.return_value.open.return_value = self.completion([
                    {"id": cue_id, "text": lines[cue_id]} for cue_id in reversed(lines)
                ])
                result = translate(self.source_path, "ta-IN", config=config)
                self.assertEqual(result, {"source_language": "en", "target_locale": "ta-IN", "cues": [
                    {**cue, "translated_text": lines[cue["id"]]} for cue in self.cues
                ]})
                request = opener.return_value.open.call_args.args[0]
                self.assertEqual(request.full_url, endpoint)
                self.assertEqual(request.get_header(header[0]), header[1])
                payload = json.loads(request.data)
                self.assertEqual(payload["model"], config.chat_model)
                self.assertEqual(payload["response_format"], {"type": "json_object"})
                self.assertEqual(json.loads(payload["messages"][1]["content"]), {
                    "target_locale": "ta-IN",
                    "cues": [{"id": cue["id"], "text": cue["text"]} for cue in self.cues],
                })

    @patch("llm.build_opener")
    def test_invalid_translation_identifies_the_affected_cue(self, opener):
        first = {"id": "cue-001", "text": "மீனா 25 ரூபாய் செலுத்தினார்."}
        for translations, error in (
            ([first, {"id": "cue-002", "text": "  "}], "Empty or invalid translation for cue cue-002"),
            ([first], "Missing translation for cues cue-002"),
            ([first, first], "Duplicate translation for cue cue-001"),
            ([first, {"id": "cue-003", "text": "வேண்டாம்."}], "unknown cue cue-003"),
        ):
            with self.subTest(error=error):
                opener.return_value.open.return_value = self.completion(translations)
                with self.assertRaisesRegex(ValueError, error):
                    translate(self.source_path, "ta-IN", config=self.sarvam)
