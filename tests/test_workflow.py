import base64
import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from urllib.error import HTTPError

from dub import main
from llm import AIConfig
from test_mix import make_video, picture, tone
from test_synthesis import wav_bytes
from test_transcription import Response


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.source = self.root / "source.mp4"
        self.output = self.root / "runs"
        self.argv = ["dub.py", "--input", str(self.source), "--target-locale", "ta-IN", "--output", str(self.output)]
        make_video(self.source)
        self.openai = AIConfig("test-key", "https://api.openai.com/v1", "whisper-1", "gpt-4.1-mini", "gpt-4o-mini-tts")
        self.sarvam = AIConfig("test-key", "https://api.sarvam.ai", "saaras:v4", "sarvam-105b", "bulbul:v4-flash")

    def completion(self, request):
        payload = json.loads(request.data)
        cues = json.loads(payload["messages"][1]["content"])["cues"]
        words = {"Hello.": "வணக்கம்.", "Thank you.": "நன்றி."}
        translations = [{"id": cue["id"], "text": words[cue["text"]]} for cue in cues]
        return Response({"choices": [{"message": {"content": json.dumps({"translations": translations})}}]})

    def respond(self, request, *, timeout):
        path = request.selector
        if path == "/v1/chat/completions":
            return self.completion(request)
        if path == "/v1/audio/transcriptions":
            return Response({"text": "Hello. Thank you.", "segments": [
                {"start": 0.45, "end": 0.70, "text": "Hello."},
                {"start": 1.20, "end": 1.50, "text": "Thank you."},
            ]})
        if path == "/v1/audio/speech":
            return Response(tone(440, 4800).tobytes(), "audio/pcm")
        if path == "/text-to-speech":
            return Response({"audios": [base64.b64encode(wav_bytes(tone(440, 4800).tobytes())).decode()]})
        responses = {
            "/speech-to-text/job/v1": {"job_id": "job_1", "storage_container_type": "Azure_V1"},
            "/speech-to-text/job/v1/upload-files": {"upload_urls": {"analysis.wav": {
                "file_url": "https://storage.example/upload.wav",
            }}},
            "/speech-to-text/job/v1/job_1/start": {"job_state": "Running"},
            "/speech-to-text/job/v1/job_1/status": {"job_state": "Completed", "job_details": [{
                "state": "Success", "outputs": [{"file_name": "0.json"}],
            }]},
            "/speech-to-text/job/v1/download-files": {"download_urls": {"0.json": {
                "file_url": "https://storage.example/0.json",
            }}},
            "/0.json": {"transcript": "Hello. Thank you.", "diarized_transcript": {"entries": [
                {"start_time_seconds": 0.45, "end_time_seconds": 0.70, "transcript": "Hello.", "speaker_id": "0"},
                {"start_time_seconds": 1.20, "end_time_seconds": 1.50, "transcript": "Thank you.", "speaker_id": "0"},
            ]}},
        }
        if path == "/upload.wav":
            self.assertEqual(request.get_method(), "PUT")
            self.assertEqual(request.data.read(4), b"RIFF")
            return Response(b"", "application/octet-stream")
        return Response(responses[path])

    @patch("llm.build_opener")
    def test_same_command_produces_playable_dubs_under_both_configurations(self, opener):
        opener.return_value.open.side_effect = self.respond
        for config in (self.sarvam, self.openai):
            with self.subTest(provider=config.provider_name):
                stdout = io.StringIO()
                with patch("dub.AIConfig.from_env", return_value=config), patch("sys.argv", self.argv), redirect_stdout(stdout):
                    main()
                run_dir = self.output / config.provider_name
                report_path = run_dir / "run.json"
                report = json.loads(report_path.read_text())
                self.assertEqual(report["target_locale"], "ta-IN")
                self.assertEqual(report["status"], "complete")
                self.assertEqual(report["placed_cue_ids"], ["cue-000001", "cue-000002"])
                self.assertTrue(report["playback"]["verified"])
                self.assertGreater(report["playback"]["audio_frames"], 0)
                for artifact in report["artifacts"].values():
                    self.assertTrue((run_dir / artifact).is_file(), artifact)
                video = run_dir / report["artifacts"]["video"]
                self.assertEqual(picture(video), picture(self.source))
                self.assertEqual(stdout.getvalue().splitlines(), [str(video), str(report_path)])

    @patch("llm.build_opener")
    def test_provider_failure_exits_nonzero_and_names_the_stage(self, opener):
        opener.return_value.open.side_effect = HTTPError(
            "https://api.openai.com/v1/audio/transcriptions", 401, "provider details", None, io.BytesIO(),
        )
        stderr = io.StringIO()
        with patch("dub.AIConfig.from_env", return_value=self.openai), patch("sys.argv", self.argv), redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as caught:
                main()
        self.assertEqual(caught.exception.code, 1)
        self.assertIn("Transcription failed: OpenAI transcription failed (HTTP 401)", stderr.getvalue())
        self.assertEqual(list(self.output.iterdir()), [])
