import io
import json
import os
import unittest
import wave
from email.message import Message
from email.parser import BytesParser
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from urllib.error import HTTPError

from llm import AIConfig, ProviderError, TranscriptionRequest, TranscriptionResult, call_ai
from transcribe import transcribe


class Response(io.BytesIO):
    def __init__(self, body, content_type="application/json"):
        super().__init__(json.dumps(body).encode() if content_type == "application/json" else body)
        self.headers = Message()
        self.headers["Content-Type"] = content_type


def form_fields(request):
    header = f"Content-Type: {request.get_header('Content-type')}\r\n\r\n".encode()
    return {
        part.get_param("name", header="content-disposition"): part.get_payload(decode=True)
        for part in BytesParser().parsebytes(header + request.data).get_payload()
    }


class ProviderTests(unittest.TestCase):
    def test_config_reads_dotenv_with_shell_overrides(self):
        with TemporaryDirectory() as directory:
            dotenv_path = Path(directory) / ".env"
            dotenv_path.write_text(
                'AI_API_KEY="file-${TOKEN}"\n'
                'AI_BASE_URL=https://api.sarvam.ai\n'
                'AI_STT_MODEL=saaras:v4\n',
                encoding="utf-8",
            )
            with patch("llm.PROJECT_ENV_PATH", dotenv_path), patch.dict(os.environ, {}, clear=True):
                self.assertEqual(AIConfig.from_env(), AIConfig("file-${TOKEN}", "https://api.sarvam.ai", "saaras:v4"))
                with patch.dict(os.environ, {"AI_API_KEY": "exported-key"}):
                    self.assertEqual(AIConfig.from_env().api_key, "exported-key")

    @patch("llm.build_opener")
    def test_openai_dispatch_and_timestamped_result(self, opener):
        transport = opener.return_value
        transport.open.return_value = Response({
            "text": "Hello.", "segments": [{"start": 0.25, "end": 1, "text": "Hello."}],
        })
        request = TranscriptionRequest(b"wav contents")
        openai = AIConfig("test-key", "https://proxy.OpenAI.example/gateway/", "whisper-1")
        result = call_ai(request, config=openai)
        self.assertIsInstance(result, TranscriptionResult)
        self.assertEqual((result.segments[0].start_us, result.segments[0].end_us), (250_000, 1_000_000))

        first = transport.open.call_args.args[0]
        self.assertEqual(first.full_url.lower(), "https://proxy.openai.example/gateway/audio/transcriptions")
        self.assertEqual(first.get_header("Authorization"), "Bearer test-key")
        self.assertEqual(form_fields(first)["timestamp_granularities[]"], b"segment")
        self.assertEqual(form_fields(first)["file"], request.wav)

    @patch("llm.build_opener")
    def test_http_failure_redacts_credentials_and_response_body(self, opener):
        config = AIConfig("test-key", "https://api.openai.com", "whisper-1")
        opener.return_value.open.side_effect = HTTPError(
            config.base_url, 401, "test-key", None, io.BytesIO(b"test-key"),
        )
        with self.assertRaises(ProviderError) as caught:
            call_ai(TranscriptionRequest(b"wav contents"), config=config)
        self.assertEqual(str(caught.exception), "OpenAI transcription failed (HTTP 401).")
        self.assertNotIn("test-key", repr(config))
        with self.assertRaises(ValueError):
            AIConfig("test-key", "https://user:secret@openai.example", "whisper-1")


class CueTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.manifest_path = Path(directory.name) / "media.json"
        self.manifest_path.write_text(json.dumps({
            "video_duration_us": 1_500_000,
            "analysis_audio": {"path": "analysis.wav", "offset_us": -100_000},
        }))
        self.wav_path = self.manifest_path.with_name("analysis.wav")

    def write_audio(self, pcm):
        with wave.open(str(self.wav_path), "wb") as wav:
            wav.setparams((1, 2, 16000, 0, "NONE", ""))
            wav.writeframes(pcm)

    @patch("llm.build_opener")
    def test_text_only_results_are_transcribed_in_deterministic_speech_windows(self, opener):
        signal, silence = b"\xe8\x03", b"\x00\x00"
        self.write_audio(signal * 8000 + silence * 8000 + signal * 9600)
        opener.return_value.open.side_effect = [
            Response({"text": "First phrase."}), Response(b"Second phrase.", "text/plain"),
        ]
        config = AIConfig("test-key", "https://api.openai.com/v1", "gpt-transcribe")
        cues = transcribe(self.manifest_path, config=config)
        self.assertEqual([(cue.start_us, cue.end_us) for cue in cues], [(0, 400_000), (900_000, 1_500_000)])
        self.assertEqual([cue.id for cue in cues], ["cue-000001", "cue-000002"])
        self.assertEqual([cue.text for cue in cues], ["First phrase.", "Second phrase."])
        self.assertTrue(all(cue.review_flags == ("approximate_timing",) for cue in cues))
        for call, count in zip(opener.return_value.open.call_args_list, (8000, 9600), strict=True):
            fields = form_fields(call.args[0])
            self.assertEqual(fields["languages[]"], b"en")
            with wave.open(io.BytesIO(fields["file"])) as wav:
                self.assertEqual(wav.getnframes(), count)

    @patch("llm.build_opener")
    def test_expected_timestamps_can_fall_back_to_windows(self, opener):
        self.write_audio(b"\x00\x00" * 8000 + b"\xe8\x03" * 8000)
        opener.return_value.open.side_effect = [
            Response({"text": "A full upload without timestamps."}),
            Response({"text": "A speech window."}),
        ]
        config = AIConfig("test-key", "https://api.openai.com", "whisper-1")
        cue, = transcribe(self.manifest_path, config=config)
        self.assertEqual((cue.start_us, cue.end_us, cue.text), (400_000, 900_000, "A speech window."))
        self.assertEqual(cue.timestamp_provenance, "speech_window")

    @patch("llm.sleep")
    @patch("llm.build_opener")
    def test_sarvam_batches_the_entire_wav_and_preserves_speakers(self, opener, sleep):
        self.write_audio(b"\xe8\x03" * 16000 * 29)
        manifest = json.loads(self.manifest_path.read_text())
        manifest["video_duration_us"] = 28_800_000
        self.manifest_path.write_text(json.dumps(manifest))
        upload_url = "https://storage.example/audio.wav?signature=upload"
        download_url = "https://storage.example/result.json?signature=download"
        responses = iter((
            Response({"job_id": "job_123", "storage_container_type": "Azure_V1"}),
            Response({"upload_urls": {"analysis.wav": {"file_url": upload_url}}}),
            Response(b"", "application/octet-stream"),
            Response({"job_state": "Running"}),
            Response({"job_state": "Running"}),
            Response({"job_state": "Completed", "job_details": [
                {"state": "Success", "outputs": [{"file_name": "0.json"}]},
            ]}),
            Response({"download_urls": {"0.json": {"file_url": download_url}}}),
            Response({"transcript": "First. Second. First again.", "diarized_transcript": {"entries": [
                {"start_time_seconds": 28.7, "end_time_seconds": 29, "transcript": "First again.", "speaker_id": "0"},
                {"start_time_seconds": 0, "end_time_seconds": 0.7, "transcript": "First.", "speaker_id": "0"},
                {"start_time_seconds": 0.6, "end_time_seconds": 1.2, "transcript": "Second.", "speaker_id": 1},
            ]}}),
        ))

        def respond(request, *, timeout):
            if request.get_method() == "PUT":
                self.assertEqual(request.data.read(), self.wav_path.read_bytes())
                self.assertEqual(request.get_header("Content-length"), str(self.wav_path.stat().st_size))
                self.assertEqual(request.get_header("X-ms-blob-type"), "BlockBlob")
            return next(responses)

        transport = opener.return_value
        transport.open.side_effect = respond
        config = AIConfig("test-key", "https://proxy.example/openai", "saaras:v4")
        cues = transcribe(self.manifest_path, config=config)
        self.assertEqual([(cue.start_us, cue.end_us, cue.speaker_id) for cue in cues], [
            (0, 600_000, "0"), (500_000, 1_100_000, "1"), (28_600_000, 28_800_000, "0"),
        ])
        requests = [call.args[0] for call in transport.open.call_args_list]
        self.assertEqual([request.get_method() for request in requests],
                         ["POST", "POST", "PUT", "POST", "GET", "GET", "POST", "GET"])
        job_url = "https://proxy.example/openai/speech-to-text/job/v1"
        self.assertEqual([request.full_url for request in requests], [
            job_url, f"{job_url}/upload-files", upload_url, f"{job_url}/job_123/start",
            f"{job_url}/job_123/status", f"{job_url}/job_123/status", f"{job_url}/download-files", download_url,
        ])
        parameters = json.loads(requests[0].data)["job_parameters"]
        self.assertTrue(parameters["with_diarization"])
        self.assertEqual(parameters["model"], "saaras:v4")
        self.assertEqual(json.loads(requests[1].data), {"job_id": "job_123", "files": ["analysis.wav"]})
        self.assertEqual(json.loads(requests[6].data), {"job_id": "job_123", "files": ["0.json"]})
        for request in requests:
            expected = "test-key" if request.full_url.startswith(job_url) else None
            self.assertEqual(request.get_header("Api-subscription-key"), expected)
        sleep.assert_called_once_with(5)

    @patch("llm.build_opener")
    def test_digital_silence_needs_no_request(self, opener):
        self.write_audio(b"\x00\x00" * 16000)
        config = AIConfig("test-key", "https://api.sarvam.ai", "saaras:v4")
        self.assertEqual(transcribe(self.manifest_path, config=config), [])
        opener.assert_not_called()
