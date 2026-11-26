"""Provider adapter for the pipeline's AI requests."""

import base64
import io
import json
import os
import sys
import wave
from array import array
from dataclasses import asdict, dataclass, field
from decimal import Decimal, DecimalException
from http.client import HTTPException
from pathlib import Path
from time import monotonic, sleep
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
from uuid import uuid4

from dotenv import dotenv_values


PROJECT_ENV_PATH = Path(__file__).resolve().with_name(".env")
SPEECH_SAMPLE_RATE_HZ = 24_000
SPEECH_SAMPLE_WIDTH_BYTES = 2
SARVAM_SPEECH_LOCALES = frozenset({
    "bn-IN", "en-IN", "gu-IN", "hi-IN", "kn-IN", "ml-IN",
    "mr-IN", "od-IN", "pa-IN", "ta-IN", "te-IN",
})
OPENAI_SPEECH_LOCALES = frozenset({"en-IN", "hi-IN", "kn-IN", "mr-IN", "ta-IN", "ur-IN"})


@dataclass(frozen=True)
class AIConfig:
    api_key: str = field(repr=False)
    base_url: str
    stt_model: str = ""
    chat_model: str = ""
    tts_model: str = ""

    def __post_init__(self):
        if not self.api_key or any(not 33 <= ord(char) <= 126 for char in self.api_key):
            raise ValueError("AI requests require a valid AI_API_KEY.")
        try:
            url = urlsplit(self.base_url)
            port = url.port
        except ValueError:
            raise ValueError("AI requests require a valid HTTPS AI_BASE_URL.") from None
        if (url.scheme != "https" or not url.hostname or url.username is not None
                or port == 0 or any(char in self.base_url for char in "\\?#")
                or any(not 33 <= ord(char) <= 126 for char in self.base_url)):
            raise ValueError("AI requests require an HTTPS AI_BASE_URL without credentials, query, or fragment.")

    @classmethod
    def from_env(cls):
        settings = dotenv_values(PROJECT_ENV_PATH, interpolate=False)
        return cls(
            os.getenv("AI_API_KEY", settings.get("AI_API_KEY") or ""),
            os.getenv("AI_BASE_URL", settings.get("AI_BASE_URL") or ""),
            os.getenv("AI_STT_MODEL", settings.get("AI_STT_MODEL") or ""),
            os.getenv("AI_CHAT_MODEL", settings.get("AI_CHAT_MODEL") or ""),
            os.getenv("AI_TTS_MODEL", settings.get("AI_TTS_MODEL") or ""),
        )

    @property
    def _is_openai(self):
        return "openai" in urlsplit(self.base_url).hostname.casefold()

    @property
    def provider_name(self):
        return "openai" if self._is_openai else "sarvam"

    @property
    def supports_stt_timestamps(self):
        return self.uses_batch_stt or self.stt_model in {"whisper-1", "gpt-4o-transcribe-diarize"}

    @property
    def uses_batch_stt(self):
        return not self._is_openai


@dataclass(frozen=True)
class TranscriptionRequest:
    wav: bytes | Path = field(repr=False)


@dataclass(frozen=True)
class TranscriptSegment:
    start_us: int
    end_us: int
    text: str
    speaker_id: str | None = None


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    segments: tuple[TranscriptSegment, ...] = ()


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


@dataclass(frozen=True)
class ChatRequest:
    messages: tuple[ChatMessage, ...]
    json_object: bool = False


@dataclass(frozen=True)
class ChatResult:
    content: str


@dataclass(frozen=True)
class SpeechRequest:
    text: str
    target_locale: str


@dataclass(frozen=True)
class SpeechResult:
    pcm: bytes = field(repr=False)
    voice: str

    @property
    def sample_count(self):
        return len(self.pcm) // SPEECH_SAMPLE_WIDTH_BYTES


class ProviderError(RuntimeError):
    """A sanitized provider error suitable for CLI output."""


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _post_transcription(url, provider, auth, fields, wav):
    boundary = uuid4().hex
    parts = [
        f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()
        for name, value in fields.items()
    ]
    parts.extend((
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="audio.wav"\r\n'
        'Content-Type: audio/wav\r\n\r\n'.encode(),
        wav,
        f"\r\n--{boundary}--\r\n".encode(),
    ))
    request = Request(url, data=b"".join(parts), headers={
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "Accept": "application/json",
        **auth,
    })
    return _send(request, f"{provider} transcription", response_kind="transcription")


def _send(request, operation, *, response_kind="json"):
    try:
        with build_opener(_NoRedirect()).open(request, timeout=90) as response:
            if response_kind == "discard":
                return None
            if response_kind == "pcm":
                if response.headers.get_content_type() not in {"audio/pcm", "audio/x-pcm", "application/octet-stream"}:
                    raise ProviderError(f"{operation} returned an unexpected audio content type.")
                return response.read()
            if response_kind == "transcription" and response.headers.get_content_type() == "text/plain":
                return response.read().decode("utf-8")
            return json.load(response, parse_float=Decimal)
    except HTTPError as exc:
        exc.close()
        raise ProviderError(f"{operation} failed (HTTP {exc.code}).") from None
    except (URLError, OSError, HTTPException, ValueError):
        raise ProviderError(f"{operation} failed (connection or invalid response).") from None


def _segment(start, end, text, speaker_id=None):
    if not isinstance(text, str):
        raise ValueError("Invalid segment text")
    start_us = round(Decimal(str(start)) * 1_000_000)
    end_us = round(Decimal(str(end)) * 1_000_000)
    if end_us <= start_us:
        raise ValueError("Invalid segment bounds")
    if speaker_id is not None:
        if type(speaker_id) not in (str, int):
            raise ValueError("Invalid speaker label")
        speaker_id = str(speaker_id).strip() or None
    return TranscriptSegment(start_us, end_us, text.strip(), speaker_id)


def _storage_request(url, **options):
    parsed = urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username is not None:
        raise ValueError("Invalid Sarvam storage URL")
    return Request(url, **options)


def _post_json(url, payload, auth, operation, *, response_kind="json"):
    request = Request(url, data=json.dumps(payload, ensure_ascii=False).encode(), headers={
        "Content-Type": "application/json", **auth,
    })
    return _send(request, operation, response_kind=response_kind)


def _post_chat(request, model, url, provider, auth, **options):
    payload = {"model": model, "messages": [asdict(message) for message in request.messages], **options}
    if request.json_object:
        payload["response_format"] = {"type": "json_object"}
    operation = f"{provider} chat completion"
    data = _post_json(url, payload, {"Accept": "application/json", **auth}, operation)
    try:
        choice = data["choices"][0]
        if choice.get("finish_reason") == "length":
            raise ProviderError(f"{operation} failed (output exceeded the token limit).")
        content = choice["message"]["content"]
        if not isinstance(content, str):
            raise ValueError
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        raise ProviderError(f"{operation} failed (invalid response format).") from None
    return ChatResult(content)


def _validated_speech_pcm(pcm, voice, operation):
    if len(pcm) % SPEECH_SAMPLE_WIDTH_BYTES:
        raise ProviderError(f"{operation} returned an incomplete PCM sample.")
    if not any(pcm):
        raise ProviderError(f"{operation} returned empty or silent audio.")
    return SpeechResult(pcm, voice)


def _decode_speech_wav(data, voice, operation):
    try:
        with wave.open(io.BytesIO(data), "rb") as wav:
            if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (
                1, SPEECH_SAMPLE_WIDTH_BYTES, SPEECH_SAMPLE_RATE_HZ,
            ):
                raise ProviderError(f"{operation} returned an unexpected WAV format. Expected mono 24 kHz PCM16.")
            frames = wav.getnframes()
            pcm = wav.readframes(frames + 1)
    except (EOFError, wave.Error):
        raise ProviderError(f"{operation} returned an unreadable WAV.") from None
    if len(pcm) != frames * SPEECH_SAMPLE_WIDTH_BYTES:
        raise ProviderError(f"{operation} returned a truncated WAV.")
    return _validated_speech_pcm(pcm, voice, operation)


def call_ai(
    request: TranscriptionRequest | ChatRequest | SpeechRequest, *, config: AIConfig,
) -> TranscriptionResult | ChatResult | SpeechResult:
    """Dispatch an AI request by hostname and normalize the provider response."""
    base_url = config.base_url.rstrip("/")

    def call_openai():
        prefix = base_url if urlsplit(base_url).path else f"{base_url}/v1"
        auth = {"Authorization": f"Bearer {config.api_key}"}
        if isinstance(request, ChatRequest):
            return _post_chat(request, config.chat_model, f"{prefix}/chat/completions", "OpenAI", auth)
        if isinstance(request, SpeechRequest):
            voice = "alloy"
            if request.target_locale not in OPENAI_SPEECH_LOCALES:
                raise ProviderError(f"OpenAI speech synthesis does not support {request.target_locale} with {voice}.")
            if len(request.text) > 4096:
                raise ValueError("OpenAI speech synthesis accepts at most 4096 characters per cue.")
            operation = f"OpenAI speech synthesis for {request.target_locale} with {voice}"
            pcm = _post_json(f"{prefix}/audio/speech", {
                "model": config.tts_model, "voice": voice, "input": request.text,
                "response_format": "pcm", "speed": 1.0,
            }, auth, operation, response_kind="pcm")
            result = _validated_speech_pcm(pcm, voice, operation)
            if sys.byteorder == "big":
                samples = array("h", result.pcm)
                samples.byteswap()
                return SpeechResult(samples.tobytes(), voice)
            return result

        model = config.stt_model
        fields = {"model": model, "response_format": "json"}
        if model == "gpt-transcribe":
            fields["languages[]"] = "en"
        else:
            fields["language"] = "en"
        if model == "whisper-1":
            fields.update({"response_format": "verbose_json", "timestamp_granularities[]": "segment"})
        if model == "gpt-4o-transcribe-diarize":
            fields.update(response_format="diarized_json", chunking_strategy="auto")
        wav = request.wav.read_bytes() if isinstance(request.wav, Path) else request.wav
        data = _post_transcription(
            f"{prefix}/audio/transcriptions", "OpenAI", auth, fields, wav,
        )
        if isinstance(data, str):
            return data, ()
        segments = tuple(
            _segment(item["start"], item["end"], item["text"]) for item in (data.get("segments") or ())
        )
        return data["text"], segments

    def call_sarvam():
        auth = {"api-subscription-key": config.api_key}
        if isinstance(request, ChatRequest):
            versioned_url = base_url if urlsplit(base_url).path.endswith(("/v1", "/v2")) else f"{base_url}/v1"
            options = {"max_tokens": 8192}
            if config.chat_model == "sarvam-105b":
                options["reasoning_effort"] = None
            return _post_chat(request, config.chat_model, f"{versioned_url}/chat/completions",
                              "Sarvam", auth, **options)
        if isinstance(request, SpeechRequest):
            model, locale = config.tts_model, request.target_locale
            if model == "bulbul:v4-flash":
                voice = "gokul_ta_narration" if locale == "ta-IN" else "shubh_enhi_ads"
            elif model == "bulbul:v3":
                voice = "ratan" if locale == "ta-IN" else "shubh"
            else:
                raise ProviderError("Unsupported Sarvam AI_TTS_MODEL. Use bulbul:v4-flash or bulbul:v3.")
            if locale not in SARVAM_SPEECH_LOCALES:
                raise ProviderError(f"Sarvam speech synthesis does not support {locale} with {voice}.")
            if len(request.text) > 2500:
                raise ValueError("Sarvam speech synthesis accepts at most 2500 characters per cue.")
            operation = f"Sarvam speech synthesis for {locale} with {voice}"
            data = _post_json(f"{base_url}/text-to-speech", {
                "model": model, "text": request.text, "language_code": locale, "speaker": voice,
                "pace": 1.0, "speech_sample_rate": SPEECH_SAMPLE_RATE_HZ, "output_audio_codec": "wav",
            }, auth, operation)
            try:
                audios = data["audios"]
                if not isinstance(audios, list) or len(audios) != 1:
                    raise ValueError
                encoded = audios[0]
                wav = base64.b64decode(encoded, validate=True)
            except (KeyError, TypeError, ValueError):
                raise ProviderError(f"{operation} returned invalid base64 audio.") from None
            return _decode_speech_wav(wav, voice, operation)

        job_url = f"{base_url}/speech-to-text/job/v1"
        auth["Content-Type"] = "application/json"

        def api(operation, path="", payload=None, method="POST"):
            body = json.dumps(payload).encode() if payload is not None else None
            return _send(Request(job_url + path, data=body, headers=auth, method=method),
                         f"Sarvam Batch transcription {operation}")

        parameters = {
            "model": config.stt_model, "language_code": "en-IN",
            "with_timestamps": True, "with_diarization": True,
        }
        if config.stt_model.startswith("saaras:"):
            parameters["mode"] = "verbatim"
        job = api("initiation", payload={"job_parameters": parameters})
        job_id = job["job_id"]
        if not isinstance(job_id, str) or not job_id or job_id in {".", ".."}:
            raise ValueError("Invalid batch job ID")
        job_path = f"/{quote(job_id, safe='')}"
        audio_name = "analysis.wav"
        upload = api("upload link", "/upload-files", {"job_id": job_id, "files": [audio_name]})
        upload_url = upload["upload_urls"][audio_name]["file_url"]
        headers = {"Content-Type": "audio/wav"}
        if upload.get("storage_container_type", job.get("storage_container_type")) in {"Azure", "Azure_V1"}:
            headers["x-ms-blob-type"] = "BlockBlob"
        if isinstance(request.wav, Path):
            with request.wav.open("rb") as audio:
                headers["Content-Length"] = str(os.fstat(audio.fileno()).st_size)
                _send(_storage_request(upload_url, data=audio, headers=headers, method="PUT"),
                      "Sarvam Batch transcription upload", response_kind="discard")
        else:
            _send(_storage_request(upload_url, data=request.wav, headers=headers, method="PUT"),
                  "Sarvam Batch transcription upload", response_kind="discard")

        api("start", f"{job_path}/start")
        deadline = monotonic() + 600
        while True:
            status = api("status", f"{job_path}/status", method="GET")
            state = status["job_state"]
            if state in {"Completed", "PartiallyCompleted"}:
                break
            if state == "Failed":
                raise ProviderError("Sarvam Batch transcription job failed. Check the job in the Sarvam dashboard.")
            if state not in {"Accepted", "Pending", "Running"}:
                raise ValueError("Invalid batch job state")
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise ProviderError("Sarvam Batch transcription timed out. Check the job in the Sarvam dashboard.")
            sleep(min(5, remaining))

        outputs = [
            output["file_name"] for detail in (status.get("job_details") or ())
            if detail.get("state") == "Success" for output in detail["outputs"]
        ]
        if len(outputs) != 1:
            raise ProviderError("Sarvam Batch transcription returned no unique result for the uploaded audio.")
        name = outputs[0]
        download = api("download link", "/download-files", {"job_id": job_id, "files": [name]})
        data = _send(
            _storage_request(download["download_urls"][name]["file_url"]),
            "Sarvam Batch transcription download",
        )
        entries = (data.get("diarized_transcript") or {}).get("entries")
        if entries:
            return data["transcript"], tuple(
                _segment(entry["start_time_seconds"], entry["end_time_seconds"],
                         entry["transcript"], entry.get("speaker_id")) for entry in entries
            )
        timestamps = data.get("timestamps")
        segments = ()
        if timestamps:
            segments = tuple(
                _segment(start, end, text) for start, end, text in zip(
                    timestamps["start_time_seconds"], timestamps["end_time_seconds"], timestamps["chunks"],
                    strict=True,
                )
            )
        return data["transcript"], segments

    dispatch = call_openai if config._is_openai else call_sarvam
    if isinstance(request, SpeechRequest):
        if not config.tts_model.strip():
            raise ValueError("Speech synthesis requires AI_TTS_MODEL.")
        if not request.text.strip():
            raise ValueError("Speech synthesis requires nonempty text.")
        return dispatch()
    if isinstance(request, ChatRequest):
        if not config.chat_model.strip():
            raise ValueError("Chat completion requires AI_CHAT_MODEL.")
        return dispatch()
    if not isinstance(request, TranscriptionRequest):
        raise TypeError("Unsupported AI request type.")
    if not config.stt_model.strip():
        raise ValueError("Transcription requires AI_STT_MODEL.")
    provider = "OpenAI" if config._is_openai else "Sarvam Batch"
    try:
        text, segments = dispatch()
        if not isinstance(text, str):
            raise ValueError("Invalid transcript text")
        return TranscriptionResult(text.strip(), tuple(segment for segment in segments if segment.text))
    except (AttributeError, KeyError, TypeError, ValueError, DecimalException, OverflowError):
        raise ProviderError(f"{provider} transcription failed (invalid response format).") from None
