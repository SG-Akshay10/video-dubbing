# video-dubbing
An agentic workflow that dubs English videos into Indian languages. It transcribes and translates speech, generates time-aligned voiceovers, and mixes them with the original music and sound effects where possible.

## Ingest a source video

From the project directory, run the following in a Python environment.

```sh
python -m pip install -r requirements.txt
python ingest.py --input input/podcast.mp4 --output output/podcast
```

Use a fresh output directory for each run. If the video has several audio tracks, the CLI lists their container indices. Pass `--audio-stream INDEX` to choose one.

- `media.json` records the absolute source video path for export, picture duration, selected audio stream, and analysis audio format. Durations and offsets use integer microseconds. Offsets are relative to the first picture timestamp. The analysis audio offset locates WAV sample zero on that timeline.
- `analysis.wav` contains mono 16 kHz, signed 16-bit PCM audio for transcription. Its path in the manifest is relative to `media.json`.

## Transcribe the analysis audio

Install the current dependencies with `python -m pip install -r requirements.txt`. Add your API key to `.env` in the project root and set `AI_BASE_URL` and `AI_STT_MODEL` for the same provider. Copy `.env.example` to `.env` if the file is missing. The CLI loads it automatically, and exported shell variables take precedence. Git ignores `.env`.

| Provider | `AI_BASE_URL` | `AI_STT_MODEL` example |
| --- | --- | --- |
| OpenAI | `https://api.openai.com/v1` | `gpt-transcribe` for approximate timing, or `whisper-1` for provider timestamps while it is available |
| Sarvam | `https://api.sarvam.ai` | `saaras:v4` |

```sh
python transcribe.py --input output/podcast/media.json --output output/podcast/transcript.json
python -m unittest discover -s tests
```

Use a new transcript filename for each run. The transcript contains ordered English cues with unique IDs and source-relative `start_us` and `end_us` values.

Sarvam sends the full analysis WAV as one Batch job with diarization enabled, then waits for the result. Each cue carries the provider's `speaker_id` for that recording. If Sarvam returns timestamps without a speaker label, the cue has a null `speaker_id` and a `speaker_unavailable` review flag. Batch accepts up to two hours of audio per file.

OpenAI uses uploads of up to 28 seconds. Cues from text-only responses use bounded audio windows separated by low-energy pauses and carry the `approximate_timing` review flag.

## Translate the English cues

Set `AI_CHAT_MODEL` in `.env` for the provider that matches your `AI_BASE_URL` and `AI_API_KEY`. The examples use `sarvam-105b` for Sarvam and `gpt-4.1-mini` for OpenAI.

```sh
python translate.py --input output/podcast/transcript.json --target-locale ta-IN --output output/podcast/translation-ta-IN.json
python -m unittest discover -s tests
```

The output keeps the English `text` and existing cue fields, adds `translated_text` to every cue, and records `target_locale` for the file. Use a new output filename when rerunning translation.

## Synthesize the translated cues

Set `AI_TTS_MODEL` in `.env` to `bulbul:v4-flash` for Sarvam or `gpt-4o-mini-tts` for OpenAI. The adapter chooses one stock voice for the locale and uses it at standard speed for every cue. For Tamil, these models use Sarvam's `gokul_ta_narration` and OpenAI's `alloy`, respectively. OpenAI schedules its Speech API models for retirement on January 6, 2027. [OpenAI Speech API](https://developers.openai.com/api/reference/python/resources/audio/subresources/speech/methods/create) · [Sarvam voice example](https://docs.sarvam.ai/api/cookbook/example-voice-agents/tutor-agent)

Pass the filename you used for Task 3 and choose a fresh speech output directory.

```sh
python synthesize.py --input output/podcast/translation-ta-IN.json --output output/podcast/speech-ta-IN
```

The output includes `speech.json` and one mono 24 kHz PCM16 WAV per cue. The manifest records each clip's path, cue ID, original cue bounds, decoded sample count, and full clip duration in microseconds.

## Export a dubbed video

```sh
python mix.py --media output/podcast/media.json --speech output/podcast/speech-ta-IN/speech.json --output output/podcast/mix-ta-IN
```

Use a fresh output directory. The mixer copies the picture stream and replaces the audio with the synthesized clips placed at their cue starts. `mix.json` records sample placements and boundary crossings. A clean mix produces `dubbed.mp4`. A mix with crossings produces `preview.mp4` for review. Overlapping clips are summed, and audio past the picture end is omitted from the preview.

## Run the v0 workflow end to end

Configure `AI_API_KEY`, `AI_BASE_URL`, `AI_STT_MODEL`, `AI_CHAT_MODEL`, and `AI_TTS_MODEL` in `.env` using one provider's settings. Use the same command for either provider.

```sh
python dub.py --input input/podcast-v0.mp4 --target-locale ta-IN --output output/v0-smoke
```

`input/podcast-v0.mp4` is the local five-second smoke fixture cut from `input/podcast.mp4`, with a silent tail. For other runs, pass your source video and choose a fresh output root. Add `--audio-stream INDEX` when the source has multiple audio tracks.

Each provider writes its video and `run.json` under `output/v0-smoke/sarvam` or `output/v0-smoke/openai`. The run report lists placed cue IDs, artifact paths relative to the report, and the results of decoding the entire video and replacement audio. Exit status `0` means a verified `dubbed.mp4`, `1` means a stage or verification failed, and `2` means a decodable `preview.mp4` was written for cue boundary review.
