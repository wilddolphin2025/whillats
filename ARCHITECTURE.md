# Whillats Architecture

## Overview

Whillats provides real-time AI speech processing for WebRTC applications. It integrates Speech-to-Text (Whisper), Language Model (LLM), and Text-to-Speech (Piper/StyleTTS2) into a process-separated architecture that isolates AI workloads from the WebRTC process.

As of **talkingface5**, the LLM backend is handled by a separately-deployed **llama-server** process (llama.cpp's OpenAI-compatible HTTP server). `whillats_server` communicates with it over localhost HTTP rather than embedding llama.cpp directly.

## Process Architecture

```
Browser
  │  WebRTC (audio + video)
  ▼
┌─────────────────────────────────┐
│           directcall            │  WebRTC process
│   (Clang/libc++, -fno-except)  │
│                                 │
│  ┌─────────────────────────┐   │
│  │  whillats client        │   │
│  │  (compiled in-tree GN)  │   │
│  │  - IPC stubs (pipes)    │   │
│  │  - TalkingFace (video)  │   │
│  └──────────┬──────────────┘   │
└─────────────┼───────────────────┘
              │ pipes (binary IPC)
              ▼
┌─────────────────────────────────┐
│        whillats_server          │  GCC/libstdc++ process
│                                 │
│  Whisper STT (whisper.cpp)      │
│  Piper TTS (ONNX)               │
│  LlamaHttpClient ───────────────┼──► http://127.0.0.1:8080
│  espeak-ng (fallback)           │        │
└─────────────────────────────────┘        │
                                           │ HTTP/SSE
                                  ┌────────┴────────┐
                                  │   llama-server   │  separate process
                                  │  (llama.cpp)     │
                                  │  Gemma-4 Q2_K   │
                                  │  /v1/chat/       │
                                  │  completions     │
                                  └──────────────────┘
```

### Why Three Processes?

**WebRTC ↔ whillats_server split** — WebRTC is compiled Clang/libc++ with `-fno-exceptions`. AI libraries (whisper.cpp, ONNX Runtime) require GCC/libstdc++ with exceptions. Mixing these in a single process causes `std::string` ABI crashes.

**whillats_server ↔ llama-server split** — Decouples LLM lifecycle from STT/TTS. llama-server can be restarted, swapped, or scaled independently. whillats_server binary no longer links llama.cpp or ggml — significantly smaller and faster to start.

### In-Tree Client Build

The whillats client is compiled **directly by WebRTC's GN build system** using the same Clang compiler, libc++, sysroot, and `-fno-exceptions` as directcall. Zero ABI mismatch. No `libwhillats.so` needed at runtime.

## IPC Protocol (whillats_client ↔ whillats_server)

Length-prefixed binary messages over Unix pipes (fork/exec on first use):

```
┌──────────┬──────────────┬────────────────┐
│ type (1B)│ length (4B)  │ payload (var)  │
└──────────┴──────────────┴────────────────┘
```

| Type | Code | Direction | Description |
|------|------|-----------|-------------|
| MSG_CONFIG | 0x30 | client→server | Model paths, thread counts, language, llama-server URL |
| MSG_WHISPER_START | 0x01 | client→server | Confirm Whisper ready |
| MSG_WHISPER_AUDIO | 0x03 | client→server | Audio chunk (len=0 = flush) |
| MSG_WHISPER_RESULT | 0x04 | server→client | Transcription text |
| MSG_WHISPER_LANGUAGE | 0x05 | server→client | Detected language tag |
| MSG_LLAMA_START | 0x10 | client→server | Confirm Llama ready |
| MSG_LLAMA_ASK | 0x12 | client→server | Prompt text |
| MSG_LLAMA_RESPONSE | 0x13 | server→client | Response token |
| MSG_LLAMA_DONE | 0x15 | server→client | Generation complete |
| MSG_LLAMA_VIDEO_FRAME | 0x14 | client→server | YUV frame for multimodal |
| MSG_TTS_START | 0x20 | client→server | Start TTS engine |
| MSG_TTS_SPEAK | 0x22 | client→server | Text + language tag |
| MSG_TTS_AUDIO | 0x23 | server→client | PCM int16 audio |
| MSG_TTS_DONE | 0x24 | server→client | Synthesis complete |
| MSG_TTS_VOICE | 0x25 | client→server | Select a cloned voice (clip name in `POCKET_TTS_VOICES_DIR`) |
| MSG_TTS_INTERRUPT | 0x26 | client→server | Drop queued text, abort current utterance (barge-in) |
| MSG_SHUTDOWN | 0xFF | client→server | Shutdown server |

**Note:** `MSG_CONFIG.llama_model` field carries the llama-server URL (e.g. `http://127.0.0.1:8080`) rather than a model file path.

Key IPC design details:
- Header + payload written atomically (single `write()` via malloc'd buffer)
- Server writes protected by mutex (Whisper + TTS callbacks on different threads)
- Reader thread uses only C `malloc/free` — no C++ allocations (`-fno-exceptions` safe)
- Server redirects `stdout→stderr` (`dup2`) before any library code runs

## llama-server HTTP Protocol

`LlamaHttpClient` connects to llama-server using raw POSIX sockets — no libcurl or other HTTP library. It implements SSE (Server-Sent Events) streaming parsing of `/v1/chat/completions`.

```
POST /v1/chat/completions  HTTP/1.1
Content-Type: application/json
Accept: text/event-stream

{"stream":true,"messages":[
  {"role":"system","content":"..."},
  {"role":"user","content":"..."}
]}

→ data: {"choices":[{"delta":{"content":"Hello"}}]}
→ data: {"choices":[{"delta":{"content":" there"}}]}
→ data: [DONE]
```

For multimodal (video frame attached): user content is an array with a `text` item and an `image_url` item containing a base64-encoded JPEG (YUV420 → RGB → JPEG via stb_image_write).

## Audio + Video Pipeline

```
Browser mic → WebRTC RTP → directcall
                               │
                     WhillatsTranscriber
                               │ MSG_WHISPER_AUDIO (16kHz PCM)
                               ▼
                        whillats_server
                         Whisper STT
                               │
                    MSG_WHISPER_RESULT + MSG_WHISPER_LANGUAGE
                               │
                     WhillatsLlama.askLlama()
                               │ MSG_LLAMA_ASK
                               ▼
                        whillats_server
                        LlamaHttpClient
                               │ HTTP POST /v1/chat/completions (SSE)
                               ▼
                          llama-server
                          Gemma-4 Q2_K
                               │
                     MSG_LLAMA_RESPONSE (per token)
                               │
                        WhisperAudioDevice
                         speakText()
                               │ MSG_TTS_SPEAK
                               ▼
                        whillats_server
                          Piper TTS
                               │ MSG_TTS_AUDIO (16kHz int16 PCM)
                               ▼
                        directcall
                    TalkingFace (lip-sync)
                               │ WebRTC RTP
                               ▼
                            Browser
                     (audio + animated video)
```

Browser video track (when enabled) → `MSG_LLAMA_VIDEO_FRAME` → `LlamaDeviceBase.receiveVideoFrame()` → stored as `_lastFrame` → attached to next `MSG_LLAMA_ASK` as JPEG base64 in multimodal message.

## Source Files

### Client (compiled in-tree by WebRTC GN)

| File | Description |
|------|-------------|
| `whillats.h` | Public API: WhillatsTTS, WhillatsTranscriber, WhillatsLlama |
| `whillats.cc` | API implementation — routes calls to IPC stubs |
| `whillats_client.h/cc` | IPC client: fork/exec server, pipe I/O (C malloc only) |
| `whillats_ipc.h` | Binary protocol: message types, read/write helpers |
| `whillats_utils.h/cc` | Audio resampling, YUV utilities |
| `talking_face.h/cc` | Video lip-sync animation (10ms frame-by-frame) |
| `stb_image_impl.c` | stb_image compiled as C (avoids C++ warnings) |

### Server (compiled by CMake/GCC — links whisper.cpp + Piper only)

| File | Description |
|------|-------------|
| `whillats_server.cc` | Main IPC loop → dispatch to AI backends |
| `whisper_transcription.h/cc` | Whisper STT: VAD, ring buffer, language detection |
| `llama_device_base.h/cc` | LLM interface: request queue, video frame store, YUV→JPEG |
| `llama_http_client.h/cc` | HTTP/SSE client for llama-server (raw POSIX sockets) |
| `stb_image_write.h` | Bundled: JPEG encoding for video frames |
| `piper_tts.h/cc` | Piper TTS controller: per-language subprocess routing |
| `piper_subprocess.h/cc` | Piper ONNX isolation via fork (closes inherited fds) |
| `styletts2_tts.h/cc` | StyleTTS2 neural TTS (ONNX, alternative to Piper) |
| `pocket_voice_tts.h/cc` | Pocket TTS zero-shot voice cloning (`WHILLATS_POCKET_TTS`, pocket-tts-raven runtime) |
| `espeak_tts.h/cc` | espeak-ng rule-based TTS fallback |
| `whisper_helpers.h` | Logging macros (stderr only), timing |

## Performance (Contabo vServer, x86_64, CPU-only)

| Component | Observed |
|-----------|----------|
| Whisper small — language detection + transcription | ~2–4s per utterance |
| Piper TTS (low quality, 16kHz) | ~0.3–0.8s per sentence |
| Gemma-4 Q2_K via llama-server (first token) | ~1–2s |
| llama-server cold start (model load) | ~10–20s |
| whillats_server cold start | ~1s (Whisper preloads in background) |

Models preload at startup. llama-server and whillats_server start independently; directcall waits for llama-server via `/health` polling (up to 60s).

## Personalized Voices (Pocket TTS)

Build with `-DWHILLATS_POCKET_TTS=ON` (CMake ≥ 3.28). `whillats_server` then
loads Kyutai's Pocket TTS (100M parameters, weights CC-BY-4.0) through the MIT
[pocket-tts-raven](https://github.com/pkalogiros/pocket-tts-raven) ONNX runtime
next to Piper/StyleTTS2:

| Variable | Meaning |
|----------|---------|
| `POCKET_TTS_MODELS_DIR` | Bundle from `third_party/pocket-tts-raven/tools/prepare_models.sh` |
| `POCKET_TTS_VOICES_DIR` | Reference clips (6–15 s of one consenting speaker, wav/mp3) |
| `POCKET_TTS_VOICE` | Default clip name |
| `POCKET_TTS_LANGUAGE` | Bundle language (default `en`); other languages fall back to Piper |

The clip is encoded once (cached in `voices/.cache/`), so switching voices
mid-call costs one background encode. Output is 24 kHz and converted to the
pipeline's 16 kHz with an anti-aliasing filter. `MSG_TTS_INTERRUPT` aborts
generation within one decode batch, which makes barge-in immediate.
