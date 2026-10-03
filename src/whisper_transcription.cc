/*
 *  (c) 2025, wilddolphin2025
 *  For WebRTCsays.ai project
 *  https://github.com/wilddolphin2025
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "whisper_transcription.h"

#include <algorithm>

#include <cctype>
#include <chrono>
#include <cmath>
#include <cstring>
#include <regex>

#ifdef WHILLATS_INPROCESS
// whillats_ios.mm: mirrors every transcript to NSNotificationCenter as
// "WhillatsTranscriptNotification" {text, final, language} for the host app.
void whillats_post_transcript(const char* text, bool is_final, const char* language);
#endif

namespace {

constexpr int kSamplesPerMs = WHISPER_SAMPLE_RATE / 1000;

// Decodes one UTF-8 code point and advances p; returns 0 at end of string.
uint32_t nextCodePoint(const unsigned char*& p) {
    unsigned char c = *p;
    if (!c) return 0;
    int n = c < 0x80 ? 1 : (c & 0xE0) == 0xC0 ? 2 : (c & 0xF0) == 0xE0 ? 3 : (c & 0xF8) == 0xF0 ? 4 : 1;
    uint32_t cp = n == 1 ? c : c & (0xFF >> (n + 1));
    for (int i = 1; i < n && p[i]; i++) cp = (cp << 6) | (p[i] & 0x3F);
    for (int i = 0; i < n && *p; i++) p++;
    return cp;
}

std::string trim(const std::string& s) {
    size_t b = s.find_first_not_of(" \t\r\n"), e = s.find_last_not_of(" \t\r\n");
    return b == std::string::npos ? std::string() : s.substr(b, e - b + 1);
}

}  // namespace

WhisperTranscriber::WhisperTranscriber(const char* modelPath,
    WhillatsSetResponseCallback callback,
    WhillatsSetLanguageCallback languageCallback)
    : _responseCallback(callback),
      _languageCallback(languageCallback),
      _ring(kSampleRate * 30) {
    _frame.reserve(kFrameSamples);
    _preroll.assign(kPrerollMs * kSamplesPerMs, 0.0f);
    _utt.reserve(kMaxUtteranceMs * kSamplesPerMs);
    if (const char* prompt = getenv("WHISPER_INITIAL_PROMPT")) _initialPrompt = prompt;
    if (const char* langs = getenv("WHISPER_LANGUAGES")) setAllowedLanguages(langs);
    if (!modelPath || !initModel(modelPath)) LOG_E("Failed to initialize Whisper");
}

WhisperTranscriber::~WhisperTranscriber() {
    stop();
    if (_detState) whisper_free_state(_detState);
    if (_state) whisper_free_state(_state);
    if (_ctx) whisper_free(_ctx);
}

bool WhisperTranscriber::initModel(const std::string& modelPath) {
    whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = true;
    // Flash attention pads the cross-attention cache to 256 frames and leaves
    // stale encoder output in the padding when audio_ctx shrinks between
    // calls, which garbles trimmed decodes ("Hang up." -> "Anio.").
    cparams.flash_attn = false;
    _ctx = whisper_init_from_file_with_params(modelPath.c_str(), cparams);
    if (!_ctx) {
        LOG_E("Failed to load Whisper model from " << modelPath);
        return false;
    }
    _state = whisper_init_state(_ctx);
    if (!_state) {
        LOG_E("Failed to initialize Whisper state");
        whisper_free(_ctx);
        _ctx = nullptr;
        return false;
    }
    // Compiles GPU kernels before the first utterance.
    warmUp(_state, 64);
    LOG_I("Whisper streaming STT ready: " << modelPath);
    return true;
}

void WhisperTranscriber::setAllowedLanguages(const std::string& codes) {
    std::vector<int> ids;
    size_t start = 0;
    while (start <= codes.size()) {
        size_t end = codes.find(',', start);
        if (end == std::string::npos) end = codes.size();
        std::string code = trim(codes.substr(start, end - start));
        int id = code.empty() ? -1 : whisper_lang_id(code.c_str());
        if (id >= 0) ids.push_back(id);
        start = end + 1;
    }
    std::lock_guard<std::mutex> l(_cfgMutex);
    _allowedLangs = std::move(ids);
}

// Language auto-detection encodes with whatever audio_ctx the state's last
// decode used (0 = a full 30 s window); a tiny decode pins it.
void WhisperTranscriber::warmUp(whisper_state* state, int audioCtx) {
    std::vector<float> silence(kSampleRate / 2, 0.0f);
    whisper_full_params p = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    p.print_progress = p.print_realtime = p.print_timestamps = p.print_special = false;
    p.no_timestamps = p.single_segment = true;
    p.language = "en";
    p.audio_ctx = audioCtx;
    p.max_tokens = 1;
    whisper_full_with_state(_ctx, state, p, silence.data(), (int)silence.size());
}

// Whisper's built-in detection chooses among ~100 languages (short commands
// come out as Korean) and inflates the no-speech probability of the decode
// that follows, so detect here, restricted to the call's languages, and
// always decode with an explicit language.
std::string WhisperTranscriber::detectLanguage(const std::vector<float>& pcm, const std::vector<int>& allowed) {
    const int threads = _nThreads > 0 ? _nThreads : (int)std::min(4u, std::max(1u, std::thread::hardware_concurrency()));
    std::vector<float> probs(whisper_lang_max_id() + 1, 0.0f);
    // A separate state: detecting on the decode state skews the no-speech
    // probability of the next decode, so real speech gets dropped.
    if (!_detState && (_detState = whisper_init_state(_ctx))) warmUp(_detState, kLangDetectMs / 20);
    if (!_detState || whisper_pcm_to_mel_with_state(_ctx, _detState, pcm.data(), (int)pcm.size(), threads) != 0 ||
        whisper_lang_auto_detect_with_state(_ctx, _detState, 0, threads, probs.data()) < 0) {
        return !_convLang.empty() ? _convLang : allowed.empty() ? "en" : whisper_lang_str(allowed[0]);
    }
    int best = allowed.empty() ? 0 : allowed[0];
    if (allowed.empty()) { for (int id = 0; id < (int)probs.size(); ++id) if (probs[id] > probs[best]) best = id; }
    else { for (int id : allowed) if (probs[id] > probs[best]) best = id; }
    return whisper_lang_str(best);
}

void WhisperTranscriber::setTranscriptCallback(WhillatsSetTranscriptCallback cb) {
    std::lock_guard<std::mutex> l(_cbMutex);
    _transcriptCallback = cb;
}

void WhisperTranscriber::setResponseCallback(WhillatsSetResponseCallback cb) {
    std::lock_guard<std::mutex> l(_cbMutex);
    _responseCallback = cb;
}

void WhisperTranscriber::setLanguageCallback(WhillatsSetLanguageCallback cb) {
    std::lock_guard<std::mutex> l(_cbMutex);
    _languageCallback = cb;
}

// Runs on the real-time audio thread: convert and enqueue only.
void WhisperTranscriber::processAudioBuffer(uint8_t* buffer, size_t bytes) {
    if (bytes == 0 || !buffer) {
        _flushRequested = true;
        _wake.notify_one();
        return;
    }
    if (bytes % 2) return;
    const size_t n = bytes / 2;
    if (_scratch.size() < n) _scratch.resize(n);
    const int16_t* pcm = reinterpret_cast<const int16_t*>(buffer);
    for (size_t i = 0; i < n; ++i) _scratch[i] = pcm[i] * (1.0f / 32768.0f);
    _ring.write(_scratch.data(), n);
    _wake.notify_one();
}

bool WhisperTranscriber::start() {
    if (!_ctx) {
        LOG_E("Whisper context is not initialized");
        return false;
    }
    std::lock_guard<std::mutex> lock(_threadMutex);
    if (_running) return true;
    if (_worker.joinable()) _worker.join();
    _ring.clear();
    resetUtterance();
    _convLang.clear();
    _running = true;
    _worker = std::thread(&WhisperTranscriber::run, this);
    return true;
}

void WhisperTranscriber::stop() {
    std::lock_guard<std::mutex> lock(_threadMutex);
    if (!_running) return;
    _running = false;
    _wake.notify_all();
    if (_worker.joinable()) {
        if (std::this_thread::get_id() == _worker.get_id()) _worker.detach();
        else _worker.join();
    }
}

void WhisperTranscriber::run() {
    std::vector<float> chunk;
    while (_running) {
        {
            std::unique_lock<std::mutex> l(_wakeMutex);
            _wake.wait_for(l, std::chrono::milliseconds(50), [this] {
                return !_running || _flushRequested || _ring.availableToRead() >= (size_t)kFrameSamples;
            });
        }
        if (!_running) break;

        const size_t n = _ring.availableToRead();
        if (n) {
            chunk.resize(n);
            _ring.read(chunk.data(), n);
            for (size_t i = 0; i < n;) {
                const size_t take = std::min(kFrameSamples - _frame.size(), n - i);
                _frame.insert(_frame.end(), chunk.begin() + i, chunk.begin() + i + take);
                i += take;
                if (_frame.size() == (size_t)kFrameSamples) {
                    consumeFrame(_frame.data());
                    _frame.clear();
                }
            }
        }
        if (_flushRequested.exchange(false)) finalizeUtterance();
        // Interims run after the backlog is consumed so endpointing never waits on them.
        maybeEmitInterim();
    }
    resetUtterance();
}

// Adaptive energy VAD: the noise floor follows quiet frames quickly and loud
// frames slowly; speech opens after kOnsetMs of voiced frames.
void WhisperTranscriber::consumeFrame(const float* f) {
    float e = 0.0f;
    for (int i = 0; i < kFrameSamples; ++i) e += f[i] * f[i];
    const float rms = std::sqrt(e / kFrameSamples);
    const bool voiced = rms > std::max(kAbsMinRms, _noise * _vadSnr);

    if (!_inSpeech) {
        _noise = rms < _noise ? 0.9f * _noise + 0.1f * rms : 0.995f * _noise + 0.005f * rms;
        _noise = std::max(_noise, 1e-4f);
        std::memcpy(&_preroll[_prerollPos], f, kFrameSamples * sizeof(float));
        _prerollPos = (_prerollPos + kFrameSamples) % _preroll.size();
        _onsetRun = voiced ? _onsetRun + kFrameMs : 0;
        if (_onsetRun < kOnsetMs) return;

        _inSpeech = true;
        _utt.assign(_preroll.begin() + _prerollPos, _preroll.end());
        _utt.insert(_utt.end(), _preroll.begin(), _preroll.begin() + _prerollPos);
        _speechMs = _onsetRun;
        _silenceMs = 0;
        _msSinceInterim = 0;
        _newAudioSinceInterim = true;
        return;
    }

    if (rms < _noise) _noise = std::max(0.9f * _noise + 0.1f * rms, 1e-4f);
    _utt.insert(_utt.end(), f, f + kFrameSamples);
    _newAudioSinceInterim = true;
    _msSinceInterim += kFrameMs;
    if (voiced) {
        _speechMs += kFrameMs;
        _silenceMs = 0;
    } else {
        _silenceMs += kFrameMs;
    }
    if (_silenceMs >= _endpointSilenceMs || _utt.size() >= (size_t)kMaxUtteranceMs * kSamplesPerMs)
        finalizeUtterance();
}

void WhisperTranscriber::maybeEmitInterim() {
    if (!_inSpeech || !_newAudioSinceInterim || _msSinceInterim < _interimIntervalMs ||
        _speechMs < kMinInterimMs || _silenceMs >= kFrameMs * 10)
        return;
    _msSinceInterim = 0;
    _newAudioSinceInterim = false;
    Decoded d;
    if (decode(_utt, false, d) && isValidText(d.text) && d.text != _lastInterim) {
        _lastInterim = d.text;
        emit(d.text, false, d.language);
    }
}

void WhisperTranscriber::finalizeUtterance() {
    if (_inSpeech && _speechMs >= kMinSpeechMs) {
        const size_t tail = (size_t)std::max(0, _silenceMs - kTailKeepMs) * kSamplesPerMs;
        _utt.resize(_utt.size() - std::min(tail, _utt.size()));
        Decoded d;
        const auto t0 = std::chrono::steady_clock::now();
        const bool ok = decode(_utt, true, d) && isValidText(d.text);
        LOG_V("Final decode " << _utt.size() / kSamplesPerMs << " ms audio in "
              << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count() << " ms");
        // An empty final still closes the utterance so UIs can clear interim text.
        if (ok || !_lastInterim.empty()) emit(ok ? d.text : std::string(), true, d.language);
    }
    resetUtterance();
}

void WhisperTranscriber::resetUtterance() {
    _inSpeech = false;
    _utt.clear();
    _onsetRun = _silenceMs = _speechMs = _msSinceInterim = 0;
    _newAudioSinceInterim = false;
    _uttLang.clear();
    _lastInterim.clear();
}

bool WhisperTranscriber::abortCallback(void* user_data) {
    return !static_cast<WhisperTranscriber*>(user_data)->_running.load(std::memory_order_relaxed);
}

bool WhisperTranscriber::decode(const std::vector<float>& pcm, bool final, Decoded& out) {
    if (!_ctx || !_state || pcm.empty()) return false;
    std::string lang, prompt;
    std::vector<int> allowed;
    {
        std::lock_guard<std::mutex> l(_cfgMutex);
        lang = _language;
        prompt = _initialPrompt;
        allowed = _allowedLangs;
    }
    const bool autoLang = lang.empty() || lang == "auto";
    if (autoLang) lang = _uttLang.empty() ? detectLanguage(pcm, allowed) : _uttLang;

    whisper_full_params p = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    p.n_threads = _nThreads > 0 ? _nThreads : (int)std::min(4u, std::max(1u, std::thread::hardware_concurrency()));
    p.print_progress = p.print_realtime = p.print_timestamps = p.print_special = false;
    p.translate = false;
    p.no_context = true;
    p.no_timestamps = true;
    p.single_segment = true;
    p.suppress_blank = true;
    p.suppress_nst = true;
    p.temperature = 0.0f;
    p.temperature_inc = final ? 0.2f : 0.0f;  // fallback re-decodes only for finals
    p.no_speech_thold = 0.6f;
    // The prompt is English; it would drag other languages into English.
    p.initial_prompt = (lang == "en" && !prompt.empty()) ? prompt.c_str() : nullptr;
    p.language = lang.c_str();
    // Encoder cost scales with audio_ctx (50 frames/s, 1500 = 30 s). Trimming
    // it to the audio actually heard makes interims cheap and short finals
    // (voice commands) 5-8x faster; long finals keep the full window because
    // a trimmed window can drop whole clauses there.
    const int frames = (int)(pcm.size() / (kSamplesPerMs * 20));
    if (!final) p.audio_ctx = std::min(whisper_n_audio_ctx(_ctx), frames + 64);
    else if (frames <= kFastFinalMs / 20) p.audio_ctx = std::min(whisper_n_audio_ctx(_ctx), frames + 256);
    // A shortened audio_ctx occasionally sends Whisper into a loop; capping
    // tokens by duration (~8/s covers fast Cyrillic speech) bounds the cost.
    p.max_tokens = (int)(pcm.size() / (kSamplesPerMs * 125)) + 12;
    p.abort_callback = abortCallback;
    p.abort_callback_user_data = this;

    if (whisper_full_with_state(_ctx, _state, p, pcm.data(), (int)pcm.size()) != 0) return false;
    if (autoLang) {
        if (!final && _uttLang.empty() && pcm.size() >= (size_t)kLangLockMs * kSamplesPerMs) _uttLang = lang;
        if (final) _convLang = lang;
    }

    std::string text;
    const int n = whisper_full_n_segments_from_state(_state);
    for (int i = 0; i < n; ++i) {
        if (whisper_full_get_segment_no_speech_prob_from_state(_state, i) > p.no_speech_thold) continue;
        text += whisper_full_get_segment_text_from_state(_state, i);
    }
    out.language = lang;
    out.text = trim(collapseRepeats(text));
    return true;
}

// Cuts a phrase repeated three or more times back to one ("a b. a b. a b." -> "a b.").
std::string WhisperTranscriber::collapseRepeats(const std::string& s) {
    const size_t n = s.size();
    for (size_t len = 4; len * 3 <= n; ++len)
        for (size_t i = 0; i + len * 3 <= n; ++i)
            if (s.compare(i, len, s, i + len, len) == 0 && s.compare(i, len, s, i + 2 * len, len) == 0)
                return s.substr(0, i + len);
    return s;
}

void WhisperTranscriber::emit(const std::string& text, bool final, const std::string& lang) {
    LOG_I((final ? "Final" : "Interim") << " [" << lang << "]: " << text);
    std::lock_guard<std::mutex> l(_cbMutex);
    if (final && !text.empty()) {
        std::string configured = getLanguage();
        if (configured == "auto" || lang != _lastReportedLang) _languageCallback.OnLanguageDetected(true, lang);
        _lastReportedLang = lang;
    }
    _transcriptCallback.OnTranscript(text.c_str(), final, lang.c_str());
#ifdef WHILLATS_INPROCESS
    whillats_post_transcript(text.c_str(), final, lang.c_str());
#endif
    if (final && !text.empty()) _responseCallback.OnResponseComplete(true, text.c_str());
}

bool WhisperTranscriber::isValidText(std::string& text) {
    static const std::regex kBracketed(R"(\[[^\]]*\]|\([^)]*\)|\*[^*]*\*)");
    static const char* kHallucinations[] = {
        "thanks for watching", "thank you for watching", "please subscribe",
        "subtitles by", "amara.org", "transcribed by",
    };

    std::string cleaned = trim(std::regex_replace(text, kBracketed, ""));
    if (cleaned.empty()) return false;

    std::string lower(cleaned);
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) { return std::tolower(c); });
    for (const char* h : kHallucinations)
        if (lower.find(h) != std::string::npos) return false;

    // Need >= 2 letters/digits; IPA modifiers, punctuation and math symbols
    // are Whisper noise. A run of 3+ identical non-ASCII code points ("ღღღ")
    // is a noise hallucination too.
    int letters = 0, run = 0;
    uint32_t prev = 0;
    const unsigned char* p = reinterpret_cast<const unsigned char*>(cleaned.c_str());
    while (uint32_t cp = nextCodePoint(p)) {
        run = cp == prev ? run + 1 : 1;
        prev = cp;
        if (cp < 0x80) {
            letters += std::isalnum((int)cp) ? 1 : 0;
            continue;
        }
        if (run >= 3) return false;
        const bool symbol = (cp >= 0x0250 && cp <= 0x02FF) || (cp >= 0x2000 && cp <= 0x206F) ||
                            (cp >= 0x2200 && cp <= 0x22FF);
        letters += symbol ? 0 : 1;
    }
    if (letters < 2) return false;
    text = cleaned;
    return true;
}
