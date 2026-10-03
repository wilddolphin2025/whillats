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

#include <memory>
#include <string>
#include <thread>
#include <mutex>
#include <vector>
#include <unistd.h>
#include <fcntl.h>
#include <sys/wait.h>
#include <errno.h>

// #include "silence_finder.h" // removed: not needed for iOS build
#include "whillats.h"
#if defined(__APPLE__)
#include <TargetConditionals.h>
#endif
#include "whillats_utils.h"

#ifdef WHILLATS_INPROCESS
#include "whisper_transcription.h"
#include "llama_device_base.h"
#else
#include "whillats_client.h"
#include "whillats_ipc.h"

// Thin client library: Whisper/Llama run in the whillats_server subprocess

// Shared server connection (lazy-initialized, one per process)
static std::shared_ptr<WhillatsServerConnection> s_serverConn;
static std::mutex s_serverMutex;

static std::shared_ptr<WhillatsServerConnection> getOrCreateServer() {
    std::lock_guard<std::mutex> lock(s_serverMutex);
    if (s_serverConn && s_serverConn->isRunning())
        return s_serverConn;

    s_serverConn = std::make_shared<WhillatsServerConnection>();

    // Find whillats_server binary next to libwhillats.so
    std::string server_path;
    char exe_path[1024];
    ssize_t len = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
    if (len > 0) {
        exe_path[len] = '\0';
        std::string dir(exe_path);
        size_t slash = dir.find_last_of('/');
        if (slash != std::string::npos) dir = dir.substr(0, slash);
        // Try same dir, then ../bin/Debug
        std::string candidates[] = {
            dir + "/whillats_server",
            dir + "/../bin/Debug/whillats_server",
            dir + "/../../modules/third_party/whillats/build/bin/Debug/whillats_server",
        };
        for (auto& c : candidates) {
            if (access(c.c_str(), X_OK) == 0) { server_path = c; break; }
        }
    }
    // Fallback: env var
    if (server_path.empty()) {
        const char* env = getenv("WHILLATS_SERVER");
        if (env && access(env, X_OK) == 0) server_path = env;
    }
    if (server_path.empty()) {
        fprintf(stderr, "[whillats] ERROR: Cannot find whillats_server binary. "
                "Set WHILLATS_SERVER env var.\n");
        return nullptr;
    }

    whillats_ipc::ConfigMsg cfg{};
    const char* e;
    if ((e = getenv("WHISPER_MODEL")))     strncpy(cfg.whisper_model, e, sizeof(cfg.whisper_model)-1);
    if ((e = getenv("LLAMA_MODEL")))       strncpy(cfg.llama_model, e, sizeof(cfg.llama_model)-1);
    if ((e = getenv("LLAMA_MMPROJ")))      strncpy(cfg.llama_mmproj, e, sizeof(cfg.llama_mmproj)-1);
    if ((e = getenv("PIPER_MODEL")))       strncpy(cfg.piper_model, e, sizeof(cfg.piper_model)-1);
    if ((e = getenv("ESPEAK_DATA_PATH")))  strncpy(cfg.espeak_data, e, sizeof(cfg.espeak_data)-1);
    // Leave cfg.language empty so Whisper uses auto-detect mode
    cfg.whisper_threads = 4;
    cfg.llama_threads = 6;
    cfg.tts_threads = 2;
    // Per-language Piper models: PIPER_MODEL_ES, PIPER_MODEL_RU, etc.
    {
        const char* lang_envs[][2] = {
            { "en", "PIPER_MODEL_EN" },
            { "es", "PIPER_MODEL_ES" },
            { "ru", "PIPER_MODEL_RU" },
            { "de", "PIPER_MODEL_DE" },
            { "fr", "PIPER_MODEL_FR" },
            { "zh", "PIPER_MODEL_ZH" },
            { "ja", "PIPER_MODEL_JA" },
        };
        cfg.piper_lang_model_count = 0;
        for (auto& le : lang_envs) {
            const char* path = getenv(le[1]);
            if (path && path[0] && cfg.piper_lang_model_count < whillats_ipc::PIPER_LANG_MAX) {
                int idx = cfg.piper_lang_model_count++;
                strncpy(cfg.piper_lang_models[idx].lang, le[0], sizeof(cfg.piper_lang_models[idx].lang)-1);
                strncpy(cfg.piper_lang_models[idx].path, path, sizeof(cfg.piper_lang_models[idx].path)-1);
            }
        }
    }

    if (!s_serverConn->start(server_path, cfg)) {
        fprintf(stderr, "[whillats] ERROR: Failed to start whillats_server at %s\n", server_path.c_str());
        s_serverConn.reset();
        return nullptr;
    }
    fprintf(stderr, "[whillats] Server started: %s\n", server_path.c_str());
    return s_serverConn;
}
#endif  // WHILLATS_INPROCESS

// ================================================================
// TTS — server-backed on Linux, platform-native on Apple
// ================================================================
#if defined(__APPLE__) && TARGET_OS_IPHONE
#import "whillats_synth.h"
WhillatsTTS::WhillatsTTS(WhillatsSetAudioCallback callback)
    : _callback(callback), _wrapper(std::make_unique<WhillatsSpeechSynthesizerWrapper>()) {}
WhillatsTTS::~WhillatsTTS() { stop(); }
bool WhillatsTTS::start(bool ep) { _wrapper->initialize(&_callback, ep); return true; }
bool WhillatsTTS::start() { return start(true); }
void WhillatsTTS::stop() { _wrapper->stop(); }
int WhillatsTTS::getSampleRate() { return 16000; }
void WhillatsTTS::queueText(const char* t) { _wrapper->synthesize(std::string(t), "en"); }
void WhillatsTTS::queueText(const char* t, const char* l) { _wrapper->synthesize(std::string(t), std::string(l)); }
void WhillatsTTS::enableSpeakerphone() { _wrapper->enableSpeakerphone(); }
void WhillatsTTS::disableSpeakerphone() { _wrapper->disableSpeakerphone(); }
void WhillatsTTS::setThreadCount(int) {}
// The synthesizer wrapper holds &_callback, so updating it in place retargets audio.
void WhillatsTTS::setCallback(WhillatsSetAudioCallback cb) { _callback = cb; }

#elif defined(__APPLE__) && TARGET_OS_OSX
#include "synthesis.h"
WhillatsTTS::WhillatsTTS(WhillatsSetAudioCallback callback)
    : _callback(callback), _synth(std::make_unique<Synthesis>(callback)) {}
WhillatsTTS::~WhillatsTTS() = default;
bool WhillatsTTS::start(bool) { return _synth->start(); }
bool WhillatsTTS::start() { return start(true); }
void WhillatsTTS::stop() { _synth->stop(); }
int WhillatsTTS::getSampleRate() { return Synthesis::getSampleRate(); }
void WhillatsTTS::queueText(const char* t) { _synth->queueText(std::string(t), "en"); }
void WhillatsTTS::queueText(const char* t, const char* l) { _synth->queueText(std::string(t), std::string(l)); }
void WhillatsTTS::enableSpeakerphone() {}
void WhillatsTTS::disableSpeakerphone() {}
void WhillatsTTS::setThreadCount(int) {}

#else
// Linux: always use whillats_server for TTS
WhillatsTTS::WhillatsTTS(WhillatsSetAudioCallback callback) : _callback(callback) {}
WhillatsTTS::~WhillatsTTS() {}
bool WhillatsTTS::start() {
    _conn = getOrCreateServer();
    if (!_conn) return false;
    _conn->setTtsCallback(_callback.callback_, _callback.user_data_);
    _ttsClient = std::make_unique<WhillatsTTSClient>(*_conn);
    return _ttsClient->start();
}
bool WhillatsTTS::start(bool) { return start(); }
void WhillatsTTS::stop() { if (_ttsClient) _ttsClient->stop(); }
void WhillatsTTS::setCallback(WhillatsSetAudioCallback cb) {
    _callback = cb;
    if (_conn) _conn->setTtsCallback(cb.callback_, cb.user_data_);
}
void WhillatsTTS::queueText(const char* t) { if (_ttsClient) _ttsClient->queueText(t, "en"); }
void WhillatsTTS::queueText(const char* t, const char* l) { if (_ttsClient) _ttsClient->queueText(t, l); }
int WhillatsTTS::getSampleRate() { return 16000; }
void WhillatsTTS::setThreadCount(int) {}
void WhillatsTTS::enableSpeakerphone() {}
void WhillatsTTS::disableSpeakerphone() {}
#endif

// ================================================================
// Callback — out-of-line definition
// ================================================================
void WhillatsSetResponseCallback::OnResponseComplete(bool success, const char* response) {
    if (callback_) callback_(success, response, user_data_);
}

#ifdef WHILLATS_INPROCESS
// ================================================================
// Whisper Transcriber — in-process (iOS)
// ================================================================
WhillatsTranscriber::WhillatsTranscriber(const char* model_path,
    WhillatsSetResponseCallback callback,
    WhillatsSetLanguageCallback language_callback)
    : _callback(callback), _language_callback(language_callback),
      _model_path(model_path ? model_path : ""), _language("auto") {}

WhillatsTranscriber::WhillatsTranscriber(const char* model_path,
    WhillatsSetResponseCallback callback)
    : WhillatsTranscriber(model_path, callback, {nullptr, nullptr}) {}

WhillatsTranscriber::~WhillatsTranscriber() { stop(); }

bool WhillatsTranscriber::start() {
    if (!_local) {
        _local = std::make_unique<WhisperTranscriber>(_model_path.c_str(), _callback, _language_callback);
        _local->setTranscriptCallback(_transcript_callback);
        _local->setLanguage(_language);
        if (_threadCount > 0) _local->setThreadCount(_threadCount);
    }
    return _local->start();
}

void WhillatsTranscriber::stop() { if (_local) _local->stop(); }
void WhillatsTranscriber::setCallback(WhillatsSetResponseCallback cb) {
    _callback = cb;
    if (_local) _local->setResponseCallback(cb);
}
void WhillatsTranscriber::setLanguageCallback(WhillatsSetLanguageCallback cb) {
    _language_callback = cb;
    if (_local) _local->setLanguageCallback(cb);
}
void WhillatsTranscriber::setTranscriptCallback(WhillatsSetTranscriptCallback cb) {
    _transcript_callback = cb;
    if (_local) _local->setTranscriptCallback(cb);
}
void WhillatsTranscriber::processAudioBuffer(uint8_t* buf, const size_t size) {
    if (_local) _local->processAudioBuffer(buf, size);
}
std::string WhillatsTranscriber::getLanguage() const { return _local ? _local->getLanguage() : _language; }
void WhillatsTranscriber::setLanguage(const char* lang) {
    if (!lang) return;
    _language = lang;
    if (_local) _local->setLanguage(_language);
}
void WhillatsTranscriber::setThreadCount(int n) {
    _threadCount = n;
    if (_local) _local->setThreadCount(n);
}

// ================================================================
// Llama — in-process (iOS)
// ================================================================
// One model stays resident for the app's lifetime: the audio device recreates
// WhillatsLlama for every call, and reloading GGUF weights costs seconds.
static std::mutex s_llamaMutex;
static std::shared_ptr<LlamaDeviceBase> s_llama;

WhillatsLlama::WhillatsLlama(const char* model_path, WhillatsSetResponseCallback callback)
    : WhillatsLlama(model_path, "", callback) {}

WhillatsLlama::WhillatsLlama(const char* model_path, const char* mmproj_path, WhillatsSetResponseCallback callback)
    : _callback(callback), _model_path(model_path ? model_path : ""), _mmproj_path(mmproj_path ? mmproj_path : "") {}

WhillatsLlama::~WhillatsLlama() {
    stop();
    if (_local) _local->setCallback({nullptr, nullptr});
}

bool WhillatsLlama::start() {
    if (!_local) {
        std::lock_guard<std::mutex> lock(s_llamaMutex);
        if (!s_llama || s_llama->modelPath() != _model_path || s_llama->mmprojPath() != _mmproj_path)
            s_llama = std::make_shared<LlamaDeviceBase>(_model_path.c_str(), _mmproj_path.c_str(), _callback);
        _local = s_llama;
    }
    _local->setCallback(_callback);
    if (_threadCount > 0) _local->setThreadCount(_threadCount);
    return _local->start();
}

bool WhillatsLlama::isRunning() const { return _local && _local->isRunning(); }
void WhillatsLlama::setThreadCount(int n) { _threadCount = n; }
void WhillatsLlama::stop() { if (_local) _local->stop(); }
void WhillatsLlama::setCallback(WhillatsSetResponseCallback cb) {
    _callback = cb;
    if (_local) _local->setCallback(cb);
}
void WhillatsLlama::askLlama(const char* prompt) { if (_local) _local->askLlama(prompt); }
void WhillatsLlama::askWithImageFile(const char*, const char*, int, int) {}
void WhillatsLlama::askWithYUVRaw(const char*, const uint8_t*, const uint8_t*,
    const uint8_t*, int, int, size_t, size_t) {}
void WhillatsLlama::receiveVideoFrame(const YUVData& yuv) { if (_local) _local->receiveVideoFrame(yuv); }

#else
// ================================================================
// Whisper Transcriber — server-backed (Linux, macOS)
// ================================================================
WhillatsTranscriber::WhillatsTranscriber(const char* model_path,
    WhillatsSetResponseCallback callback,
    WhillatsSetLanguageCallback language_callback)
    : _callback(callback), _language_callback(language_callback) {
    if (model_path && model_path[0]) setenv("WHISPER_MODEL", model_path, 1);
}

WhillatsTranscriber::WhillatsTranscriber(const char* model_path,
    WhillatsSetResponseCallback callback)
    : WhillatsTranscriber(model_path, callback, {nullptr, nullptr}) {}

WhillatsTranscriber::~WhillatsTranscriber() {}

bool WhillatsTranscriber::start() {
    _conn = getOrCreateServer();
    if (!_conn) return false;
    _conn->setWhisperCallback(_callback.callback_, _callback.user_data_);
    _conn->setLanguageCallback(_language_callback.callback_, _language_callback.user_data_);
    _conn->setTranscriptCallback(_transcript_callback.callback_, _transcript_callback.user_data_);
    _whisperClient = std::make_unique<WhillatsTranscriberClient>(*_conn);
    return _whisperClient->start();
}

void WhillatsTranscriber::stop() { if (_whisperClient) _whisperClient->stop(); }
void WhillatsTranscriber::setCallback(WhillatsSetResponseCallback cb) {
    _callback = cb;
    if (_conn) _conn->setWhisperCallback(cb.callback_, cb.user_data_);
}
void WhillatsTranscriber::setLanguageCallback(WhillatsSetLanguageCallback cb) {
    _language_callback = cb;
    if (_conn) _conn->setLanguageCallback(cb.callback_, cb.user_data_);
}
void WhillatsTranscriber::setTranscriptCallback(WhillatsSetTranscriptCallback cb) {
    _transcript_callback = cb;
    if (_conn) _conn->setTranscriptCallback(cb.callback_, cb.user_data_);
}

void WhillatsTranscriber::processAudioBuffer(uint8_t* buf, const size_t size) {
    if (_whisperClient) _whisperClient->processAudioBuffer(buf, size);
}

std::string WhillatsTranscriber::getLanguage() const { return _language; }
void WhillatsTranscriber::setLanguage(const char* lang) { if (lang) _language = lang; }
void WhillatsTranscriber::setThreadCount(int n) { _threadCount = n; }

// ================================================================
// Llama — server-backed (Linux, macOS)
// ================================================================
WhillatsLlama::WhillatsLlama(const char* model_path, WhillatsSetResponseCallback callback)
    : _callback(callback) {
    if (model_path && model_path[0]) setenv("LLAMA_MODEL", model_path, 1);
}

WhillatsLlama::WhillatsLlama(const char* model_path, const char* mmproj_path, WhillatsSetResponseCallback callback)
    : _callback(callback) {
    if (model_path && model_path[0]) setenv("LLAMA_MODEL", model_path, 1);
    if (mmproj_path && mmproj_path[0]) setenv("LLAMA_MMPROJ", mmproj_path, 1);
}

WhillatsLlama::~WhillatsLlama() {}

bool WhillatsLlama::start() {
    _conn = getOrCreateServer();
    if (!_conn) return false;
    _conn->setLlamaCallback(_callback.callback_, _callback.user_data_);
    _llamaClient = std::make_unique<WhillatsLlamaClient>(*_conn);
    return _llamaClient->start();
}

bool WhillatsLlama::isRunning() const { return _llamaClient != nullptr; }
void WhillatsLlama::setThreadCount(int) {}
void WhillatsLlama::stop() { if (_llamaClient) _llamaClient->stop(); }
void WhillatsLlama::setCallback(WhillatsSetResponseCallback cb) {
    _callback = cb;
    if (_conn) _conn->setLlamaCallback(cb.callback_, cb.user_data_);
}
void WhillatsLlama::askLlama(const char* prompt) { if (_llamaClient) _llamaClient->askLlama(prompt); }
void WhillatsLlama::askWithImageFile(const char*, const char*, int, int) {}
void WhillatsLlama::askWithYUVRaw(const char*, const uint8_t*, const uint8_t*,
    const uint8_t*, int, int, size_t, size_t) {}

void WhillatsLlama::receiveVideoFrame(const YUVData& yuv) {
    if (_llamaClient) _llamaClient->receiveVideoFrame(yuv);
}

#endif  // WHILLATS_INPROCESS

bool WHILLATS_API save_yuv_as_bmp(const YUVData& yuv, const char* path) {
    (void)yuv; (void)path;
    return false;
}
