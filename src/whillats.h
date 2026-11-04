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

#ifndef WHILLATS_H
#define WHILLATS_H

#include "whillats_export.h"
#if defined(__APPLE__)
#include <TargetConditionals.h>
#endif

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

// Callback function pointer types
typedef void (*ResponseCallback)(bool success, const char* response, void* user_data);
typedef void (*AudioCallback)(bool success, const uint16_t* buffer, size_t buffer_size, void* user_data);
typedef void (*LanguageCallback)(bool success, const char* language, void* user_data);
// Streaming STT: interim hypotheses (is_final == false) refine while the user
// speaks; exactly one final result (is_final == true) is emitted per utterance.
typedef void (*TranscriptCallback)(const char* text, bool is_final, const char* language, void* user_data);

#if defined(__APPLE__) && TARGET_OS_IPHONE
// iOS forbids spawning helper processes: run Whisper/Llama inside the app.
#define WHILLATS_INPROCESS 1
#endif

struct clip_image_u8 {
    int width;
    int height;
    uint8_t* data; // RGB, interleaved [R,G,B,R,G,B,...]
};

// Simple YUV frame container used by llama video ingest
struct YUVData {
    int width = 0;
    int height = 0;
    size_t y_size = 0;
    size_t uv_size = 0;
    std::unique_ptr<uint8_t[]> y;
    std::unique_ptr<uint8_t[]> u;
    std::unique_ptr<uint8_t[]> v;
};

class WhillatsSpeechSynthesizerWrapper;
class WhisperTranscriber;
class LlamaDeviceBase;
class Synthesis;
class WhillatsServerConnection;
class WhillatsTranscriberClient;
class WhillatsLlamaClient;
class WhillatsTTSClient;

class WHILLATS_API WhillatsSetResponseCallback {
public:
    WhillatsSetResponseCallback(ResponseCallback callback, void* user_data)
        : callback_(callback), user_data_(user_data) {}

    void OnResponseComplete(bool success, const char* response);

    ResponseCallback callback_;
    void* user_data_;
};

class WHILLATS_API WhillatsSetAudioCallback {
public:
    WhillatsSetAudioCallback(AudioCallback callback, void* user_data)
        : callback_(callback), user_data_(user_data) {}

    void OnBufferComplete(bool success, const std::vector<uint16_t>& buffer) {
        if (callback_) {
            callback_(success, buffer.data(), buffer.size(), user_data_);
        }
    }

    void OnSynthesisComplete() {
        if (callback_) {
            callback_(false, nullptr, 0, user_data_);
        }
    }

    AudioCallback callback_;
    void* user_data_;
};

class WHILLATS_API WhillatsSetLanguageCallback {
public:
    WhillatsSetLanguageCallback(LanguageCallback callback, void* user_data)
        : callback_(callback), user_data_(user_data) {}

    void OnLanguageChanged(bool success, const char* language) {
        if (callback_) {
            callback_(success, language, user_data_);
        }
    }

    void OnLanguageDetected(bool success, const std::string& language) {
        OnLanguageChanged(success, language.c_str());
    }

    LanguageCallback callback_;
    void* user_data_;
};

class WHILLATS_API WhillatsSetTranscriptCallback {
public:
    WhillatsSetTranscriptCallback(TranscriptCallback callback = nullptr, void* user_data = nullptr)
        : callback_(callback), user_data_(user_data) {}

    void OnTranscript(const char* text, bool is_final, const char* language) const {
        if (callback_) callback_(text, is_final, language, user_data_);
    }

    TranscriptCallback callback_;
    void* user_data_;
};

class WHILLATS_API WhillatsTTS {
  public:
    WhillatsTTS(WhillatsSetAudioCallback callback);
    ~WhillatsTTS();

    bool start();
    bool start(bool withAudio);
    void stop();
    void queueText(const char* text);
    void queueText(const char* text, const char* language);
    void setThreadCount(int n);
    void enableSpeakerphone();
    void disableSpeakerphone();
    void setCallback(WhillatsSetAudioCallback cb);
    // Personalized voice: name of a reference clip known to the TTS server.
    // Returns false when the build has no voice-cloning engine.
    bool setVoice(const char* voice);
    // Stops speaking now and drops queued text (barge-in).
    void interrupt();

    static int getSampleRate();

  private:
    WhillatsSetAudioCallback _callback;
#ifndef WHILLATS_INPROCESS
    std::shared_ptr<WhillatsServerConnection> _conn;
    std::unique_ptr<WhillatsTTSClient> _ttsClient;
#endif
#if defined(__APPLE__) && TARGET_OS_IPHONE
    std::unique_ptr<WhillatsSpeechSynthesizerWrapper> _wrapper;
#elif defined(__APPLE__) && TARGET_OS_OSX
    std::unique_ptr<Synthesis> _synth;
#endif
};

class WHILLATS_API WhillatsTranscriber {
  public:
    WhillatsTranscriber(const char* model_path, WhillatsSetResponseCallback callback);
    WhillatsTranscriber(const char* model_path, WhillatsSetResponseCallback callback, WhillatsSetLanguageCallback languageCallback);
    ~WhillatsTranscriber();

    bool start();
    void stop();

    void processAudioBuffer(uint8_t* playoutBuffer, const size_t playoutBufferSize);

    // Language control used by factory
    std::string getLanguage() const;
    void setLanguage(const char* language);
    void setThreadCount(int n);
    // The response callback receives final transcripts only; subscribe here to
    // also receive interim hypotheses.
    void setCallback(WhillatsSetResponseCallback cb);
    void setLanguageCallback(WhillatsSetLanguageCallback cb);
    void setTranscriptCallback(WhillatsSetTranscriptCallback cb);

  private:
    WhillatsSetResponseCallback _callback; 
    WhillatsSetLanguageCallback _language_callback;
    WhillatsSetTranscriptCallback _transcript_callback;
#ifndef WHILLATS_INPROCESS
    std::shared_ptr<WhillatsServerConnection> _conn;
    std::unique_ptr<WhillatsTranscriberClient> _whisperClient;
#endif
#ifdef WHILLATS_INPROCESS
    std::string _model_path;
    std::unique_ptr<WhisperTranscriber> _local;
#endif
    std::string _language = "en";
    int _threadCount = 0;
};

class WHILLATS_API WhillatsLlama {
  public:
    WhillatsLlama(const char* model_path, WhillatsSetResponseCallback callback);
    WhillatsLlama(const char* model_path, const char* mmproj_path, WhillatsSetResponseCallback callback);
    ~WhillatsLlama();

    bool start();
    void stop();
    bool isRunning() const;
    void setThreadCount(int n);
    void askLlama(const char* prompt, const char* lang = nullptr);
    void askWithImageFile(const char* prompt, const char* image_file, int width, int height);
    void askWithYUVRaw(
        const char* prompt,
        const uint8_t* y_plane,
        const uint8_t* u_plane,
        const uint8_t* v_plane,
        int width,
        int height,
        size_t y_size,
        size_t uv_size);

    void receiveVideoFrame(const YUVData& yuv);
    void setCallback(WhillatsSetResponseCallback cb);

  private:
    WhillatsSetResponseCallback _callback;
#ifndef WHILLATS_INPROCESS
    std::shared_ptr<WhillatsServerConnection> _conn;
    std::unique_ptr<WhillatsLlamaClient> _llamaClient;
#endif
#ifdef WHILLATS_INPROCESS
    std::string _model_path, _mmproj_path;
    int _threadCount = 0;
    std::shared_ptr<LlamaDeviceBase> _local;
#endif
};

// Helper functions
bool WHILLATS_API save_yuv_as_bmp(const YUVData& yuv, const char* path);

#endif // WHILLATS_H
