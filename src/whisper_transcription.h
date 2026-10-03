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

#ifndef WHISPER_TRANSCRIPTION_H
#define WHISPER_TRANSCRIPTION_H

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <whisper.h>
#include "whisper_helpers.h"
#include "whillats.h"

// Streaming Whisper STT.
//
// Audio (16 kHz mono PCM16) is segmented into utterances by an adaptive
// energy VAD. While an utterance is in progress, a cheap decode over the
// audio heard so far is emitted as an interim hypothesis every
// interimIntervalMs. When the VAD sees endpointSilenceMs of trailing silence
// (or the utterance hits kMaxUtteranceMs), one higher-quality decode produces
// the final result. Final results go to both the transcript callback and the
// legacy response callback; interims go to the transcript callback only, so
// downstream LLM/command consumers act on final results exclusively.
class WhisperTranscriber {
public:
    WhisperTranscriber(const char* modelPath,
        WhillatsSetResponseCallback callback,
        WhillatsSetLanguageCallback languageCallback);
    ~WhisperTranscriber();

    bool start();
    // Aborts any in-flight decode and drops the unfinished utterance.
    void stop();

    // PCM16 little-endian mono at 16 kHz. A zero-length buffer finalizes the
    // utterance in progress.
    void processAudioBuffer(uint8_t* playoutBuffer, size_t kPlayoutBufferSize);

    void setTranscriptCallback(WhillatsSetTranscriptCallback cb);
    void setResponseCallback(WhillatsSetResponseCallback cb);
    void setLanguageCallback(WhillatsSetLanguageCallback cb);
    void setLanguage(const std::string& language) { std::lock_guard<std::mutex> l(_cfgMutex); _language = language; }
    std::string getLanguage() { std::lock_guard<std::mutex> l(_cfgMutex); return _language; }
    void setDetectLanguage(bool detectLanguage) { if (detectLanguage) setLanguage("auto"); }
    // Biases decoding toward expected vocabulary, e.g. "dial, hang up, mute".
    void setInitialPrompt(const std::string& prompt) { std::lock_guard<std::mutex> l(_cfgMutex); _initialPrompt = prompt; }
    // Comma-separated ISO codes ("en,es,ru") that auto-detection may choose from.
    void setAllowedLanguages(const std::string& codes);

    // Speech/noise ratio required to open the VAD (linear RMS ratio).
    void setVADThreshold(float threshold) { _vadSnr = std::max(1.5f, threshold * 4.0f); }
    float getVADThreshold() { return _vadSnr / 4.0f; }
    void setThreadCount(int n) { _nThreads = n; }
    void setInterimIntervalMs(int ms) { _interimIntervalMs = std::max(200, ms); }
    void setEndpointSilenceMs(int ms) { _endpointSilenceMs = std::max(200, ms); }

private:
    struct Decoded {
        std::string text;
        std::string language;
    };

    bool initModel(const std::string& modelPath);
    void run();
    void consumeFrame(const float* frame);
    void maybeEmitInterim();
    void finalizeUtterance();
    void resetUtterance();
    bool decode(const std::vector<float>& pcm, bool final, Decoded& out);
    void emit(const std::string& text, bool final, const std::string& lang);
    static bool isValidText(std::string& text);
    static bool abortCallback(void* user_data);
    void warmUp(whisper_state* state, int audioCtx);
    std::string detectLanguage(const std::vector<float>& pcm, const std::vector<int>& allowed);
    static std::string collapseRepeats(const std::string& s);

    whisper_context* _ctx = nullptr;
    whisper_state* _state = nullptr;
    whisper_state* _detState = nullptr;  // language detection only, created on first auto-detect

    WhillatsSetResponseCallback _responseCallback;
    WhillatsSetLanguageCallback _languageCallback;
    WhillatsSetTranscriptCallback _transcriptCallback;
    std::mutex _cbMutex;

    std::mutex _cfgMutex;
    std::string _language = "auto";
    std::string _initialPrompt;    // English-only bias (e.g. phone commands)
    std::vector<int> _allowedLangs; // auto-detect candidates; empty = all

    // Capture side (audio thread) -> worker thread.
    AudioRingBuffer<float> _ring;
    std::vector<float> _scratch;
    std::mutex _wakeMutex;
    std::condition_variable _wake;
    std::atomic<bool> _flushRequested{false};

    std::thread _worker;
    std::mutex _threadMutex;
    std::atomic<bool> _running{false};

    // Worker-thread state.
    std::vector<float> _frame;     // partially filled VAD frame
    std::vector<float> _preroll;   // circular history kept before speech onset
    size_t _prerollPos = 0;
    std::vector<float> _utt;       // current utterance audio
    bool _inSpeech = false;
    int _onsetRun = 0;
    int _silenceMs = 0;
    int _speechMs = 0;
    int _msSinceInterim = 0;
    bool _newAudioSinceInterim = false;
    float _noise = 0.003f;
    std::string _uttLang;          // language locked for the current utterance
    std::string _convLang;         // language of the last final (fallback for short commands)
    std::string _lastInterim;
    std::string _lastReportedLang;

    float _vadSnr = 3.0f;
    int _nThreads = 0;
    int _interimIntervalMs = 600;
    int _endpointSilenceMs = 700;

    static constexpr int kSampleRate = WHISPER_SAMPLE_RATE;
    static constexpr int kFrameMs = 20;
    static constexpr int kFrameSamples = kSampleRate * kFrameMs / 1000;
    static constexpr int kPrerollMs = 300;
    static constexpr int kOnsetMs = 60;
    static constexpr int kMinSpeechMs = 250;
    static constexpr int kMinInterimMs = 800;
    static constexpr int kMaxUtteranceMs = 15000;
    static constexpr int kFastFinalMs = 4000;
    static constexpr int kLangLockMs = 1500;    // shorter interims are too short to trust detection
    static constexpr int kLangDetectMs = 3000;  // audio window used for language detection
    static constexpr int kTailKeepMs = 200;
    static constexpr float kAbsMinRms = 0.004f;
};

#endif
