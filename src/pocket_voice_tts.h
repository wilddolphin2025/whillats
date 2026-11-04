/*
 *  (c) 2026, wilddolphin2025
 *  For WebRTCsays.ai project
 *  https://github.com/wilddolphin2025
 *
 *  Pocket TTS: Kyutai's 100M-parameter TTS (weights CC-BY-4.0) run through
 *  the pocket-tts-raven ONNX runtime (MIT). Clones a voice zero-shot from a
 *  6-15 s reference clip, so every call can speak in a personalized voice.
 */

#ifndef POCKET_VOICE_TTS_H
#define POCKET_VOICE_TTS_H

#include "whillats.h"

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// Streaming 24 kHz -> 16 kHz converter (2:3) with an anti-aliasing FIR.
// Output sample n sits at input position 1.5 n, so only two fractional
// phases (0 and 0.5) are ever needed.
class Resampler24kTo16k {
public:
    Resampler24kTo16k();
    void process(const float* in, size_t n, std::vector<int16_t>& out);
    // Drains the filter tail and rewinds for the next utterance.
    void flush(std::vector<int16_t>& out);

private:
    void reset();
    static constexpr int kHalf = 16;  // taps per side
    float taps_[2][2 * kHalf];
    std::vector<float> buf_;
    double pos_ = 0;
};

class PocketTTS {
public:
    explicit PocketTTS(WhillatsSetAudioCallback callback);
    ~PocketTTS();

    // models_dir: bundle prepared by pocket-tts-raven's tools/prepare_models.sh.
    // voices_dir: reference clips (wav/mp3) plus the runtime's .cache/.
    // language:   language of the bundle ("en", "fr", ...).
    bool start(const std::string& models_dir, const std::string& voices_dir,
               const std::string& default_voice, const std::string& language,
               int threads);
    void stop();

    // Selects the voice for subsequent utterances. Only bare file names that
    // exist inside voices_dir are accepted, since the name can originate from
    // a remote peer. Encoding the clip happens in the background.
    bool setVoice(const std::string& voice);
    std::string voice() const;

    void queueText(const char* text, const char* language);
    // Drops queued text and aborts the utterance in progress (barge-in).
    void interrupt();

    bool supportsLanguage(const std::string& language) const;
    static constexpr int kSampleRate = 16000;

private:
    struct Job {
        enum Kind { kSpeak, kPrepareVoice } kind;
        std::string text;
    };
    void run();
    void emit(std::vector<int16_t>& pcm);

    WhillatsSetAudioCallback _callback;
    void* _handle = nullptr;
    std::string _voicesDir;
    std::string _language;

    mutable std::mutex _mutex;
    std::condition_variable _cv;
    std::deque<Job> _queue;
    std::string _voice;
    void* _activeStream = nullptr;
    bool _running = false;
    std::thread _worker;
};

#endif  // POCKET_VOICE_TTS_H
