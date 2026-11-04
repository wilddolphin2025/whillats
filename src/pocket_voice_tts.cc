/*
 *  (c) 2026, wilddolphin2025
 *  For WebRTCsays.ai project
 *  https://github.com/wilddolphin2025
 */

#include "pocket_voice_tts.h"

#include <sys/stat.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

#include "pocket_tts.h"  // pocket-tts-raven C API

namespace {

constexpr double kPi = 3.14159265358979323846;
// Cutoff relative to the 24 kHz input: 7.2 kHz, under the 8 kHz output Nyquist.
constexpr double kCutoff = 0.3;
constexpr size_t kMaxVoiceName = 128;

bool IsSafeVoiceName(const std::string& name) {
    if (name.empty() || name.size() > kMaxVoiceName || name[0] == '.') return false;
    return std::all_of(name.begin(), name.end(), [](char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
               (c >= '0' && c <= '9') || c == '_' || c == '-' || c == '.';
    });
}

bool FileExists(const std::string& path) {
    struct stat st;
    return stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

}  // namespace

Resampler24kTo16k::Resampler24kTo16k() {
    for (int phase = 0; phase < 2; ++phase) {
        const double frac = phase * 0.5;
        double sum = 0;
        for (int j = 0; j < 2 * kHalf; ++j) {
            const double t = j - kHalf + 1 - frac;
            const double x = 2 * kCutoff * t;
            const double sinc = t == 0 ? 1.0 : std::sin(kPi * x) / (kPi * x);
            // Blackman window over [-kHalf, kHalf].
            const double w = 0.42 + 0.5 * std::cos(kPi * t / kHalf) +
                             0.08 * std::cos(2 * kPi * t / kHalf);
            taps_[phase][j] = static_cast<float>(2 * kCutoff * sinc * w);
            sum += taps_[phase][j];
        }
        for (int j = 0; j < 2 * kHalf; ++j) taps_[phase][j] /= static_cast<float>(sum);
    }
    reset();
}

void Resampler24kTo16k::reset() {
    buf_.assign(kHalf - 1, 0.0f);
    pos_ = kHalf - 1;
}

void Resampler24kTo16k::process(const float* in, size_t n, std::vector<int16_t>& out) {
    buf_.insert(buf_.end(), in, in + n);
    while (static_cast<size_t>(pos_) + kHalf < buf_.size()) {
        const size_t i = static_cast<size_t>(pos_);
        const float* taps = taps_[(pos_ - i) > 0.25 ? 1 : 0];
        const float* x = &buf_[i - kHalf + 1];
        float acc = 0;
        for (int j = 0; j < 2 * kHalf; ++j) acc += taps[j] * x[j];
        acc = std::max(-1.0f, std::min(1.0f, acc));
        out.push_back(static_cast<int16_t>(std::lrint(acc * 32767.0f)));
        pos_ += 1.5;
    }
    const long drop = static_cast<long>(pos_) - kHalf + 1;
    if (drop > 0) {
        buf_.erase(buf_.begin(), buf_.begin() + drop);
        pos_ -= drop;
    }
}

void Resampler24kTo16k::flush(std::vector<int16_t>& out) {
    const std::vector<float> tail(kHalf, 0.0f);
    process(tail.data(), tail.size(), out);
    reset();
}

PocketTTS::PocketTTS(WhillatsSetAudioCallback callback) : _callback(callback) {}

PocketTTS::~PocketTTS() { stop(); }

bool PocketTTS::start(const std::string& models_dir, const std::string& voices_dir,
                      const std::string& default_voice, const std::string& language,
                      int threads) {
    if (_handle) return true;
    const std::string tokenizer = models_dir + "/tokenizer.model";
    _handle = ptt_create(models_dir.c_str(), voices_dir.c_str(), tokenizer.c_str(),
                         "int8", /*temperature=*/0.7f, /*lsd_steps=*/1, threads);
    if (!_handle) {
        fprintf(stderr, "[pocket_tts] failed to load models from %s\n", models_dir.c_str());
        return false;
    }
    ptt_warmup(_handle);
    _voicesDir = voices_dir;
    _language = language.empty() ? "en" : language;
    {
        std::lock_guard<std::mutex> lock(_mutex);
        _running = true;
    }
    _worker = std::thread([this] { run(); });
    if (!default_voice.empty() && !setVoice(default_voice)) {
        fprintf(stderr, "[pocket_tts] default voice '%s' not found in %s\n",
                default_voice.c_str(), voices_dir.c_str());
    }
    fprintf(stderr, "[pocket_tts] ready (lang=%s voice=%s)\n", _language.c_str(),
            voice().c_str());
    return true;
}

void PocketTTS::stop() {
    {
        std::lock_guard<std::mutex> lock(_mutex);
        if (!_running) return;
        _running = false;
        _queue.clear();
        if (_activeStream) ptt_stream_stop(_activeStream);
    }
    _cv.notify_all();
    if (_worker.joinable()) _worker.join();
    ptt_destroy(_handle);
    _handle = nullptr;
}

bool PocketTTS::setVoice(const std::string& voice) {
    if (!IsSafeVoiceName(voice) || !FileExists(_voicesDir + "/" + voice)) {
        fprintf(stderr, "[pocket_tts] rejected voice '%s'\n", voice.c_str());
        return false;
    }
    {
        std::lock_guard<std::mutex> lock(_mutex);
        if (_voice == voice) return true;
        _voice = voice;
        // Encode the clip now so the first sentence in this voice is not slow.
        _queue.push_front({Job::kPrepareVoice, voice});
    }
    _cv.notify_one();
    return true;
}

std::string PocketTTS::voice() const {
    std::lock_guard<std::mutex> lock(_mutex);
    return _voice;
}

bool PocketTTS::supportsLanguage(const std::string& language) const {
    return language.empty() || language.compare(0, 2, _language, 0, 2) == 0;
}

void PocketTTS::queueText(const char* text, const char* /*language*/) {
    if (!text || !*text) return;
    {
        std::lock_guard<std::mutex> lock(_mutex);
        if (!_running) return;
        _queue.push_back({Job::kSpeak, text});
    }
    _cv.notify_one();
}

void PocketTTS::interrupt() {
    std::lock_guard<std::mutex> lock(_mutex);
    _queue.erase(std::remove_if(_queue.begin(), _queue.end(),
                                [](const Job& j) { return j.kind == Job::kSpeak; }),
                 _queue.end());
    if (_activeStream) ptt_stream_stop(_activeStream);
}

void PocketTTS::emit(std::vector<int16_t>& pcm) {
    if (pcm.empty()) return;
    std::vector<uint16_t> out(pcm.size());
    std::memcpy(out.data(), pcm.data(), pcm.size() * sizeof(int16_t));
    _callback.OnBufferComplete(true, out);
    pcm.clear();
}

void PocketTTS::run() {
    Resampler24kTo16k resampler;
    std::vector<int16_t> pcm;
    for (;;) {
        Job job;
        std::string voice;
        {
            std::unique_lock<std::mutex> lock(_mutex);
            _cv.wait(lock, [this] { return !_running || !_queue.empty(); });
            if (!_running) return;
            job = std::move(_queue.front());
            _queue.pop_front();
            voice = _voice;
        }
        if (voice.empty()) {
            fprintf(stderr, "[pocket_tts] no voice selected, dropping text\n");
            if (job.kind == Job::kSpeak) _callback.OnSynthesisComplete();
            continue;
        }
        if (job.kind == Job::kPrepareVoice) {
            if (ptt_prepare_voice(_handle, job.text.c_str()) != 0)
                fprintf(stderr, "[pocket_tts] could not encode voice '%s'\n", job.text.c_str());
            continue;
        }

        void* stream = ptt_stream_start(_handle, job.text.c_str(), voice.c_str());
        if (!stream) {
            fprintf(stderr, "[pocket_tts] synthesis failed for voice '%s'\n", voice.c_str());
            _callback.OnSynthesisComplete();
            continue;
        }
        {
            std::lock_guard<std::mutex> lock(_mutex);
            _activeStream = stream;
        }
        float* chunk = nullptr;
        int n = 0;
        while (ptt_stream_read(stream, &chunk, &n) == 1) {
            resampler.process(chunk, static_cast<size_t>(n), pcm);
            ptt_free_audio(chunk);
            emit(pcm);
        }
        resampler.flush(pcm);
        emit(pcm);
        {
            std::lock_guard<std::mutex> lock(_mutex);
            _activeStream = nullptr;
        }
        ptt_stream_end(stream);
        _callback.OnSynthesisComplete();
    }
}
