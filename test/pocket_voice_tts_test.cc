// Standalone checks for the Pocket TTS adapter: resampler response and the
// voice-name allow list. Links against stub ptt_* symbols, no models needed.
//   c++ -std=c++17 -Isrc -Ithird_party/pocket-tts-raven/include \
//       test/pocket_voice_tts_test.cc src/pocket_voice_tts.cc -o /tmp/pvt && /tmp/pvt

#include <cmath>
#include <cstdio>
#include <vector>

#include "pocket_tts.h"
#include "pocket_voice_tts.h"

extern "C" {
void* ptt_create(const char*, const char*, const char*, const char*, float, int, int) { return nullptr; }
double ptt_warmup(void*) { return 0; }
void ptt_destroy(void*) {}
int ptt_prepare_voice(void*, const char*) { return 0; }
void* ptt_stream_start(void*, const char*, const char*) { return nullptr; }
int ptt_stream_read(void*, float**, int*) { return 0; }
void ptt_stream_stop(void*) {}
void ptt_stream_end(void*) {}
void ptt_free_audio(float*) {}
}

static double ToneRms(double freq_hz, size_t* out_len, size_t in_len = 24000) {
    std::vector<float> in(in_len);
    for (size_t i = 0; i < in.size(); ++i)
        in[i] = 0.5f * static_cast<float>(std::sin(2 * M_PI * freq_hz * i / 24000.0));
    Resampler24kTo16k rs;
    std::vector<int16_t> out;
    // Feed in uneven chunks, as the TTS stream does.
    for (size_t pos = 0, step = 997; pos < in.size(); pos += step)
        rs.process(in.data() + pos, std::min(step, in.size() - pos), out);
    rs.flush(out);
    *out_len = out.size();
    double acc = 0;
    // Skip filter warm-up / tail.
    for (size_t i = 100; i + 100 < out.size(); ++i) acc += double(out[i]) * out[i];
    return std::sqrt(acc / (out.size() - 200)) / 32767.0;
}

int main() {
    int failures = 0;
    auto check = [&](bool ok, const char* what) {
        printf("%s %s\n", ok ? "PASS" : "FAIL", what);
        failures += !ok;
    };
    size_t len = 0;
    const double pass_rms = ToneRms(1000, &len);
    // 0.5 amplitude sine -> RMS 0.354.
    check(std::fabs(pass_rms - 0.3536) < 0.01, "1 kHz passes at unity gain");
    check(len >= 16000 && len <= 16000 + 12, "24000 in -> ~16000 out");
    const double stop_rms = ToneRms(10000, &len);
    check(stop_rms < 0.3536 * 0.01, "10 kHz (aliasing band) attenuated > 40 dB");
    printf("     1 kHz rms=%.4f, 10 kHz rms=%.6f (%.1f dB)\n", pass_rms, stop_rms,
           20 * std::log10(stop_rms / 0.3536));

    PocketTTS tts(WhillatsSetAudioCallback(nullptr, nullptr));
    check(!tts.setVoice("../../etc/passwd"), "rejects path traversal");
    check(!tts.setVoice("/etc/passwd"), "rejects absolute path");
    check(!tts.setVoice(".hidden.wav"), "rejects dot files");
    check(!tts.setVoice("missing.wav"), "rejects unknown clip");
    return failures;
}
