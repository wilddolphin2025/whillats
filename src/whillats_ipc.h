#ifndef WHILLATS_IPC_H
#define WHILLATS_IPC_H

#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <unistd.h>

namespace whillats_ipc {

enum MsgType : uint8_t {
    MSG_WHISPER_START      = 0x01,
    MSG_WHISPER_STOP       = 0x02,
    MSG_WHISPER_AUDIO      = 0x03,
    MSG_WHISPER_RESULT     = 0x04,
    MSG_WHISPER_LANGUAGE   = 0x05,
    MSG_WHISPER_TRANSCRIPT = 0x06,  // payload: 'I'|'F', language, '\t', text

    MSG_LLAMA_START        = 0x10,
    MSG_LLAMA_STOP         = 0x11,
    MSG_LLAMA_ASK          = 0x12,
    MSG_LLAMA_RESPONSE     = 0x13,
    MSG_LLAMA_VIDEO_FRAME  = 0x14,
    MSG_LLAMA_DONE         = 0x15,  // sent after all tokens for a prompt are delivered

    MSG_TTS_START          = 0x20,
    MSG_TTS_STOP           = 0x21,
    MSG_TTS_SPEAK          = 0x22,
    MSG_TTS_AUDIO          = 0x23,
    MSG_TTS_DONE           = 0x24,
    MSG_TTS_VOICE          = 0x25,  // payload: voice clip name inside the voices dir
    MSG_TTS_INTERRUPT      = 0x26,  // drop queued text, abort current utterance

    MSG_CONFIG             = 0x30,
    MSG_SHUTDOWN           = 0xFF,
};

struct Header {
    uint8_t  type;
    uint32_t len;
} __attribute__((packed));

static constexpr size_t HEADER_SIZE = sizeof(Header);

inline bool write_msg(int fd, uint8_t type, const void* data, uint32_t len) {
    Header h;
    h.type = type;
    h.len  = len;
    
    size_t total = HEADER_SIZE + len;
    uint8_t* buf = (uint8_t*)malloc(total);
    if (!buf) return false;
    memcpy(buf, &h, HEADER_SIZE);
    if (len > 0 && data)
        memcpy(buf + HEADER_SIZE, data, len);
    
    const uint8_t* ptr = buf;
    size_t rem = total;
    while (rem > 0) {
        ssize_t n = ::write(fd, ptr, rem);
        if (n <= 0) { free(buf); return false; }
        ptr += n; rem -= n;
    }
    free(buf);
    return true;
}

inline bool read_exact(int fd, void* buf, size_t count) {
    uint8_t* ptr = reinterpret_cast<uint8_t*>(buf);
    while (count > 0) {
        ssize_t n = ::read(fd, ptr, count);
        if (n <= 0) return false;
        ptr += n; count -= n;
    }
    return true;
}

inline bool read_header(int fd, Header& h) {
    return read_exact(fd, &h, HEADER_SIZE);
}

struct PiperLangEntry {
    char lang[8];    // e.g. "en", "es", "ru"
    char path[504];  // path to .onnx model
} __attribute__((packed));  // 512 bytes each

static constexpr int32_t PIPER_LANG_MAX = 8;

struct ConfigMsg {
    char whisper_model[512];
    char llama_model[512];
    char llama_mmproj[512];
    char piper_model[512];
    char espeak_data[512];
    char language[32];
    int32_t whisper_threads;
    int32_t llama_threads;
    int32_t tts_threads;
    // Per-language Piper models (supplement piper_model default)
    PiperLangEntry piper_lang_models[PIPER_LANG_MAX];
    int32_t piper_lang_model_count;
} __attribute__((packed));

struct TtsAudioMsg {
    int32_t sample_rate;
    uint32_t num_samples;
    // followed by num_samples * sizeof(int16_t) bytes of audio
} __attribute__((packed));

}  // namespace whillats_ipc

#endif  // WHILLATS_IPC_H
