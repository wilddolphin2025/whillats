#include "whillats_client.h"
#include "whillats_ipc.h"

#include <unistd.h>
#include <signal.h>
#include <sys/wait.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>

using namespace whillats_ipc;

// --- WhillatsServerConnection ---

WhillatsServerConnection::WhillatsServerConnection() {}

WhillatsServerConnection::~WhillatsServerConnection() { stop(); }

bool WhillatsServerConnection::start(const std::string& server_path, const ConfigMsg& cfg) {
    int parent_to_child[2], child_to_parent[2];
    if (pipe(parent_to_child) != 0 || pipe(child_to_parent) != 0) {
        perror("[whillats_client] pipe");
        return false;
    }

    _serverPid = fork();
    if (_serverPid < 0) {
        perror("[whillats_client] fork");
        return false;
    }

    if (_serverPid == 0) {
        close(parent_to_child[1]);
        close(child_to_parent[0]);

        char read_fd_str[16], write_fd_str[16];
        snprintf(read_fd_str, sizeof(read_fd_str), "%d", parent_to_child[0]);
        snprintf(write_fd_str, sizeof(write_fd_str), "%d", child_to_parent[1]);

        execl(server_path.c_str(), "whillats_server", read_fd_str, write_fd_str, nullptr);
        perror("[whillats_client] execl failed");
        _exit(1);
    }

    close(parent_to_child[0]);
    close(child_to_parent[1]);
    _writeFd = parent_to_child[1];
    _readFd  = child_to_parent[0];

    _running = true;

    {
        std::lock_guard<std::mutex> lock(_writeMutex);
        write_msg(_writeFd, MSG_CONFIG, &cfg, sizeof(cfg));
    }

    _reader = std::thread([this]{ readerThread(); });

    fprintf(stderr, "[whillats_client] Server started pid=%d\n", _serverPid);
    return true;
}

void WhillatsServerConnection::stop() {
    if (!_running) return;
    _running = false;

    {
        std::lock_guard<std::mutex> lock(_writeMutex);
        write_msg(_writeFd, MSG_SHUTDOWN, nullptr, 0);
    }

    if (_writeFd >= 0) { close(_writeFd); _writeFd = -1; }

    if (_reader.joinable()) _reader.join();

    if (_readFd >= 0) { close(_readFd); _readFd = -1; }

    if (_serverPid > 0) {
        int status;
        waitpid(_serverPid, &status, 0);
        fprintf(stderr, "[whillats_client] Server exited status=%d\n", status);
        _serverPid = -1;
    }
}

bool WhillatsServerConnection::sendMsg(uint8_t type, const void* data, uint32_t len) {
    std::lock_guard<std::mutex> lock(_writeMutex);
    return write_msg(_writeFd, type, data, len);
}

// Reader thread: uses ONLY C malloc/free — no std::vector, no std::string.
// This runs inside directcall's address space which has -fno-exceptions.
// Any C++ allocation that fails would call std::terminate().
void WhillatsServerConnection::readerThread() {
    Header h;
    while (_running && read_header(_readFd, h)) {
        // Sanity check
        if (h.type == 0 || h.len > 2 * 1024 * 1024) {
            fprintf(stderr, "[whillats_client] corrupt msg type=0x%02x len=%u\n", h.type, h.len);
            break;
        }

        // C malloc for payload — no exceptions possible
        uint8_t* payload = NULL;
        if (h.len > 0) {
            payload = (uint8_t*)malloc(h.len + 1);
            if (!payload) {
                fprintf(stderr, "[whillats_client] malloc(%u) failed\n", h.len);
                break;
            }
            if (!read_exact(_readFd, payload, h.len)) {
                free(payload);
                break;
            }
            payload[h.len] = 0; // null-terminate for string safety
        }

        {
            std::lock_guard<std::mutex> cbLock(_cbMutex);

            switch (h.type) {

            case MSG_WHISPER_RESULT:
                if (_whisperFn && payload)
                    _whisperFn(true, (const char*)payload, _whisperUd);
                break;

            case MSG_WHISPER_TRANSCRIPT:
                if (_transcriptFn && payload && h.len >= 1) {
                    char* text = strchr((char*)payload + 1, '\t');
                    if (text) {
                        *text++ = 0;
                        _transcriptFn(text, payload[0] == 'F', (const char*)payload + 1, _transcriptUd);
                    }
                }
                break;

            case MSG_WHISPER_LANGUAGE:
                if (_langFn && payload)
                    _langFn(true, (const char*)payload, _langUd);
                break;

            case MSG_LLAMA_RESPONSE:
                if (_llamaFn && payload)
                    _llamaFn(true, (const char*)payload, _llamaUd);
                break;

            case MSG_LLAMA_DONE:
                if (_llamaFn)
                    _llamaFn(false, nullptr, _llamaUd);
                break;

            case MSG_TTS_AUDIO:
                if (_ttsFn && payload && h.len >= sizeof(TtsAudioMsg)) {
                    TtsAudioMsg hdr;
                    memcpy(&hdr, payload, sizeof(hdr));
                    const uint16_t* samples = (const uint16_t*)(payload + sizeof(TtsAudioMsg));
                    size_t n = hdr.num_samples;
                    if (sizeof(TtsAudioMsg) + n * sizeof(int16_t) <= h.len)
                        _ttsFn(true, samples, n, _ttsUd);
                }
                break;

            case MSG_TTS_DONE:
                if (_ttsFn)
                    _ttsFn(false, NULL, 0, _ttsUd);
                break;

            default:
                break;
            }
        }

        free(payload);
    }
    fprintf(stderr, "[whillats_client] Reader thread exiting\n");
}

// --- WhillatsTranscriberClient ---

WhillatsTranscriberClient::WhillatsTranscriberClient(WhillatsServerConnection& conn)
    : _conn(conn) {}

WhillatsTranscriberClient::~WhillatsTranscriberClient() { stop(); }

bool WhillatsTranscriberClient::start() {
    _conn.sendMsg(MSG_WHISPER_START, nullptr, 0);
    _started = true;
    return true;
}

void WhillatsTranscriberClient::stop() {
    if (_started) {
        _conn.sendMsg(MSG_WHISPER_STOP, nullptr, 0);
        _started = false;
    }
}

void WhillatsTranscriberClient::processAudioBuffer(uint8_t* buffer, size_t size) {
    if (!_started) return;
    _conn.sendMsg(MSG_WHISPER_AUDIO, buffer, (uint32_t)size);
}

// --- WhillatsLlamaClient ---

WhillatsLlamaClient::WhillatsLlamaClient(WhillatsServerConnection& conn)
    : _conn(conn) {}

WhillatsLlamaClient::~WhillatsLlamaClient() { stop(); }

bool WhillatsLlamaClient::start() {
    _conn.sendMsg(MSG_LLAMA_START, nullptr, 0);
    _started = true;
    return true;
}

void WhillatsLlamaClient::stop() {
    if (_started) {
        _conn.sendMsg(MSG_LLAMA_STOP, nullptr, 0);
        _started = false;
    }
}

// Payload: prompt, optionally followed by '\0' and the language code.
void WhillatsLlamaClient::askLlama(const char* prompt, const char* lang) {
    if (!_started || !prompt) return;
    std::string payload(prompt);
    if (lang && *lang) { payload += '\0'; payload += lang; }
    _conn.sendMsg(MSG_LLAMA_ASK, payload.data(), (uint32_t)payload.size());
}

void WhillatsLlamaClient::receiveVideoFrame(const YUVData& yuv) {
    if (!_started || !yuv.y || !yuv.u || !yuv.v) return;
    size_t payload_sz = 16 + yuv.y_size + 2 * yuv.uv_size;
    uint8_t* buf = (uint8_t*)malloc(payload_sz);
    if (!buf) return;
    int32_t w = yuv.width, ht = yuv.height;
    int32_t ys = (int32_t)yuv.y_size, uvs = (int32_t)yuv.uv_size;
    memcpy(buf,    &w,  4);
    memcpy(buf+4,  &ht, 4);
    memcpy(buf+8,  &ys, 4);
    memcpy(buf+12, &uvs,4);
    memcpy(buf+16, yuv.y.get(), yuv.y_size);
    memcpy(buf+16+yuv.y_size, yuv.u.get(), yuv.uv_size);
    memcpy(buf+16+yuv.y_size+yuv.uv_size, yuv.v.get(), yuv.uv_size);
    _conn.sendMsg(MSG_LLAMA_VIDEO_FRAME, buf, (uint32_t)payload_sz);
    free(buf);
}

// --- WhillatsTTSClient ---

WhillatsTTSClient::WhillatsTTSClient(WhillatsServerConnection& conn)
    : _conn(conn) {}

WhillatsTTSClient::~WhillatsTTSClient() { stop(); }

bool WhillatsTTSClient::start() {
    _conn.sendMsg(MSG_TTS_START, nullptr, 0);
    _started = true;
    return true;
}

void WhillatsTTSClient::stop() {
    if (_started) {
        _conn.sendMsg(MSG_TTS_STOP, nullptr, 0);
        _started = false;
    }
}

void WhillatsTTSClient::queueText(const char* text, const char* language) {
    if (!_started || !text) return;
    const char* lang = language ? language : "en";
    uint16_t lang_len = (uint16_t)strlen(lang);
    uint32_t text_len = (uint32_t)strlen(text);
    uint32_t total = 2 + lang_len + text_len;
    uint8_t* buf = (uint8_t*)malloc(total);
    if (!buf) return;
    memcpy(buf, &lang_len, 2);
    memcpy(buf+2, lang, lang_len);
    memcpy(buf+2+lang_len, text, text_len);
    _conn.sendMsg(MSG_TTS_SPEAK, buf, total);
    free(buf);
}
