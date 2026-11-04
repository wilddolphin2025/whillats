#ifndef WHILLATS_CLIENT_H
#define WHILLATS_CLIENT_H

#include <string>
#include <thread>
#include <atomic>
#include <mutex>
#include <sys/types.h>

#include "whillats.h"
#include "whillats_ipc.h"

class WhillatsServerConnection {
public:
    WhillatsServerConnection();
    ~WhillatsServerConnection();

    bool start(const std::string& server_path, const whillats_ipc::ConfigMsg& cfg);
    void stop();
    bool isRunning() const { return _running; }

    bool sendMsg(uint8_t type, const void* data, uint32_t len);

    void setWhisperCallback(ResponseCallback cb, void* ud) { std::lock_guard<std::mutex> l(_cbMutex); _whisperFn = cb; _whisperUd = ud; }
    void setLanguageCallback(LanguageCallback cb, void* ud) { std::lock_guard<std::mutex> l(_cbMutex); _langFn = cb; _langUd = ud; }
    void setTranscriptCallback(TranscriptCallback cb, void* ud) { std::lock_guard<std::mutex> l(_cbMutex); _transcriptFn = cb; _transcriptUd = ud; }
    void setLlamaCallback(ResponseCallback cb, void* ud) { std::lock_guard<std::mutex> l(_cbMutex); _llamaFn = cb; _llamaUd = ud; }
    void setTtsCallback(AudioCallback cb, void* ud) { std::lock_guard<std::mutex> l(_cbMutex); _ttsFn = cb; _ttsUd = ud; }

    void clearAllCallbacks() {
        std::lock_guard<std::mutex> l(_cbMutex);
        _whisperFn = nullptr; _whisperUd = nullptr;
        _langFn = nullptr; _langUd = nullptr;
        _transcriptFn = nullptr; _transcriptUd = nullptr;
        _llamaFn = nullptr; _llamaUd = nullptr;
        _ttsFn = nullptr; _ttsUd = nullptr;
    }

private:
    void readerThread();

    pid_t _serverPid = -1;
    int _writeFd = -1;
    int _readFd = -1;
    std::atomic<bool> _running{false};
    std::thread _reader;
    std::mutex _writeMutex;
    std::mutex _cbMutex;

    ResponseCallback _whisperFn = nullptr;
    void* _whisperUd = nullptr;
    LanguageCallback _langFn = nullptr;
    void* _langUd = nullptr;
    TranscriptCallback _transcriptFn = nullptr;
    void* _transcriptUd = nullptr;
    ResponseCallback _llamaFn = nullptr;
    void* _llamaUd = nullptr;
    AudioCallback _ttsFn = nullptr;
    void* _ttsUd = nullptr;
};

class WhillatsTranscriberClient {
public:
    WhillatsTranscriberClient(WhillatsServerConnection& conn);
    ~WhillatsTranscriberClient();

    bool start();
    void stop();
    void processAudioBuffer(uint8_t* buffer, size_t size);
    std::string getLanguage() const { return _language; }
    void setLanguage(const char* lang) { if (lang) _language = lang; }
    void setThreadCount(int) {}

private:
    WhillatsServerConnection& _conn;
    std::string _language = "en";
    bool _started = false;
};

class WhillatsLlamaClient {
public:
    WhillatsLlamaClient(WhillatsServerConnection& conn);
    ~WhillatsLlamaClient();

    bool start();
    void stop();
    bool isRunning() const { return _started; }
    void setThreadCount(int) {}
    void askLlama(const char* prompt, const char* lang = nullptr);
    void receiveVideoFrame(const YUVData& yuv);

private:
    WhillatsServerConnection& _conn;
    bool _started = false;
};

class WhillatsTTSClient {
public:
    WhillatsTTSClient(WhillatsServerConnection& conn);
    ~WhillatsTTSClient();

    bool start();
    void stop();
    void queueText(const char* text, const char* language);
    void setVoice(const char* voice);
    void interrupt();
    static int getSampleRate() { return 16000; }

private:
    WhillatsServerConnection& _conn;
    bool _started = false;
};

#endif  // WHILLATS_CLIENT_H
