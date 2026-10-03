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

#ifndef LLAMA_DEVICE_BASE_H
#define LLAMA_DEVICE_BASE_H

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <atomic>
#include <queue>
#include <thread>
#include <functional>
#include <condition_variable>
#include <deque>
#include <set>
#include <algorithm>
#include <memory>
#include <chrono>
#include <mutex>

#include "whillats.h"
#include "whisper_helpers.h"
#include "llama.h"
#include "mtmd.h"

struct llama_model;
struct llama_context;
struct llama_sampler;
struct llama_vocab;
typedef int32_t llama_token;

class LlamaSimpleChat {
public:
  LlamaSimpleChat();
  ~LlamaSimpleChat();

  LlamaSimpleChat(const LlamaSimpleChat&) = delete;
  LlamaSimpleChat& operator=(const LlamaSimpleChat&) = delete;

  bool SetModelPaths(const std::string &path, const std::string &mmproj_path);
  bool SetNGL(int layers);
  bool SetContextSize(int size);
  void SetThreadCount(int n);
  void StopGeneration();

  bool Initialize();
  // Both keep the running conversation in the KV cache; old turns are shifted
  // out (system prompt preserved) when the context fills up.
  std::string generate(const std::string& request, WhillatsSetResponseCallback callback,
                       const std::string& lang = "");
  std::string generateFromImage(YUVData* yuv, const std::string& prompt, WhillatsSetResponseCallback callback,
                                const std::string& lang = "");
  void ResetConversation();

  bool InitializeContext();
  void FreeContext();
  bool LoadModel();

  std::string model_path_;
  std::string mmproj_path_;
  int ngl_ = 999;
  int n_ctx_ = 4096;
  int n_threads_ = 0;

  llama_model* model_ = nullptr;
  const llama_vocab* vocab_ = nullptr;
  llama_context* ctx_ = nullptr;

  struct MtmdContextDeleter {
    void operator()(mtmd_context* p) const { if (p) mtmd_free(p); }
  };
  std::unique_ptr<mtmd_context, MtmdContextDeleter> ctx_mtmd_;

  llama_sampler* smpl_ = nullptr;
  int n_past_ = 0;   // tokens currently in the KV cache
  int n_sys_ = 0;    // leading tokens that are never shifted out
  bool image_in_context_ = false;
  std::atomic<bool> continue_{true};

  enum class ChatFormat { GEMMA, GEMMA4, CHATML, LLAMA3 };
  ChatFormat chat_format_ = ChatFormat::CHATML;

  std::set<llama_token> stopping_token_ids_;
  std::vector<std::string> stopping_token_strings_;

  std::chrono::steady_clock::time_point _lastResponseStart;

  void DetectStoppingTokens();
  void DetectChatFormat();
  bool isRepetitive(const std::string &text, size_t minPatternLength = 5);
  bool isCompleteSentence(const std::string &text);

private:
  std::vector<llama_token> Tokenize(const std::string& text, bool add_bos, bool special) const;
  bool Feed(std::vector<llama_token> tokens);
  bool EnsureRoom(int n_tokens);
  bool BeginConversation();
  bool PrepareTurn(int n_tokens);
  void CloseTurn();
  std::string UserPrefix() const;
  std::string UserSuffix() const;
  std::string RunTurn(WhillatsSetResponseCallback& callback, int max_tokens, int min_gen);
};

class LlamaDeviceBase {
public:
  LlamaDeviceBase(const char* model_path,
                  const char* mmproj_path,
                  WhillatsSetResponseCallback callback);
  virtual ~LlamaDeviceBase();

  // start() after stop() reuses the loaded weights, so barge-in and
  // back-to-back calls do not pay the model load again.
  bool start();
  void stop();
  bool isRunning() const { return _running; }
  void setThreadCount(int n);
  void setCallback(WhillatsSetResponseCallback cb);
  void resetConversation();

  const std::string& modelPath() const { return _model_path; }
  const std::string& mmprojPath() const { return _mmproj_path; }

  // lang: STT language code of the utterance (optional), named in the reply reminder.
  void askLlama(const char* prompt, const char* lang = nullptr);
  void receiveVideoFrame(const YUVData& yuv);

  bool hasMultimodalSupport() const { return _hasMultimodalModel; }

private:
  std::string _model_path;
  std::string _mmproj_path;
  WhillatsSetResponseCallback _responseCallback;
  std::mutex _callbackMutex;
  std::unique_ptr<LlamaSimpleChat> _chat;
  int _threadCount = 0;

  std::atomic<bool> _running{false};
  bool _hasMultimodalModel = false;
  int _imageRetentionMs = 5000;

  std::thread _processingThread;

  struct Request {
    std::string prompt;
    std::shared_ptr<YUVData> yuv; // null = text only
    std::string lang;
  };
  std::deque<Request> _requestQueue;
  std::mutex _queueMutex;
  std::condition_variable _queueCondition;

  std::shared_ptr<YUVData> _lastFrame;
  std::mutex _frameMutex;
  std::chrono::steady_clock::time_point _lastFrameTime;

  std::atomic<bool> _destructing_{false};

  bool RunProcessingThread();
};

#endif // LLAMA_DEVICE_BASE_H
