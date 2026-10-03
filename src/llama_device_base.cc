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

#include <thread>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#include <mutex>
#include <queue>
#include <regex>
#include <set>
#include <cstdint>
#include <cstring>
#include <memory>
#include <cstdlib>
#include <cctype>

#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

#include "llama.h"
#include "mtmd.h"
#include "llama_device_base.h"
#include "whisper_helpers.h"
#include "mtmd-helper.h"

#ifdef GGML_USE_CUDA
#include <cuda_runtime.h>
#endif

// ---------------------------------------------------------------------------
// Strip markdown formatting so TTS speaks clean text
// e.g. "**bold**" → "bold", "*item*" → "item", "# Title" → "Title"
// ---------------------------------------------------------------------------
static std::string strip_markdown(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    size_t i = 0;
    bool line_start = true;
    while (i < s.size()) {
        // Skip heading markers at line start (# ## ###)
        if (line_start && s[i] == '#') {
            while (i < s.size() && s[i] == '#') ++i;
            while (i < s.size() && s[i] == ' ') ++i;
            continue;
        }
        char c = s[i];
        line_start = (c == '\n');
        // Bold/italic: ** or * or _ — skip the markers, keep the text
        if ((c == '*' || c == '_') ) {
            // peek ahead to find matching marker and strip both
            size_t j = i + 1;
            bool double_marker = (j < s.size() && s[j] == c);
            if (double_marker) j++;
            // find closing
            size_t close = s.find(double_marker ? std::string(2, c) : std::string(1, c), j);
            if (close != std::string::npos) {
                // copy inner text
                out += s.substr(j, close - j);
                i = close + (double_marker ? 2 : 1);
                continue;
            }
            // no closing marker found — just skip the marker char
            i++;
            continue;
        }
        // Inline code: `text` → text
        if (c == '`') {
            size_t close = s.find('`', i + 1);
            if (close != std::string::npos) {
                out += s.substr(i + 1, close - i - 1);
                i = close + 1;
                continue;
            }
            i++;
            continue;
        }
        // Block quote marker at line start
        if (c == '>' && (i == 0 || s[i-1] == '\n')) {
            i++;
            while (i < s.size() && s[i] == ' ') ++i;
            continue;
        }
        out += c;
        ++i;
    }
    return out;
}

// ---------------------------------------------------------------------------
// Clean model output: remove control tokens + markdown
// ---------------------------------------------------------------------------
static std::string clean_response(const std::string& response) {
    std::string cleaned = response;
    // Remove special tokens
    static const std::regex re_ctrl(
        "<\\|eot_id\\|>|<\\|im_end\\|>|<\\|endoftext\\|>|<end_of_turn>"
        "|<start_of_turn>[^<]*>?|<turn\\|>|<think>[\\s\\S]*?</think>"
        "|<\\|channel>[\\s\\S]*?<channel\\|>|<\\|?channel\\|?>");
    cleaned = std::regex_replace(cleaned, re_ctrl, "");
    // Strip markdown
    cleaned = strip_markdown(cleaned);
    // Trim trailing whitespace
    auto last = cleaned.find_last_not_of(" \n\r\t");
    if (last != std::string::npos)
        cleaned = cleaned.substr(0, last + 1);
    else
        cleaned.clear();
    return cleaned;
}


// ============================================================================
// LlamaSimpleChat
// ============================================================================

namespace {

const char* kInstruction =
    "You are a friendly voice assistant on a live phone or video call. "
    "Always reply in the language of the user's latest message, even if earlier turns used another language. "
    "Keep replies to one or two short spoken sentences. "
    "If an image is attached it is the live camera view; use it only when relevant. "
    "Never use markdown, lists, emoji, or special formatting.";

constexpr int kMaxReplyTokens = 256;
constexpr int kMaxImageReplyTokens = 160;
constexpr int kImageMaxTokens = 256;  // caps vision tokens per frame on the phone

constexpr int kImageMaxSide = 448;

// I420 -> packed RGB, nearest-neighbour downscaled so the long side is at most
// max_side (aspect ratio kept; the projector does its own final resize).
std::vector<uint8_t> YuvToRgb(const YUVData& yuv, int max_side, int& w, int& h) {
    if (!yuv.y || !yuv.u || !yuv.v || yuv.width <= 1 || yuv.height <= 1) return {};
    const float scale = std::min(1.0f, (float)max_side / std::max(yuv.width, yuv.height));
    w = std::max(1, (int)(yuv.width * scale));
    h = std::max(1, (int)(yuv.height * scale));
    const int cw = (yuv.width + 1) / 2;
    std::vector<uint8_t> rgb((size_t)w * h * 3);
    uint8_t* out = rgb.data();
    auto clamp = [](int v) { return (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v); };
    for (int y = 0; y < h; ++y) {
        const int sy = std::min(yuv.height - 1, (int)(y / scale));
        for (int x = 0; x < w; ++x) {
            const int sx = std::min(yuv.width - 1, (int)(x / scale));
            const int c = (yuv.y[(size_t)sy * yuv.width + sx] - 16) * 298 + 128;
            const size_t uv = (size_t)(sy / 2) * cw + sx / 2;
            const int d = yuv.u[uv] - 128, e = yuv.v[uv] - 128;
            *out++ = clamp((c + 409 * e) >> 8);
            *out++ = clamp((c - 100 * d - 208 * e) >> 8);
            *out++ = clamp((c + 516 * d) >> 8);
        }
    }
    return rgb;
}

bool IsMostlyNonLatin(const std::string& s) {
    int latin = 0, other = 0;
    for (size_t i = 0; i < s.size();) {
        const unsigned char c = s[i];
        if (c < 0x80) { latin += std::isalpha(c) ? 1 : 0; ++i; continue; }
        const int len = c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : 2;
        // U+0080..U+036F (2-byte lead < 0xCD) are Latin accents, not another script.
        if (c >= 0xCD) ++other; else ++latin;
        i += len;
    }
    return other > latin;
}

}  // namespace

LlamaSimpleChat::LlamaSimpleChat() = default;

LlamaSimpleChat::~LlamaSimpleChat() {
    if (smpl_) llama_sampler_free(smpl_);
    FreeContext();
    if (model_) llama_model_free(model_);
}

bool LlamaSimpleChat::SetModelPaths(const std::string &path, const std::string &mmproj_path) {
    model_path_ = path;
    mmproj_path_ = mmproj_path;
    return true;
}

bool LlamaSimpleChat::SetNGL(int layers) { ngl_ = layers; return true; }
bool LlamaSimpleChat::SetContextSize(int size) { n_ctx_ = size; return true; }
void LlamaSimpleChat::StopGeneration() { continue_ = false; }

void LlamaSimpleChat::SetThreadCount(int n) {
    n_threads_ = n;
    if (ctx_ && n > 0) llama_set_n_threads(ctx_, n, n);
}

bool LlamaSimpleChat::Initialize() {
    llama_backend_init();
    if (!LoadModel()) { LOG_E("Failed to load model."); return false; }
    if (!InitializeContext()) { LOG_E("Failed to initialize context."); return false; }
    if (smpl_) llama_sampler_free(smpl_);
    smpl_ = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (!smpl_) { LOG_E("Failed to initialize sampler."); return false; }
    llama_sampler_chain_add(smpl_, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(smpl_, llama_sampler_init_top_p(0.9f, 1));
    llama_sampler_chain_add(smpl_, llama_sampler_init_temp(0.7f));
    llama_sampler_chain_add(smpl_, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
    DetectStoppingTokens();
    DetectChatFormat();
    return BeginConversation();
}

bool LlamaSimpleChat::LoadModel() {
    if (model_path_.empty()) { LOG_E("Model path not set."); return false; }

    llama_model_params model_params = llama_model_default_params();
#ifdef __APPLE__
    // Unified memory: Metal maps the same mmap'd weights the CPU would use, so
    // full offload costs no extra RAM and is several times faster on A/M chips.
    const char *env_ngl = std::getenv("LLAMA_METAL_GPU_LAYERS");
    model_params.n_gpu_layers = (env_ngl && std::atoi(env_ngl) >= 0) ? std::atoi(env_ngl) : ngl_;
#elif defined(GGML_USE_CUDA)
    size_t free_mem, total_mem;
    cudaError_t cuda_err = cudaMemGetInfo(&free_mem, &total_mem);
    int max_layers = 0;
    if (cuda_err == cudaSuccess) {
        size_t available_mb = free_mem / (1024 * 1024);
        size_t total_mb = total_mem / (1024 * 1024);
        if (total_mb <= 4096) max_layers = 0;
        else if (available_mb >= 11000) max_layers = 30;
        else if (available_mb >= 9000)  max_layers = 20;
        else if (available_mb >= 7000)  max_layers = 5;
        else if (available_mb >= 5000)  max_layers = 2;
        else max_layers = 0;
    }
    model_params.n_gpu_layers = std::min(ngl_, max_layers);
    model_params.main_gpu = 0;
    model_params.split_mode = LLAMA_SPLIT_MODE_LAYER;
#else
    model_params.n_gpu_layers = 0;
#endif

    model_ = llama_model_load_from_file(model_path_.c_str(), model_params);
    if (!model_) { LOG_E("Unable to load model."); return false; }
    vocab_ = llama_model_get_vocab(model_);
    return true;
}

bool LlamaSimpleChat::InitializeContext() {
    if (ctx_) FreeContext();
    if (!model_ || !vocab_) { LOG_E("Model or vocab not loaded."); return false; }
    n_past_ = n_sys_ = 0;
    image_in_context_ = false;

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx   = std::min<int>(n_ctx_, llama_model_n_ctx_train(model_));
    ctx_params.n_batch = 512;
    ctx_params.n_ubatch = 512;
    ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
    ctx_params.no_perf = true;

#if defined(__APPLE__)
    int perf_cores = 0;
    size_t len = sizeof(perf_cores);
    if (sysctlbyname("hw.perflevel0.logicalcpu", &perf_cores, &len, NULL, 0) == 0 && perf_cores > 0)
        ctx_params.n_threads = perf_cores;
    else
        ctx_params.n_threads = std::max(1u, std::thread::hardware_concurrency());
#else
    ctx_params.n_threads = std::max(1u, std::thread::hardware_concurrency());
#endif
    if (n_threads_ > 0) ctx_params.n_threads = n_threads_;
    ctx_params.n_threads_batch = ctx_params.n_threads;

    ctx_ = llama_init_from_model(model_, ctx_params);
    if (!ctx_) { LOG_E("Failed to create llama_context."); return false; }

    if (!mmproj_path_.empty()) {
        mtmd_context_params mtmd_params = mtmd_context_params_default();
        mtmd_params.n_threads        = ctx_params.n_threads;
        mtmd_params.use_gpu          = true;
        mtmd_params.print_timings    = false;
        mtmd_params.flash_attn_type  = LLAMA_FLASH_ATTN_TYPE_AUTO;
        mtmd_params.image_max_tokens = kImageMaxTokens;
        ctx_mtmd_.reset(mtmd_init_from_file(mmproj_path_.c_str(), model_, mtmd_params));
        if (!ctx_mtmd_) LOG_E("Failed to load MMProj model; continuing text-only");
    }
    return true;
}

void LlamaSimpleChat::FreeContext() {
    ctx_mtmd_.reset();
    if (ctx_) { llama_free(ctx_); ctx_ = nullptr; }
}

bool LlamaSimpleChat::isRepetitive(const std::string &text, size_t minPatternLength) {
    if (text.length() < minPatternLength * 2) return false;
    for (size_t len = minPatternLength; len <= text.length() / 2; ++len) {
        if (text.compare(text.length() - 2 * len, len, text, text.length() - len, len) == 0) return true;
    }
    return false;
}

bool LlamaSimpleChat::isCompleteSentence(const std::string &text) {
    size_t end = text.find_last_not_of(" \n\r\t\"')");
    if (end == std::string::npos) return false;
    char last = text[end];
    return last == '.' || last == '!' || last == '?' || last == '\n';
}

// ---------------------------------------------------------------------------
// Prompt building. Markers are tokenized with parse_special=true; user text
// is tokenized with parse_special=false so it can never inject chat markers.
// ---------------------------------------------------------------------------

std::vector<llama_token> LlamaSimpleChat::Tokenize(const std::string& text, bool add_bos, bool special) const {
    int n = -llama_tokenize(vocab_, text.c_str(), (int)text.size(), nullptr, 0, add_bos, special);
    std::vector<llama_token> out(std::max(n, 0));
    if (n > 0 && llama_tokenize(vocab_, text.c_str(), (int)text.size(), out.data(), n, add_bos, special) < 0)
        out.clear();
    return out;
}

std::string LlamaSimpleChat::UserPrefix() const {
    switch (chat_format_) {
        case ChatFormat::GEMMA4: return "<|turn>user\n";
        case ChatFormat::GEMMA:  return "<start_of_turn>user\n";
        case ChatFormat::CHATML: return "<|im_start|>user\n";
        default:                 return "<|start_header_id|>user<|end_header_id|>\n\n";
    }
}

std::string LlamaSimpleChat::UserSuffix() const {
    switch (chat_format_) {
        case ChatFormat::GEMMA4: return "<turn|>\n<|turn>model\n";
        case ChatFormat::GEMMA:  return "<end_of_turn>\n<start_of_turn>model\n";
        case ChatFormat::CHATML: return "<|im_end|>\n<|im_start|>assistant\n";
        default:                 return "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
    }
}

void LlamaSimpleChat::CloseTurn() {
    const char* eot = chat_format_ == ChatFormat::GEMMA4 ? "<turn|>\n"
                    : chat_format_ == ChatFormat::GEMMA  ? "<end_of_turn>\n"
                    : chat_format_ == ChatFormat::CHATML ? "<|im_end|>\n" : "<|eot_id|>";
    Feed(Tokenize(eot, false, true));
}

bool LlamaSimpleChat::Feed(std::vector<llama_token> tokens) {
    if (tokens.empty()) return true;
    if (!EnsureRoom((int)tokens.size())) return false;
    const int n_batch = (int)llama_n_batch(ctx_);
    for (size_t i = 0; i < tokens.size(); i += n_batch) {
        int n = std::min<int>(n_batch, (int)(tokens.size() - i));
        if (llama_decode(ctx_, llama_batch_get_one(&tokens[i], n))) { LOG_E("llama_decode failed"); return false; }
        n_past_ += n;
    }
    return true;
}

// Makes room for n_tokens by dropping the oldest half of the conversation
// (system prompt kept). Image embeddings use non-linear positions (M-RoPE)
// and cannot be shifted, so those conversations are restarted instead.
bool LlamaSimpleChat::EnsureRoom(int n_tokens) {
    const int n_ctx = (int)llama_n_ctx(ctx_);
    if (n_past_ + n_tokens <= n_ctx) return true;
    llama_memory_t mem = llama_get_memory(ctx_);
    const int n_hist = n_past_ - n_sys_;
    if (image_in_context_ || !llama_memory_can_shift(mem) || n_hist <= 0 ||
        n_sys_ + n_hist / 2 + n_tokens > n_ctx) {
        return false;
    }
    const int n_discard = std::max(n_hist / 2, n_past_ + n_tokens - n_ctx);
    llama_memory_seq_rm(mem, 0, n_sys_, n_sys_ + n_discard);
    llama_memory_seq_add(mem, 0, n_sys_ + n_discard, n_past_, -n_discard);
    n_past_ -= n_discard;
    LOG_I("LlamaSimpleChat: context shift, dropped " << n_discard << " tokens");
    return true;
}

bool LlamaSimpleChat::BeginConversation() {
    llama_memory_clear(llama_get_memory(ctx_), true);
    n_past_ = n_sys_ = 0;
    image_in_context_ = false;
    llama_sampler_reset(smpl_);

    std::string sys;
    switch (chat_format_) {
        case ChatFormat::GEMMA4: sys = std::string("<|turn>system\n") + kInstruction + "<turn|>\n"; break;
        case ChatFormat::GEMMA:
            sys = std::string("<start_of_turn>user\n") + kInstruction +
                  "<end_of_turn>\n<start_of_turn>model\nUnderstood.<end_of_turn>\n";
            break;
        case ChatFormat::CHATML: sys = std::string("<|im_start|>system\n") + kInstruction + "<|im_end|>\n"; break;
        default: sys = std::string("<|start_header_id|>system<|end_header_id|>\n\n") + kInstruction + "<|eot_id|>"; break;
    }
    if (!Feed(Tokenize(sys, true, true))) return false;
    n_sys_ = n_past_;
    return true;
}

void LlamaSimpleChat::ResetConversation() {
    if (ctx_ && smpl_) BeginConversation();
}

bool LlamaSimpleChat::PrepareTurn(int n_tokens) {
    const int need = n_tokens + kMaxReplyTokens + 8;
    if (EnsureRoom(need)) return true;
    LOG_I("LlamaSimpleChat: context full, starting a new conversation");
    return BeginConversation() && n_past_ + need <= (int)llama_n_ctx(ctx_);
}

// Shared sampling loop. Streams each finished sentence to the callback so TTS
// starts speaking while the rest of the reply is still being generated.
std::string LlamaSimpleChat::RunTurn(WhillatsSetResponseCallback& callback, int max_tokens, int min_gen) {
    std::string response, phrase, recent;
    int generated = 0, repeats = 0;
    bool in_thought = false;
    char piece_buf[128];

    auto flush = [&](bool force) {
        if (phrase.empty() || (!force && !isCompleteSentence(phrase))) return;
        std::string cleaned = clean_response(phrase);
        if (!cleaned.empty()) {
            callback.OnResponseComplete(true, cleaned.c_str());
            LOG_I("LlamaSimpleChat: '" << cleaned << "' in "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::steady_clock::now() - _lastResponseStart).count() << "ms");
        }
        response += phrase;
        phrase.clear();
    };

    while (continue_ && generated < max_tokens) {
        llama_token tok = llama_sampler_sample(smpl_, ctx_, -1);
        const bool is_stop = llama_vocab_is_eog(vocab_, tok) ||
                             (generated >= min_gen && stopping_token_ids_.count(tok));
        if (is_stop) break;

        int n = llama_token_to_piece(vocab_, tok, piece_buf, sizeof(piece_buf), 0, true);
        if (n < 0) break;
        std::string piece(piece_buf, n);
        if (generated >= min_gen) {
            bool stop = false;
            for (const auto &ss : stopping_token_strings_)
                if (piece.find(ss) != std::string::npos) { stop = true; break; }
            if (stop) break;
        }

        if (!Feed({tok})) break;
        ++generated;

        // Reasoning ("thought" channel) is kept in the KV cache but never spoken.
        if (piece.find("<|channel>") != std::string::npos) { in_thought = true; phrase.clear(); continue; }
        if (piece.find("<channel|>") != std::string::npos) { in_thought = false; phrase.clear(); continue; }
        if (in_thought) continue;

        phrase += piece;
        recent += piece;
        if (recent.size() > 64) recent.erase(0, recent.size() - 64);
        if (isRepetitive(recent, 8)) { if (++repeats > 2) break; } else repeats = 0;
        flush(false);
    }
    if (!in_thought && continue_) flush(true);  // a barged-in turn drops its unfinished sentence
    CloseTurn();
    return clean_response(response);
}

std::string LlamaSimpleChat::generate(const std::string &prompt,
                                      WhillatsSetResponseCallback callback) {
    if (!ctx_ || !vocab_ || !smpl_) { LOG_E("generate: not initialized"); return ""; }
    continue_ = true;
    _lastResponseStart = std::chrono::steady_clock::now();

    std::vector<llama_token> toks = Tokenize(UserPrefix(), false, true);
    // Small models drift back to the language of earlier turns; a per-turn
    // reminder keeps non-Latin-script speakers (ru, uk, zh, ar, ...) answered in kind.
    std::vector<llama_token> body = Tokenize(
        IsMostlyNonLatin(prompt) ? prompt + "\n(Reply in the language of this message.)" : prompt, false, false);
    std::vector<llama_token> tail = Tokenize(UserSuffix(), false, true);
    toks.insert(toks.end(), body.begin(), body.end());
    toks.insert(toks.end(), tail.begin(), tail.end());

    if (!PrepareTurn((int)toks.size()) || !Feed(std::move(toks))) return "";
    return RunTurn(callback, kMaxReplyTokens, 0);
}

std::string LlamaSimpleChat::generateFromImage(YUVData* yuv, const std::string& prompt,
                                               WhillatsSetResponseCallback callback) {
    if (!ctx_mtmd_ || !yuv || !yuv->y || !yuv->u || !yuv->v) return generate(prompt, callback);
    if (!ctx_ || !vocab_ || !smpl_) { LOG_E("generateFromImage: not initialized"); return ""; }
    continue_ = true;
    _lastResponseStart = std::chrono::steady_clock::now();

    int w = 0, h = 0;
    std::vector<uint8_t> rgb = YuvToRgb(*yuv, kImageMaxSide, w, h);
    if (rgb.empty()) return generate(prompt, callback);
    mtmd::bitmap bitmap(w, h, rgb.data());
    if (!bitmap.ptr) return generate(prompt, callback);

    // User text cannot be parsed for specials here, so neutralise marker look-alikes.
    const std::string marker = mtmd_default_marker();
    std::string text = prompt;
    for (const std::string& bad : {std::string("<|"), std::string("|>"), std::string("<start_of_turn>"),
                                   std::string("<end_of_turn>"), marker}) {
        for (size_t pos; (pos = text.find(bad)) != std::string::npos;) text.erase(pos, bad.size());
    }
    const std::string full = UserPrefix() + marker + "\n" + text + UserSuffix();
    mtmd_input_text input_text = { full.c_str(), full.size(), false, true };
    const mtmd_bitmap* bitmaps[] = { bitmap.ptr.get() };
    mtmd::input_chunks chunks(mtmd_input_chunks_init());
    if (!chunks.ptr || mtmd_tokenize(ctx_mtmd_.get(), chunks.ptr.get(), &input_text, bitmaps, 1) != 0) {
        LOG_E("mtmd_tokenize failed");
        return generate(prompt, callback);
    }

    if (!PrepareTurn((int)mtmd_helper_get_n_pos(chunks.ptr.get()))) return "";
    llama_pos new_n_past = n_past_;
    if (mtmd_helper_eval_chunks(ctx_mtmd_.get(), ctx_, chunks.ptr.get(), n_past_, 0,
                                (int)llama_n_batch(ctx_), true, &new_n_past) != 0) {
        LOG_E("mtmd eval failed");
        BeginConversation();
        return "";
    }
    n_past_ = new_n_past;
    image_in_context_ = true;
    return RunTurn(callback, kMaxImageReplyTokens, 4);
}

void LlamaSimpleChat::DetectStoppingTokens() {
    if (!vocab_) return;
    stopping_token_ids_.clear();
    stopping_token_strings_.clear();

    static const std::vector<std::string> known = {
        "<|eot_id|>", "<|end_of_text|>", "<|end|>", "</s>",
        "<|im_end|>", "<|endoftext|>", "</think>",
        "<end_of_turn>", "<turn|>", "<|turn>"
    };

    int n_vocab = llama_vocab_n_tokens(vocab_);
    for (int id = 0; id < n_vocab; ++id) {
        char buf[128] = {};
        int len = llama_token_to_piece(vocab_, id, buf, sizeof(buf), 0, true);
        if (len < 0) continue;
        buf[std::min(len, (int)sizeof(buf)-1)] = '\0';
        std::string ts(buf);
        for (const auto &s : known) {
            if (ts.find(s) != std::string::npos) {
                stopping_token_ids_.insert(id);
                stopping_token_strings_.push_back(s);
                break;
            }
        }
    }
    llama_token eos = llama_vocab_eos(vocab_);
    if (eos != -1) stopping_token_ids_.insert(eos);
}

void LlamaSimpleChat::DetectChatFormat() {
    if (!vocab_) return;
    // Scan vocab directly — llama_tokenize is unreliable for special tokens in
    // quantized models (may return multiple pieces even for single-token entries).
    auto vocabContains = [&](const char* text) -> bool {
        int n_vocab = llama_vocab_n_tokens(vocab_);
        for (llama_token id = 0; id < n_vocab; ++id) {
            const char* s = llama_vocab_get_text(vocab_, id);
            if (s && strcmp(s, text) == 0) return true;
        }
        return false;
    };
    // Gemma-4 (unsloth) uses <|turn> / <turn|> markers
    if (vocabContains("<|turn>"))         { chat_format_ = ChatFormat::GEMMA4; LOG_I("Chat format: Gemma4"); return; }
    // Classic Gemma uses <start_of_turn>
    if (vocabContains("<start_of_turn>")) { chat_format_ = ChatFormat::GEMMA;  LOG_I("Chat format: Gemma"); return; }
    if (vocabContains("<|im_start|>"))    { chat_format_ = ChatFormat::CHATML; LOG_I("Chat format: ChatML"); return; }
    if (vocabContains("<|start_header_id|>")) { chat_format_ = ChatFormat::LLAMA3; LOG_I("Chat format: Llama3"); return; }
    chat_format_ = ChatFormat::CHATML;
    LOG_I("Chat format: ChatML (default)");
}

// ============================================================================
// LlamaDeviceBase
// ============================================================================

LlamaDeviceBase::LlamaDeviceBase(const char* model_path,
                                 const char* mmproj_path,
                                 WhillatsSetResponseCallback callback)
    : _model_path(model_path ? model_path : "")
    , _mmproj_path(mmproj_path ? mmproj_path : "")
    , _responseCallback(callback)
{
    LOG_I("LlamaDeviceBase: model=" << _model_path
          << " mmproj=" << (_mmproj_path.empty() ? "(none)" : _mmproj_path));
}

LlamaDeviceBase::~LlamaDeviceBase() {
    if (_destructing_.exchange(true)) return;
    stop();
}

void LlamaDeviceBase::setCallback(WhillatsSetResponseCallback cb) {
    std::lock_guard<std::mutex> lk(_callbackMutex);
    _responseCallback = cb;
}

void LlamaDeviceBase::setThreadCount(int n) {
    _threadCount = n;
    if (_chat && !_running) _chat->SetThreadCount(n);
}

void LlamaDeviceBase::resetConversation() {
    if (_chat) _chat->StopGeneration();
    std::lock_guard<std::mutex> lk(_queueMutex);
    _requestQueue.clear();
    _requestQueue.push_back({std::string(), nullptr});  // empty prompt = reset marker
    _queueCondition.notify_one();
}

bool LlamaDeviceBase::start() {
    if (_running) return true;
    if (_processingThread.joinable()) _processingThread.join();

    if (!_chat) {
        auto chat = std::make_unique<LlamaSimpleChat>();
        chat->SetModelPaths(_model_path, _mmproj_path);
        if (_threadCount > 0) chat->SetThreadCount(_threadCount);
        LOG_I("LlamaDeviceBase: loading model...");
        if (!chat->Initialize()) { LOG_E("LlamaDeviceBase: model init failed"); return false; }
        _chat = std::move(chat);
        _hasMultimodalModel = _chat->ctx_mtmd_ && mtmd_support_vision(_chat->ctx_mtmd_.get());
        LOG_I("LlamaDeviceBase: multimodal=" << (_hasMultimodalModel ? "yes" : "no"));
    } else if (_threadCount > 0) {
        _chat->SetThreadCount(_threadCount);
    }

    {
        std::lock_guard<std::mutex> lk(_queueMutex);
        _requestQueue.clear();
    }
    _running = true;
    _processingThread = std::thread([this]() { RunProcessingThread(); });
    LOG_I("LlamaDeviceBase: started");
    return true;
}

void LlamaDeviceBase::stop() {
    if (!_running.exchange(false)) return;
    if (_chat) _chat->StopGeneration();
    _queueCondition.notify_all();
    if (_processingThread.joinable() && std::this_thread::get_id() != _processingThread.get_id())
        _processingThread.join();
}

void LlamaDeviceBase::askLlama(const char* prompt) {
    if (!prompt || !*prompt || !_running) return;
    if (_chat) _chat->StopGeneration();  // barge-in: newest utterance wins

    Request req{prompt, nullptr};
    if (_hasMultimodalModel) {
        std::lock_guard<std::mutex> lk(_frameMutex);
        if (_lastFrame && std::chrono::steady_clock::now() - _lastFrameTime <=
                              std::chrono::milliseconds(_imageRetentionMs)) {
            req.yuv = std::move(_lastFrame);  // each frame is sent once; KV keeps it afterwards
        }
    }
    {
        std::lock_guard<std::mutex> lk(_queueMutex);
        // Drop stale prompts but keep pending conversation resets.
        _requestQueue.erase(std::remove_if(_requestQueue.begin(), _requestQueue.end(),
                                           [](const Request& r) { return !r.prompt.empty(); }),
                            _requestQueue.end());
        _requestQueue.push_back(std::move(req));
    }
    _queueCondition.notify_one();
}

void LlamaDeviceBase::receiveVideoFrame(const YUVData& yuv) {
    if (!_hasMultimodalModel) return;
    auto frame = std::make_shared<YUVData>();
    frame->width   = yuv.width;
    frame->height  = yuv.height;
    frame->y_size  = yuv.y_size;
    frame->uv_size = yuv.uv_size;
    frame->y = std::make_unique<uint8_t[]>(yuv.y_size);
    frame->u = std::make_unique<uint8_t[]>(yuv.uv_size);
    frame->v = std::make_unique<uint8_t[]>(yuv.uv_size);
    memcpy(frame->y.get(), yuv.y.get(), yuv.y_size);
    memcpy(frame->u.get(), yuv.u.get(), yuv.uv_size);
    memcpy(frame->v.get(), yuv.v.get(), yuv.uv_size);

    std::lock_guard<std::mutex> lk(_frameMutex);
    _lastFrame     = std::move(frame);
    _lastFrameTime = std::chrono::steady_clock::now();
}

bool LlamaDeviceBase::RunProcessingThread() {
    while (_running) {
        Request req;
        {
            std::unique_lock<std::mutex> lk(_queueMutex);
            _queueCondition.wait(lk, [this] { return !_requestQueue.empty() || !_running; });
            if (!_running) break;
            req = std::move(_requestQueue.front());
            _requestQueue.pop_front();
        }
        if (req.prompt.empty()) { _chat->ResetConversation(); continue; }

        WhillatsSetResponseCallback cb(nullptr, nullptr);
        {
            std::lock_guard<std::mutex> lk(_callbackMutex);
            cb = _responseCallback;
        }
        LOG_I("LlamaDeviceBase: prompt='" << req.prompt.substr(0, 80) << "'"
              << (req.yuv ? " [+image]" : ""));

        if (req.yuv) _chat->generateFromImage(req.yuv.get(), req.prompt, cb);
        else         _chat->generate(req.prompt, cb);
        cb.OnResponseComplete(false, nullptr);  // signal done
    }
    return true;
}
