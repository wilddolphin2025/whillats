/*
 *  (c) 2025, wilddolphin2025 
 *  For WebRTCsays.ai project
 *  https://github.com/wilddolphin2025
 */

#include <fstream>
#include <vector>
#include <cstring>
#include <filesystem>
#include <cstdint>
#include <inttypes.h>
#include <memory>
#include <span>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <dlfcn.h>

#include "whillats.h"
#include "whillats_utils.h"
// whisper_helpers.h removed from thin library
#define LOG_I(x) do { fprintf(stderr, "[INFO] %s\n", #x); } while(0)
#define LOG_E(x) do { fprintf(stderr, "[ERROR] %s\n", #x); } while(0)
#define LOG_V(x) do {} while(0)
#define LOG_W(x) do { fprintf(stderr, "[WARN] %s\n", #x); } while(0)

// Compute perceptual hash for clip_image_u8
std::string compute_image_hash(const clip_image_u8& image) {
    if (image.width <= 0 || image.height <= 0 || !image.data) {
        LOG_E("Invalid or empty image provided for hashing");
        return "";
    }
    const int target_size = 32;
    std::vector<uint8_t> resized(target_size * target_size * 3);
    {
        float x_ratio = static_cast<float>(image.width) / target_size;
        float y_ratio = static_cast<float>(image.height) / target_size;
        for (int y = 0; y < target_size; ++y) {
            for (int x = 0; x < target_size; ++x) {
                int src_x = static_cast<int>(x * x_ratio);
                int src_y = static_cast<int>(y * y_ratio);
                src_x = src_x < image.width ? src_x : image.width - 1;
                src_y = src_y < image.height ? src_y : image.height - 1;
                size_t src_idx = (src_y * image.width + src_x) * 3;
                size_t dst_idx = (y * target_size + x) * 3;
                resized[dst_idx]     = image.data[src_idx];
                resized[dst_idx + 1] = image.data[src_idx + 1];
                resized[dst_idx + 2] = image.data[src_idx + 2];
            }
        }
    }
    std::vector<float> gray(target_size * target_size);
    for (int i = 0; i < target_size * target_size; ++i) {
        size_t rgb_idx = i * 3;
        gray[i] = 0.299f * resized[rgb_idx] + 0.587f * resized[rgb_idx + 1] + 0.114f * resized[rgb_idx + 2];
    }
    std::vector<float> dct_result(target_size * target_size);
    const float pi = 3.14159265358979323846f;
    for (int u = 0; u < target_size; ++u) {
        for (int v = 0; v < target_size; ++v) {
            float sum = 0.0f;
            for (int x = 0; x < target_size; ++x) {
                for (int y = 0; y < target_size; ++y) {
                    float cos_x = std::cos(pi * (2 * x + 1) * u / (2.0f * target_size));
                    float cos_y = std::cos(pi * (2 * y + 1) * v / (2.0f * target_size));
                    sum += gray[y * target_size + x] * cos_x * cos_y;
                }
            }
            float cu = (u == 0) ? 1.0f / std::sqrt(2.0f) : 1.0f;
            float cv = (v == 0) ? 1.0f / std::sqrt(2.0f) : 1.0f;
            dct_result[v * target_size + u] = sum * 2.0f * cu * cv / target_size;
        }
    }
    float coefficients[64];
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            coefficients[i * 8 + j] = dct_result[i * target_size + j];
        }
    }
    std::sort(coefficients, coefficients + 64);
    float median = coefficients[31];
    uint64_t hash = 0;
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            hash <<= 1;
            if (dct_result[i * target_size + j] > median) {
                hash |= 1;
            }
        }
    }
    char hash_buf[32];
    snprintf(hash_buf, sizeof(hash_buf), "%016" PRIx64, hash);
    return std::string(hash_buf);
}

clip_image_u8* yuv_to_clip(const YUVData& yuv) {
    if (!yuv.y || !yuv.u || !yuv.v) return nullptr;
    const int target_width = 224;
    const int target_height = 224;
    auto* clip = new clip_image_u8;
    clip->width = target_width;
    clip->height = target_height;
    size_t rgb_size = static_cast<size_t>(target_width) * target_height * 3;
    clip->data = (uint8_t*)malloc(rgb_size);
    if (!clip->data) { delete clip; return nullptr; }
    float x_ratio = static_cast<float>(yuv.width) / target_width;
    float y_ratio = static_cast<float>(yuv.height) / target_height;
    for (int y = 0; y < target_height; ++y) {
        for (int x = 0; x < target_width; ++x) {
            int src_x = static_cast<int>(x * x_ratio);
            int src_y = static_cast<int>(y * y_ratio);
            src_x = src_x < yuv.width ? src_x : yuv.width - 1;
            src_y = src_y < yuv.height ? src_y : yuv.height - 1;
            size_t rgb_idx = (y * target_width + x) * 3;
            size_t y_idx = src_y * yuv.width + src_x;
            size_t uv_idx = (src_y / 2) * (yuv.width / 2) + (src_x / 2);
            int Y = yuv.y[y_idx] - 16;
            int U = yuv.u[uv_idx] - 128;
            int V = yuv.v[uv_idx] - 128;
            int C = Y * 298 + 128;
            int D = U * 517;
            int E = V * 409;
            int R = (C + E) >> 8;
            int G = (C - ((U * 100 + V * 208) >> 8)) >> 8;
            int B = (C + D) >> 8;
            clip->data[rgb_idx]     = static_cast<uint8_t>(R < 0 ? 0 : R > 255 ? 255 : R);
            clip->data[rgb_idx + 1] = static_cast<uint8_t>(G < 0 ? 0 : G > 255 ? 255 : G);
            clip->data[rgb_idx + 2] = static_cast<uint8_t>(B < 0 ? 0 : B > 255 ? 255 : B);
        }
    }
    return clip;
}

void free_clip(clip_image_u8* clip) {
    if (clip) { free(clip->data); delete clip; }
}

bool load_yuv(YUVData& yuv, const char* filename, int width, int height) {
    FILE* file = fopen(filename, "rb");
    if (!file) return false;
    fseek(file, 0, SEEK_END);
    size_t file_size = ftell(file);
    fseek(file, 0, SEEK_SET);
    size_t expected_size = static_cast<size_t>(width * height) * 3 / 2;
    if (file_size < expected_size) { fclose(file); return false; }
    yuv.width = width;
    yuv.height = height;
    yuv.y_size = static_cast<size_t>(width) * height;
    yuv.uv_size = yuv.y_size / 4;
    yuv.y = std::make_unique<uint8_t[]>(yuv.y_size);
    yuv.u = std::make_unique<uint8_t[]>(yuv.uv_size);
    yuv.v = std::make_unique<uint8_t[]>(yuv.uv_size);
    if (fread(yuv.y.get(), 1, yuv.y_size, file) != yuv.y_size ||
        fread(yuv.u.get(), 1, yuv.uv_size, file) != yuv.uv_size ||
        fread(yuv.v.get(), 1, yuv.uv_size, file) != yuv.uv_size) {
        fclose(file);
        return false;
    }
    fclose(file);
    return true;
}

void free_yuv(YUVData& yuv) {
    yuv.y.reset();
    yuv.u.reset();
    yuv.v.reset();
}

std::vector<uint8_t> yuv_to_flat_array(const YUVData& yuv) {
    if (!yuv.y || !yuv.u || !yuv.v || yuv.width <= 0 || yuv.height <= 0) {
        LOG_E("Invalid YUVData: null pointers or invalid dimensions");
        return std::vector<uint8_t>();
    }
    size_t expected_y_size = static_cast<size_t>(yuv.width) * yuv.height;
    size_t expected_uv_size = expected_y_size / 4;
    if (yuv.y_size != expected_y_size || yuv.uv_size != expected_uv_size) {
        LOG_E("YUVData plane size mismatch: y_size=" << yuv.y_size << ", expected=" << expected_y_size
              << ", uv_size=" << yuv.uv_size << ", expected=" << expected_uv_size);
        return std::vector<uint8_t>();
    }
    size_t total_size = yuv.y_size + 2 * yuv.uv_size;
    std::vector<uint8_t> flat_array(total_size);
    std::memcpy(flat_array.data(), yuv.y.get(), yuv.y_size);
    std::memcpy(flat_array.data() + yuv.y_size, yuv.u.get(), yuv.uv_size);
    std::memcpy(flat_array.data() + yuv.y_size + yuv.uv_size, yuv.v.get(), yuv.uv_size);
    LOG_I("Converted YUVData to flat array: size=" << total_size << " bytes");
    return flat_array;
}

bool save_clip_as_bmp(const clip_image_u8& clip, const char* filename) {
    if (clip.width <= 0 || clip.height <= 0 || !clip.data) {
        LOG_E("Invalid or empty clip image for BMP saving");
        return false;
    }
    FILE* fp = fopen(filename, "wb");
    if (!fp) { LOG_E("Failed to open file for BMP writing"); return false; }
    int bytes_per_pixel = 3;
    int row_size = ((clip.width * bytes_per_pixel + 3) / 4) * 4;
    int pixel_data_size = row_size * clip.height;
    int file_size = 14 + 40 + pixel_data_size;
    uint8_t file_header[14] = { 'B','M',
        static_cast<uint8_t>(file_size), static_cast<uint8_t>(file_size >> 8),
        static_cast<uint8_t>(file_size >> 16), static_cast<uint8_t>(file_size >> 24),
        0,0,0,0, 54,0,0,0 };
    if (fwrite(file_header, 1, 14, fp) != 14) { fclose(fp); LOG_E("Failed to write BMP file header"); return false; }
    uint8_t info_header[40] = { 40,0,0,0,
        static_cast<uint8_t>(clip.width), static_cast<uint8_t>(clip.width >> 8), static_cast<uint8_t>(clip.width >> 16), static_cast<uint8_t>(clip.width >> 24),
        static_cast<uint8_t>(clip.height), static_cast<uint8_t>(clip.height >> 8), static_cast<uint8_t>(clip.height >> 16), static_cast<uint8_t>(clip.height >> 24),
        1,0, 24,0, 0,0,0,0,
        static_cast<uint8_t>(pixel_data_size), static_cast<uint8_t>(pixel_data_size >> 8), static_cast<uint8_t>(pixel_data_size >> 16), static_cast<uint8_t>(pixel_data_size >> 24),
        0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0 };
    if (fwrite(info_header, 1, 40, fp) != 40) { fclose(fp); LOG_E("Failed to write BMP info header"); return false; }
    std::vector<uint8_t> row_buffer(row_size, 0);
    for (int y = clip.height - 1; y >= 0; --y) {
        for (int x = 0; x < clip.width; ++x) {
            size_t rgb_idx = (y * clip.width + x) * 3;
            size_t bmp_idx = x * 3;
            row_buffer[bmp_idx]     = clip.data[rgb_idx + 2];
            row_buffer[bmp_idx + 1] = clip.data[rgb_idx + 1];
            row_buffer[bmp_idx + 2] = clip.data[rgb_idx];
        }
        if (fwrite(row_buffer.data(), 1, row_size, fp) != static_cast<size_t>(row_size)) { fclose(fp); LOG_E("Failed to write BMP pixel data"); return false; }
    }
    fclose(fp);
    return true;
}

std::vector<int16_t> resampleAudio(const int16_t* data, size_t count,
                                   int src_rate, int dst_rate) {
    if (src_rate == dst_rate || count == 0 || !data) {
        return std::vector<int16_t>(data, data + count);
    }

    auto gcd_fn = [](int a, int b) { while (b) { int t = b; b = a % b; a = t; } return a; };
    int g = gcd_fn(src_rate, dst_rate);
    int up   = dst_rate / g;   // upsample factor
    int down = src_rate / g;   // decimate factor

    // The polyphase FIR below requires NTAPS >= up to fire at least one tap per
    // output sample. For high up values (e.g. 22050->16000 gives up=320) the FIR
    // degrades to zero output on most samples, causing a comb-filter / metallic
    // artefact. Fall back to linear interpolation for those cases — correct and
    // artefact-free even if slightly less sharp in the stopband.
    if (up > 63) {
        double ratio = static_cast<double>(src_rate) / dst_rate;
        size_t out_len = static_cast<size_t>(count / ratio);
        std::vector<int16_t> out(out_len);
        for (size_t i = 0; i < out_len; ++i) {
            double pos  = i * ratio;
            size_t lo   = static_cast<size_t>(pos);
            size_t hi   = (lo + 1 < count) ? lo + 1 : lo;
            double frac = pos - lo;
            double v    = data[lo] + frac * (data[hi] - data[lo]);
            out[i] = static_cast<int16_t>(std::max(-32768.0, std::min(32767.0, v)));
        }
        return out;
    }

    // FIR low-pass: cutoff at min(1/up, 1/down) with Kaiser window (beta=5)
    constexpr int NTAPS = 63;
    constexpr int HALF  = NTAPS / 2;
    constexpr double BETA = 5.0;
    double fc = std::min(1.0 / up, 1.0 / down) * 0.90;

    static thread_local int    cached_up = 0, cached_down = 0;
    static thread_local double fir[NTAPS];

    if (cached_up != up || cached_down != down) {
        auto bessel_i0 = [](double x) -> double {
            double sum = 1.0, term = 1.0;
            for (int k = 1; k < 25; ++k) {
                term *= (x / (2.0 * k)) * (x / (2.0 * k));
                sum += term;
            }
            return sum;
        };
        double denom = bessel_i0(BETA);
        double fir_sum = 0.0;
        for (int n = 0; n < NTAPS; ++n) {
            int k = n - HALF;
            double sinc = (k == 0) ? 2.0 * fc
                                   : sin(2.0 * M_PI * fc * k) / (M_PI * k);
            double t = 2.0 * n / (NTAPS - 1) - 1.0;
            double win = bessel_i0(BETA * sqrt(1.0 - t * t)) / denom;
            fir[n] = sinc * win;
            fir_sum += fir[n];
        }
        for (int n = 0; n < NTAPS; ++n) fir[n] /= fir_sum;
        cached_up = up;
        cached_down = down;
    }

    // Pad to avoid startup/tail transients
    size_t pad = static_cast<size_t>(HALF);
    (void)(count + 2 * pad); // padded count for reference

    // Build zero-stuffed upsampled stream (only the non-zero entries matter)
    // For each output sample i, compute which upsampled index to read: i * down
    // Then convolve around that index in the (conceptual) upsampled+padded stream.
    size_t out_len = static_cast<size_t>(
        static_cast<double>(count) * up / down);
    std::vector<int16_t> out(out_len);

    for (size_t i = 0; i < out_len; ++i) {
        // Position in the upsampled stream (with padding offset)
        double pos = static_cast<double>(i) * down;
        // Shift by pad*up to account for front padding
        pos += pad * up;

        double acc = 0.0;
        for (int j = 0; j < NTAPS; ++j) {
            // Index in upsampled stream that this tap reads
            double ui = pos - HALF + j;
            // Only non-zero entries are at multiples of 'up'
            // Find nearest input sample: ui / up
            int src_idx = static_cast<int>(round(ui / up)) - static_cast<int>(pad);
            double remainder = ui - static_cast<double>(
                (src_idx + static_cast<int>(pad))) * up;
            // Only accumulate if this is a non-zero sample (remainder ~ 0)
            if (fabs(remainder) < 0.5 && src_idx >= 0 &&
                static_cast<size_t>(src_idx) < count) {
                acc += static_cast<double>(data[src_idx]) * fir[j];
            }
        }
        acc *= up;
        acc = std::max(-32768.0, std::min(32767.0, acc));
        out[i] = static_cast<int16_t>(acc);
    }

    return out;
}

std::string getDylibPath() {
    Dl_info info;
    if (dladdr((void *)getDylibPath, &info)) {
        std::string path = info.dli_fname;
        size_t lastSlash = path.find_last_of('/');
        if (lastSlash != std::string::npos) {
            return path.substr(0, lastSlash);
        }
        return path;
    }
    return "";
}
