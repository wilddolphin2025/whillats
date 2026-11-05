/*
 *  (c) 2025, wilddolphin2022
 *  For WebRTCsays.ai project
 *  https://github.com/wilddolphin2022
 */

#include "talking_face.h"
#include <cmath>
#include <algorithm>
#include <cstring>

#ifndef STB_IMAGE_IMPLEMENTATION_PROVIDED
#define STB_IMAGE_IMPLEMENTATION
#endif
#include "stb_image.h"

TalkingFace::TalkingFace() = default;

TalkingFace::~TalkingFace() = default;

bool TalkingFace::loadImage(const char* imagePath) {
    FILE* f = fopen(imagePath, "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (len <= 0) {
        fclose(f);
        return false;
    }
    std::vector<uint8_t> buf(static_cast<size_t>(len));
    const size_t nread = fread(buf.data(), 1, buf.size(), f);
    fclose(f);
    if (nread != buf.size()) return false;

    return loadImageFromMemory(buf.data(), (int)buf.size(), 0, 0);
}

bool TalkingFace::loadImageFromMemory(const uint8_t* data,
                                      int size, int dummy1, int dummy2) {
    int w, h, ch;
    uint8_t* pixels = stbi_load_from_memory(data, size, &w, &h, &ch, 3);
    if (!pixels) return false;

    img_w_ = w;
    img_h_ = h;
    base_rgb_.resize(w * h * 3);
    std::memcpy(base_rgb_.data(), pixels, w * h * 3);
    stbi_image_free(pixels);

    detectMouthRegion();
    return true;
}

void TalkingFace::detectMouthRegion() {
    // The robot logo has a small rectangular mouth in the lower-center area.
    // Scan the center column of the lower half to find the navy-colored mouth
    // element by looking for a dark region below the visor.
    int cx = img_w_ / 2;
    int search_top = img_h_ * 55 / 100;  // start below visor
    int search_bot = img_h_ * 75 / 100;

    auto isDark = [&](int x, int y) -> bool {
        int idx = (y * img_w_ + x) * 3;
        int r = base_rgb_[idx], g = base_rgb_[idx+1], b = base_rgb_[idx+2];
        return (r + g + b) < 200;
    };

    // Find the mouth rectangle: scan downward for dark region
    int mouth_top = -1, mouth_bot = -1;
    for (int y = search_top; y < search_bot; y++) {
        if (isDark(cx, y)) {
            if (mouth_top < 0) mouth_top = y;
            mouth_bot = y;
        } else if (mouth_top > 0) {
            break;
        }
    }

    if (mouth_top < 0) {
        // Fallback: place mouth at 62% height, 8% width
        mouth_cx_ = img_w_ / 2;
        mouth_cy_ = img_h_ * 62 / 100;
        mouth_w_  = img_w_ * 8 / 100;
        mouth_max_h_ = img_h_ * 8 / 100;
        return;
    }

    // Scan horizontally to find mouth width
    int mouth_mid_y = (mouth_top + mouth_bot) / 2;
    int left = cx, right = cx;
    while (left > 0 && isDark(left - 1, mouth_mid_y)) left--;
    while (right < img_w_ - 1 && isDark(right + 1, mouth_mid_y)) right++;

    mouth_cx_ = (left + right) / 2;
    mouth_cy_ = (mouth_top + mouth_bot) / 2;
    mouth_w_  = right - left;
    mouth_max_h_ = img_h_ * 8 / 100;

    // Sample background color just above the mouth
    int bg_y = std::max(0, mouth_top - 5);
    int bg_idx = (bg_y * img_w_ + cx) * 3;
    bg_r_ = base_rgb_[bg_idx];
    bg_g_ = base_rgb_[bg_idx + 1];
    bg_b_ = base_rgb_[bg_idx + 2];
}

void TalkingFace::feedAudio(const int16_t* samples, size_t count) {
    if (!samples || count == 0) {
        smoothed_energy_ = 0.0f;
        mouth_openness_.store(0.0f);
        return;
    }

    double sum_sq = 0.0;
    for (size_t i = 0; i < count; i++) {
        double s = samples[i] / 32768.0;
        sum_sq += s * s;
    }
    float rms = static_cast<float>(std::sqrt(sum_sq / count));

    // Smoothed envelope with fast attack, slow release
    float target = std::min(1.0f, rms * 25.0f);
    if (target > smoothed_energy_) {
        smoothed_energy_ = smoothed_energy_ * 0.2f + target * 0.8f;
    } else {
        smoothed_energy_ = smoothed_energy_ * 0.7f + target * 0.3f;
    }

    mouth_openness_.store(smoothed_energy_);
}

void TalkingFace::drawFilledRect(std::vector<uint8_t>& rgb, int imgW, int imgH,
                                 int x, int y, int w, int h,
                                 uint8_t r, uint8_t g, uint8_t b) {
    int x0 = std::max(0, x);
    int y0 = std::max(0, y);
    int x1 = std::min(imgW, x + w);
    int y1 = std::min(imgH, y + h);
    for (int py = y0; py < y1; py++) {
        for (int px = x0; px < x1; px++) {
            int idx = (py * imgW + px) * 3;
            rgb[idx]     = r;
            rgb[idx + 1] = g;
            rgb[idx + 2] = b;
        }
    }
}

void TalkingFace::drawRoundedRect(std::vector<uint8_t>& rgb, int imgW, int imgH,
                                  int cx, int cy, int w, int h, int radius,
                                  uint8_t r, uint8_t g, uint8_t b) {
    int x0 = cx - w / 2;
    int y0 = cy - h / 2;
    int x1 = x0 + w;
    int y1 = y0 + h;
    int rad = std::min(radius, std::min(w / 2, h / 2));

    for (int py = std::max(0, y0); py < std::min(imgH, y1); py++) {
        for (int px = std::max(0, x0); px < std::min(imgW, x1); px++) {
            // Check if this pixel is inside the rounded corners
            bool inside = true;
            int dx = 0, dy = 0;
            if (px < x0 + rad && py < y0 + rad) {
                dx = px - (x0 + rad); dy = py - (y0 + rad);
            } else if (px >= x1 - rad && py < y0 + rad) {
                dx = px - (x1 - rad - 1); dy = py - (y0 + rad);
            } else if (px < x0 + rad && py >= y1 - rad) {
                dx = px - (x0 + rad); dy = py - (y1 - rad - 1);
            } else if (px >= x1 - rad && py >= y1 - rad) {
                dx = px - (x1 - rad - 1); dy = py - (y1 - rad - 1);
            }
            if (dx != 0 || dy != 0) {
                inside = (dx * dx + dy * dy) <= (rad * rad);
            }
            if (inside) {
                int idx = (py * imgW + px) * 3;
                rgb[idx]     = r;
                rgb[idx + 1] = g;
                rgb[idx + 2] = b;
            }
        }
    }
}

void TalkingFace::renderMouth(std::vector<uint8_t>& rgb, int w, int h,
                              float openness) {
    // Erase the mouth area with the background color
    int erase_h = mouth_max_h_ + 4;
    int erase_w = mouth_w_ + 8;
    drawFilledRect(rgb, w, h,
                   mouth_cx_ - erase_w / 2, mouth_cy_ - erase_h / 2,
                   erase_w, erase_h,
                   bg_r_, bg_g_, bg_b_);

    // Draw the mouth shape based on openness
    int mw = mouth_w_;
    int mh;

    if (openness < 0.05f) {
        // Closed: thin line
        mh = std::max(2, mouth_max_h_ / 8);
        int radius = mh / 2;
        drawRoundedRect(rgb, w, h, mouth_cx_, mouth_cy_, mw, mh, radius,
                        kFaceR, kFaceG, kFaceB);
    } else if (openness < 0.3f) {
        // Slightly open: small rounded rect
        float t = openness / 0.3f;
        mh = static_cast<int>(mouth_max_h_ * (0.15f + t * 0.25f));
        int radius = mh / 3;
        drawRoundedRect(rgb, w, h, mouth_cx_, mouth_cy_, mw, mh, radius,
                        kFaceR, kFaceG, kFaceB);
    } else if (openness < 0.7f) {
        // Medium open: wider rounded rect
        float t = (openness - 0.3f) / 0.4f;
        mh = static_cast<int>(mouth_max_h_ * (0.4f + t * 0.35f));
        mw = static_cast<int>(mouth_w_ * (1.0f + t * 0.15f));
        int radius = mh / 3;
        drawRoundedRect(rgb, w, h, mouth_cx_, mouth_cy_, mw, mh, radius,
                        kFaceR, kFaceG, kFaceB);
    } else {
        // Wide open: large oval
        mh = static_cast<int>(mouth_max_h_ * (0.75f + (openness - 0.7f)));
        mw = static_cast<int>(mouth_w_ * 1.2f);
        int radius = std::min(mw, mh) / 2;
        drawRoundedRect(rgb, w, h, mouth_cx_, mouth_cy_, mw, mh, radius,
                        kFaceR, kFaceG, kFaceB);
    }
}

void TalkingFace::rgbToYuv420(const uint8_t* rgb, int w, int h, YUVData& yuv) {
    yuv.width   = w;
    yuv.height  = h;
    yuv.y_size  = w * h;
    yuv.uv_size = (w / 2) * (h / 2);
    yuv.y = std::make_unique<uint8_t[]>(yuv.y_size);
    yuv.u = std::make_unique<uint8_t[]>(yuv.uv_size);
    yuv.v = std::make_unique<uint8_t[]>(yuv.uv_size);

    for (int py = 0; py < h; py++) {
        for (int px = 0; px < w; px++) {
            int idx = (py * w + px) * 3;
            uint8_t r = rgb[idx], g = rgb[idx + 1], b = rgb[idx + 2];

            int Y = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
            yuv.y[py * w + px] = static_cast<uint8_t>(std::max(0, std::min(255, Y)));

            if ((py % 2 == 0) && (px % 2 == 0)) {
                int uv_idx = (py / 2) * (w / 2) + (px / 2);
                int U = ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
                int V = ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
                yuv.u[uv_idx] = static_cast<uint8_t>(std::max(0, std::min(255, U)));
                yuv.v[uv_idx] = static_cast<uint8_t>(std::max(0, std::min(255, V)));
            }
        }
    }
}

bool TalkingFace::renderFrame(YUVData& out) {
    std::lock_guard<std::mutex> lock(render_mutex_);
    if (base_rgb_.empty()) return false;

    // Work on a copy so the base image stays clean
    std::vector<uint8_t> frame = base_rgb_;
    float openness = mouth_openness_.load();

    renderMouth(frame, img_w_, img_h_, openness);

    // Scale to output size if different from source
    if (img_w_ == out_w_ && img_h_ == out_h_) {
        rgbToYuv420(frame.data(), img_w_, img_h_, out);
    } else {
        // Nearest-neighbor scale
        std::vector<uint8_t> scaled(out_w_ * out_h_ * 3);
        for (int y = 0; y < out_h_; y++) {
            int sy = y * img_h_ / out_h_;
            for (int x = 0; x < out_w_; x++) {
                int sx = x * img_w_ / out_w_;
                int src = (sy * img_w_ + sx) * 3;
                int dst = (y * out_w_ + x) * 3;
                scaled[dst]     = frame[src];
                scaled[dst + 1] = frame[src + 1];
                scaled[dst + 2] = frame[src + 2];
            }
        }
        rgbToYuv420(scaled.data(), out_w_, out_h_, out);
    }

    return true;
}
