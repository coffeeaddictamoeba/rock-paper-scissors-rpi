#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <opencv2/opencv.hpp>


#include "../include/camera.h"

void CameraPreprocessor::capture_to_float(float* dst, int input_width, int input_height) {
    capture_and_resize(input_width, input_height);
    for (int y = 0; y < input_height; ++y) {
        for (int x = 0; x < input_width; ++x) {
            cv::Vec3b pixel = resized_rgb_.at<cv::Vec3b>(y, x);
            int index = (y * input_width + x) * 3;
            dst[index + 0] = static_cast<float>(pixel[0]) / 255.0f; // R
            dst[index + 1] = static_cast<float>(pixel[1]) / 255.0f; // G
            dst[index + 2] = static_cast<float>(pixel[2]) / 255.0f; // B
        }
    }
}

void CameraPreprocessor::capture_to_uint8(uint8_t* dst, int input_width, int input_height) {
    capture_and_resize(input_width, input_height);
    const int byte_count = input_width * input_height * 3;

    if (resized_rgb_.isContinuous()) {
        std::memcpy(dst, resized_rgb_.data, byte_count);
        return;
    }

    for (int y = 0; y < input_height; ++y) {
        const uint8_t* row = resized_rgb_.ptr<uint8_t>(y);
        std::memcpy(
            dst + y * input_width * 3,
            row,
            input_width * 3
        );
    }
}

void CameraPreprocessor::capture_to_int8(int8_t* dst, int input_width, int input_height, float scale, int zero_point) {
    if (scale == 0.0f) {
        fprintf(stderr, "[ERROR] Quantization scale cannot be zero\n");
    }

    capture_and_resize(input_width, input_height);

    for (int y = 0; y < input_height; ++y) {
        for (int x = 0; x < input_width; ++x) {
            cv::Vec3b pixel = resized_rgb_.at<cv::Vec3b>(y, x);

            int index = (y * input_width + x) * 3;

            for (int c = 0; c < 3; ++c) {
                float real_value = static_cast<float>(pixel[c]) / 255.0f;
                int q = static_cast<int>(std::round(real_value / scale)) + zero_point;
                q = std::max(-128, std::min(127, q));
                dst[index + c] = static_cast<int8_t>(q);
            }
        }
    }
}

void CameraPreprocessor::capture_and_resize(int input_width, int input_height) {
    cap_.read(frame_bgr_);
    if (frame_bgr_.empty()) {
        fprintf(stderr, "[ERROR] Could not capture image\n");
    }
    
    cv::resize(
        frame_bgr_,
        resized_bgr_,
        cv::Size(input_width, input_height),
        0,
        0,
        cv::INTER_LINEAR
    );

    cv::cvtColor(
        resized_bgr_,
        resized_rgb_,
        cv::COLOR_BGR2RGB
    );
}