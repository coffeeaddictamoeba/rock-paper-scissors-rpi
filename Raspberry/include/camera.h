#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <opencv2/opencv.hpp>

#include "defaults.h"

class CameraPreprocessor {
public:
    explicit CameraPreprocessor(const std::string& image_path) {
        image_path_ = std::move(image_path);
        height_ = IMG_HEIGHT;
        width_  = IMG_WIDTH;
    }

    explicit CameraPreprocessor(const std::string& image_path, int height, int width) {
        image_path_ = std::move(image_path);
        height_ = height;
        width_  = width;
    }

    void capture_to_float(
        float* dst,
        int input_width,
        int input_height
    );

    void capture_to_uint8(
        uint8_t* dst,
        int input_width,
        int input_height
    );

    void capture_to_int8(
        int8_t* dst, 
        int input_width, 
        int input_height, 
        float scale, 
        int zero_point
    );

private:
    void capture_and_resize(
        int input_width,
        int input_height
    );
    
private:
    cv::Mat frame_bgr_;
    cv::Mat resized_bgr_;
    cv::Mat resized_rgb_;
    std::string& image_path_;
    int height_;
    int width_;
};