#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <opencv2/opencv.hpp>

class CameraPreprocessor {
public:
    explicit CameraPreprocessor() {
        std::string pipeline =
            "libcamerasrc ! "
            "video/x-raw,width=640,height=480,framerate=15/1,format=NV12 ! "
            "videoconvert ! "
            "video/x-raw,format=BGR ! "
            "appsink drop=true max-buffers=1 sync=false";

        cap_.open(pipeline, cv::CAP_GSTREAMER);
        if (!cap_.isOpened()) { 
            fprintf(
                stderr, 
                "[ERROR] Could not open Raspberry Pi camera\n"
            );
        }
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
    cv::VideoCapture cap_;
    cv::Mat frame_bgr_;
    cv::Mat resized_bgr_;
    cv::Mat resized_rgb_;
};