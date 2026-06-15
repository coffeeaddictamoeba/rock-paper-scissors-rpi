#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/lite/c/common.h"

class CameraPreprocessor;

struct ImageModelInputInfo {
    TfLiteType type = kTfLiteNoType;

    int height = 0;
    int width = 0;
    int channels = 0;

    float scale = 1.0f;
    int zero_point = 0;
};

struct ClassificationResult {
    int class_index = -1;
    float score = 0.0f;

    // These may be probabilities, logits, or scores depending on your model.
    std::vector<float> scores;
};

class TfliteImageClassifier {
public:
    explicit TfliteImageClassifier(const std::string& model_path);
    explicit TfliteImageClassifier(const std::string& model_path, int height, int width) {
        input_info_.height = height;
        input_info_.width  = width;
    }

    ~TfliteImageClassifier();

    bool ok() const;
    const std::string& errmsg() const;
    const ImageModelInputInfo& input_info() const;
    int numclasses() const;

    ClassificationResult predict(CameraPreprocessor& camera);

private:
    struct Impl;

    bool load(const std::string& model_path);
    bool fill_input_from_camera(CameraPreprocessor& camera);
    bool invoke();
    std::vector<float> read_output_scores() const;

private:
    std::unique_ptr<Impl> impl_;

    bool ok_ = false;
    std::string error_message_;

    ImageModelInputInfo input_info_;
    int class_count_ = 0;
};