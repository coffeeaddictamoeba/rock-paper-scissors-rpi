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
    ~TfliteImageClassifier();

    bool ok() const;
    const std::string& error_message() const;

    const ImageModelInputInfo& input_info() const;
    int class_count() const;

    ClassificationResult Predict(CameraPreprocessor& camera);

private:
    struct Impl;

    bool Load(const std::string& model_path);
    bool FillInputFromCamera(CameraPreprocessor& camera);
    bool Invoke();
    std::vector<float> ReadOutputScores() const;

private:
    std::unique_ptr<Impl> impl_;

    bool ok_ = false;
    std::string error_message_;

    ImageModelInputInfo input_info_;
    int class_count_ = 0;
};