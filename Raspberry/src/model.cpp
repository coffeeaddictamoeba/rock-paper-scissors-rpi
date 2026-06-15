#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "../include/camera.h"
#include "../include/model.h"

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model_builder.h"

bool is_tensor_type_supported(TfLiteType type) {
    return type == kTfLiteFloat32 || type == kTfLiteInt8;
}

const char* get_tensor_type_name(TfLiteType type) {
    switch (type) {
        case kTfLiteFloat32:
            return "float32";
        case kTfLiteInt8:
            return "int8";
        case kTfLiteUInt8:
            return "uint8";
        default:
            return "unsupported";
    }
}

bool has_valid_qint8(const TfLiteTensor* tensor) {
    return tensor != nullptr && tensor->type == kTfLiteInt8 && tensor->params.scale > 0.0f;
}

float deqint8(int8_t value, float scale, int zero_point) {
    return scale * static_cast<float>(static_cast<int>(value) - zero_point);
}

struct TfliteImageClassifier::Impl {
    std::unique_ptr<tflite::FlatBufferModel> model;
    std::unique_ptr<tflite::Interpreter> interpreter;
};

TfliteImageClassifier::TfliteImageClassifier(const std::string& model_path) : impl_(std::make_unique<Impl>()) {
    ok_ = load(model_path);
}

TfliteImageClassifier::~TfliteImageClassifier() = default;

bool TfliteImageClassifier::ok() const { return ok_; }

const std::string& TfliteImageClassifier::errmsg() const {
    return error_message_;
}

const ImageModelInputInfo& TfliteImageClassifier::input_info() const {
    return input_info_;
}

int TfliteImageClassifier::numclasses() const { return class_count_; }

bool TfliteImageClassifier::load(const std::string& model_path) {
    impl_->model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());

    if (!impl_->model) {
        error_message_ = "Failed to load model: " + model_path;
        return false;
    }

    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*impl_->model, resolver);

    if (builder(&impl_->interpreter) != kTfLiteOk || !impl_->interpreter) {
        error_message_ = "Failed to create TensorFlow Lite interpreter.";
        return false;
    }

    impl_->interpreter->SetNumThreads(1);

    if (impl_->interpreter->AllocateTensors() != kTfLiteOk) {
        error_message_ = "Failed to allocate tensors.";
        return false;
    }

    if (impl_->interpreter->inputs().size() != 1) {
        error_message_ = "Model must have exactly one input tensor.";
        return false;
    }

    if (impl_->interpreter->outputs().size() != 1) {
        error_message_ = "Model must have exactly one output tensor.";
        return false;
    }

    const TfLiteTensor* input = impl_->interpreter->input_tensor(0);
    const TfLiteTensor* output = impl_->interpreter->output_tensor(0);

    if (input == nullptr) {
        error_message_ = "Input tensor is null.";
        return false;
    }

    if (output == nullptr) {
        error_message_ = "Output tensor is null.";
        return false;
    }

    auto check_tensor_type = [&](const TfLiteTensor* tensor, const std::string& label) -> bool {
        if (!is_tensor_type_supported(tensor->type)) {
            error_message_ = label + " tensor must be float32 or int8; got ";
            error_message_ += get_tensor_type_name(input->type);
            error_message_ += ".";
            return false;
        }
        return true;
    }

    if (!check_tensor_type(input, "Input") && !check_tensor_type(output, "Output")) {
        return false;
    }

    if (input->type == kTfLiteInt8 && !has_valid_qint8(input)) {
        error_message_ = "Input int8 tensor has invalid quantization params.";
        return false;
    }

    if (output->type == kTfLiteInt8 && !has_valid_qint8(output)) {
        error_message_ = "Output int8 tensor has invalid quantization params.";
        return false;
    }

    // Expected image input layout:
    //
    //     [1, height, width, channels]
    //
    // Usually channels is 1 for grayscale or 3 for RGB.
    if (input->dims == nullptr ||
        input->dims->size != 4 ||
        input->dims->data[0] != 1) {
        error_message_ = "Expected input shape [1, height, width, channels].";
        return false;
    }

    const int input_height = input->dims->data[1];
    const int input_width  = input->dims->data[2];
    const int input_channels = input->dims->data[3];

    if (input_height <= 0 || input_width <= 0) {
        error_message_ = "Input width/height must be positive.";
        return false;
    }

    if (input_channels != 1 && input_channels != 3) {
        error_message_ = "Input channels must be 1 grayscale or 3 RGB.";
        return false;
    }

    // Expected classifier output layout:
    //
    //     [1, class_count]
    if (output->dims == nullptr ||
        output->dims->size != 2 ||
        output->dims->data[0] != 1) {
        error_message_ = "Expected output shape [1, class_count].";
        return false;
    }

    const int output_classes = output->dims->data[1];

    if (output_classes <= 0) {
        error_message_ = "Output class count must be positive.";
        return false;
    }

    input_info_.type   = input->type;
    input_info_.height = input_height;
    input_info_.width  = input_width;
    input_info_.channels = input_channels;
    input_info_.scale = input->params.scale;
    input_info_.zero_point = input->params.zero_point;

    class_count_ = output_classes;

    return true;
}

bool TfliteImageClassifier::fill_input_from_camera(CameraPreprocessor& camera) {
    TfLiteTensor* input_tensor = impl_->interpreter->input_tensor(0);

    if (input_tensor == nullptr) {
        error_message_ = "Input tensor is null.";
        return false;
    }

    if (input_tensor->type == kTfLiteFloat32) {
        float* input =
            impl_->interpreter->typed_input_tensor<float>(0);

        if (input == nullptr) {
            error_message_ = "Could not get float32 input tensor buffer.";
            return false;
        }

        camera.capture_to_float(
            input,
            input_info_.width,
            input_info_.height,
            input_info_.channels
        );

        return true;
    }

    if (input_tensor->type == kTfLiteInt8) {
        int8_t* input = impl_->interpreter->typed_input_tensor<int8_t>(0);

        if (input == nullptr) {
            error_message_ = "Could not get int8 input tensor buffer.";
            return false;
        }

        camera.capture_to_int8(
            input,
            input_info_.width,
            input_info_.height,
            input_info_.channels,
            input_tensor->params.scale,
            input_tensor->params.zero_point
        );

        return true;
    }

    error_message_ = "Unsupported input tensor type.";
    return false;
}

bool TfliteImageClassifier::invoke() {
    if (impl_->interpreter->Invoke() != kTfLiteOk) {
        error_message_ = "TensorFlow Lite invocation failed.";
        return false;
    }

    return true;
}

std::vector<float> TfliteImageClassifier::read_output_scores() const {
    const TfLiteTensor* output_tensor = impl_->interpreter->output_tensor(0);

    std::vector<float> scores(class_count_, 0.0f);

    if (output_tensor == nullptr) return scores;

    if (output_tensor->type == kTfLiteFloat32) {
        const float* output = impl_->interpreter->typed_output_tensor<float>(0);

        if (output == nullptr) return scores;

        std::copy(
            output,
            output + class_count_,
            scores.begin()
        );

        return scores;
    }

    if (output_tensor->type == kTfLiteInt8) {
        const int8_t* output = impl_->interpreter->typed_output_tensor<int8_t>(0);

        if (output == nullptr) return scores;

        for (int idx = 0; idx < class_count_; ++idx) {
            scores[idx] = deqint8(
                output[idx],
                output_tensor->params.scale,
                output_tensor->params.zero_point
            );
        }

        return scores;
    }

    return scores;
}

ClassificationResult TfliteImageClassifier::predict(CameraPreprocessor& camera) {
    ClassificationResult result;

    if (!ok_) return result;

    try {
        if (!fill_input_from_camera(camera)) {
            return result;
        }
    } catch (const std::exception& e) {
        error_message_ = e.what();
        return result;
    }

    if (!invoke()) return result;

    result.scores = read_output_scores();

    if (result.scores.empty()) {
        return result;
    }

    auto best = std::max_element(
        result.scores.begin(),
        result.scores.end()
    );

    result.class_index = static_cast<int>(
        std::distance(result.scores.begin(), best)
    );

    result.score = *best;

    return result;
}