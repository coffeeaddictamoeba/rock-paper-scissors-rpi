#include "../include/model.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model_builder.h"

namespace {

bool IsSupportedTensorType(TfLiteType type) {
  return type == kTfLiteFloat32 || type == kTfLiteInt8;
}

const char* TensorTypeName(TfLiteType type) {
  switch (type) {
    case kTfLiteFloat32:
      return "float32";
    case kTfLiteInt8:
      return "int8";
    default:
      return "unsupported";
  }
}

int8_t QuantizeInt8(float value, float scale, int zero_point) {
  int quantized = static_cast<int>(
      std::lround(value / scale + static_cast<float>(zero_point)));

  const int min_value = static_cast<int>(std::numeric_limits<int8_t>::min());
  const int max_value = static_cast<int>(std::numeric_limits<int8_t>::max());

  quantized = std::max(min_value, std::min(max_value, quantized));

  return static_cast<int8_t>(quantized);
}

float DequantizeInt8(int8_t value, float scale, int zero_point) {
  return scale * static_cast<float>(static_cast<int>(value) - zero_point);
}

}  // namespace

struct TfliteChoiceClassifier::Impl {
  std::unique_ptr<tflite::FlatBufferModel> model;
  std::unique_ptr<tflite::Interpreter> interpreter;
};

TfliteChoiceClassifier::TfliteChoiceClassifier(
    const std::string& model_path)
    : impl_(std::make_unique<Impl>()) {
  ok_ = Load(model_path);
}

TfliteChoiceClassifier::~TfliteChoiceClassifier() = default;

bool TfliteChoiceClassifier::Load(const std::string& model_path) {
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
    error_message_ = "The model must have exactly one input tensor.";
    return false;
  }

  if (impl_->interpreter->outputs().size() != 1) {
    error_message_ = "The model must have exactly one output tensor.";
    return false;
  }

  const TfLiteTensor* input = impl_->interpreter->input_tensor(0);
  const TfLiteTensor* output = impl_->interpreter->output_tensor(0);

  if (input == nullptr || !IsSupportedTensorType(input->type)) {
    error_message_ = "The model input must be float32 or int8; got ";
    error_message_ += input == nullptr ? "null" : TensorTypeName(input->type);
    error_message_ += ".";
    return false;
  }

  if (output == nullptr || !IsSupportedTensorType(output->type)) {
    error_message_ = "The model output must be float32 or int8; got ";
    error_message_ += output == nullptr ? "null" : TensorTypeName(output->type);
    error_message_ += ".";
    return false;
  }

  // Expected input shape: [1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS]
  if (input->dims == nullptr ||
      input->dims->size != 4 ||
      input->dims->data[0] != 1 ||
      input->dims->data[1] != IMG_HEIGHT ||
      input->dims->data[2] != IMG_WIDTH ||
      input->dims->data[3] != IMG_CHANNELS) {
    error_message_ =
        "Expected input shape [1, " +
        std::to_string(IMG_HEIGHT) + ", " +
        std::to_string(IMG_WIDTH) + ", " +
        std::to_string(IMG_CHANNELS) + "].";
    return false;
  }

  // Expected output shape: [1, NUM_CLASSES]
  if (output->dims == nullptr ||
      output->dims->size != 2 ||
      output->dims->data[0] != 1 ||
      output->dims->data[1] != NUM_CLASSES) {
    error_message_ =
        "Expected output shape [1, " +
        std::to_string(NUM_CLASSES) + "].";
    return false;
  }

  return true;
}

bool TfliteChoiceClassifier::CopyInput(
    const ImageMatrix& normalized_image_bmp) {
  TfLiteTensor* input_tensor = impl_->interpreter->input_tensor(0);

  if (input_tensor == nullptr) {
    error_message_ = "Input tensor is null.";
    return false;
  }

  if (input_tensor->type == kTfLiteFloat32) {
    float* input = impl_->interpreter->typed_input_tensor<float>(0);

    std::size_t idx = 0;

    for (int y = 0; y < IMG_HEIGHT; ++y) {
      for (int x = 0; x < IMG_WIDTH; ++x) {
        input[idx++] = normalized_image_bmp[y][x];
      }
    }

    return true;
  }

  if (input_tensor->type == kTfLiteInt8) {
    int8_t* input = impl_->interpreter->typed_input_tensor<int8_t>(0);

    std::size_t idx = 0;

    for (int y = 0; y < IMG_HEIGHT; ++y) {
      for (int x = 0; x < IMG_WIDTH; ++x) {
        input[idx++] = QuantizeInt8(
            normalized_image_bmp[y][x],
            input_tensor->params.scale,
            input_tensor->params.zero_point);
      }
    }

    return true;
  }

  error_message_ = "Unsupported input tensor type.";
  return false;
}

std::vector<float> TfliteChoiceClassifier::ReadOutput() const {
  const TfLiteTensor* output_tensor = impl_->interpreter->output_tensor(0);

  if (output_tensor->type == kTfLiteFloat32) {
    const float* output = impl_->interpreter->typed_output_tensor<float>(0);

    return std::vector<float>(
        output,
        output + NUM_CLASSES);
  }

  std::vector<float> probabilities(NUM_CLASSES, 0.0F);

  if (output_tensor->type == kTfLiteInt8) {
    const int8_t* output =
        impl_->interpreter->typed_output_tensor<int8_t>(0);

    for (std::size_t idx = 0; idx < probabilities.size(); ++idx) {
      probabilities[idx] = DequantizeInt8(
          output[idx],
          output_tensor->params.scale,
          output_tensor->params.zero_point);
    }

    return probabilities;
  }

  return probabilities;
}

ChoicePrediction TfliteChoiceClassifier::Predict(const ImageMatrix& normalized_image_bmp) {
  ChoicePrediction prediction;
  prediction.choice = -1;
  prediction.confidence = 0.0F;

  if (!ok_) {
    return prediction;
  }

  if (!CopyInput(normalized_image_bmp)) {
    ok_ = false;
    return prediction;
  }

  if (impl_->interpreter->Invoke() != kTfLiteOk) {
    error_message_ = "TensorFlow Lite invocation failed.";
    ok_ = false;
    return prediction;
  }

  prediction.probabilities = ReadOutput();

  const auto best = std::max_element(
      prediction.probabilities.begin(),
      prediction.probabilities.end());

  prediction.choice = static_cast<int>(
      std::distance(prediction.probabilities.begin(), best));

  prediction.confidence = *best;

  return prediction;
}