#include "model.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
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

// -----------------------------------------------------------------------------
// IMPORTANT:
//
// Replace these values with the real numbers from:
//
//   artifacts/gesture/normalization.json
//   artifacts/bracelet/normalization.json
//
// Example:
//   "mean": [123.4, 118.2, 104.7]
//   "std":  [52.1, 50.2, 49.3]
//
// Then write:
//   constexpr std::array<float, 3> GESTURE_INPUT_MEAN = {
//       123.4F, 118.2F, 104.7F
//   };
//
// Do NOT leave these as 0/1, or the model will still receive the wrong input.
// -----------------------------------------------------------------------------

constexpr std::array<float, 3> GESTURE_INPUT_MEAN = {
    96.131F,
    102.354F,
    101.926F
};

constexpr std::array<float, 3> GESTURE_INPUT_STD = {
    31.8431F,
    42.4998F,
    47.59517F
};

constexpr std::array<float, 3> BRACELET_INPUT_MEAN = {
    96.131F,
    102.354F,
    101.926F
};

constexpr std::array<float, 3> BRACELET_INPUT_STD = {
    31.843F,
    42.5F,
    47.595F
};

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

std::string TensorShapeString(const TfLiteTensor* tensor) {
  if (tensor == nullptr || tensor->dims == nullptr) {
    return "null";
  }

  std::string result = "[";

  for (int i = 0; i < tensor->dims->size; ++i) {
    if (i > 0) {
      result += ", ";
    }

    result += std::to_string(tensor->dims->data[i]);
  }

  result += "]";
  return result;
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

  int num_classes = 0;

  std::array<float, 3> input_mean = {0.0F, 0.0F, 0.0F};
  std::array<float, 3> input_std = {1.0F, 1.0F, 1.0F};
};

TfliteChoiceClassifier::TfliteChoiceClassifier(
    const std::string& model_path,
    int num_classes)
    : impl_(std::make_unique<Impl>()) {
  ok_ = Load(model_path, num_classes);
}

TfliteChoiceClassifier::~TfliteChoiceClassifier() = default;

bool TfliteChoiceClassifier::Load(
    const std::string& model_path,
    int num_classes) {
  impl_->model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  impl_->num_classes = num_classes;

  // Pick normalization based on model type.
  //
  // gesture model:
  //   output classes = 3
  //
  // bracelet model:
  //   output classes = 2
  if (num_classes == 3) {
    impl_->input_mean = GESTURE_INPUT_MEAN;
    impl_->input_std = GESTURE_INPUT_STD;
  } else if (num_classes == 2) {
    impl_->input_mean = BRACELET_INPUT_MEAN;
    impl_->input_std = BRACELET_INPUT_STD;
  } else {
    error_message_ =
        "Unsupported num_classes: " + std::to_string(num_classes) +
        ". Expected 3 for gesture or 2 for bracelet.";
    return false;
  }

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
        std::to_string(IMG_CHANNELS) + "], but model input shape is " +
        TensorShapeString(input) + ".";
    return false;
  }

  // Expected output shape: [1, num_classes]
  if (output->dims == nullptr ||
      output->dims->size != 2 ||
      output->dims->data[0] != 1 ||
      output->dims->data[1] != impl_->num_classes) {
    error_message_ =
        "Expected output shape [1, " +
        std::to_string(impl_->num_classes) +
        "], but model output shape is " +
        TensorShapeString(output) + ".";
    return false;
  }

  return true;
}

bool TfliteChoiceClassifier::CopyInput(const ImageMatrix& image_bmp) {
  TfLiteTensor* input_tensor = impl_->interpreter->input_tensor(0);

  if (input_tensor == nullptr) {
    error_message_ = "Input tensor is null.";
    return false;
  }

  auto normalize_channel = [&](uint8_t value, int channel) -> float {
    const std::size_t idx = static_cast<std::size_t>(channel);
    const float stddev = impl_->input_std[idx];

    if (stddev == 0.0F) {
      return 0.0F;
    }

    // This matches Python:
    //
    //   normalized = (pixel - mean[channel]) / std[channel]
    //
    // Pixel value is raw 0..255, NOT divided by 255.
    return (static_cast<float>(value) - impl_->input_mean[idx]) / stddev;
  };

  auto grayscale = [&](const Pixel& pixel) -> float {
    const float r = normalize_channel(pixel.r, 0);
    const float g = normalize_channel(pixel.g, 1);
    const float b = normalize_channel(pixel.b, 2);

    return 0.299F * r + 0.587F * g + 0.114F * b;
  };

  auto write_float_pixel = [&](float* input,
                               std::size_t& idx,
                               const Pixel& pixel) {
    if (IMG_CHANNELS == 1) {
      input[idx++] = grayscale(pixel);
    } else if (IMG_CHANNELS == 3) {
      input[idx++] = normalize_channel(pixel.r, 0);
      input[idx++] = normalize_channel(pixel.g, 1);
      input[idx++] = normalize_channel(pixel.b, 2);
    }
  };

  auto write_int8_pixel = [&](int8_t* input,
                              std::size_t& idx,
                              const Pixel& pixel) {
    if (IMG_CHANNELS == 1) {
      input[idx++] = QuantizeInt8(
          grayscale(pixel),
          input_tensor->params.scale,
          input_tensor->params.zero_point);
    } else if (IMG_CHANNELS == 3) {
      input[idx++] = QuantizeInt8(
          normalize_channel(pixel.r, 0),
          input_tensor->params.scale,
          input_tensor->params.zero_point);

      input[idx++] = QuantizeInt8(
          normalize_channel(pixel.g, 1),
          input_tensor->params.scale,
          input_tensor->params.zero_point);

      input[idx++] = QuantizeInt8(
          normalize_channel(pixel.b, 2),
          input_tensor->params.scale,
          input_tensor->params.zero_point);
    }
  };

  if (IMG_CHANNELS != 1 && IMG_CHANNELS != 3) {
    error_message_ =
        "Only 1-channel grayscale or 3-channel RGB input is supported.";
    return false;
  }

  if (input_tensor->type == kTfLiteFloat32) {
    float* input = impl_->interpreter->typed_input_tensor<float>(0);

    std::size_t idx = 0;

    for (int y = 0; y < IMG_HEIGHT; ++y) {
      for (int x = 0; x < IMG_WIDTH; ++x) {
        write_float_pixel(input, idx, image_bmp[y][x]);
      }
    }

    return true;
  }

  if (input_tensor->type == kTfLiteInt8) {
    int8_t* input = impl_->interpreter->typed_input_tensor<int8_t>(0);

    std::size_t idx = 0;

    for (int y = 0; y < IMG_HEIGHT; ++y) {
      for (int x = 0; x < IMG_WIDTH; ++x) {
        write_int8_pixel(input, idx, image_bmp[y][x]);
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
        output + impl_->num_classes);
  }

  std::vector<float> probabilities(
      static_cast<std::size_t>(impl_->num_classes),
      0.0F);

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

ChoicePrediction TfliteChoiceClassifier::Predict(
    const ImageMatrix& image_bmp) {
  ChoicePrediction prediction;
  prediction.choice = -1;
  prediction.confidence = 0.0F;

  if (!ok_) {
    return prediction;
  }

  if (!CopyInput(image_bmp)) {
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