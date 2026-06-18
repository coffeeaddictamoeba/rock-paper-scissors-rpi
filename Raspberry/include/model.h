#include <memory>
#include <string>
#include <vector>

#include "bmp.h"

struct ChoicePrediction {
  int choice;
  float confidence;
  std::vector<float> probabilities;
};

class TfliteChoiceClassifier {
 public:
  explicit TfliteChoiceClassifier(const std::string& model_path);
  ~TfliteChoiceClassifier();

  bool ok() const { return ok_; }
  const std::string& error_message() const { return error_message_; }

  ChoicePrediction Predict(const ImageMatrix& normalized_image_bmp);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;

  bool ok_ = false;
  std::string error_message_;

  bool Load(const std::string& model_path);
  bool CopyInput(const ImageMatrix& normalized_image_bmp);
  std::vector<float> ReadOutput() const;
};