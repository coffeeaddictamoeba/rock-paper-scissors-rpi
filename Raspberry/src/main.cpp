#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "../include/model.h"

constexpr int NO_CHOICE = -1;

struct config {
    std::string model     = MODEL_DEFAULT;
    std::string pred_file = PREDICTION_FILE_DEFAULT;
    std::string rec_file  = RECORDING_DATA_DEFAULT;
    bool verbose = false;
};

int parse_args(int argc, char** argv, config& cf) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--predict") == 0 && i+1 < argc) {
            cf.pred_file = argv[++i];
        } else if (strcmp(argv[i], "--model") == 0 && i+1 < argc) {
            cf.model = argv[++i];
        } else if (strcmp(argv[i], "--record-to") == 0 && i+1 < argc) {
            cf.rec_file = argv[++i];
        } else if (strcmp(argv[i], "--verbose") == 0) {
            cf.verbose = true;
        } else {
            fprintf(
                stderr, 
                "[ERROR] Unknown argument: %s\n", 
                argv[i]
            );
        }
    }
}

int main(int argc, char* argv[]) {
  config cf;
  parse_args(argc, argv, cf);

  if (cf.verbose) {
    printf(
      "[INFO] Running rock-paper-scissors with:\n\t"
      " Model:  %s\n\t "
      " Predict: %s\n\t "
      " Record: %s\n",
      cf.model.c_str(), 
      cf.pred_file.c_str(), 
      cf.rec_file.c_str()
    );
  }

  try {
    ImageMatrix image = readBMP(cf.pred_file);

    TfliteChoiceClassifier classifier(cf.model);

    if (!classifier.ok()) {
        fprintf(
            stderr, 
            "[ERROR] Model error: %s\n", 
            classifier.error_message().c_str()
        );
        return NO_CHOICE;
    }

    ChoicePrediction prediction = classifier.Predict(image);

    if (prediction.choice < 0) {
        fprintf(stderr, "[ERROR] Prediction failed: %s\n", classifier.error_message().c_str());
        return NO_CHOICE;
    }

    if (cf.verbose) {
        printf("[INFO] Pi's choice: %d\n", prediction.choice);
        printf("[INFO] Pi's confidence: %f\n", prediction.confidence);
        printf("[INFO] Probabilities: \n");
        for (std::size_t i = 0; i < prediction.probabilities.size(); ++i) {
            printf("\t%zu : %f\n", i, prediction.probabilities[i]);
        }
    }
  } catch (const std::exception& e) {
    fprintf(stderr, "[ERROR] Failed to get a prediction: %s\n", e.what());
    return NO_CHOICE;
  }

  return prediction.choice;
}