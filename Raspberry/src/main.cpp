#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "../include/defaults.h"
#include "../include/model.h"

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

int main(int argc, char** argv) {
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

    CameraPreprocessor camera(cf.pred_file);

    TfliteImageClassifier classifier(cf.model);
    if (!classifier.ok()) {
        fprintf(stderr, "[ERROR] %s\n", classifier.errmsg().c_str());
        return EXIT_FAILURE;
    }

    const ImageModelInputInfo& input = classifier.input_info();

    if (cf.verbose) {
        printf(
            "[INFO] Input: %dx%dx%d\n", 
            input.height,
            input.width,
            input.channels
        );
    }

    ClassificationResult result = classifier.predict(camera);

    printf("[INFO] Class: %d, score: %f\n", result.class_index, result.score);

    return EXIT_SUCCESS;
}