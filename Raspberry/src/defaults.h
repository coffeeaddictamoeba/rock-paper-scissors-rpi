
#include <cstddef>

#define MODEL_DEFAULT "rock_paper_scissors_model_f32.tflite"
#define PREDICTION_FILE_DEFAULT "test.png"
#define RECORDING_DATA_DEFAULT "rock_paper_scissors_data_recorded.csv"

inline constexpr int IMG_WIDTH = 128;
inline constexpr int IMG_HEIGHT = 96;
inline constexpr int IMG_CHANNELS = 3;
inline constexpr int NUM_CLASSES = 3;

inline constexpr std::size_t IMG_INPUT_SIZE =
    static_cast<std::size_t>(IMG_WIDTH) *
    static_cast<std::size_t>(IMG_HEIGHT) *
    static_cast<std::size_t>(IMG_CHANNELS);