import subprocess
import time
from pathlib import Path
from datetime import datetime

from sense_hat import SenseHat


sense = SenseHat()
sense.clear()

IMAGE_DIR = Path.home() / "Documents" / "rps_runtime"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

EXECUTABLE = str(Path.home() / "Documents" / "Project" / "." / "rockpaperscissors")

BRACELET_MODEL_PATH = str(Path.home() / "Documents" / "Project" / "bracelet" / "model_i8.tflite")
GESTURE_MODEL_PATH  = str(Path.home() / "Documents" / "Project" / "gesture"  / "model_i8.tflite")

GESTURE_NUM_CLASSES = 3
BRACELET_NUM_CLASSES = 2

# Expected class mapping:
# Gesture model:
#   0 = ROCK
#   1 = PAPER
#   2 = SCISSORS
#
# Bracelet model:
#   0 = no bracelet
#   1 = bracelet detected
BRACELET_DETECTED_CLASS = 1

DELAY_AFTER_COUNTDOWN = 0.05

ROCK_CODE = 0
PAPER_CODE = 1
SCISSORS_CODE = 2

MOVE_NAMES = {
    ROCK_CODE: "ROCK",
    PAPER_CODE: "PAPER",
    SCISSORS_CODE: "SCISSORS",
}

# Pi move that beats the player's move.
PI_WIN_MOVE = {
    ROCK_CODE: PAPER_CODE,       # paper beats rock
    PAPER_CODE: SCISSORS_CODE,   # scissors beats paper
    SCISSORS_CODE: ROCK_CODE,    # rock beats scissors
}

# Pi move that loses to the player's move.
PI_LOSE_MOVE = {
    ROCK_CODE: SCISSORS_CODE,    # scissors loses to rock
    PAPER_CODE: ROCK_CODE,       # rock loses to paper
    SCISSORS_CODE: PAPER_CODE,   # paper loses to scissors
}


W = (255, 255, 255)
B = (0, 0, 0)
G = (0, 255, 0)
R = (255, 0, 0)


ROCK = [
    B,B,W,W,W,W,B,B,
    B,W,W,W,W,W,W,B,
    W,W,W,W,W,W,W,W,
    W,W,W,W,W,W,W,W,
    W,W,W,W,W,W,W,W,
    B,W,W,W,W,W,W,B,
    B,B,W,W,W,W,B,B,
    B,B,B,B,B,B,B,B
]

PAPER = [
    B,W,B,W,B,W,B,B,
    B,W,B,W,B,W,B,B,
    B,W,B,W,B,W,B,B,
    B,W,W,W,W,W,W,B,
    B,W,W,W,W,W,W,B,
    B,W,W,W,W,W,W,B,
    B,W,W,W,W,W,W,B,
    B,B,W,W,W,W,B,B
]

SCISSORS = [
    W,B,B,B,B,B,B,W,
    B,W,B,B,B,B,W,B,
    B,B,W,B,B,W,B,B,
    B,B,B,W,W,B,B,B,
    B,B,B,W,W,B,B,B,
    B,B,W,B,B,W,B,B,
    W,B,B,W,W,B,B,W,
    B,W,W,B,B,W,W,B
]

SYMBOLS = {
    ROCK_CODE: ROCK,
    PAPER_CODE: PAPER,
    SCISSORS_CODE: SCISSORS,
}


def show_start():
    sense.clear(G)
    time.sleep(0.5)
    sense.clear()


def show_end():
    sense.clear(R)
    time.sleep(0.5)
    sense.clear()


def show_countdown():
    for number in ["1", "2", "3"]:
        sense.show_letter(number, text_colour=W)
        print(f"[COUNTDOWN] {number}")
        time.sleep(0.9)
        sense.clear()


def show_symbol(code):
    name = MOVE_NAMES.get(code, "UNKNOWN")
    symbol = SYMBOLS.get(code)

    print(f"[PI] Pi chose: {name}")

    if symbol is None:
        sense.show_letter("?", text_colour=R)
        print(f"[ERROR] Unknown symbol code: {code}")
    else:
        sense.set_pixels(symbol)

    time.sleep(2)
    sense.clear()


def show_win():
    print("[GAME] Player WON")
    sense.clear(G)
    time.sleep(0.2)
    sense.clear()


def show_loss():
    print("[GAME] Player LOST")
    sense.clear(R)
    time.sleep(0.2)
    sense.clear()


def capture_image():
    image_path = IMAGE_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.bmp"

    cmd = [
        "rpicam-still",
        "-n",
        "--immediate",
        "--shutter", "5000",
        "--width", "128",
        "--height", "96",
        "--encoding", "bmp",
        "-o", str(image_path),
    ]

    print("[INFO] Capturing image with:")
    print("[INFO]", " ".join(cmd))

    subprocess.run(cmd, check=True)

    print(f"[INFO] Captured image: {image_path}")
    return image_path


def run_prediction(image_path, model_path, num_classes, model_name):
    cmd = [
        EXECUTABLE,
        "--predict", str(image_path),
        "--model", model_path,
        "--num-classes", str(num_classes),
    ]

    print(f"[INFO] Running {model_name} model")

    # This assumes your C++ executable returns the predicted class as the exit code,
    # like your original wrapper did.
    result = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    prediction_code = result.returncode

    if prediction_code < 0 or prediction_code >= num_classes:
        raise RuntimeError(
            f"{model_name} model returned invalid class code: {prediction_code}"
        )

    print(f"[INFO] {model_name} prediction: {prediction_code}")
    return prediction_code


def get_player_gesture(image_path):
    return run_prediction(
        image_path=image_path,
        model_path=GESTURE_MODEL_PATH,
        num_classes=GESTURE_NUM_CLASSES,
        model_name="gesture",
    )


def detect_bracelet(image_path):
    prediction = run_prediction(
        image_path=image_path,
        model_path=BRACELET_MODEL_PATH,
        num_classes=BRACELET_NUM_CLASSES,
        model_name="bracelet",
    )

    bracelet_detected = prediction == BRACELET_DETECTED_CLASS
    print(f"[INFO] Bracelet detected: {bracelet_detected}")
    return bracelet_detected


def choose_pi_move(player_gesture, bracelet_detected):
    if bracelet_detected:
        # Bracelet detected: Pi loses against the player.
        pi_move = PI_LOSE_MOVE[player_gesture]
    else:
        # No bracelet detected: Pi wins to the player.
        pi_move = PI_WIN_MOVE[player_gesture]
        

    print(f"[GAME] Player gesture: {MOVE_NAMES[player_gesture]}")
    print(f"[GAME] Pi gesture: {MOVE_NAMES[pi_move]}")

    return pi_move


def did_player_win(player_gesture, pi_move):
    return pi_move == PI_LOSE_MOVE[player_gesture]


def play_round():
    print("[INFO] Game started")

    show_countdown()
    time.sleep(DELAY_AFTER_COUNTDOWN)

    image_path = capture_image()

    # Decision is made immediately after capture.
    # No thinking animation, no fake delay.
    player_gesture    = get_player_gesture(image_path)
    bracelet_detected = detect_bracelet(image_path)

    pi_move = choose_pi_move(player_gesture, bracelet_detected)

    show_symbol(pi_move)

    if did_player_win(player_gesture, pi_move):
        show_win()
    else:
        show_loss()

    print("[INFO] Round finished")
    print(f"[INFO] Saved image: {image_path}")


def main():
    show_start()

    print("[INFO] RPS Sense HAT wrapper started")
    print("[INFO] Press joystick UP to play")
    print("[INFO] Press joystick MIDDLE to exit")

    try:
        while True:
            for event in sense.stick.get_events():
                if event.action == "pressed" and event.direction == "up":
                    try:
                        play_round()
                    except Exception as e:
                        print(f"[ERROR] Round failed: {e}")
                        sense.show_letter("!", text_colour=R)
                        time.sleep(1)
                        sense.clear()

                elif event.action == "pressed" and event.direction == "middle":
                    print("[INFO] Exiting wrapper")
                    show_end()
                    return

            time.sleep(0.03)

    except KeyboardInterrupt:
        print("[INFO] Stopping wrapper")
        show_end()


if __name__ == "__main__":
    main()