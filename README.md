# rock-paper-scissors-rpi

Real-time game of Rock-Paper-Scissors with RPI

### Setup (with `tflite-demo-container`)

---

- On your local machine, train the model with the `train_and_convert.py` script (optional).

  - If using `tflite-demo-container` (recommended), run `make train`
  - Don't forget to prepare the data and change the name of data directory (`DEFAULT_DATA_DIR ="EAI4IL-project-data"`) inside the script!
- If models are ready (you can take them from `Models/` folder here), compile the C++ code with `make build`, placing it inside `src` directory of `tflite-demo-container`.

  - As the default name for building is `pi-demo`, you may want to change it inside `.env`.
  - I recommend setting `APP_NAME=rockpaperscissors`.
  - You will find this executable inside `build` direcory.
- On Raspberry Pi with SenseHat, copy the C++ executable (`pi_demo`, `rockpaperscissors` etc.) and both **gesture classification** and **bracelet detection** models.

  - As the systemd service mentiones `Documents/Project/` directory inside Pi's `/home`, it is highly recommended to place executable and models there.
  - Models should be placed under specified direcories `gesture/` and `bracelet/` under `Project/`
- Place `rps-camera.service` under `/etc/systemd/system`
- Also copy the `Sense_hat_wrapper.py` script to `/usr/bin/` or modify the service so it accepts the script right from `Documents` (recommended for debugging and taken image review)
- Run `sudo systemctl daemon-reload` so service is able to start
- Start the service with `sudo systemctl start rps-camera`
- Then move the joystick of SenseHat up - the countdown will start, and right after "3" it will make a photo and try to beat/lose to you

  - Both models expect a hand on a white wall with day light.
  - The full hand and wrist should be visible.
  - Model expects normal Rock-Paper-Scissors gestures - if you show something else, model will be confuzed and give you the wrong result.
  - As the resolution of image the Pi is taking is pretty small (128x96), the distance matters. The best distance to show the gesture is from **45 to 65 sm** from the Pi's camera centre.

### C++ Executable for TFLite Model(s)

---

```
./rockpaperscissors --model <model_path> --predict <image_bmp> --num_classes <n> --verbose
```

- `--model` - allows you to specify the model path
- `--predict` - allows you to specify the BMP 128x96 image to get a model's prediction
- `--num-classes` - allow you to select number of classes that model predicts depending on model. Default: 2.
- `--verbose` - enables all the logs from the app. It is recommnded to run with this flag for debugging.

The application itself returns the id of prediction in `int` format. If application fails, `-1` is returned.
