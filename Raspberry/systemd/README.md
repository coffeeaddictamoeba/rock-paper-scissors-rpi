# Systemd Service

This folder contains the service file that starts the Raspberry Pi game automatically after boot.

Chosen names:

- Service file: `brace-rps.service`
- Wrapper: `rockpaperscissors_wrapper.py`
- Install directory: `/usr/local/bin`

`brace-rps` means "bracelet rock-paper-scissors". It is used for the service name. The service starts the Python wrapper provided by the team workflow.

## Expected Raspberry Pi Layout

The service expects the wrapper here:

```bash
/usr/local/bin/rockpaperscissors_wrapper.py
```

The wrapper can then call the predictor executable, for example:

```bash
/usr/local/bin/rockpaperscissors --predict /dev/shm/brace-rps/latest_frame.bmp --model /usr/local/bin/model.tflite --verbose
```

The service runs from this working directory:

```bash
/usr/local/bin
```

If the executable is placed somewhere else, update these lines in `brace-rps.service`:

```ini
WorkingDirectory=/usr/local/bin
ExecStart=/usr/bin/python3 /usr/local/bin/rockpaperscissors_wrapper.py
```

## Install

Run these commands on the Raspberry Pi from the repository root:

```bash
sudo mkdir -p /dev/shm/brace-rps
sudo cp Raspberry/systemd/brace-rps.service /etc/systemd/system/brace-rps.service
sudo systemctl daemon-reload
sudo systemctl enable brace-rps.service
sudo systemctl start brace-rps.service
```

The wrapper must be copied to `/usr/local/bin/rockpaperscissors_wrapper.py` before `systemctl start` can work.

The predictor executable and model must also be available wherever the wrapper expects them.

## Check Status

```bash
systemctl status brace-rps.service
```

## Show Logs

```bash
journalctl -u brace-rps.service -f
```

## Stop Or Disable

```bash
sudo systemctl stop brace-rps.service
sudo systemctl disable brace-rps.service
```

## Runtime Image Path

The service reads this RAM-backed image file:

```bash
/dev/shm/brace-rps/latest_frame.bmp
```

The image capture process should keep overwriting that file with the latest frame.

The service starts the wrapper like this:

```bash
/usr/bin/python3 /usr/local/bin/rockpaperscissors_wrapper.py
```

The wrapper should use the latest frame path as a `.bmp` file because the shared image format should be BMP.

## Notes

The service follows the same structure as the team-provided wrapper service. It runs as user/group `1000` from `/usr/local/bin` and restarts after 3 seconds if it exits.
