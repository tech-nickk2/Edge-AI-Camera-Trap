# Edge-AI Camera Trap: Arduino App Lab app

The app that runs on the Arduino UNO Q. When the PIR sensor detects motion, the
microcontroller checks the light level (LDR) and switches on the IR LEDs if it is dark.
The Linux side then captures a short burst of photos and runs two Edge Impulse models in
a cascade:

1. **Detector**: finds animals, people and vehicles. Frames with nothing in them are discarded.
2. **Species classifier**: names the species of each animal it finds, or reports "uncertain"
   when it isn't confident.

Kept photos are saved to `captures/<date>/` as a clean copy (for retraining) and a labelled
copy (with boxes), and every trigger is logged in `captures/log.csv`.

```
camera-trap-app/
├── app.yaml              # App Lab app definition
├── python/
│   ├── main.py           # the cascade (Linux side)
│   └── requirements.txt
├── sketch/
│   └── sketch.ino        # PIR, LDR and IR LEDs (microcontroller side)
└── models/               # optional: put .eim files here (see models/README.md)
```

## Hardware and wiring

| Part | Connection |
|---|---|
| PIR sensor (e.g. HC-SR501) | VCC → 5V, GND → GND, OUT → D2 |
| LDR + 10 kΩ resistor | LDR between 3.3V and A0; 10 kΩ from A0 to GND |
| IR LEDs (850 nm) | Switched by a logic-level N-channel MOSFET: gate → D3 via 100 Ω, 10 kΩ gate → GND; LEDs (with resistors) from 5V to drain; source → GND |
| USB camera | USB-C hub with power input. Must be able to see infrared (NoIR / no IR-cut filter) for night photos |

## Install on the UNO Q

The repository is small (the training datasets are never stored in it), so cloning it on
the board is fine. Only the `camera-trap-app` folder is needed there.

```bash
ssh arduino@<your-board>.local
git clone --depth 1 https://github.com/<user>/edge-ai-camera-trap.git /tmp/edge-ai-camera-trap
cp -r /tmp/edge-ai-camera-trap/camera-trap-app ~/ArduinoApps/camera-trap
rm -rf /tmp/edge-ai-camera-trap
```

`~/ArduinoApps` is where App Lab keeps your apps on the board (check with `ls ~/ArduinoApps`).
The app then appears under **My Apps** in App Lab. If `git` is missing: `sudo apt install git`.

## Get the models

The app needs one object detection model and one classification model from Edge Impulse,
both built for the UNO Q (Linux aarch64).

- **Use the published models:** clone the two public Edge Impulse projects
  ([detector](<EDGE IMPULSE DETECTOR PROJECT LINK>), [species classifier](<EDGE IMPULSE CLASSIFIER PROJECT LINK>))
  into your own Edge Impulse account, or
- **Train your own** with the scripts in [`../training`](../training).

Then link App Lab to your Edge Impulse account (any brick's **AI Models** tab →
**Train new AI model** → sign in) and download both models to the board, for example
through the "Detect objects on Camera" and image classification examples.

On start-up the app scans the downloaded models and picks the most recently downloaded
object detection model and classification model. It prints its choice; to force specific
files, set `DETECTOR_MODEL` / `CLASSIFIER_MODEL` at the top of `python/main.py` or copy the
files into `models/`.

## Run

Open the app in App Lab and click **Run**. Watch the log for:

```
Using detector: .../ei-model-XXXXXXX-N/model.eim
Using classifier: .../ei-model-YYYYYYY-N/model.eim
Camera trap ready (pir mode). Saving to .../captures
```

## Settings (top of `python/main.py`)

| Setting | Default | Meaning |
|---|---|---|
| `TRIGGER_MODE` | `"pir"` | `"timer"` captures every `TIMER_INTERVAL_S` seconds, for testing without sensors |
| `DETECTOR_THRESHOLD` | `0.5` | Minimum confidence for a detection |
| `CLASSIFIER_THRESHOLD` | `0.6` | Below this the species is reported as `uncertain` |
| `BURST_FRAMES` | `3` | Frames analysed per trigger |
| `SAVE_EMPTY` | `False` | Keep frames where nothing was detected |

Day/night thresholds (`DARK_BELOW`, `LIGHT_ABOVE`) are in `sketch/sketch.ino`. The light
value is logged with every trigger, so use the log to calibrate them.

## Troubleshooting

| Problem | Fix |
|---|---|
| `No .eim models found` | Download the models in App Lab, or copy them into `models/` |
| Wrong model picked | Set `DETECTOR_MODEL` / `CLASSIFIER_MODEL` in `main.py` |
| `Could not open camera 0` | Check the camera on the USB-C hub; try `CAMERA_INDEX = 1` |
| No triggers | Test with `TRIGGER_MODE = "timer"`; check PIR wiring and its sensitivity/delay knobs |
| Dark night photos | The camera's IR-cut filter is blocking the IR LEDs; use a NoIR camera |
