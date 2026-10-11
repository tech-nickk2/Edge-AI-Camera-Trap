# Edge-AI Camera Trap: Arduino App Lab app

The app that runs on the Arduino UNO Q. When the PIR sensor detects motion, the
microcontroller checks the light level (LDR) and switches on the IR LEDs if it is dark.
The Linux side then captures a short burst of photos and runs two Edge Impulse models in
a cascade:

1. **Detector** (`arduino:object_detection` brick): finds animals, people and vehicles.
   Frames with nothing in them are discarded.
2. **Species classifier** (`arduino:image_classification` brick): names the species of each
   animal, or reports `uncertain` when it isn't confident.

Kept photos are saved to `data/captures/<date>/` as a clean copy (for retraining) and a
labelled copy (with boxes), and every trigger is logged in `data/captures/log.csv`.

```
camera-trap-app/
├── app.yaml              # App Lab app definition: web UI + the two AI bricks and their models
├── assets/               # the test web page (index.html, app.js, app.css, libs/)
├── python/
│   ├── main.py           # the cascade (Linux side)
│   └── requirements.txt  # empty: everything comes with App Lab
└── sketch/
    ├── sketch.ino        # PIR, LDR and IR LEDs (microcontroller side)
    └── sketch.yaml       # sketch project file (the Bridge library comes with the board package)
```

## Hardware and wiring

| Part | Connection |
|---|---|
| PIR sensor (e.g. HC-SR501) | VCC → 5V, GND → GND, OUT → D2 |
| LDR + 10 kΩ resistor | LDR between 3.3V and A0; 10 kΩ from A0 to GND |
| IR LEDs (850 nm) | Switched by a logic-level N-channel MOSFET: gate → D3 via 100 Ω, 10 kΩ gate → GND; LEDs (with resistors) from 5V to drain; source → GND |
| USB camera | USB-C hub with power input. Must be able to see infrared (NoIR / no IR-cut filter) for night photos |

## 1. Install the app on the UNO Q

The repository is small (the training datasets are never stored in it), so cloning it on
the board is fine. Only the `camera-trap-app` folder is needed there. Over SSH:

```bash
git clone --depth 1 https://github.com/<user>/edge-ai-camera-trap.git /tmp/edge-ai-camera-trap
cp -r /tmp/edge-ai-camera-trap/camera-trap-app ~/ArduinoApps/camera-trap
rm -rf /tmp/edge-ai-camera-trap
```

The app then appears under **My Apps** in App Lab. If `git` is missing: `sudo apt install git`.

## 2. Get the models

You need one **object detection** model and one **image classification** model from Edge
Impulse:

- **Use the published models:** clone the two public Edge Impulse projects
  ([detector](<EDGE IMPULSE DETECTOR PROJECT LINK>), [species classifier](<EDGE IMPULSE CLASSIFIER PROJECT LINK>))
  into your own Edge Impulse account, or
- **Train your own** with the scripts in [`../training`](../training).

Link App Lab to your Edge Impulse account (a brick's **AI Models** tab → **Train new AI
model** → sign in) and download both models to the board.

## 3. Pick the models in the app

The model ids are the folder names in `~/.arduino-bricks/models/custom-ei/`. To see which
is which:

```bash
grep -H "^name:" ~/.arduino-bricks/models/custom-ei/*/model.yaml
```

Put your ids into `app.yaml` (the model id and the `.eim` path of each brick). On the board
this can be done in one go, e.g.:

```bash
sed -i 's/DETECTOR_ID-VERSION/1119962-1/g; s/CLASSIFIER_ID-VERSION/1121401-3/g' app.yaml
```

The result looks like this:

```yaml
bricks:
- arduino:web_ui
- arduino:object_detection:
    model: ei-model-1119962-1
    variables:
      CUSTOM_MODEL_PATH: /home/arduino/.arduino-bricks/models/custom-ei
      EI_OBJ_DETECTION_MODEL: /home/arduino/.arduino-bricks/models/custom-ei/ei-model-1119962-1/model.eim
- arduino:image_classification:
    model: ei-model-1121401-3
    variables:
      CUSTOM_MODEL_PATH: /home/arduino/.arduino-bricks/models/custom-ei
      EI_CLASSIFICATION_MODEL: /home/arduino/.arduino-bricks/models/custom-ei/ei-model-1121401-3/model.eim
```

**Why `CUSTOM_MODEL_PATH` points at the parent `custom-ei` folder:** App Lab gives all
bricks in an app one shared `CUSTOM_MODEL_PATH`, the board folder that is shared into the
brick containers. Left alone, it becomes the folder of the *last* brick's model only, so
with two custom models the first brick can't see its model and fails to start (see
Troubleshooting). The parent folder contains both models, so both bricks find theirs.

Selecting a model in a brick's **AI Models** tab in App Lab sets only the `model:` line.
If you change models later, update the `.eim` paths to match.

## 4. Run

Click **Run**. On start-up the app prints the two models it is using (name, project id,
type, input size, labels), so you can confirm they're the right ones:

```
Starting models ...
  ObjectDetection: <detector project> (project 1119962), type=object_detection, input=320x320, ...
  ImageClassification: <species project> (project 1121401), type=classification, input=160x160, ...
Camera trap ready (pir mode). Saving to /app/data/captures
[14:02:11] pir trigger (night=False, light=612)
  frame 0: animal/zebra 0.87 (1.2s)
```

## 5. Test it from the web page

The app serves a test page (the `arduino:web_ui` brick). Open it from any device on the same
network as the board:

```
http://<board-name>.local:7000      (or http://<board-ip>:7000)
```

<!-- Add a screenshot of the test page here -->
![Test page](../images/test-page.png)

| Panel | What it does |
|---|---|
| **Live camera** | Live view from the camera. **Capture now** runs the full cascade on demand; tick **with IR LEDs** to test a night capture. |
| **PIR / Timer / Manual** | Switches the trigger mode while the app runs: PIR sensor, a capture every 10 s, or the button only |
| **Sensors** | Live light level (LDR), day/night decision, PIR state and IR LEDs, refreshed every second. **On/Off** tests the IR LEDs (they switch off by themselves after 15 s) |
| **Thresholds** | Detector and species confidence sliders, applied to the next capture. Also shows which models the bricks loaded |
| **Statistics** | Triggers, frames, empty frames discarded, animals / people / vehicles, and species counts |
| **Results** | Every capture, newest first: the labelled frames, each detection with its confidence, the species with its top 3 guesses, and whether the frame was saved |

**Suggested test plan:**

1. **Models:** check the Thresholds panel shows your detector and species classifier.
2. **Cascade without sensors:** in **Manual** mode, point the camera at printed or on-screen
   animals, people and vehicles and press **Capture now**. An empty scene should show
   "Nothing detected: frame discarded".
3. **Thresholds:** move the sliders and capture again to see how many detections and
   species guesses change.
4. **Sensors:** wave a hand at the PIR (the PIR dot lights up), cover the LDR (it switches to
   Night), and press IR **On/Off**.
5. **Night capture:** cover the LDR, switch to **PIR** mode and walk past: the result should be
   tagged *night* and *IR on*. Use the light readings to tune `DARK_BELOW` / `LIGHT_ABOVE` in
   the sketch.

## Settings (top of `python/main.py`)

| Setting | Default | Meaning |
|---|---|---|
| `TRIGGER_MODE` | `"pir"` | Mode at start-up: `"pir"`, `"timer"` (every `TIMER_INTERVAL_S` seconds) or `"manual"`. Can be switched on the test page |
| `DETECTOR_THRESHOLD` | `0.5` | Minimum confidence for a detection (start-up value; adjustable on the test page) |
| `CLASSIFIER_THRESHOLD` | `0.6` | Below this the species is reported as `uncertain` (start-up value; adjustable on the test page) |
| `BURST_FRAMES` | `3` | Frames analysed per trigger |
| `CAPTURE_WIDTH`, `CAPTURE_HEIGHT` | `1280`, `720` | Camera resolution |
| `SAVE_EMPTY` | `False` | Keep frames where nothing was detected |

Day/night thresholds (`DARK_BELOW`, `LIGHT_ABOVE`) are in `sketch/sketch.ino`. The light
value is logged with every trigger, so use the log to calibrate them.

## Seeing the Python output

App Lab's console shows the sketch build and the containers starting. The Python output
(`print` lines and errors) is in the app's main container:

```bash
docker logs <app-folder-name>-main-1 2>&1 | tail -40     # e.g. camera-trap-main-1
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `container ...-ei-obj-detection-runner-1 is unhealthy`, and its log (`docker logs <app>-ei-obj-detection-runner-1`) shows `ENOENT: no such file or directory, chmod .../model.eim` | The brick can't see its model folder. Check that both `CUSTOM_MODEL_PATH` variables in `app.yaml` point at `/home/arduino/.arduino-bricks/models/custom-ei` (step 3). `docker inspect <app>-ei-obj-detection-runner-1 --format '{{range .Mounts}}{{println .Source}}{{end}}'` shows which folder was shared |
| A brick runs the wrong model (e.g. generic objects such as "cup") | Select your model in that brick's **AI Models** tab, or fix the id in `app.yaml` |
| `No solution found when resolving dependencies` | Leave `requirements.txt` empty: OpenCV is already provided and listing it conflicts |
| Camera errors | Check the camera on the USB-C hub. Don't add the Video Object Detection brick: it would take the camera |
| No triggers | Test with `TRIGGER_MODE = "timer"`; check the PIR wiring and its sensitivity/delay knobs |
| Dark night photos | The camera's IR-cut filter is blocking the IR LEDs; use a NoIR camera |
