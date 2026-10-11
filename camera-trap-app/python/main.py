"""
Edge-AI camera trap: detector -> species classifier cascade (Linux side of the UNO Q).

Flow on every trigger:
  PIR fires -> sketch reads the LDR, switches the IR LEDs on if it's dark ->
  this script grabs a short burst of frames -> detector finds animal / person /
  vehicle boxes -> each animal box is cropped (with padding) and passed to the
  species classifier -> labelled photos + a CSV log are saved; empty frames are
  discarded -> the IR LEDs are switched off again.

Models: App Lab downloads Edge Impulse models to
/home/arduino/.arduino-bricks/models/custom-ei/<ei-model-PROJECT-VERSION>/model.eim.
This script inspects every .eim it finds there (and in the app's own models/ folder)
and picks the most recently downloaded object detection model as the detector and
the most recently downloaded classification model as the classifier. Set
DETECTOR_MODEL / CLASSIFIER_MODEL below to force specific files.
"""

import csv
import datetime
import glob
import os
import queue
import time

import cv2
import numpy as np
from edge_impulse_linux.image import ImageImpulseRunner

try:
    from arduino.app_utils import App, Bridge
except ImportError:  # running outside App Lab (e.g. over SSH): timer mode only
    App = Bridge = None

# ------------------------------------------------------------------ settings

TRIGGER_MODE = "pir"         # "pir" = wait for the PIR sensor, "timer" = capture every TIMER_INTERVAL_S
TIMER_INTERVAL_S = 10        # desk testing without the sensors wired up

CAMERA_INDEX = 0             # /dev/video0
CAPTURE_WIDTH, CAPTURE_HEIGHT = 1280, 720
WARMUP_FRAMES = 5            # frames thrown away after a trigger so exposure adapts (e.g. to the IR LEDs)
BURST_FRAMES = 3             # frames analysed per trigger
BURST_GAP_S = 0.4            # pause between burst frames

DETECTOR_THRESHOLD = 0.5     # minimum detector confidence for a box to count
CLASSIFIER_THRESHOLD = 0.6   # below this the species is reported as "uncertain"
CROP_PAD = 0.25              # padding around animal boxes; matches the training crops

SAVE_EMPTY = False           # keep frames where nothing was detected (normally discarded)

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIRS = [os.path.join(APP_DIR, "models"), "/home/arduino/.arduino-bricks/models/custom-ei"]
DETECTOR_MODEL = None        # e.g. "/home/arduino/.arduino-bricks/models/custom-ei/ei-model-1119962-1/model.eim"
CLASSIFIER_MODEL = None
OUT_DIR = os.path.join(APP_DIR, "captures")


# -------------------------------------------------------------------- models

def load_runner(path):
    runner = ImageImpulseRunner(path)
    info = runner.init()
    return runner, info


def describe(info):
    project = info.get("project", {})
    params = info.get("model_parameters", {})
    return (f"{project.get('owner', '?')} / {project.get('name', '?')} "
            f"(project {project.get('id', '?')}, v{project.get('deploy_version', '?')}), "
            f"type={params.get('model_type', '?')}, "
            f"input={params.get('image_input_width')}x{params.get('image_input_height')}, "
            f"labels={params.get('labels')}")


def find_models():
    """Return ((detector_runner, info), (classifier_runner, info))."""
    if DETECTOR_MODEL and CLASSIFIER_MODEL:
        det, cls = load_runner(DETECTOR_MODEL), load_runner(CLASSIFIER_MODEL)
        print("Detector:  ", describe(det[1]))
        print("Classifier:", describe(cls[1]))
        return det, cls

    paths = []
    for d in MODEL_DIRS:
        paths += glob.glob(os.path.join(d, "**", "*.eim"), recursive=True)
    if not paths:
        raise SystemExit(f"No .eim models found in {MODEL_DIRS}. Download them in App Lab first, "
                         f"or copy them into {os.path.join(APP_DIR, 'models')}.")

    # newest download of each model type wins
    best = {}
    print("Models found:")
    for path in sorted(paths, key=os.path.getmtime):
        try:
            runner, info = load_runner(path)
        except Exception as e:
            print(f"  {path}: could not load ({e})")
            continue
        model_type = info.get("model_parameters", {}).get("model_type", "")
        kind = "detector" if "object_detection" in model_type else \
               "classifier" if model_type == "classification" else None
        print(f"  {path}\n    {describe(info)}")
        if kind is None:
            runner.stop()
            continue
        if kind in best:
            best[kind][0].stop()
        best[kind] = (runner, info, path)

    if "detector" not in best or "classifier" not in best:
        raise SystemExit("Need one object detection model and one classification model.")
    for kind in ("detector", "classifier"):
        print(f"Using {kind}: {best[kind][2]}")
    return best["detector"][:2], best["classifier"][:2]


def input_size(info):
    p = info["model_parameters"]
    return p["image_input_width"], p["image_input_height"]


def letterbox(img, width, height):
    """Studio's "Fit longest axis": scale to fit, pad the rest with black.
    Returns the padded image plus the scale and offsets needed to map boxes back."""
    h, w = img.shape[:2]
    s = min(width / w, height / h)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    canvas = np.zeros((height, width, 3), np.uint8)
    x0, y0 = (width - nw) // 2, (height - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return canvas, s, x0, y0


def detect(runner, info, rgb):
    """Run the detector on a full-resolution RGB frame; boxes come back in frame pixels."""
    w, h = input_size(info)
    canvas, s, x0, y0 = letterbox(rgb, w, h)
    features, _ = runner.get_features_from_image(canvas)
    result = runner.classify(features)["result"]
    boxes = []
    for b in result.get("bounding_boxes", []):
        if b["value"] < DETECTOR_THRESHOLD:
            continue
        boxes.append({
            "label": b["label"], "conf": b["value"],
            "x": (b["x"] - x0) / s, "y": (b["y"] - y0) / s,
            "w": b["width"] / s, "h": b["height"] / s,
        })
    return boxes


def classify_crop(runner, info, rgb, box):
    """Crop the box (padded like the training crops) and return (species, confidence)."""
    H, W = rgb.shape[:2]
    px, py = box["w"] * CROP_PAD, box["h"] * CROP_PAD
    x1, y1 = max(0, int(box["x"] - px)), max(0, int(box["y"] - py))
    x2, y2 = min(W, int(box["x"] + box["w"] + px)), min(H, int(box["y"] + box["h"] + py))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None, 0.0
    w, h = input_size(info)
    canvas, _, _, _ = letterbox(rgb[y1:y2, x1:x2], w, h)
    features, _ = runner.get_features_from_image(canvas)
    scores = runner.classify(features)["result"]["classification"]
    species, conf = max(scores.items(), key=lambda kv: kv[1])
    return species, conf


# ---------------------------------------------------------------- the trap

class CameraTrap:
    def __init__(self):
        (self.det, self.det_info), (self.cls, self.cls_info) = find_models()
        self.cam = cv2.VideoCapture(CAMERA_INDEX)
        if not self.cam.isOpened():
            raise SystemExit(f"Could not open camera {CAMERA_INDEX}")
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
        self.cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.events = queue.Queue()
        self.last_timer = 0.0
        os.makedirs(OUT_DIR, exist_ok=True)
        self.log_path = os.path.join(OUT_DIR, "log.csv")
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", newline="") as f:
                csv.writer(f).writerow(["time", "trigger", "night", "light", "frame", "label",
                                        "det_conf", "species", "species_conf", "x", "y", "w", "h", "file"])

    # called by the sketch through Bridge.notify("motion", night, light)
    def on_motion(self, night, light):
        self.events.put(("pir", bool(night), int(light)))

    def grab(self):
        ok, frame = self.cam.read()
        return frame if ok else None

    def handle_trigger(self, trigger, night, light):
        stamp = datetime.datetime.now()
        print(f"[{stamp:%H:%M:%S}] {trigger} trigger (night={night}, light={light})")
        for _ in range(WARMUP_FRAMES):  # flush stale frames, let exposure settle
            self.grab()
        day_dir = os.path.join(OUT_DIR, f"{stamp:%Y-%m-%d}")
        os.makedirs(day_dir, exist_ok=True)

        rows = []
        for n in range(BURST_FRAMES):
            bgr = self.grab()
            if bgr is None:
                print("  camera read failed")
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            t0 = time.time()
            boxes = detect(self.det, self.det_info, rgb)
            for b in boxes:
                b["species"], b["species_conf"] = (None, 0.0)
                if b["label"] == "animal":
                    sp, conf = classify_crop(self.cls, self.cls_info, rgb, b)
                    b["species"] = sp if conf >= CLASSIFIER_THRESHOLD else "uncertain"
                    b["species_conf"] = conf
            took = time.time() - t0

            name = f"{stamp:%H%M%S}_{n}"
            if boxes or SAVE_EMPTY:
                # clean copy for retraining, annotated copy for people
                cv2.imwrite(os.path.join(day_dir, name + ".jpg"), bgr)
                cv2.imwrite(os.path.join(day_dir, name + "_labelled.jpg"), annotate(bgr, boxes))
            summary = ", ".join(f"{b['label']}{'/' + b['species'] if b['species'] else ''} "
                                f"{b['conf']:.2f}" for b in boxes) or "empty, discarded"
            print(f"  frame {n}: {summary} ({took:.1f}s)")

            base = [stamp.isoformat(timespec="seconds"), trigger, night, light, n]
            if not boxes:
                rows.append(base + ["empty", "", "", "", "", "", "", "", ""])
            for b in boxes:
                rows.append(base + [b["label"], f"{b['conf']:.3f}", b["species"] or "",
                                    f"{b['species_conf']:.3f}" if b["species"] else "",
                                    int(b["x"]), int(b["y"]), int(b["w"]), int(b["h"]),
                                    name + ".jpg"])
            time.sleep(BURST_GAP_S)

        with open(self.log_path, "a", newline="") as f:
            csv.writer(f).writerows(rows)
        if night and Bridge is not None:
            Bridge.call("ir_off")
        # motion that happened while we were busy is stale; drop it
        while not self.events.empty():
            self.events.get_nowait()

    def loop(self):
        try:
            self.handle_trigger(*self.events.get(timeout=1.0))
        except queue.Empty:
            if TRIGGER_MODE == "timer" and time.time() - self.last_timer >= TIMER_INTERVAL_S:
                self.last_timer = time.time()
                self.handle_trigger("timer", None, None)


def annotate(bgr, boxes):
    out = bgr.copy()
    colours = {"animal": (80, 200, 80), "person": (60, 60, 230), "vehicle": (230, 150, 50)}
    for b in boxes:
        c = colours.get(b["label"], (200, 200, 200))
        x, y, w, h = int(b["x"]), int(b["y"]), int(b["w"]), int(b["h"])
        cv2.rectangle(out, (x, y), (x + w, y + h), c, 2)
        text = f"{b['label']} {b['conf']:.2f}"
        if b["species"]:
            text += f" | {b['species']} {b['species_conf']:.2f}"
        cv2.putText(out, text, (x, max(15, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
    return out


if Bridge is None and TRIGGER_MODE == "pir":
    print("No App Lab Bridge (running outside App Lab?): the PIR can't reach this script, "
          "falling back to timer mode")
    TRIGGER_MODE = "timer"

trap = CameraTrap()

if Bridge is not None:
    Bridge.provide("motion", trap.on_motion)

print(f"Camera trap ready ({TRIGGER_MODE} mode). Saving to {OUT_DIR}")
if App is not None:
    App.run(user_loop=trap.loop)
else:
    while True:
        trap.loop()
