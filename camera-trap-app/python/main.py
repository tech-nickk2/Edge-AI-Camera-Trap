"""
Edge-AI camera trap: detector -> species classifier cascade (Linux side of the UNO Q).

Flow on every trigger:
  PIR fires -> sketch reads the LDR, switches the IR LEDs on if it's dark ->
  this script grabs a short burst of frames -> detector finds animal / person /
  vehicle boxes -> each animal box is cropped (with padding) and passed to the
  species classifier -> labelled photos + a CSV log are saved; empty frames are
  discarded -> the IR LEDs are switched off again.

The two models run in App Lab bricks, each in its own container:
  arduino:object_detection      -> the detector           (model set in app.yaml)
  arduino:image_classification  -> the species classifier (model set in app.yaml)
A test web page (arduino:web_ui brick, files in ../assets) shows the live camera, every
result, live sensor readings and the statistics, and lets you capture on demand, switch
the trigger mode, change the thresholds and test the IR LEDs. Open http://<board>:7000
"""

import base64
import collections
import csv
import datetime
import os
import queue
import threading
import time

import cv2
import numpy as np
import requests

from arduino.app_utils import App, Bridge
from arduino.app_bricks.object_detection import ObjectDetection
from arduino.app_bricks.image_classification import ImageClassification
from arduino.app_bricks.web_ui import WebUI

try:
    from arduino.app_peripherals.camera import Camera
except ImportError:  # older App Lab without the Camera peripheral (no live stream then)
    Camera = None

# ------------------------------------------------------------------ settings

TRIGGER_MODE = "pir"         # "pir", "timer" (every TIMER_INTERVAL_S) or "manual" (web page button only)
TIMER_INTERVAL_S = 10

CAPTURE_WIDTH, CAPTURE_HEIGHT = 1280, 720
WARMUP_FRAMES = 5            # frames thrown away after a trigger so exposure adapts (e.g. to the IR LEDs)
BURST_FRAMES = 3             # frames analysed per trigger
BURST_GAP_S = 0.4            # pause between burst frames

DETECTOR_THRESHOLD = 0.5     # minimum detector confidence for a box to count
CLASSIFIER_THRESHOLD = 0.6   # below this the species is reported as "uncertain"
CROP_PAD = 0.25              # padding around animal boxes; matches the training crops

SAVE_EMPTY = False           # keep frames where nothing was detected (normally discarded)
SENSOR_POLL_S = 1.0          # how often the web page's sensor readings are refreshed

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(APP_DIR, "data", "captures")   # "data" is App Lab's persistent app folder


# ------------------------------------------------------------ model helpers

def model_params(brick):
    """Input size, resize mode and a short description of the model behind a brick."""
    info = requests.get(f"{brick.url}/api/info", timeout=10).json()
    p = info.get("modelParameters", {})
    project = info.get("project", {})
    desc = {"name": project.get("name", "?"), "project": project.get("id", "?"),
            "type": p.get("model_type"), "input": f"{p.get('image_input_width')}x{p.get('image_input_height')}",
            "resize": p.get("image_resize_mode", "not reported"), "labels": p.get("labels", [])}
    print(f"  {type(brick).__name__}: {desc}")
    return (p["image_input_width"], p["image_input_height"], p.get("image_resize_mode", "squash")), desc


def to_model_input(bgr, width, height, mode):
    """Resize a frame exactly as Edge Impulse Studio did for this model. Returns the
    resized image and the transform model_xy = frame_xy * scale + offset, so detector
    boxes can be mapped back onto the full-resolution frame."""
    H, W = bgr.shape[:2]
    if mode == "fit-longest":      # scale to fit inside, pad with black
        s = min(width / W, height / H)
        nw, nh = int(W * s), int(H * s)
        top, left = (height - nh) // 2, (width - nw) // 2
        img = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        img = cv2.copyMakeBorder(img, top, height - nh - top, left, width - nw - left,
                                 cv2.BORDER_CONSTANT, value=0)
        return img, (s, s, left, top)
    if mode == "fit-shortest":     # scale to cover, crop the centre
        s = max(width / W, height / H)
        nw, nh = int(np.ceil(W * s)), int(np.ceil(H * s))
        cx, cy = (nw - width) // 2, (nh - height) // 2
        img = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)[cy:cy + height, cx:cx + width]
        return img, (s, s, -cx, -cy)
    img = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)   # "squash"
    return img, (width / W, height / H, 0, 0)


def png_bytes(bgr):
    ok, buf = cv2.imencode(".png", bgr)
    return buf.tobytes() if ok else None


def jpeg_data_url(bgr, max_width=640, quality=75):
    """Small JPEG preview for the web page."""
    h, w = bgr.shape[:2]
    if w > max_width:
        bgr = cv2.resize(bgr, (max_width, int(h * max_width / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode() if ok else None


# ---------------------------------------------------------------- the trap

class CameraTrap:
    def __init__(self):
        self.mode = TRIGGER_MODE
        self.det_thr = DETECTOR_THRESHOLD
        self.cls_thr = CLASSIFIER_THRESHOLD
        self.stats = collections.Counter()
        self.species_counts = collections.Counter()
        self.recent = collections.deque(maxlen=8)   # last results, re-sent to newly opened pages
        self.sensors = None
        self.busy = False
        self.events = queue.Queue()
        self.last_timer = 0.0
        self.last_poll = 0.0
        self.bridge_lock = threading.Lock()
        self.bridge_warned = set()

        print("Starting models ...")
        self.detector = ObjectDetection(confidence=0.05)        # threshold applied per call
        self.classifier = ImageClassification(confidence=0.0)    # keep all scores; threshold applied below
        self.det_size, self.det_desc = model_params(self.detector)
        _, self.cls_desc = model_params(self.classifier)

        if Camera is not None:
            self.cam = Camera(resolution=(CAPTURE_WIDTH, CAPTURE_HEIGHT), fps=10)
            self.cam.start()
            self.read = self.cam.capture
        else:
            self.cam = cv2.VideoCapture(0)
            self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
            self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
            self.read = lambda: (lambda r: r[1] if r[0] else None)(self.cam.read())

        os.makedirs(OUT_DIR, exist_ok=True)
        self.log_path = os.path.join(OUT_DIR, "log.csv")
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", newline="") as f:
                csv.writer(f).writerow(["time", "trigger", "night", "light", "frame", "label",
                                        "det_conf", "species", "species_conf", "x", "y", "w", "h", "file"])
        self.setup_ui()

    # ---- web page

    def setup_ui(self):
        self.ui = WebUI()
        if Camera is not None and hasattr(self.ui, "expose_camera"):
            self.ui.expose_camera("/camera", self.cam, jpeg_quality=70)
        self.ui.on_connect(lambda sid: self.send_state())
        self.ui.on_message("capture", lambda sid, data: self.manual_capture(bool(data.get("ir"))))
        self.ui.on_message("set_mode", lambda sid, data: self.set_mode(data.get("mode")))
        self.ui.on_message("set_thresholds", lambda sid, data: self.set_thresholds(data))
        self.ui.on_message("ir", lambda sid, data: self.bridge_call("ir_on" if data.get("on") else "ir_off"))

    def state(self):
        return {"mode": self.mode, "timer_s": TIMER_INTERVAL_S, "busy": self.busy,
                "det_thr": self.det_thr, "cls_thr": self.cls_thr,
                "detector": self.det_desc, "classifier": self.cls_desc,
                "stats": dict(self.stats), "species": self.species_counts.most_common(),
                "sensors": self.sensors, "stream": Camera is not None}

    def send_state(self):
        self.ui.send_message("state", self.state())
        for r in list(self.recent):
            self.ui.send_message("result", r)

    def manual_capture(self, with_ir):
        self.events.put(("manual", None, None, with_ir))
        return {"queued": True}

    def set_mode(self, mode):
        if mode in ("pir", "timer", "manual"):
            self.mode = mode
            self.last_timer = time.time()
            print(f"Trigger mode -> {mode}")
        self.ui.send_message("state", self.state())

    def set_thresholds(self, data):
        self.det_thr = min(max(float(data.get("det_thr", self.det_thr)), 0.05), 0.99)
        self.cls_thr = min(max(float(data.get("cls_thr", self.cls_thr)), 0.0), 0.99)
        self.ui.send_message("state", self.state())

    def bridge_call(self, name):
        try:
            with self.bridge_lock:
                return Bridge.call(name, timeout=2)
        except Exception as e:   # sketch not running / function missing: the page just shows no data
            if name not in self.bridge_warned:   # warn once per function, not every poll
                self.bridge_warned.add(name)
                print(f"Bridge call {name} failed: {e} (is the latest sketch flashed?)")
            return None

    def poll_sensors(self):
        raw = self.bridge_call("read_sensors")
        if isinstance(raw, str) and raw.count(",") == 3:
            light, night, pir, ir = raw.split(",")
            self.sensors = {"light": int(light), "night": night == "1", "pir": pir == "1", "ir": ir == "1"}
            self.ui.send_message("sensors", self.sensors)

    # ---- the two stages

    def detect(self, bgr):
        """Stage 1: boxes in full-resolution frame pixels."""
        width, height, mode = self.det_size
        img, (sx, sy, ox, oy) = to_model_input(bgr, width, height, mode)
        res = self.detector.detect(png_bytes(img), image_type="png", confidence=self.det_thr) or {}
        boxes = []
        for d in res.get("detection", []):
            x1, y1, x2, y2 = d["bounding_box_xyxy"]          # in model-input pixels
            boxes.append({
                "label": d["class_name"], "conf": float(d["confidence"]) / 100,
                "x": (x1 - ox) / sx, "y": (y1 - oy) / sy,
                "w": (x2 - x1) / sx, "h": (y2 - y1) / sy,
            })
        return boxes

    def classify(self, bgr, box):
        """Stage 2: crop the box (padded like the training crops) -> (species, confidence, top 3)."""
        H, W = bgr.shape[:2]
        px, py = box["w"] * CROP_PAD, box["h"] * CROP_PAD
        x1, y1 = max(0, int(box["x"] - px)), max(0, int(box["y"] - py))
        x2, y2 = min(W, int(box["x"] + box["w"] + px)), min(H, int(box["y"] + box["h"] + py))
        if x2 - x1 < 4 or y2 - y1 < 4:
            return None, 0.0, []
        res = self.classifier.classify(png_bytes(bgr[y1:y2, x1:x2]), image_type="png") or {}
        scores = sorted(((c["class_name"], float(c["confidence"]) / 100) for c in res.get("classification", [])),
                        key=lambda kv: -kv[1])
        if not scores:
            return None, 0.0, []
        return scores[0][0], scores[0][1], scores[:3]

    # ---- triggers

    # called by the sketch through Bridge.notify("motion", night, light)
    def on_motion(self, night, light):
        self.events.put(("pir", bool(night), int(light), False))

    def handle_trigger(self, trigger, night, light, with_ir=False):
        stamp = datetime.datetime.now()
        print(f"[{stamp:%H:%M:%S}] {trigger} trigger (night={night}, light={light})")
        self.busy = True
        self.ui.send_message("busy", {"trigger": trigger, "time": f"{stamp:%H:%M:%S}"})
        if with_ir:
            self.bridge_call("ir_on")
        for _ in range(WARMUP_FRAMES):  # flush stale frames, let exposure settle
            self.read()
        day_dir = os.path.join(OUT_DIR, f"{stamp:%Y-%m-%d}")
        os.makedirs(day_dir, exist_ok=True)

        rows, frames = [], []
        for n in range(BURST_FRAMES):
            bgr = self.read()
            if bgr is None:
                print("  camera read failed")
                continue
            t0 = time.time()
            boxes = self.detect(bgr)
            for b in boxes:
                b["species"], b["species_conf"], b["top3"] = None, 0.0, []
                if b["label"] == "animal":
                    sp, conf, top3 = self.classify(bgr, b)
                    b["species"] = sp if conf >= self.cls_thr else "uncertain"
                    b["species_conf"], b["top3"] = conf, top3
            took = time.time() - t0

            name = f"{stamp:%H%M%S}_{n}"
            labelled = annotate(bgr, boxes)
            if boxes or SAVE_EMPTY:
                # clean copy for retraining, annotated copy for people
                cv2.imwrite(os.path.join(day_dir, name + ".jpg"), bgr)
                cv2.imwrite(os.path.join(day_dir, name + "_labelled.jpg"), labelled)
            summary = ", ".join(f"{b['label']}{'/' + b['species'] if b['species'] else ''} "
                                f"{b['conf']:.2f}" for b in boxes) or "empty, discarded"
            print(f"  frame {n}: {summary} ({took:.1f}s)")

            self.stats["frames"] += 1
            self.stats["empty" if not boxes else "kept"] += 1
            for b in boxes:
                self.stats[b["label"]] += 1
                if b["species"]:
                    self.species_counts[b["species"]] += 1
            frames.append({"n": n, "took": round(took, 2), "saved": bool(boxes or SAVE_EMPTY),
                           "preview": jpeg_data_url(labelled),
                           "boxes": [{"label": b["label"], "conf": round(b["conf"], 3),
                                      "species": b["species"], "species_conf": round(b["species_conf"], 3),
                                      "top3": [[s, round(c, 3)] for s, c in b["top3"]]} for b in boxes]})

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
        if night or with_ir:
            self.bridge_call("ir_off")
        self.stats["triggers"] += 1
        self.stats[f"trigger_{trigger}"] += 1

        result = {"time": f"{stamp:%Y-%m-%d %H:%M:%S}", "trigger": trigger, "night": night,
                  "light": light, "ir": bool(night or with_ir), "frames": frames}
        self.recent.append(result)
        self.busy = False
        self.ui.send_message("result", result)
        self.ui.send_message("state", self.state())
        # motion that happened while we were busy is stale; drop it (manual requests are kept)
        pending = []
        while not self.events.empty():
            ev = self.events.get_nowait()
            if ev[0] == "manual":
                pending.append(ev)
        for ev in pending:
            self.events.put(ev)

    def loop(self):
        try:
            ev = self.events.get(timeout=0.2)
            if ev[0] == "pir" and self.mode != "pir":
                return  # PIR events are ignored in timer/manual mode
            self.handle_trigger(*ev)
        except queue.Empty:
            now = time.time()
            if self.mode == "timer" and now - self.last_timer >= TIMER_INTERVAL_S:
                self.last_timer = now
                self.handle_trigger("timer", None, None)
            elif now - self.last_poll >= SENSOR_POLL_S:
                self.last_poll = now
                self.poll_sensors()


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


trap = CameraTrap()
Bridge.provide("motion", trap.on_motion)
print(f"Camera trap ready ({trap.mode} mode). Saving to {OUT_DIR}. Test page: http://<board>:7000")
App.run(user_loop=trap.loop)
