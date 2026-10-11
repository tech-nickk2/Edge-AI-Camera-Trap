"""
Create synthetic person / vehicle training frames for the Stage 1 detector by pasting
real COCO cutouts onto real, empty Kenyan camera trap frames.

Why: public camera trap data has almost no usable Kenyan people or vehicles (LILA
removes images of people; NACTI is US farmland). Copy-paste is an established way to
fill rare classes (Dwibedi 2017 "Cut, Paste and Learn"; Ghiasi CVPR 2021 "Simple
Copy-Paste"; Beery WACV 2020 "Synthetic Examples Improve Generalization for Rare
Classes"). See docs/notes_synthetic_data.md.

Precautions built in:
  - backgrounds: empty frames from Snapshot Enonkishu + Biome Health Maasai Mara
    (Kenya), half day / half night, only from TRAINING camera sites
  - cutouts: full-body, untruncated, unoccluded COCO objects with polygon masks;
    motorcycles keep their riders (rider = person box, bike = vehicle box)
  - feathered mask edges, brightness/contrast matched to the background, grayscale on
    night IR frames, slight blur, sensor noise and JPEG re-compression
  - objects stand on the ground and shrink towards the horizon
  - animals are pasted too (label "animal"), so "pasted" is never a class cue
  - output is TRAINING data only; always test on real frames

Output (drag into Edge Impulse Studio as Training):
  <outdir>/training/synth_*.jpg + bounding_boxes.labels
  <outdir>/preview/   the first --preview images with boxes drawn, for checking

Example:
  python synth_paste.py --per-class 300 --outdir ./ei_synth

First run downloads the COCO 2017 annotations (~250 MB, cached in .lila_cache/).
Requires Pillow and numpy.
"""

import argparse
import collections
import io
import json
import os
import random
import sys
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import lila_sample_puller as lila

COCO_ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
CUTOUT_DIR = os.path.join(lila.CACHE_DIR, "synth_cutouts")
BG_DIR = os.path.join(lila.CACHE_DIR, "synth_backgrounds")

# COCO category -> (our label, sampling weight within the group)
GROUPS = {
    "person": {"person": 1.0},
    # boda bodas are the most common vehicle around Kenyan conservancies; no buses/trains
    "vehicle": {"motorcycle": 0.45, "car": 0.30, "truck": 0.25},
    # East African-looking animals, pasted so that pasting itself isn't a person/vehicle cue
    "animal": {"elephant": 1, "zebra": 1, "giraffe": 1, "cow": 1.5, "sheep": 1, "horse": 0.5, "dog": 0.5},
}
LABEL_OF = {"person": "person", "motorcycle": "vehicle", "car": "vehicle", "truck": "vehicle"}

# object height as a fraction of image height, far (near horizon) -> near (bottom of frame)
HEIGHT_RANGE = {"person": (0.10, 0.55), "vehicle": (0.08, 0.45), "animal": (0.08, 0.50)}
GROUND_Y = (0.45, 0.93)  # where objects stand; stays above the camera's timestamp strip


# ----------------------------------------------------------------- COCO

def download_to_file(url, path):
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"Downloading {url} (one-time) ...")
    req = urllib.request.Request(url, headers={"User-Agent": "synth-paste/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(path + ".part", "wb") as f:
        total, done = int(resp.headers.get("Content-Length", 0)), 0
        while chunk := resp.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e6:.0f}/{total / 1e6:.0f} MB", end="", flush=True)
    print()
    os.replace(path + ".part", path)


def load_coco(split):
    path = os.path.join(lila.CACHE_DIR, "annotations_trainval2017.zip")
    download_to_file(COCO_ANN_URL, path)
    with zipfile.ZipFile(path) as zf:
        return json.loads(zf.read(f"annotations/instances_{split}.json"))


def usable(a, im, min_h, min_fill):
    """Whole, clearly visible object: polygon mask, not cut by the image border, big enough."""
    if a["iscrowd"] or not isinstance(a["segmentation"], list):
        return False
    x, y, w, h = a["bbox"]
    if h < min_h or w < 12:
        return False
    if x < 2 or y < 2 or x + w > im["width"] - 2 or y + h > im["height"] - 2:
        return False
    return a["area"] / (w * h) >= min_fill


def box_overlap(a, b):
    """Fraction of COCO bbox a covered by bbox b."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    return ix * iy / (aw * ah + 1e-9)


def plan_cutouts(coco, per_group, rng):
    """Pick COCO annotations to cut out. Returns {group: [plan]}, plan = dict with the
    image, the annotations forming the cutout and their labels."""
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    sports = {c["id"] for c in coco["categories"] if c["supercategory"] == "sports"}
    images = {im["id"]: im for im in coco["images"]}
    by_image = collections.defaultdict(list)
    for a in coco["annotations"]:
        by_image[a["image_id"]].append(a)

    plans = {g: [] for g in GROUPS}
    for iid, anns in by_image.items():
        im = images[iid]
        # skiers, tennis players, surfers... odd poses/outfits for a Kenyan camera trap
        if any(a["category_id"] in sports for a in anns):
            continue
        for a in anns:
            name = cats[a["category_id"]]
            group = next((g for g, c in GROUPS.items() if name in c), None)
            if group is None:
                continue
            if group == "person":
                ok = usable(a, im, 80, 0.30) and 1.3 <= a["bbox"][3] / a["bbox"][2] <= 5  # standing/walking
            else:
                ok = usable(a, im, 50, 0.40)
            if not ok:
                continue
            parts = [a]
            if name == "motorcycle":
                # keep riders: they're part of what a camera sees, labelled as person
                riders = [o for o in anns if cats[o["category_id"]] == "person"
                          and box_overlap(o["bbox"], a["bbox"]) > 0.3 and usable(o, im, 30, 0.2)]
                parts += riders
            # anything else overlapping would leave holes or stray pixels in the cutout
            others = [o for o in anns if o not in parts and box_overlap(a["bbox"], o["bbox"]) > 0.05]
            if others:
                continue
            plans[group].append({"id": a["id"], "coco_name": name, "image": im, "parts": parts,
                                 "labels": [LABEL_OF.get(cats[p["category_id"]], "animal") for p in parts]})

    for g in plans:
        # weighted sample within the group (e.g. more motorcycles than trucks)
        weights = GROUPS[g]
        rng.shuffle(plans[g])
        plans[g].sort(key=lambda p: -weights[p["coco_name"]] * rng.random())
        plans[g] = plans[g][:per_group]
        print(f"  {g}: {len(plans[g])} cutouts "
              f"({dict(collections.Counter(p['coco_name'] for p in plans[g]))})")
    return plans


def make_cutout(plan):
    """Download the COCO image and return (RGBA cutout, [(label, x, y, w, h)] in cutout px)."""
    png = os.path.join(CUTOUT_DIR, f"{plan['id']}.png")
    meta = png[:-4] + ".json"
    if os.path.exists(png) and os.path.exists(meta):
        with open(meta) as f:
            return Image.open(png).copy(), [tuple(b) for b in json.load(f)]
    img = Image.open(io.BytesIO(lila.download_bytes(plan["image"]["coco_url"]))).convert("RGB")
    if lila.is_night(img):  # black-and-white photo: looks wrong pasted into colour daytime frames
        raise ValueError("grayscale COCO photo")
    mask = Image.new("L", img.size, 0)
    draw = ImageDraw.Draw(mask)
    for p in plan["parts"]:
        for poly in p["segmentation"]:
            if len(poly) >= 6:
                draw.polygon(poly, fill=255)
    x0 = min(p["bbox"][0] for p in plan["parts"])
    y0 = min(p["bbox"][1] for p in plan["parts"])
    x1 = max(p["bbox"][0] + p["bbox"][2] for p in plan["parts"])
    y1 = max(p["bbox"][1] + p["bbox"][3] for p in plan["parts"])
    crop = (int(x0), int(y0), int(x1 + 1), int(y1 + 1))
    rgba = img.crop(crop)
    rgba.putalpha(mask.crop(crop))
    boxes = [(lab, p["bbox"][0] - crop[0], p["bbox"][1] - crop[1], p["bbox"][2], p["bbox"][3])
             for lab, p in zip(plan["labels"], plan["parts"])]
    os.makedirs(CUTOUT_DIR, exist_ok=True)
    rgba.save(png)
    with open(meta, "w") as f:
        json.dump(boxes, f)
    return rgba, boxes


def fetch_cutouts(plans, workers):
    out = {}
    for g, items in plans.items():
        print(f"Preparing {g} cutouts ...")
        with ThreadPoolExecutor(workers) as pool:
            results = list(pool.map(lambda p: _safe(make_cutout, p), items))
        out[g] = [r for r in results if r]
        print(f"  {len(out[g])} ready")
    return out


def _safe(fn, arg):
    try:
        return fn(arg)
    except Exception as e:
        print(f"    skip ({e})")
        return None


# ----------------------------------------------------------- backgrounds

def fetch_backgrounds(args):
    """Empty Kenyan frames from TRAINING camera sites only, half day / half night."""
    ns = argparse.Namespace(task="detector", box_conf=0.5, review_conf=1.0,
                            min_box_area=args.min_box_area, empty_max_conf=0.1)
    records = lila.load_eno(ns) + lila.load_mara(ns)
    empties = [r for r in records if lila.detector_class(r, ns) == "empty"
               and lila.split_for(r["location"], args.test_frac) == "training"]
    random.shuffle(empties)
    print(f"Backgrounds: {len(empties)} empty training-site frames available, using {args.backgrounds}")
    os.makedirs(BG_DIR, exist_ok=True)
    path_of = lambda r: os.path.join(BG_DIR, os.path.splitext(r["key"])[0] + ".jpg")
    fetch = lila.resumable(lambda r: lila.fetch_image(r["url"], args.max_size), path_of)
    picked = lila.sample_balanced(empties, args.backgrounds, fetch, "background", args.workers)
    bgs = []
    for rec, img, half in picked:
        if not os.path.exists(path_of(rec)):
            img.save(path_of(rec), quality=92)
        bgs.append((img, half == "night"))
    return bgs


# ------------------------------------------------------------ compositing

def luminance(arr):
    return arr[..., 0] * 0.299 + arr[..., 1] * 0.587 + arr[..., 2] * 0.114


def match_to_background(cut, bg_region, night, rng):
    """Pull the cutout's brightness/contrast (and colour cast) part-way towards the
    background frame, so it doesn't look lit by a different sun. Uses the whole frame:
    matching to the local patch over-tints objects (e.g. an orange elephant on dry grass)."""
    rgb = np.asarray(cut.convert("RGB"), dtype=np.float32)
    alpha = np.asarray(cut.getchannel("A"), dtype=np.float32) / 255
    if night:
        rgb = np.repeat(luminance(rgb)[..., None], 3, axis=2)
    m = alpha > 0.5
    if m.sum() < 20:
        return cut
    bg = np.asarray(bg_region.convert("RGB"), dtype=np.float32)
    lum_c, lum_b = luminance(rgb)[m], luminance(bg).ravel()
    mean_c, std_c = lum_c.mean(), lum_c.std() + 1e-3
    mean_b, std_b = lum_b.mean(), lum_b.std() + 1e-3
    k_std = rng.uniform(0.3, 0.6)    # how far to pull contrast
    k_mean = rng.uniform(0.3, 0.6)   # how far to pull brightness
    scale = 1 + k_std * (std_b / std_c - 1)
    target_mean = mean_c + k_mean * (mean_b - mean_c)
    rgb = (rgb - mean_c) * scale + target_mean
    if not night:  # nudge the colour cast towards the background's
        cast = bg.reshape(-1, 3).mean(0) - bg.reshape(-1, 3).mean()
        rgb += 0.15 * cast
    out = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
    out.putalpha(cut.getchannel("A"))
    return out


def paste_object(canvas, cutout, boxes, group, night, placed, rng):
    """Scale/place one cutout on the ground plane. Returns new label boxes, or None if it
    couldn't be placed without overlapping objects already pasted."""
    W, H = canvas.size
    for _ in range(15):
        ground = rng.uniform(*GROUND_Y)
        t = (ground - GROUND_Y[0]) / (GROUND_Y[1] - GROUND_Y[0])  # 0 = horizon, 1 = near camera
        lo, hi = HEIGHT_RANGE[group]
        target_h = H * (lo + (hi - lo) * t) * rng.uniform(0.8, 1.2)
        s = target_h / cutout.height
        w, h = max(1, int(cutout.width * s)), max(1, int(cutout.height * s))
        if w > 0.9 * W:
            continue
        x = rng.randint(0, W - w)
        y = int(ground * H) - h
        if y < 0:
            continue
        if any(not (x + w < px or px + pw < x or y + h < py or py + ph < y) for px, py, pw, ph in placed):
            continue
        break
    else:
        return None

    flip = rng.random() < 0.5
    cut = cutout.transpose(Image.FLIP_LEFT_RIGHT) if flip else cutout
    cut = cut.resize((w, h), Image.LANCZOS)
    cut = match_to_background(cut, canvas, night, rng)
    # camera traps are soft: COCO photos are much sharper than trap images at any distance
    cut = cut.filter(ImageFilter.GaussianBlur(rng.uniform(0.5, 1.1)))
    # soft contact shadow under the feet/wheels so objects don't look like they float
    shadow = Image.new("L", canvas.size, 0)
    sw, sh = w * rng.uniform(0.6, 0.9), max(3, h * 0.06)
    cx, cy = x + w / 2, y + h - sh * 0.3
    ImageDraw.Draw(shadow).ellipse((cx - sw / 2, cy - sh / 2, cx + sw / 2, cy + sh / 2),
                                   fill=int(255 * (0.12 if night else rng.uniform(0.25, 0.4))))
    shadow = shadow.filter(ImageFilter.GaussianBlur(max(2, sh / 2)))
    canvas.paste(Image.new("RGB", canvas.size, (0, 0, 0)), (0, 0), shadow)
    # feather the mask: shrink 1px (drops COCO background fringe), then soften the edge
    alpha = cut.getchannel("A").filter(ImageFilter.MinFilter(3)).filter(ImageFilter.GaussianBlur(1.0))
    canvas.paste(cut, (x, y), alpha)
    placed.append((x, y, w, h))

    out = []
    for lab, bx, by, bw, bh in boxes:
        if flip:
            bx = cutout.width - bx - bw
        out.append({"label": lab, "x": round(x + bx * s), "y": round(y + by * s),
                    "width": max(1, round(bw * s)), "height": max(1, round(bh * s))})
    return out


def camera_finish(img, rng):
    """Whole-frame sensor noise + JPEG re-compression so pasted and real pixels match."""
    arr = np.asarray(img, dtype=np.float32)
    arr += np.random.default_rng(rng.randrange(1 << 30)).normal(0, rng.uniform(1.5, 4.5), arr.shape)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=rng.randint(75, 92))
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def compose(target, bgs, cutouts, args, rng):
    bg, night = rng.choice(bgs)
    canvas = bg.copy()
    placed, labels = [], []
    n = {"person": rng.choices([1, 2, 3], [0.6, 0.25, 0.15])[0],
         "vehicle": rng.choices([1, 2], [0.85, 0.15])[0],
         "animal": rng.choices([1, 2, 3], [0.5, 0.3, 0.2])[0]}[target]
    todo = [target] * n
    if target != "animal" and rng.random() < args.distractor_prob:
        todo.append("animal")
    for group in todo:  # paste_object keeps objects from overlapping, so order doesn't matter
        cut, boxes = rng.choice(cutouts[group])
        res = paste_object(canvas, cut, boxes, group, night, placed, rng)
        if res:
            labels += res
    if not any(b["label"] == target for b in labels):
        return None
    W, H = canvas.size
    if any(b["width"] * b["height"] < args.min_box_area * W * H for b in labels):
        return None  # too small to learn at the model's input resolution
    return camera_finish(canvas, rng), labels, night


def draw_preview(img, labels, path):
    img = img.copy()
    d = ImageDraw.Draw(img)
    colours = {"person": (255, 60, 60), "vehicle": (60, 160, 255), "animal": (80, 230, 80)}
    for b in labels:
        d.rectangle((b["x"], b["y"], b["x"] + b["width"], b["y"] + b["height"]), outline=colours[b["label"]], width=2)
        d.text((b["x"] + 2, b["y"] + 2), b["label"], fill=colours[b["label"]])
    img.save(path, quality=90)


# ------------------------------------------------------------------ main

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--per-class", type=int, default=300, help="synthetic frames for person and for vehicle")
    parser.add_argument("--animal-count", type=int, default=None,
                        help="animal-only synthetic frames (default: half of --per-class)")
    parser.add_argument("--distractor-prob", type=float, default=0.3,
                        help="chance a person/vehicle frame also gets a pasted animal")
    parser.add_argument("--backgrounds", type=int, default=300, help="distinct Kenyan empty frames to use")
    parser.add_argument("--cutouts", type=int, default=400, help="max COCO cutouts per group")
    parser.add_argument("--coco-split", default="train2017", choices=["val2017", "train2017"],
                        help="train2017 has ~20x more usable cutouts (needs ~4 GB RAM); "
                             "val2017 is lighter but only has ~150 people / ~110 vehicles")
    parser.add_argument("--max-size", type=int, default=640)
    parser.add_argument("--min-box-area", type=float, default=0.002)
    parser.add_argument("--test-frac", type=float, default=0.2, help="must match lila_sample_puller.py")
    parser.add_argument("--preview", type=int, default=40, help="write this many images with boxes drawn")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    animal_count = args.per_class // 2 if args.animal_count is None else args.animal_count

    random.seed(args.seed)
    rng = random.Random(args.seed)

    bgs = fetch_backgrounds(args)
    if not bgs:
        sys.exit("No backgrounds downloaded.")
    print(f"Loading COCO {args.coco_split} annotations ...")
    plans = plan_cutouts(load_coco(args.coco_split), args.cutouts, rng)
    cutouts = fetch_cutouts(plans, args.workers)
    if not all(cutouts.values()):
        sys.exit("Missing cutouts for some group; check the network and retry.")

    train_dir = os.path.join(args.outdir, "training")
    prev_dir = os.path.join(args.outdir, "preview")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(prev_dir, exist_ok=True)
    labels_path = os.path.join(train_dir, "bounding_boxes.labels")
    labels = {}
    if os.path.exists(labels_path):  # resume
        with open(labels_path) as f:
            labels = json.load(f)["boundingBoxes"]

    def save_labels():
        with open(labels_path, "w") as f:
            json.dump({"version": 1, "type": "bounding-box-labels", "boundingBoxes": labels}, f)

    stats = collections.Counter()
    jobs = [("person", args.per_class), ("vehicle", args.per_class), ("animal", animal_count)]
    n_prev = 0
    for target, count in jobs:
        print(f"\nGenerating {count} {target} frames ...")
        i = 0
        while i < count:
            name = f"synth_{target}_{i:04d}.jpg"
            i += 1
            if name in labels and os.path.exists(os.path.join(train_dir, name)):
                continue
            res = None
            for _ in range(10):
                res = compose(target, bgs, cutouts, args, rng)
                if res:
                    break
            if not res:
                print(f"  could not place {target} objects, skipping {name}")
                continue
            img, boxes, night = res
            img.save(os.path.join(train_dir, name), quality=92)
            labels[name] = boxes
            stats[(target, "night" if night else "day")] += 1
            if n_prev < args.preview:
                draw_preview(img, boxes, os.path.join(prev_dir, name))
                n_prev += 1
            if i % 25 == 0:
                save_labels()
                print(f"  {i}/{count}")
        save_labels()

    print("\nSummary (new frames this run, day/night):")
    for t, _ in jobs:
        print(f"  {t:8s} {stats[(t, 'day')]:4d}/{stats[(t, 'night')]:<4d}")
    print(f"\nDone. Check {prev_dir}, then upload {train_dir} to Edge Impulse as TRAINING only.")


if __name__ == "__main__":
    main()
