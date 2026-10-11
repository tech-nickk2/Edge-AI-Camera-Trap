"""
Build Edge Impulse-ready camera trap datasets from LILA BC, focused on Kenya.

Two tasks, matching the two-stage camera trap design:

  detector  Stage 1: full frames with animal bounding boxes, plus empty frames
            (false triggers). Writes an Edge Impulse bounding_boxes.labels file per
            split, so each folder can be dragged straight into Studio (Data
            acquisition > Upload data). The person and vehicle classes come from
            synth_paste.py instead: public data has almost no usable Kenyan
            people/vehicles (LILA removes images of people).

  species   Stage 2: animal crops cut out using the boxes, named "<species>.<id>.jpg"
            so Studio can infer the label from the file name.

Sources (--sources):
  eno        Snapshot Enonkishu S1, Maasai Mara, KENYA. Species labels + MegaDetector
             v5a boxes. Good source of empty frames and livestock (its human
             images are not public).
  mara       Biome Health Project Maasai Mara 2018, KENYA. Species labels +
             MegaDetector v1000 boxes. Blanks and people were removed by LILA.
  serengeti  Snapshot Serengeti S1-S6, Tanzania (same ecosystem as the Mara).
             Human-drawn animal boxes; species labels for S1.

Frames containing any person or vehicle detection are left out, so the model
never sees an unlabelled or wrongly labelled person/vehicle.

Labels are only trusted when the dataset label and the boxes agree, e.g. an
"empty" frame must be labelled empty by humans AND have no MegaDetector
detection above --empty-max-conf. Train/test is split by camera location so the
test set measures performance at sites the model has never seen.

Examples:
  # stage 1 detector dataset, 300 animal + 300 empty frames
  python lila_sample_puller.py detector --per-class 300 --outdir ./ei_detector

  # then add synthetic person/vehicle frames (see synth_paste.py)
  python synth_paste.py --per-class 300 --outdir ./ei_synth

  # stage 2 species crops: top 14 species (gazelles merged) + an other_animal class, 600 each
  python lila_sample_puller.py species --top-species 14 --per-species 600 --other \
      --merge "gazellegrants,gazellethomsons=gazelle" --outdir ./ei_species_v2

  Species crops skip cut-off animals at the frame edge and boxes smaller than
  --min-crop-px, and are padded by --pad for context. Species that can't fill 90%
  of --per-species are skipped when auto-picking; failed downloads are retried.

  # only Kenyan data, specific species
  python lila_sample_puller.py species --sources eno,mara --species zebra,wildebeest,elephant,lion,livestock --outdir ./ei_species

Requires Pillow (pip install pillow).
"""

import argparse
import collections
import hashlib
import io
import itertools
import json
import math
import os
import random
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from PIL import Image

# LILA retired the old lilablobssc Azure account (returns 403); data now lives on GCS
LILA_GCS = "https://storage.googleapis.com/public-datasets-lila"
LILA_MD = "https://lila.science/public/lila-md-results"
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".lila_cache")

DETECTOR_CLASSES = ["animal", "empty"]  # person/vehicle come from synth_paste.py

# Normalise species names across datasets (after lowercasing and removing "_" and " ")
SPECIES_ALIASES = {
    "lionmale": "lion", "lionfemale": "lion", "lioncub": "lion",
    "hyenaspotted": "hyena", "hyenastriped": "hyena", "hyenabrown": "hyena",
    "vervetmonkey": "monkeyvervet",
    "batearedfox": "foxbateared",
    "hartebeestcokes": "hartebeest",
    "jackalblackbacked": "jackal",
    "haresavannah": "hare",
    "birdother": "bird", "otherbird": "bird",
    "cattle": "livestock", "shoat": "livestock", "domesticanimal": "livestock",
    "domesticdog": "dog",
    "gazellethomsons": "gazellethomsons", "gazellegrants": "gazellegrants",
}
NON_SPECIES = {"empty", "blank", "human", "person", "vehicle", "car", "fire", "unknown",
               "animal", "group", "insectspider", "reptiles", "rodents"}


# --------------------------------------------------------------------------- IO

def download_bytes(url, timeout=60, attempts=3):
    """GET with retries for timeouts / server errors (slow links drop connections);
    404/403 are permanent (image removed by LILA) and fail straight away."""
    req = urllib.request.Request(url, headers={"User-Agent": "lila-sample-puller/2.0"})
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code < 500 and e.code != 429 or attempt == attempts - 1:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
            if attempt == attempts - 1:
                raise
        time.sleep(3 * (attempt + 1))


def load_json(url):
    """Download (with on-disk cache) a .json or single-json .zip."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, url.rsplit("/", 1)[-1])
    if not os.path.exists(path):
        print(f"  downloading {url}")
        data = download_bytes(url, timeout=600)
        with open(path + ".part", "wb") as f:
            f.write(data)
        os.replace(path + ".part", path)
    if path.endswith(".zip"):
        zf = zipfile.ZipFile(path)
        name = [n for n in zf.namelist() if n.endswith(".json")][0]
        return json.loads(zf.read(name))
    with open(path, "rb") as f:
        return json.load(f)


# ---------------------------------------------------------------- data model
#
# Every source is turned into a list of candidate records:
#   {"key": unique id, "url": image url, "location": "src/loc",
#    "species": set of normalised species names (image-level labels),
#    "boxes": [(cls, conf, x, y, w, h)] normalised 0-1, cls in animal/person/vehicle,
#    "max_conf": highest MD confidence of any detection (0 if none, 1 for human boxes),
#    "empty_label": True if humans labelled the frame empty}

def norm_species(name):
    n = name.lower().replace("_", "").replace(" ", "")
    return SPECIES_ALIASES.get(n, n)


def coco_image_labels(coco):
    """image_id -> set of normalised category names from a COCO Camera Traps file."""
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    labels = collections.defaultdict(set)
    for a in coco["annotations"]:
        labels[a["image_id"]].add(norm_species(cats[a["category_id"]]))
    return labels


def iou(a, b):
    """Intersection over union of two (cls, conf, x, y, w, h) boxes."""
    inter = overlap_frac(a, b) * a[4] * a[5]
    return inter / (a[4] * a[5] + b[4] * b[5] - inter + 1e-9)


def md_boxes(md_results, conf_thresh, review_conf):
    """file -> (boxes above conf_thresh, max conf of any detection) from MD results.
    Person/vehicle boxes are kept down to review_conf: they're rare, so worth a human look."""
    cls_map = md_results["detection_categories"]
    out = {}
    for im in md_results["images"]:
        dets = im.get("detections") or []
        boxes = [(cls_map[d["category"]], d["conf"], *d["bbox"]) for d in dets
                 if d["conf"] >= conf_thresh
                 or (cls_map[d["category"]] in ("person", "vehicle") and d["conf"] >= review_conf)]
        # MD often puts an animal AND a person box on the same close-up; keep the more confident
        boxes.sort(key=lambda b: -b[1])
        boxes = [b for i, b in enumerate(boxes) if not any(iou(b, k) > 0.7 for k in boxes[:i])]
        out[im["file"]] = (boxes, max((d["conf"] for d in dets), default=0.0))
    return out


def load_md_source(name, meta_url, md_url, image_base, args):
    coco = load_json(meta_url)
    md = md_boxes(load_json(md_url), args.box_conf, min(args.box_conf, getattr(args, "review_conf", 1.0)))
    labels = coco_image_labels(coco)
    records = []
    for im in coco["images"]:
        fn = im["file_name"]
        if fn not in md:  # MD didn't process it (corrupt / missing)
            continue
        boxes, max_conf = md[fn]
        lab = labels.get(im["id"], set())
        records.append({
            "key": f"{name}_{fn}".replace("/", "_"),
            "url": f"{image_base}/{fn}",
            "location": f"{name}/{im.get('location', 'unknown')}",
            "species": {s for s in lab if s not in NON_SPECIES},
            "human_label": bool(lab & {"human", "person"}),
            "empty_label": lab <= {"empty", "blank"} and bool(lab),
            "boxes": boxes,
            "max_conf": max_conf,
            "size": (im.get("width"), im.get("height")),
        })
    return records


def load_eno(args):
    return load_md_source(
        "eno",
        f"{LILA_GCS}/snapshot-safari/ENO/SnapshotEnonkishu_S1_v1.0.json.zip",
        f"{LILA_MD}/snapshot-safari_ENO_mdv5a.0.0_results.json.zip",
        f"{LILA_GCS}/snapshot-safari/ENO/ENO_public",
        args)


def load_mara(args):
    return load_md_source(
        "mara",
        f"{LILA_GCS}/biome-health-project-maasai-mara-2018/biome-health-project-maasai-mara-2018.json",
        f"{LILA_MD}/biome-health-mara-2025-11-11-v1000.0.0-redwood_detections.zip",
        f"{LILA_GCS}/biome-health-project-maasai-mara-2018",
        args)


def overlap_frac(a, b):
    """Fraction of box a's area that lies inside box b (boxes as (cls, conf, x, y, w, h))."""
    ix = max(0.0, min(a[2] + a[4], b[2] + b[4]) - max(a[2], b[2]))
    iy = max(0.0, min(a[3] + a[5], b[3] + b[5]) - max(a[3], b[3]))
    return ix * iy / (a[4] * a[5] + 1e-9)


def load_bbox_coco_source(name, coco, image_base, species_labels=None):
    """COCO file with human-drawn pixel boxes (Serengeti bbox subset)."""
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    images = {im["id"]: im for im in coco["images"]}
    boxes = collections.defaultdict(list)
    empty = set()
    for a in coco["annotations"]:
        if a["image_id"] not in images:  # image filtered out by the caller
            continue
        cls = cats[a["category_id"]]
        if cls == "empty":
            empty.add(a["image_id"])
            continue
        if cls == "group":  # boxes around herds, ambiguous for training
            boxes[a["image_id"]].append(None)
            continue
        im = images[a["image_id"]]
        x, y, w, h = a["bbox"]
        boxes[a["image_id"]].append((cls, 1.0, x / im["width"], y / im["height"], w / im["width"], h / im["height"]))
    records = []
    for iid, im in images.items():
        bxs = boxes.get(iid, [])
        if None in bxs:
            continue
        lab = (species_labels or {}).get(iid, set())
        records.append({
            "key": f"{name}_{im['file_name']}".replace("/", "_"),
            "url": f"{image_base}/{im['file_name']}",
            "location": f"{name}/{im.get('location', 'unknown')}",
            "species": {s for s in lab if s not in NON_SPECIES},
            "human_label": any(b[0] == "person" for b in bxs),
            "empty_label": iid in empty and not bxs,
            "boxes": bxs,
            "max_conf": 1.0 if bxs else 0.0,
            "size": (im.get("width"), im.get("height")),
        })
    return records


def load_serengeti(args):
    coco = load_json(f"{LILA_GCS}/snapshotserengeti-v-2-0/SnapshotSerengetiBboxes_20190903.json.zip")
    species = None
    if args.task == "species":
        # species labels for season 1 only; all seasons would need several GB of RAM
        species = coco_image_labels(load_json(f"{LILA_GCS}/snapshotserengeti-v-2-0/SnapshotSerengetiS01.json.zip"))
        keep = set(species)
        coco = dict(coco, images=[im for im in coco["images"] if im["id"] in keep])
    records = load_bbox_coco_source("serengeti", coco, f"{LILA_GCS}/snapshotserengeti-unzipped", species)
    # LILA took all Serengeti images containing people offline (they 404)
    return [r for r in records if not r["human_label"]]


SOURCES = {"eno": load_eno, "mara": load_mara, "serengeti": load_serengeti}


# ------------------------------------------------------------- selection

def split_for(location, test_frac):
    """Deterministic train/test assignment per camera location."""
    h = int(hashlib.md5(location.encode()).hexdigest(), 16) % 1000
    return "testing" if h < test_frac * 1000 else "training"


def detector_class(rec, args):
    """Pick the single class this frame is a candidate for, or None if ambiguous."""
    boxes = rec["boxes"]
    if not boxes:
        return "empty" if rec["empty_label"] and rec["max_conf"] < args.empty_max_conf else None
    if min(b[4] * b[5] for b in boxes) < args.min_box_area:
        return None  # tiny objects the model can't learn at low resolution
    classes = {b[0] for b in boxes}
    # person/vehicle are synthetic-only; a frame with a real (possibly wrong) person or
    # vehicle box would teach the model an unlabelled or mislabelled example, so skip it
    if classes & {"person", "vehicle"} or rec["human_label"]:
        return None
    # labels must agree with boxes: no animal boxes on frames humans said were empty
    if rec["empty_label"]:
        return None
    return "animal"


def review_candidate(rec):
    """Frames where MegaDetector (not a human) found a person or vehicle: in ENO/Mara these are
    the only Kenyan examples of those classes, but too unreliable to train on unchecked."""
    # boxes filling most of the frame are almost always animal close-ups / fur on the lens
    return any(b[0] in ("person", "vehicle") and b[1] < 1.0 and b[4] * b[5] < 0.5 for b in rec["boxes"])


def is_night(img):
    """IR night shots are (near) greyscale: R, G and B channels almost identical."""
    small = img.convert("RGB").resize((64, 64))
    px = list(small.getdata())
    diff = sum(abs(r - g) + abs(g - b) for r, g, b in px) / len(px)
    return diff < 6


def fetch_image(url, max_size):
    img = Image.open(io.BytesIO(download_bytes(url))).convert("RGB")
    img.thumbnail((max_size, max_size))
    return img


def resumable(fetch, path_of):
    """Reuse images already saved by an earlier (interrupted) run instead of downloading."""
    def wrapped(rec):
        path = path_of(rec)
        if os.path.exists(path):
            with Image.open(path) as im:
                return im.convert("RGB")
        return fetch(rec)
    return wrapped


def sample_balanced(cands, quota, fetch, label, workers):
    """Download shuffled candidates (in parallel) until `quota` succeed, aiming for half
    day / half night. Frames that would overfill one half are kept aside and used to
    top up at the end."""
    got, spare, skipped = [], [], 0
    per_half = {"day": quota - quota // 2, "night": quota // 2}
    todo = iter(cands)
    with ThreadPoolExecutor(workers) as pool:
        pending = {}

        def refill():
            # only a little more in flight than still needed, so slow links aren't wasted
            while len(pending) < min(workers, quota - len(got) + 2):
                rec = next(todo, None)
                if rec is None:
                    return
                pending[pool.submit(fetch, rec)] = rec

        refill()
        while pending and len(got) < quota:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for fut in done:
                rec = pending.pop(fut)
                try:
                    img = fut.result()
                except Exception as e:
                    skipped += 1
                    if skipped <= 3:
                        print(f"    skip {rec['key']} ({e})")
                    continue
                if len(got) >= quota:
                    continue
                half = "night" if is_night(img) else "day"
                if per_half[half] > 0:
                    per_half[half] -= 1
                    got.append((rec, img, half))
                    print(f"  [{label}] {len(got)}/{quota} {half:5s} {rec['key']}")
                elif len(spare) < quota:
                    spare.append((rec, img, half))
            if len(got) < quota:
                refill()
        for fut in pending:
            fut.cancel()
    while len(got) < quota and spare:
        got.append(spare.pop())
    if skipped:
        print(f"  [{label}] skipped {skipped} unavailable images")
    if len(got) < quota:
        print(f"  [{label}] WARNING: only {len(got)}/{quota} available")
    return got


# ------------------------------------------------------------------ tasks

def out_path(args, rec, prefix=""):
    split = split_for(rec["location"], args.test_frac)
    os.makedirs(os.path.join(args.outdir, split), exist_ok=True)
    return os.path.join(args.outdir, split, prefix + os.path.splitext(rec["key"])[0] + ".jpg")


def run_detector(records, args):
    by_class = collections.defaultdict(list)
    for r in records:
        c = detector_class(r, args)
        if c:
            by_class[c].append(r)
    print("\nCandidate frames per class:", {c: len(by_class[c]) for c in DETECTOR_CLASSES})

    labels = {"training": {}, "testing": {}}
    stats = collections.Counter()
    for cls in DETECTOR_CLASSES if args.per_class > 0 else []:  # --per-class 0 = review/ only
        cands = by_class[cls]
        random.shuffle(cands)
        print(f"\nSampling {cls} ...")
        fetch = resumable(lambda r: fetch_image(r["url"], args.max_size), lambda r: out_path(args, r))
        picked = sample_balanced(cands, args.per_class, fetch, cls, args.workers)
        for rec, img, half in picked:
            split = split_for(rec["location"], args.test_frac)
            fname = os.path.basename(out_path(args, rec))
            img.save(out_path(args, rec), quality=90)
            W, H = img.size
            labels[split][fname] = [
                {"label": b[0], "x": round(b[2] * W), "y": round(b[3] * H),
                 "width": round(b[4] * W), "height": round(b[5] * H)}
                for b in rec["boxes"]]
            stats[(cls, split, half)] += 1

        # written after every class so an interrupted run still leaves usable labels
        for split, bb in labels.items():
            if bb:
                with open(os.path.join(args.outdir, split, "bounding_boxes.labels"), "w") as f:
                    json.dump({"version": 1, "type": "bounding-box-labels", "boundingBoxes": bb}, f)
    print_stats(stats, "class")

    if args.review_conf < 1.0:
        export_review([r for r in records if review_candidate(r)], args)


def export_review(cands, args):
    """Write Kenyan MD person/vehicle candidates to <outdir>/review with their boxes, to be
    checked and fixed in Studio before being moved into training."""
    if not cands:
        return
    review_dir = os.path.join(args.outdir, "review")
    os.makedirs(review_dir, exist_ok=True)
    path_of = lambda r: os.path.join(review_dir, os.path.splitext(r["key"])[0] + ".jpg")
    print(f"\nExporting {len(cands)} person/vehicle frames for review ...")
    fetch = resumable(lambda r: fetch_image(r["url"], args.max_size), path_of)
    labels = {}
    for rec, img, half in sample_balanced(cands, len(cands), fetch, "review", args.workers):
        img.save(path_of(rec), quality=90)
        W, H = img.size
        labels[os.path.basename(path_of(rec))] = [
            {"label": b[0], "x": round(b[2] * W), "y": round(b[3] * H),
             "width": round(b[4] * W), "height": round(b[5] * H)}
            for b in rec["boxes"]]
    with open(os.path.join(review_dir, "bounding_boxes.labels"), "w") as f:
        json.dump({"version": 1, "type": "bounding-box-labels", "boundingBoxes": labels}, f)
    print(f"  {len(labels)} frames in {review_dir}: check/fix these in Studio before training on them")


OTHER_LABEL = "other_animal"


def parse_merge(spec):
    """"gazellegrants,gazellethomsons=gazelle;topi,hartebeest=topi_hartebeest" -> {src: dst}"""
    out = {}
    for group in filter(None, (spec or "").split(";")):
        srcs, dst = group.split("=")
        for src in srcs.split(","):
            out[norm_species(src)] = dst.strip()
    return out


def good_crop_boxes(rec, args):
    """Animal boxes worth cropping: confident, not cut off by the frame edge, and big
    enough in the original photo that the crop isn't just blur."""
    W, H = rec["size"]
    m = args.edge_margin
    out = []
    for b in rec["boxes"]:
        if b[0] != "animal" or b[1] < args.crop_conf or b[4] * b[5] < args.min_box_area:
            continue
        if b[2] < m or b[3] < m or b[2] + b[4] > 1 - m or b[3] + b[5] > 1 - m:
            continue  # partial animal at the frame edge
        if W and H and max(b[4] * W, b[5] * H) < args.min_crop_px:
            continue
        out.append(b)
    return out


def parse_overrides(spec, merge):
    """"wildebeest=900,impala=900" -> {"wildebeest": 900, ...} (names merged like --merge)."""
    out = {}
    for item in filter(None, (spec or "").split(",")):
        name, n = item.split("=")
        name = name.strip()
        name = name if name == OTHER_LABEL else merge.get(norm_species(name), norm_species(name))
        out[name] = int(n)
    return out


def run_species(records, args):
    merge = parse_merge(args.merge)
    by_species = collections.defaultdict(list)
    for r in records:
        # one species per frame (labels are per frame, so a crop's species is only
        # certain when there's just one), confirmed by a good animal box
        if len(r["species"]) != 1 or r["empty_label"] or r["human_label"]:
            continue
        boxes = good_crop_boxes(r, args)
        if boxes:
            sp = next(iter(r["species"]))
            by_species[merge.get(sp, sp)].append(dict(r, _boxes=boxes))

    overrides = parse_overrides(args.per_class_override, merge)
    quota_of = lambda sp: overrides.get(sp, args.per_species)
    need_of = lambda sp: math.ceil(args.min_fill * quota_of(sp))
    need = need_of(None)
    ranked = sorted(by_species, key=lambda s: -len(by_species[s]))
    print("\nAvailable species (usable frames):", ", ".join(f"{s}={len(by_species[s])}" for s in ranked))
    if args.species:
        wanted = [merge.get(norm_species(s), norm_species(s)) for s in args.species.split(",")]
        wanted = list(dict.fromkeys(wanted))  # merged names may repeat
        missing = [s for s in wanted if s not in by_species]
        if missing:
            sys.exit(f"Species not found in selected sources: {', '.join(missing)}")
        short = [f"{s}={len(by_species[s])} (need {need_of(s)})" for s in wanted
                 if len(by_species[s]) < need_of(s)]
        if short:
            print(f"WARNING: not enough usable frames for: {', '.join(short)} "
                  f"(lower --per-species / the override, or drop/merge these)")
    else:
        # only species that can actually fill ~90% of the quota
        eligible = [s for s in ranked if len(by_species[s]) >= need]
        wanted = eligible[:args.top_species]
        if len(wanted) < args.top_species:
            print(f"WARNING: only {len(wanted)} species have >= {need} usable frames")
    print("Using species:", ", ".join(f"{s}({quota_of(s)})" for s in wanted))
    unused = [s for s in overrides if s not in wanted and s != OTHER_LABEL]
    if unused:
        print(f"WARNING: --per-class-override names classes not in use: {', '.join(unused)}")
    if not args.species:  # auto-picked species already passed the default check
        for sp in wanted:
            if sp in overrides and len(by_species[sp]) < need_of(sp):
                print(f"WARNING: {sp} has only {len(by_species[sp])} usable frames for an override of {quota_of(sp)}")

    jobs = []
    for sp in wanted:
        cands = by_species[sp]
        random.shuffle(cands)
        jobs.append((sp, cands))
    if args.other:
        # round-robin across all remaining species so no single one dominates "other"
        rest = [by_species[s][:] for s in ranked if s not in wanted]
        for lst in rest:
            random.shuffle(lst)
        pool = [r for tup in itertools.zip_longest(*rest) for r in tup if r is not None]
        print(f"{OTHER_LABEL}: {len(pool)} frames from {len(rest)} other species")
        jobs.append((OTHER_LABEL, pool))

    def fetch(rec):
        full = Image.open(io.BytesIO(download_bytes(rec["url"]))).convert("RGB")
        # largest good box, padded so the classifier sees posture and surroundings
        b = max(rec["_boxes"], key=lambda b: b[4] * b[5])
        W, H = full.size
        px, py = b[4] * args.pad, b[5] * args.pad
        crop = full.crop((max(0, (b[2] - px) * W), max(0, (b[3] - py) * H),
                          min(W, (b[2] + b[4] + px) * W), min(H, (b[3] + b[5] + py) * H)))
        crop.thumbnail((args.crop_size, args.crop_size))
        return crop

    # crops created by THIS run are also copied to <outdir>/new/<split>, so a top-up run
    # (e.g. after raising an override) can be uploaded without re-uploading everything
    new_dir = os.path.join(args.outdir, "new")
    shutil.rmtree(new_dir, ignore_errors=True)
    stats = collections.Counter()
    shortfall, n_new = [], 0
    for sp, cands in jobs:
        print(f"\nSampling {sp} ...")
        path_of = lambda r, sp=sp: out_path(args, r, prefix=f"{sp}.")
        picked = sample_balanced(cands, quota_of(sp), resumable(fetch, path_of), sp, args.workers)
        for rec, img, half in picked:
            split = split_for(rec["location"], args.test_frac)
            path = path_of(rec)
            if not os.path.exists(path):
                img.save(path, quality=90)
                os.makedirs(os.path.join(new_dir, split), exist_ok=True)
                shutil.copy2(path, os.path.join(new_dir, split))
                n_new += 1
            stats[(sp, split, half)] += 1
        if len(picked) < need_of(sp):
            shortfall.append(f"{sp}={len(picked)}/{quota_of(sp)}")
    print_stats(stats, "species")
    print(f"\n{n_new} new crops this run (also copied to {new_dir} for uploading).")
    if shortfall:
        print(f"\nWARNING: below {int(args.min_fill * 100)}% of target: {', '.join(shortfall)}. "
              f"Their candidate frames ran out; rerun the same command to retry failed downloads, "
              f"or lower the target / merge these classes.")


def print_stats(stats, what):
    print(f"\nSummary ({what}: training day/night | testing day/night)")
    for name in sorted({k[0] for k in stats}):
        s = lambda sp, h: stats[(name, sp, h)]
        print(f"  {name:18s} {s('training','day'):4d}/{s('training','night'):<4d} | {s('testing','day'):4d}/{s('testing','night'):<4d}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("task", choices=["detector", "species"])
    parser.add_argument("--sources", default="eno,mara,serengeti",
                        help=f"comma-separated, from: {', '.join(SOURCES)}")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--test-frac", type=float, default=0.2, help="fraction of camera locations held out for testing")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8, help="parallel image downloads")
    parser.add_argument("--box-conf", type=float, default=0.5, help="min MegaDetector confidence to keep a box")
    parser.add_argument("--min-box-area", type=float, default=0.002, help="drop frames with boxes smaller than this fraction of the image")
    # detector
    parser.add_argument("--per-class", type=int, default=300, help="detector: frames per class")
    parser.add_argument("--empty-max-conf", type=float, default=0.1, help="detector: empty frames must have no MD detection above this")
    parser.add_argument("--max-size", type=int, default=640, help="detector: longest image side after resizing")
    parser.add_argument("--review-conf", type=float, default=1.0,
                        help="detector: export Kenyan MD person/vehicle frames above this conf to review/, "
                             "e.g. 0.3 (default 1.0 = off). Reviewed real frames make good TEST data.")
    # species
    parser.add_argument("--species", help="species: comma-separated list (default: most common)")
    parser.add_argument("--top-species", type=int, default=15)
    parser.add_argument("--per-species", type=int, default=150)
    parser.add_argument("--crop-conf", type=float, default=0.8, help="species: min box confidence for crops")
    parser.add_argument("--crop-size", type=int, default=320)
    parser.add_argument("--pad", type=float, default=0.25, help="species: context padding around the box")
    parser.add_argument("--min-crop-px", type=int, default=100,
                        help="species: min longer side of the box in the original photo, in pixels")
    parser.add_argument("--edge-margin", type=float, default=0.01,
                        help="species: skip boxes within this fraction of the frame edge (cut-off animals)")
    parser.add_argument("--merge", help='species: merge look-alikes, e.g. "gazellegrants,gazellethomsons=gazelle"')
    parser.add_argument("--other", action="store_true",
                        help=f"species: add an '{OTHER_LABEL}' class drawn evenly from all other species")
    parser.add_argument("--per-class-override",
                        help='species: per-class targets overriding --per-species, e.g. "wildebeest=900,impala=900"')
    parser.add_argument("--min-fill", type=float, default=0.9,
                        help="species: only auto-pick species that can fill this share of --per-species")
    args = parser.parse_args()

    random.seed(args.seed)
    records = []
    for src in args.sources.split(","):
        if src not in SOURCES:
            sys.exit(f"Unknown source '{src}'. Choose from: {', '.join(SOURCES)}")
        print(f"Loading {src} ...")
        recs = SOURCES[src](args)
        print(f"  {len(recs)} frames")
        records.extend(recs)

    if args.task == "detector":
        run_detector(records, args)
    else:
        run_species(records, args)
    print(f"\nDone. Upload {args.outdir}/training and {args.outdir}/testing in Edge Impulse Studio "
          f"(Data acquisition > Upload data, pick the matching category).")


if __name__ == "__main__":
    main()
