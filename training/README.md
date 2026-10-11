# Training data scripts (run on a PC, not on the UNO Q)

These scripts build the Edge Impulse datasets used to train the two models. They download
images from [LILA BC](https://lila.science) and [COCO](https://cocodataset.org), so run them
on a laptop or desktop with a few GB of free disk space and a decent connection.
**Don't run them on the UNO Q**: it doesn't have the storage for the datasets.

```bash
pip install -r requirements.txt
```

## `lila_sample_puller.py`: real camera trap data

Downloads labelled frames from Kenyan (Snapshot Enonkishu, Biome Health Maasai Mara) and
Tanzanian (Snapshot Serengeti) datasets, using MegaDetector's published boxes as labels.

- Keeps only frames where the dataset's label and MegaDetector agree
- Balances day and night (infrared) frames
- Splits training/testing **by camera location**, so tests use sites the model has never seen
- Writes Edge Impulse-ready folders (`bounding_boxes.labels` for the detector, `<species>.<id>.jpg` for the classifier)
- Resumes if interrupted; failed downloads are retried

```bash
# Stage 1 detector: animal + empty frames
python lila_sample_puller.py detector --per-class 300 --outdir ./ei_detector

# Stage 2 species classifier: animal close-ups
python lila_sample_puller.py species --per-species 450 --other \
  --merge "gazellegrants,gazellethomsons=gazelle" \
  --species gazelle,wildebeest,impala,zebra,livestock,warthog,hyena,lion,topi,dikdik,elephant,baboon,buffalo,giraffe,eland \
  --outdir ./ei_species

# Top up confusing classes later (only the new crops are copied to ./ei_species/new)
python lila_sample_puller.py species ...same options... \
  --per-class-override "wildebeest=900,impala=900,gazelle=900,topi=800"
```

Run `python lila_sample_puller.py --help` for all options.

## `synth_paste.py`: synthetic person and vehicle frames

Public camera trap data has almost no usable Kenyan people or vehicles, so this script
pastes real COCO cut-outs (people, motorbikes, cars, trucks, plus animals) onto empty
Kenyan camera trap frames, with ground placement, shadows, lighting matching and
grayscale for night frames. See [docs/notes_synthetic_data.md](docs/notes_synthetic_data.md)
for the reasoning and references.

```bash
python synth_paste.py --per-class 300 --outdir ./ei_synth
```

Check `ei_synth/preview/` and delete odd results before uploading.
**Upload synthetic images as Training only, never Testing.**

The first run downloads the COCO annotations (~250 MB). Downloads are cached in `.lila_cache/`.

## Uploading to Edge Impulse

| Folder | Project | Category |
|---|---|---|
| `ei_detector/training` | Detector (object detection) | Training |
| `ei_detector/testing` | Detector | Testing |
| `ei_synth/training` | Detector | Training |
| `ei_species/training` | Species classifier (classification, label from filename) | Training |
| `ei_species/testing` | Species classifier | Testing |
