# Notes for the Hackster write-up: synthetic person/vehicle data

## Why we needed it

- Public camera trap data has almost no usable people or vehicles for Kenya.
  LILA BC removes images containing people from every dataset (Serengeti,
  Enonkishu, Maasai Mara, Channel Islands, Idaho, SWG... all return 404).
- The only public person/vehicle frames with boxes came from NACTI (USA): 945 of
  its 1,020 person frames were drivers inside vehicles, and the vehicles/backgrounds
  are US farmland, not Kenyan conservancies. **NACTI is no longer used at all.**
- Kenyan MegaDetector person/vehicle detections (Enonkishu, Mara) were exported for
  manual review; after review only a handful were real people/vehicles.

## What we did

Pasted real person, vehicle (and animal) cutouts from COCO onto real, empty Kenyan
camera trap frames (Snapshot Enonkishu and Biome Health Maasai Mara) to create
labelled training images for the person and vehicle classes (`synth_paste.py`).

## Why this is an accepted technique

- Dwibedi et al., 2017, "Cut, Paste and Learn": pasting object cutouts onto real
  backgrounds trains detectors nearly as well as real labelled data.
- Ghiasi et al., CVPR 2021, "Simple Copy-Paste is a Strong Data Augmentation Method
  for Instance Segmentation" (Google): even simple random pasting is one of the
  strongest augmentations for detection.
- Beery et al., WACV 2020, "Synthetic Examples Improve Generalization for Rare Classes":
  synthetic examples added to camera trap data improved rare-class detection.
  Same research ecosystem as LILA/MegaDetector.

## Risks and how the script mitigates them

| Risk | Effect | Mitigation |
|---|---|---|
| Paste edges | Model learns to spot cut-out edges | Feathered (blurred) mask edges; sensor noise + slight blur over the whole frame |
| Lighting mismatch | Bright crisp person on a hazy morning looks fake | Brightness/contrast matched to the background region; grayscale for night IR frames |
| Wrong scale/placement | Floating people, giant cars in the sky | Objects stand on the ground in the lower part of the frame, smaller near the horizon |
| Shortcut learning | "Pasted = person/vehicle" | Animals are also pasted (labelled animal), so pasting is not a class cue |
| Viewpoint mismatch | COCO is eye-level photography | Full-body, untruncated, unoccluded cutouts only; favour small/distant placement |
| Wrong vehicle types | US sedans | Motorcycles (boda boda), cars/SUVs, trucks only; no buses/trains. Motorcycles keep their riders (rider = person box, bike = vehicle box) |
| Odd people | Skiers, tennis players, studio poses | COCO images containing sports equipment are skipped; standing/walking aspect ratio only |
| Black-and-white photos | Grey person in a colour daytime frame | Grayscale COCO photos are skipped |
| Remaining oddities | e.g. a pickup with a snow plough | Manual pass over the `preview/` folder before upload; delete anything implausible |

Scale of the cutout pool (COCO train2017, after filters): ~3,800 people, ~3,250 vehicles
(~1,170 motorcycles), ~5,900 animals, so cutouts are rarely reused.

Licensing note for the write-up: COCO annotations are CC BY 4.0; COCO images are Flickr
images under their individual Creative Commons licences. Credit COCO (Lin et al., 2014)
alongside LILA BC, Snapshot Enonkishu and the Biome Health Project.

## Rules we follow

1. Synthetic images go in **training only**, never testing. Testing uses real
   frames only (reviewed Kenyan frames now, own captures later).
2. Keep the mix sensible: no more than ~50% synthetic in the person/vehicle classes
   once real data exists.
3. **Prove it helps:** train with and without synthetic data, compare on the same
   real test set, and report both in the write-up.
4. Phase synthetic data out as real Kenyan person/vehicle captures accumulate.
