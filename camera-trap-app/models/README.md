# models/

Optional. The app normally finds the models App Lab downloaded to
`/home/arduino/.arduino-bricks/models/custom-ei/`.

If the app reports "No .eim models found" (or you want to pin specific models), copy the
two `.eim` files here:

```
cp /home/arduino/.arduino-bricks/models/custom-ei/ei-model-<DETECTOR_PROJECT>-<VERSION>/model.eim   detector.eim
cp /home/arduino/.arduino-bricks/models/custom-ei/ei-model-<CLASSIFIER_PROJECT>-<VERSION>/model.eim classifier.eim
```

`.eim` files are not committed to the repo (see `.gitignore`).
