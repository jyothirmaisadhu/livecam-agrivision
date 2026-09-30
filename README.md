# LiveCam AgriVision

Real-time plant disease and pest detection with damage assessment and voice assistance.

| Part | Model / tool | Output |
|---|---|---|
| Disease classification | MobileNetV3 (transfer learning) -> TFLite | 25 classes (5 crops x 4 diseases + healthy) |
| Pest detection | YOLO11n (Ultralytics) -> TFLite | boxes for 13 pests |
| Damage assessment | U-Net (MobileNetV2 encoder) -> TFLite | background / leaf / lesion mask, severity % |
| Advice | `knowledge_base/recommendations.json` + decision engine | treatment, prevention, spoken summary |
| Apps | Python + OpenCV desktop demo, Flutter mobile app | overlays + voice (pyttsx3 / flutter_tts, speech_to_text) |

Severity = lesion pixels / (leaf pixels + lesion pixels) x 100, graded
healthy < 1% <= mild <= 10% < moderate <= 25% < severe. (The base paper divides by all image pixels,
which lets the background dilute the score.)

Classes, crops and crop-pest links live in `configs/classes.yaml`; thresholds, image sizes and the
70/15/15 split live in `configs/settings.yaml`.

```
configs/            classes, settings, dataset mapping examples
knowledge_base/     disease/pest advice used by the decision engine and the app
src/agrivision/
  data/             step 1: dataset merge/split, YOLO remap, draft masks
  disease/          step 2: MobileNetV3 train / evaluate / TFLite
  pest/             step 3: YOLO11n train / TFLite
  damage/           step 4: U-Net train, severity
  decision/         step 5: recommendation engine
  pipeline/         step 6: live camera demo (models, smoothing, overlay, voice)
app/                step 7: Flutter app
notebooks/          Colab training notebook (GPU)
scripts/            demo data generator, app asset export
tests/              pytest suite
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt && pip install -e .
pytest -q && ruff check src tests scripts
```

Training on a CPU works for smoke tests only; use a GPU (Colab: `notebooks/train_colab.ipynb`).

## Step 0: smoke test with synthetic data (no downloads)

```bash
python scripts/make_demo_data.py                     # writes data/demo
python -m agrivision.disease.train --data data/demo/disease --out runs/demo/disease \
    --variant small --epochs-head 1 --epochs-finetune 1 --batch 8 --no-pretrained
python -m agrivision.damage.train --data data/demo/damage --out runs/demo/damage --epochs 1 --batch 4 --no-pretrained
python -m agrivision.pest.train --data data/demo/pest/data.yaml --out runs/demo/pest \
    --epochs 1 --batch 4 --imgsz 320 --device cpu --no-export
python -m agrivision.pipeline.live --source data/demo/demo.mp4 --no-display --save runs/demo/live.mp4 \
    --disease runs/demo/disease/disease_mobilenetv3.tflite --labels runs/demo/disease/labels.json \
    --pest runs/demo/pest/pest_yolo11n.pt --damage runs/demo/damage/damage_unet.tflite
```

The metrics are meaningless on synthetic data; this only proves the pipeline runs end to end.

## Step 1: datasets

Download the raw datasets into e.g. `raw/` (check each licence):

| Need | Suggested sources |
|---|---|
| Tomato, corn diseases | PlantVillage; PlantDoc for field photos |
| Paddy diseases | Paddy Doctor, Mendeley rice leaf disease sets |
| Cotton, chilli diseases | Kaggle / Mendeley cotton and chilli leaf sets |
| Pests (boxes) | IP102 detection subset, Roboflow Universe pest sets |
| Lesion masks | PlantSeg, or draft masks below corrected by hand |

No public set covers maize streak, sheath blight, verticillium wilt, cotton alternaria, chilli powdery
mildew or damping-off well; collect field photos for those.

```bash
# Disease: map each source folder to a canonical class, dedupe, split 70/15/15
cp configs/mappings/disease_sources.example.yaml configs/mappings/disease_sources.yaml  # edit paths
python -m agrivision.data.prepare_classification --mapping configs/mappings/disease_sources.yaml \
    --source-root raw --out data/disease

# Pests: remap YOLO class ids from several datasets into one data.yaml
cp configs/mappings/pest_sources.example.yaml configs/mappings/pest_sources.yaml
python -m agrivision.data.prepare_yolo --mapping configs/mappings/pest_sources.yaml \
    --source-root raw --out data/pest

# Damage: draft masks (0 bg, 1 leaf, 2 lesion), then fix them in CVAT/LabelMe
python -m agrivision.data.make_masks --images raw/diseased_leaves --out data/damage --preview
```

## Steps 2-4: train, evaluate, export

```bash
python -m agrivision.disease.train --data data/disease --out models/disease
python -m agrivision.disease.evaluate --data data/disease --model-dir models/disease
python -m agrivision.pest.train --data data/pest/data.yaml --out models/pest --device 0
python -m agrivision.damage.train --data data/damage --out models/damage
```

Outputs: `models/disease/{disease_mobilenetv3.tflite,labels.json,confusion_matrix.png,test_report.txt}`,
`models/pest/{pest_yolo11n.pt,pest_yolo11n.tflite}` (+ mAP50 / mAP50-95 printed),
`models/damage/damage_unet.tflite` (+ mean IoU in `history.csv`).

## Steps 5-6: advice + live desktop demo

```bash
python -m agrivision.pipeline.live --source 0 --voice \
    --disease models/disease/disease_mobilenetv3.tflite --labels models/disease/labels.json \
    --pest models/pest/pest_yolo11n.pt --damage models/damage/damage_unet.tflite
```

Every 6th frame is analysed; the disease label must win a 5-frame majority vote before it is shown or
spoken. Any model can be left out.

## Step 7: Flutter app

```bash
python scripts/export_app_assets.py --models models   # copies .tflite + labels into app/assets
cd app
flutter pub get
flutter test && flutter analyze
flutter run            # phone connected with USB debugging
flutter build apk      # build/app/outputs/flutter-apk/app-release.apk
```

The app has a live camera scan (overlay + spoken advice), photo/gallery analysis, and voice commands
("live camera", "photo", "analyze", "repeat", "treatment", "stop"). All inference runs on the phone with
TFLite. Missing models are shown on the home screen and skipped. Voice is English (en-IN).

Advice doses are general guidelines; follow the product label and your local agriculture officer.
