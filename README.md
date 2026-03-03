# Smart Recycling Bin — AI Classifier

A smart recycling bin that uses a camera to photograph disposed items, classifies them into waste categories using an AI model, and routes them to the correct bin compartment. Batteries trigger a hazardous waste alert to an external app.

## Waste Classes

| Index | Class | Bin Compartment | Alert |
|-------|---------|-------------------|-------|
| 0 | Battery | Mixed | ⚠️ Hazardous — triggers app alert |
| 1 | Cans | Cans | — |
| 2 | Organic | Organic | — |
| 3 | Paper | Paper | — |
| 4 | Plastic | Plastic | — |

## Model

- **Architecture:** MobileNetV2 (pretrained on ImageNet) + custom classification head
- **Input:** 224×224 RGB, values rescaled to 0–1
- **Deployment:** TFLite quantized (~3 MB)
- **Hardware:** Raspberry Pi 5
- **Temperature scaling:** T=1.2
- **Confidence threshold:** 0.60 (below → routed to mixed bin)
- **Test accuracy:** 99.3% on 434 images

## Repository Structure

```
project/
├── train.py                        # model training (Phase 1: frozen base, Phase 2: fine-tune)
├── evaluate.py                     # test set evaluation with confusion matrix
├── requirements_training.txt       # dependencies for training
├── requirements_pipeline.txt       # dependencies for pipeline
│
├── pipeline/
│   ├── run_pipeline_test.py        # main entry point — full pipeline on one image
│   ├── camera_input/
│   │   └── image_source.py         # loads latest image, moves to processed/
│   ├── preprocessing/
│   │   └── image_preprocessor.py   # resize, BGR→RGB, rescale, quality checks
│   ├── inference/
│   │   ├── model_interface.py      # TFLite inference + temperature scaling
│   │   ├── result_handler.py       # confidence threshold + battery routing
│   │   └── waste_classifier.tflite # deployed model (not tracked in git)
│   ├── logic/
│   │   └── event_manager.py        # bin event generation (for hardware team)
│   ├── communication/
│   │   └── api_client.py           # HTTP POST to backend API
│   └── logs/
│       └── classification_logger.py # CSV logging of every classification
│
└── AI model data/                  # dataset (not tracked in git)
    └── dataset/
        ├── train/                  # ~1589 images across 5 classes
        ├── val/                    # ~501 images
        └── test/                   # ~435 images
```

## Pipeline Flow

```
Camera Image → Quality Check → MobileNetV2 Inference → Temperature Scaling
    → Confidence Threshold → Bin Routing → API Events → CSV Log
```

**Events sent to backend:**
- `PIECE_COLLECTED` — sent for every classified item with the routed `subBin`
- `BATTERY_DETECTED` — sent when a battery is detected (always `subBin: mixed`)

## How to Run

### Pipeline (classification)

```bash
cd pipeline
python -m venv venv_pipeline
venv_pipeline\Scripts\activate        # Windows
pip install -r ../requirements_pipeline.txt
python run_pipeline_test.py
```

Place images in `pipeline/test_images/` — the pipeline picks the newest one, classifies it, and moves it to `pipeline/processed_images/`.

### Training

```bash
python -m venv venv_training
venv_training\Scripts\activate
pip install -r requirements_training.txt
python train.py
python evaluate.py
```

> **Note:** Delete `processed_general/` before retraining if you added new images to `train/` or `val/`.

## Configuration

| Setting | File | Default |
|---------|------|---------|
| `SEND_API_EVENTS` | `run_pipeline_test.py` | `False` |
| `TEMPERATURE` | `model_interface.py` / `evaluate.py` | `1.2` |
| `confidence_threshold` | `result_handler.py` | `0.60` |
| `BIN_ID` | `run_pipeline_test.py` | `BIN_001` |

## Classification Logging

Every run logs to `pipeline/logs/classifications.csv` with:

timestamp, image path, preprocessing outcome, predicted class, confidence, all 5 class probabilities, final class, accepted, alert, inference time (ms)
