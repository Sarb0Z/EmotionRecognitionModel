# Facial Emotion Recognition

A convolutional neural network that classifies facial expressions into seven emotions, with real-time webcam inference via OpenCV. Final-year project (FAST NUCES, 2024).

The same model is exported to ONNX and used for on-device inference inside a Unity game ([Sarb0Z/FYP](https://github.com/Sarb0Z/FYP)).

## Demo

`webcam_implementation.py` runs a live loop: Haar-cascade face detection → crop → 48×48 grayscale → CNN prediction → on-frame emotion label.

<!-- Add a short screen-capture GIF once recorded: ![demo](results/demo.gif) -->

## Dataset

[FER2013](https://www.kaggle.com/datasets/msambare/fer2013) — 48×48 grayscale faces, 7 classes (angry, disgust, fear, happy, neutral, sad, surprise), ~28,709 train / ~7,178 test. The dataset is **not committed** (see `.gitignore`); place it under `images/train/<label>/` and `images/test/<label>/`.

## Model architecture

Sequential CNN, 48×48×1 input:

| Stage | Layers |
|-------|--------|
| Conv blocks | Conv2D(128) → Conv2D(256) → Conv2D(512) → Conv2D(512), ReLU, with MaxPooling2D + Dropout(0.4) after each |
| Head | Flatten → Dense(512, ReLU) → Dropout → Dense(256, ReLU) → Dropout → Dense(7, softmax) |

Defined and trained in `model_training.ipynb`.

## Training setup

| Setting | Value |
|---------|-------|
| Optimizer | Adam |
| Loss | Categorical cross-entropy |
| Epochs | 100 |
| Batch size | 128 |
| Input | 48×48 grayscale, pixels scaled to [0, 1] |

Trained artifacts are committed: `emotiondetector.json` (architecture), `emotiondetector.h5` (weights, ~48 MB), and `sequential.onnx` (~16 MB) for ONNX runtimes.

## Results

Evaluate the committed model on the FER2013 test split — no retraining needed:

```bash
pip install -r requirements.txt
# place FER2013 test images under images/test/<label>/
python evaluate.py
```

`evaluate.py` prints a per-class classification report and writes `results/classification_report.txt` and `results/confusion_matrix.png`. Commit those and record the headline numbers here:

| Metric | Value |
|--------|-------|
| Test accuracy | _run `evaluate.py`_ |
| Macro F1 | _run `evaluate.py`_ |

<!-- After committing results/confusion_matrix.png, uncomment:
![Confusion matrix](results/confusion_matrix.png)
-->

> FER2013 is a hard, class-imbalanced benchmark — `disgust` has ~550 examples vs. ~7k for `happy`, and published baselines sit around 65–72% test accuracy. Calibrate against that, not near-100%.

## Repository structure

```
model_training.ipynb      # data loading, CNN definition, training
evaluate.py               # evaluate committed model -> report + confusion matrix
webcam_implementation.py  # real-time webcam inference
emotiondetector.json/.h5  # trained architecture + weights
sequential.onnx           # ONNX export for on-device / cross-runtime inference
requirements.txt
```

## Reproduce

```bash
pip install -r requirements.txt
# 1. Get FER2013, arrange as images/{train,test}/<label>/
# 2. (optional) retrain:  jupyter notebook model_training.ipynb
# 3. evaluate:            python evaluate.py
# 4. live demo:           python webcam_implementation.py
```

## Limitations

- FER2013 has noisy labels and severe class imbalance; `disgust` and `fear` are the weakest classes for most models.
- 48×48 grayscale input discards color and fine detail — this is a baseline, not production-grade.
- Webcam inference uses a Haar cascade for detection: fast, but less robust than a modern detector under pose/lighting variation.

## Status

Final-year project (2024). Training notebook, ONNX export, and real-time inference path are complete. Training-cell outputs were cleared before the original commit; `evaluate.py` reproduces metrics from the committed weights.
