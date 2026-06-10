# Facial Emotion Recognition

Final-year project. Trains a CNN on grayscale face images to classify seven emotions, then runs inference in real time through a webcam feed using OpenCV.

## What it does

`model_training.ipynb` loads the FER2013 dataset from a local `images/` directory, extracts 48×48 grayscale features, and trains a Sequential CNN with four convolutional blocks (128→256→512→512 filters), dropout regularisation, and a 7-class softmax output. The trained model is saved as `emotiondetector.json` / `emotiondetector.h5` and exported to ONNX (`sequential.onnx`). `webcam_implementation.py` loads the weights and runs a live loop: Haar cascade face detection → crop → resize → predict → overlay label.

## Architecture

| Component | Detail |
|-----------|--------|
| Input | 48×48 grayscale |
| Conv blocks | 4 × (Conv2D + MaxPool + Dropout 0.4) |
| FC layers | Dense 512 → Dense 256 → Dense 7 (softmax) |
| Loss | Categorical cross-entropy |
| Optimiser | Adam, 100 epochs, batch 128 |

## Dataset

FER2013 (images not committed — add to `images/train/` and `images/test/` with one subdirectory per emotion label). Standard FER2013 training split is approximately 28,709 images across 7 classes.

## Model artifacts

Trained weights are committed: `emotiondetector.h5` (~48 MB) and `sequential.onnx` (~16 MB). The notebook's training cells were cleared before commit, so epoch-by-epoch accuracy is not recorded in the notebook.

## Tech stack

- Python, TensorFlow/Keras, OpenCV, ONNX
- `requirements.txt` lists all dependencies

## How to run

```bash
pip install -r requirements.txt
# Place FER2013 images under images/train/<label>/ and images/test/<label>/
jupyter notebook model_training.ipynb   # to retrain
python webcam_implementation.py         # to run live inference
```

## Status

Final-year project (2024). Real-time inference path is complete. Retraining requires the FER2013 image dataset (not included in the repo per `.gitignore`).
