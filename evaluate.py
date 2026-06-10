#!/usr/bin/env python3
"""Evaluate the trained emotion-recognition CNN on the FER2013 test split.

Loads the committed model (emotiondetector.json + emotiondetector.h5) and runs
it over images/test/<label>/, writing a per-class classification report and a
confusion matrix to results/. No retraining required — this reproduces metrics
from the weights already in the repo.

Usage:
    pip install -r requirements.txt
    # place FER2013 under images/test/<label>/ for each of the 7 emotions
    python evaluate.py
"""
import os
import numpy as np
from keras.models import model_from_json

# LabelEncoder order in model_training.ipynb is alphabetical — keep it identical
# here and in webcam_implementation.py so class indices line up with the weights.
LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
TEST_DIR = os.environ.get("TEST_DIR", "images/test")
RESULTS = "results"


def load_model():
    with open("emotiondetector.json") as f:
        model = model_from_json(f.read())
    model.load_weights("emotiondetector.h5")
    return model


def load_test_set(test_dir):
    from keras_preprocessing.image import load_img
    X, y = [], []
    for idx, label in enumerate(LABELS):
        d = os.path.join(test_dir, label)
        if not os.path.isdir(d):
            continue
        for name in os.listdir(d):
            try:
                img = load_img(os.path.join(d, name), color_mode="grayscale", target_size=(48, 48))
            except Exception:
                continue
            X.append(np.array(img))
            y.append(idx)
    X = np.array(X, dtype="float32").reshape(-1, 48, 48, 1) / 255.0
    return X, np.array(y)


def main():
    if not os.path.isdir(TEST_DIR):
        raise SystemExit(f"Test data not found at '{TEST_DIR}'. "
                         "Place FER2013 under images/test/<label>/ (see README).")
    model = load_model()
    X, y_true = load_test_set(TEST_DIR)
    if len(X) == 0:
        raise SystemExit(f"No images found under '{TEST_DIR}/<label>/'.")
    print(f"Loaded {len(X)} test images across {len(set(y_true))} classes.")

    y_pred = model.predict(X, batch_size=128, verbose=1).argmax(axis=1)

    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=LABELS, digits=3)
    print(f"\nOverall test accuracy: {acc:.4f}\n{report}")

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "classification_report.txt"), "w") as f:
        f.write(f"Overall test accuracy: {acc:.4f}\n\n{report}\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(7)); ax.set_xticklabels(LABELS, rotation=45, ha="right")
    ax.set_yticks(range(7)); ax.set_yticklabels(LABELS)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"FER2013 confusion matrix (row-normalized) — acc {acc:.3f}")
    for i in range(7):
        for j in range(7):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                    color="white" if cm[i, j] > 0.5 else "black", fontsize=8)
    fig.colorbar(im); fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "confusion_matrix.png"), dpi=120)
    print(f"Wrote {RESULTS}/classification_report.txt and {RESULTS}/confusion_matrix.png")


if __name__ == "__main__":
    main()
