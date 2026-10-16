import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tensorflow import keras

IMG = 128
EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def load(path):
    return np.asarray(Image.open(path).convert("RGB").resize((IMG, IMG)), dtype="float32")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--model", default="outputs/fire_cnn.keras")
    ap.add_argument("--threshold", type=float, default=0.5)
    a = ap.parse_args()
    model = keras.models.load_model(a.model)
    p = Path(a.path)
    files = [p] if p.is_file() else sorted(f for f in p.rglob("*") if f.suffix.lower() in EXTS)
    if not files:
        raise SystemExit("No images found.")
    probs = model.predict(np.stack([load(f) for f in files]), verbose=0).ravel()
    for f, pr in zip(files, probs):
        label = "FIRE" if pr >= a.threshold else "no fire"
        print(f"{f}: {label}  (P(fire)={pr:.3f})")


if __name__ == "__main__":
    main()
