#!/usr/bin/env python3
"""Train a CNN to classify forest images as fire / no_fire.

Dataset layout:
    data/fire/*.jpg
    data/no_fire/*.jpg

Run:  python train.py --data data --epochs 15
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

IMG = 128


def build_model():
    aug = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.15),
    ], name="augmentation")
    inp = keras.Input((IMG, IMG, 3))
    x = aug(inp)
    x = layers.Rescaling(1 / 255.0)(x)
    for f in (32, 64, 128):
        x = layers.Conv2D(f, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization(momentum=0.9)(x)
        x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)   # P(fire)
    m = keras.Model(inp, out)
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy",
              metrics=["accuracy"])
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()

    common = dict(image_size=(IMG, IMG), batch_size=a.batch, seed=42,
                  validation_split=0.2, class_names=["no_fire", "fire"],
                  label_mode="binary")
    train_ds = keras.utils.image_dataset_from_directory(a.data, subset="training", **common)
    val_ds = keras.utils.image_dataset_from_directory(a.data, subset="validation", **common)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    model = build_model()
    Path("outputs").mkdir(exist_ok=True)
    cbs = [keras.callbacks.ModelCheckpoint("outputs/fire_cnn.keras", monitor="val_loss",
                                           save_best_only=True),
           keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
    hist = model.fit(train_ds, validation_data=val_ds, epochs=a.epochs, callbacks=cbs, verbose=2)

    # Final evaluation on the validation split
    y_true, y_prob = [], []
    for xb, yb in val_ds:
        y_prob.extend(model.predict(xb, verbose=0).ravel())
        y_true.extend(yb.numpy().ravel())
    y_true = np.array(y_true).astype(int)
    y_pred = (np.array(y_prob) >= 0.5).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    report = {"accuracy": round((tp + tn) / max(1, len(y_true)), 4),
              "precision": round(tp / max(1, tp + fp), 4),
              "recall": round(tp / max(1, tp + fn), 4),
              "false_positives": fp, "false_negatives": fn,
              "true_positives": tp, "true_negatives": tn}
    Path("outputs/metrics.json").write_text(json.dumps(report, indent=2))
    print("Validation metrics:", json.dumps(report, indent=2))

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(hist.history["accuracy"], label="train")
    ax[0].plot(hist.history["val_accuracy"], label="val")
    ax[0].set_title("Accuracy"); ax[0].legend()
    ax[1].plot(hist.history["loss"], label="train")
    ax[1].plot(hist.history["val_loss"], label="val")
    ax[1].set_title("Loss"); ax[1].legend()
    plt.tight_layout()
    plt.savefig("outputs/training_curves.png", dpi=150)
    print("Model saved to outputs/fire_cnn.keras")


if __name__ == "__main__":
    main()
