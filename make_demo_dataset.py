#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def forest_background(rng, size):
    img = Image.new("RGB", (size, size))
    d = ImageDraw.Draw(img)
    sky = tuple(int(v) for v in rng.integers(120, 200, 3))
    d.rectangle([0, 0, size, size // 3], fill=sky)
    for _ in range(60):  # tree blobs
        x, y, r = int(rng.integers(0, size)), int(rng.integers(size // 4, size)), int(rng.integers(8, 28))
        g = (int(rng.integers(20, 70)), int(rng.integers(80, 160)), int(rng.integers(20, 70)))
        d.ellipse([x - r, y - r, x + r, y + r], fill=g)
    return img


def add_fire(img, rng, size):
    d = ImageDraw.Draw(img)
    for _ in range(int(rng.integers(1, 4))):
        cx = int(rng.integers(size // 5, 4 * size // 5))
        cy = int(rng.integers(size // 2, 4 * size // 5))
        for r, col in [(34, (200, 40, 10)), (24, (240, 110, 15)), (14, (255, 210, 60))]:
            r = int(r * rng.uniform(0.7, 1.4))
            d.ellipse([cx - r, cy - int(r * 1.6), cx + r, cy + r // 2], fill=col)
    return img.filter(ImageFilter.GaussianBlur(1.2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--n", type=int, default=200, help="images per class")
    ap.add_argument("--size", type=int, default=128)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    for cls in ("fire", "no_fire"):
        (Path(a.out) / cls).mkdir(parents=True, exist_ok=True)
        for i in range(a.n):
            img = forest_background(rng, a.size)
            if cls == "fire":
                img = add_fire(img, rng, a.size)
            img.save(Path(a.out) / cls / f"{cls}_{i:04d}.jpg")
    print(f"Created {2 * a.n} synthetic images in '{a.out}/'")


if __name__ == "__main__":
    main()
