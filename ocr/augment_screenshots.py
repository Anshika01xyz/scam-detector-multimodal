"""Make a "messy" copy of the clean screenshots to test OCR under realistic degradation.

    python ocr/augment_screenshots.py                       # data/screenshots -> data/screenshots_messy
    python ocr/augment_screenshots.py --seed 7 --src data/screenshots --dst data/screenshots_messy

Each image gets a random mix of: downscaling, slight rotation, blur, brightness/contrast shift,
sensor noise and heavy JPEG compression (the kind of damage a forwarded or photographed
screenshot suffers). labels.csv is copied so ocr/evaluate_ocr.py can score it with --dir.
Output is always JPEG; the seed makes the result reproducible.
"""
import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def degrade(img, rng):
    h, w = img.shape[:2]
    # 1. downscale (low-res screenshot / re-shared via chat apps)
    scale = rng.uniform(0.35, 0.7)
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]
    # 2. slight rotation (screen photographed / cropped crookedly), fill with border colour
    angle = rng.uniform(-6, 6)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    # 3. blur (out of focus / motion)
    k = int(rng.choice([3, 5, 7]))
    img = cv2.GaussianBlur(img, (k, k), 0)
    # 4. brightness / contrast shift
    alpha = rng.uniform(0.75, 1.2)
    beta = rng.uniform(-25, 25)
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    # 5. sensor noise
    noise = rng.normal(0, rng.uniform(4, 12), img.shape)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/screenshots")
    ap.add_argument("--dst", default="data/screenshots_messy")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src, dst = ROOT / args.src, ROOT / args.dst
    dst.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(src / "labels.csv")
    rng = np.random.default_rng(args.seed)
    new_names = []
    for name in df["filename"]:
        img = cv2.imread(str(src / name), cv2.IMREAD_COLOR)
        out = degrade(img, rng)
        q = int(rng.integers(20, 46))  # heavy JPEG compression
        new = Path(name).stem + ".jpg"
        cv2.imwrite(str(dst / new), out, [cv2.IMWRITE_JPEG_QUALITY, q])
        new_names.append(new)
    df["filename"] = new_names
    df.to_csv(dst / "labels.csv", index=False)
    print(f"Wrote {len(df)} messy images to {dst}")


if __name__ == "__main__":
    main()
