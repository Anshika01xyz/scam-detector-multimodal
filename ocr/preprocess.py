"""Image clean-up before OCR (OpenCV).

Screenshots are often small, blurry, JPEG-compressed or in dark mode, which makes
OCR engines miss text. These steps make the text easier to read.
"""
import cv2
import numpy as np

TARGET_WIDTH = 1400  # small screenshots are scaled up to about this width


def load_image(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img


def preprocess(img, binarize=False):
    """Return a cleaned grayscale image (or black-on-white binary if binarize=True)."""
    # 1. upscale small images (OCR works best when letters are ~30 px tall)
    h, w = img.shape[:2]
    if w < TARGET_WIDTH:
        scale = TARGET_WIDTH / w
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # 2. grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. dark mode -> invert so text is dark on a light background
    if gray.mean() < 127:
        gray = cv2.bitwise_not(gray)

    # 4. remove JPEG noise, then boost local contrast
    gray = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

    if binarize:
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 31, 15)
    return gray
