"""Utility helpers for the stitcher package."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image


def load_images(directory: Path) -> List[np.ndarray]:
    """Load images from a directory sorted by filename."""
    if not directory.is_dir():
        raise FileNotFoundError(str(directory))
    images = []
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(str(path))
        images.append(img)
    return images


def parse_roi(value: str) -> Tuple[int, int, int, int]:
    """Parse ROI string formatted as ``x,y,w,h``."""
    x, y, w, h = (int(v) for v in value.split(","))
    return x, y, w, h


def save_image(image: np.ndarray, path: Path) -> Path:
    """Save an image to ``path`` in PNG format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if image.ndim == 2:
        rgba = cv2.cvtColor(image, cv2.COLOR_GRAY2RGBA)
    elif image.shape[2] == 3:
        rgba = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
    else:
        rgba = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    Image.fromarray(rgba).save(path)
    return path
