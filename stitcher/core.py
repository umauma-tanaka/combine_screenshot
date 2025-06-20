"""Core stitching logic."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from .utils import load_images, save_image


class StitchError(Exception):
    """Raised when reliable overlap cannot be determined."""


@dataclass
class MatchResult:
    offset: Tuple[int, int]
    score: float


class Stitcher:
    """Perform screenshot stitching."""

    def __init__(self, threshold: float = 0.8) -> None:
        self.threshold = threshold

    def detect_window(self, image: np.ndarray) -> Tuple[int, int, int, int]:
        """Detect ROI of the game window in ``image``."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edges = cv2.dilate(edges, None, iterations=2)
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return 0, 0, image.shape[1], image.shape[0]
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        return x, y, w, h

    def find_overlap(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        roi: Tuple[int, int, int, int],
    ) -> MatchResult:
        """Estimate translation between ``img1`` and ``img2``."""
        x, y, w, h = roi
        band_y = y + int(h * 0.2)
        band_h = int(h * 0.6)
        template = img1[band_y : band_y + band_h, x : x + w]
        res = cv2.matchTemplate(
            img2[y : y + h, x : x + w], template, cv2.TM_CCORR_NORMED
        )
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        offset = (max_loc[0], max_loc[1] + band_y)
        score = max_val
        if score >= self.threshold:
            return MatchResult(offset, score)
        # Fallback to ORB features
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(template, None)
        kp2, des2 = orb.detectAndCompute(img2[y : y + h, x : x + w], None)
        if des1 is None or des2 is None:
            raise StitchError("Not enough features")
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = matcher.match(des1, des2)
        if len(matches) < 4:
            raise StitchError("Not enough matches")
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC)
        if H is None or mask is None:
            raise StitchError("Homography failed")
        inliers = mask.ravel().astype(bool)
        if inliers.sum() / len(matches) < 0.6:
            raise StitchError("Low inlier ratio")
        dx = float(H[0, 2])
        dy = float(H[1, 2])
        return MatchResult((int(round(dx)), int(round(dy + band_y))), score)

    def solve_layout(self, overlaps: List[MatchResult]) -> List[Tuple[int, int]]:
        """Compute absolute offsets from pair-wise overlaps."""
        positions = [(0, 0)]
        for match in overlaps:
            last = positions[-1]
            positions.append((last[0] + match.offset[0], last[1] + match.offset[1]))
        return positions

    def compose(
        self,
        images: List[np.ndarray],
        offsets: List[Tuple[int, int]],
        roi: Tuple[int, int, int, int],
    ) -> np.ndarray:
        x, y, w, h = roi
        rois = [img[y : y + h, x : x + w] for img in images]
        xs = [o[0] for o in offsets]
        ys = [o[1] for o in offsets]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        canvas_w = max_x - min_x + w
        canvas_h = max_y - min_y + h
        canvas = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)
        alpha = np.zeros((canvas_h, canvas_w), dtype=np.float32)
        for roi_img, (ox, oy) in zip(rois, offsets):
            ox -= min_x
            oy -= min_y
            patch = cv2.cvtColor(roi_img, cv2.COLOR_BGR2BGRA)
            h_, w_ = patch.shape[:2]
            sub = canvas[oy : oy + h_, ox : ox + w_]
            sub_alpha = alpha[oy : oy + h_, ox : ox + w_]
            mask = patch[:, :, 3] / 255.0
            sub[:] = sub * (1 - mask[..., None]) + patch
            sub_alpha[:] = np.clip(sub_alpha + mask, 0, 1)
        return canvas

    def stitch(
        self,
        input_dir: Path | str,
        output: Optional[Path] = None,
        roi: Optional[Tuple[int, int, int, int]] = None,
        threshold: Optional[float] = None,
    ) -> Path:
        """Stitch screenshots in ``input_dir`` into a single image."""
        input_path = Path(input_dir)
        images = load_images(input_path)
        if len(images) < 2:
            raise StitchError("Need at least two images")
        if roi is None:
            roi = self.detect_window(images[0])
        if threshold is not None:
            self.threshold = threshold
        overlaps: List[MatchResult] = []
        for img1, img2 in tqdm(
            zip(images, images[1:]), total=len(images) - 1, leave=False
        ):
            overlaps.append(self.find_overlap(img1, img2, roi))
        offsets = self.solve_layout(overlaps)
        result = self.compose(images, offsets, roi)
        if output is None:
            output = input_path / "stitched.png"
        return save_image(result, output)
