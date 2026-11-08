"""Small configurable color pipeline; evidence masks never bridge painted gaps."""
from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np

from camera_calibration import BEVFrame
from .config import LaneConfig


@dataclass(frozen=True, eq=False)
class Preprocessed:
    roi: np.ndarray
    marking_mask: np.ndarray
    edges: np.ndarray
    evidence_fraction: float = 0.0
    saturated: bool = False


class Preprocessor:
    """One-entry mask cache holds original arrays to avoid object-id reuse."""

    def __init__(self, config: LaneConfig):
        self.config = config
        self._cached_masks: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if config.clahe else None
        self._opening = np.ones((config.opening_size, config.opening_size), np.uint8)

    def roi(self, frame: BEVFrame) -> np.ndarray:
        cached = self._cached_masks
        if cached is not None and cached[0] is frame.valid_mask and cached[1] is frame.support_mask:
            return cached[2]
        roi = cv2.bitwise_and(frame.valid_mask, frame.support_mask)
        margin = max(self.config.roi_margin_px, self.config.blur_size // 2)
        if margin:
            size = 2 * margin + 1
            roi = cv2.erode(roi, np.ones((size, size), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)
        roi.flags.writeable = False
        # M02 publishes read-only masks. Writable caller-owned fixtures must not be cached.
        if not frame.valid_mask.flags.writeable and not frame.support_mask.flags.writeable:
            self._cached_masks = (frame.valid_mask, frame.support_mask, roi)
        return roi

    def process(self, frame: BEVFrame) -> Preprocessed:
        cfg = self.config
        roi = self.roi(frame)
        image = frame.image
        if cfg.blur_size > 1:
            image = cv2.GaussianBlur(image, (cfg.blur_size, cfg.blur_size), 0)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hue, sat, value = cv2.split(hsv)
        luminance = self._clahe.apply(value) if self._clahe is not None else value
        white_color = sat <= cfg.white_max_saturation
        white = white_color & (luminance >= cfg.white_min_value)
        if cfg.preprocessing == "local_contrast":
            # Mask-normalized mean prevents black invalid regions from creating contrast.
            valid = (frame.support_mask != 0).astype(np.float32)
            mean_sum = cv2.boxFilter(luminance.astype(np.float32) * valid, -1,
                                     (cfg.local_window, cfg.local_window), normalize=False)
            count = cv2.boxFilter(valid, -1, (cfg.local_window, cfg.local_window), normalize=False)
            mean = mean_sum / np.maximum(count, 1)
            white = white_color & (luminance >= cfg.white_min_value) & (luminance >= mean + cfg.local_contrast)
        yellow = ((hue >= cfg.yellow_hue_min) & (hue <= cfg.yellow_hue_max)
                  & (sat >= cfg.yellow_min_saturation) & (luminance >= cfg.yellow_min_value))
        mask = ((white | yellow) & (roi != 0)).astype(np.uint8) * 255
        if cfg.opening_size > 1:
            opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._opening)
            mask &= opened  # never publish newly invented observation pixels
        # Canny on actual color evidence avoids road texture/ROI-boundary Hough votes.
        fraction = float(np.count_nonzero(mask) / max(1, np.count_nonzero(roi)))
        saturated = fraction > cfg.max_evidence_fraction
        edges = np.zeros_like(mask) if saturated else cv2.Canny(mask, cfg.canny_low, cfg.canny_high)
        edges &= roi
        mask.flags.writeable = False
        edges.flags.writeable = False
        return Preprocessed(roi, mask, edges, fraction, saturated)

    def reference_evidence(self, frame: BEVFrame, points: np.ndarray) -> bool:
        """Unblurred observed paint at both edges, supported road at the centre.

        This permits a small NEAR reference query through the processing margin;
        it neither expands any fitted domain nor adds pixels to an evidence mask.
        """
        if not np.isfinite(points).all():
            return False
        h, w = frame.image.shape[:2]
        if np.any((points[:, 0] < 0) | (points[:, 0] > w - 1) | (points[:, 1] < 0) | (points[:, 1] > h - 1)):
            return False
        u, v = np.rint(points).astype(int).T
        if not np.all((frame.valid_mask[v, u] != 0) & (frame.support_mask[v, u] != 0)):
            return False
        hsv = cv2.cvtColor(frame.image[v[:2], u[:2]].reshape(2, 1, 3), cv2.COLOR_BGR2HSV)[:, 0]
        hue, sat, value = hsv.T
        c = self.config
        white = (sat <= c.white_max_saturation) & (value >= c.white_min_value)
        yellow = ((hue >= c.yellow_hue_min) & (hue <= c.yellow_hue_max)
                  & (sat >= c.yellow_min_saturation) & (value >= c.yellow_min_value))
        return bool(np.all(white | yellow))
