"""Preserved reviewer R-09 raster protocol; extensions are explicitly named.

fragments/specks at scale=1, geometry=straight match p9_r09a.py exactly.
Other geometries reuse its clutter, and 640x480 resizes the complete raster.
"""
import cv2
import numpy as np

from .synthetic import scene


def review_case(kind, count, seed, scale=1, geometry="straight"):
    coefficients = ((0, 0, 70), (0, 0, 200))
    if geometry == "curve":
        coefficients = ((.001, -.24, 84.4), (.001, -.24, 214.4))
    elif geometry == "multi":
        coefficients = ((0, 0, 40), (0, 0, 140), (0, 0, 240))
    elif geometry == "companion":
        coefficients += ((0, 0, 215),)
    elif geometry in ("no_lane", "noise"):
        coefficients = ()
    elif geometry not in ("straight", "dashed"):
        raise ValueError("unknown geometry")
    image = scene(coefficients, dashed=geometry == "dashed")
    rng = np.random.default_rng(seed)
    for _ in range(count):
        u, v = rng.integers([5, 5], [307, 222 if kind == "fragments" else 233])
        if kind == "fragments":
            cv2.line(image, (int(u), int(v)),
                     (int(u + rng.integers(-3, 4)), int(v + rng.integers(4, 16))),
                     (230, 230, 230), 2)
        elif kind == "specks":
            cv2.circle(image, (int(u), int(v)), int(rng.integers(1, 3)), (240, 240, 240), -1)
        else:
            raise ValueError("unknown clutter kind")
    if geometry == "noise":
        image = np.clip(image.astype(float) + rng.normal(0, 65, image.shape), 0, 255).astype(np.uint8)
    return image if scale == 1 else cv2.resize(image, (320 * scale, 240 * scale), interpolation=cv2.INTER_NEAREST)
