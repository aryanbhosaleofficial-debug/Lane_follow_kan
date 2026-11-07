"""Explicit geometric trust state for M02 outputs."""
from enum import Enum


class CalibrationStatus(str, Enum):
    VALID_METRIC = "VALID_METRIC"
    METRIC_UNVERIFIED = "METRIC_UNVERIFIED"
    PIXEL_ONLY = "PIXEL_ONLY"
    UNCALIBRATED = "UNCALIBRATED"
    INVALID_OR_INCOMPATIBLE = "INVALID_OR_INCOMPATIBLE"
