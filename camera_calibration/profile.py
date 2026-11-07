"""Calibration profile: intrinsics + undistortion + road-plane homography + BEV grid, as validated JSON.

Compatibility policy (explicit, enforced by :meth:`CalibrationProfile.check_compatibility`)
------------------------------------------------------------------------------------------
* **Resolution** must equal the calibrated resolution exactly. There is no silent rescaling:
  a different resolution may be a crop or a different sensor mode, which a scale cannot model.
  :meth:`CalibrationProfile.scaled_to` exists for the case where the caller knows the new mode is
  the same view scaled uniformly; it refuses aspect-ratio changes.
* **Source**: if the profile lists ``source_ids`` (M01 ``FramePacket.source_id`` values), frames
  from any other source are refused. An empty list means "not bound to a source".
* **Session**: a changed capture ``session_id`` alone does NOT invalidate a profile. Physical
  changes (lens, zoom, mounting, crop, resolution) do, and cannot be detected from metadata:
  recalibrate whenever the camera is moved or refocused.

Files are plain JSON (no pickle / object deserialisation). Loading rejects malformed JSON,
duplicate keys, NaN/Infinity, missing fields, wrong types or shapes, inconsistent pieces, and
credentials embedded in identifiers.
"""
from __future__ import annotations

import copy
import dataclasses
import datetime as dt
import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ._util import is_real_number
from .bev import BEVConfig
from .errors import (CalibrationError, ProfileError, ResolutionMismatchError, SourceMismatchError)
from .intrinsics import IntrinsicParameters
from .roadplane import RoadPlaneCalibration
from .undistort import UndistortionSpec

FORMAT_ID = "m02-calibration-profile"
SCHEMA_VERSION = 1
MAX_PROFILE_BYTES = 8 * 1024 * 1024
# URL userinfo ("user:pass@") is refused; M01's own redaction placeholder "***@" is the one allowed form.
_CREDENTIALS = re.compile(r"://(?!\*\*\*@)[^/\s]*@")


def _now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _software() -> dict[str, str]:
    return {"opencv": cv2.__version__, "numpy": np.__version__}


def _check_text(value: Any, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProfileError(f"{what} must be a non-empty string")
    if _CREDENTIALS.search(value):
        raise ProfileError(f"{what} appears to contain URL credentials; profiles must never store credentials")
    return value


def _check_json_value(value: Any, what: str, depth: int = 0) -> None:
    if depth > 4:
        raise ProfileError(f"{what} is nested too deeply")
    if value is None or isinstance(value, (bool, str)):
        if isinstance(value, str) and _CREDENTIALS.search(value):
            raise ProfileError(f"{what} appears to contain URL credentials")
        return
    if is_real_number(value):
        if not math.isfinite(value):
            raise ProfileError(f"{what} contains a non-finite number")
        return
    if isinstance(value, dict):
        for k, v in value.items():
            if not isinstance(k, str):
                raise ProfileError(f"{what} keys must be strings")
            _check_json_value(v, f"{what}.{k}", depth + 1)
        return
    if isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            _check_json_value(v, f"{what}[{i}]", depth + 1)
        return
    raise ProfileError(f"{what} has an unsupported value of type {type(value).__name__}")


@dataclass(frozen=True, eq=False)
class CalibrationProfile:
    """Everything needed to turn a raw camera frame into a metric bird's-eye view.

    ``mounting`` is free-form documentation (e.g. measured camera height, pitch, notes). It is
    stored for humans and cross-checks and is never used in the geometry.
    """
    camera_id: str
    intrinsics: IntrinsicParameters
    undistortion: UndistortionSpec
    road_plane: RoadPlaneCalibration | None = None
    bev: BEVConfig | None = None
    source_ids: tuple[str, ...] = ()
    mounting: dict[str, Any] = field(default_factory=dict)
    created_utc: str = field(default_factory=_now_utc)
    software: dict[str, str] = field(default_factory=_software)

    def __post_init__(self):
        _check_text(self.camera_id, "camera_id")
        if not isinstance(self.intrinsics, IntrinsicParameters) or not isinstance(self.undistortion, UndistortionSpec):
            raise ProfileError("intrinsics and undistortion must be IntrinsicParameters / UndistortionSpec")
        if self.undistortion.output_size != self.intrinsics.image_size:
            raise ProfileError("undistortion output size must equal the intrinsic calibration image size")
        if self.road_plane is not None:
            rp = self.road_plane
            if rp.image_size != self.intrinsics.image_size:
                raise ProfileError(f"road plane was calibrated at {rp.image_size}, intrinsics at {self.intrinsics.image_size}")
            if not np.allclose(rp.undistorted_camera_matrix, self.undistortion.camera_matrix, rtol=1e-9, atol=1e-9):
                raise ProfileError("road-plane homography was estimated for a different undistorted camera matrix "
                                   "(e.g. the undistortion alpha changed afterwards); recalibrate the road plane")
        if self.bev is not None and self.road_plane is None:
            raise ProfileError("a BEV grid needs a road-plane calibration")
        object.__setattr__(self, "source_ids", tuple(_check_text(s, "source_ids entry") for s in self.source_ids))
        _check_json_value(self.mounting, "mounting")
        object.__setattr__(self, "mounting", copy.deepcopy(dict(self.mounting)))
        try:
            dt.datetime.fromisoformat(self.created_utc)
        except (TypeError, ValueError):
            raise ProfileError("created_utc must be an ISO-8601 timestamp") from None
        _check_json_value(self.software, "software")

    # -- derived state ---------------------------------------------------------------------
    @property
    def image_size(self) -> tuple[int, int]:
        return self.intrinsics.image_size

    @property
    def is_metric(self) -> bool:
        """True when a road-plane calibration and a BEV grid are both present."""
        return self.road_plane is not None and self.bev is not None

    def require_metric(self) -> None:
        if self.road_plane is None:
            raise ProfileError("profile has no road-plane calibration; run the road-plane step first")
        if self.bev is None:
            raise ProfileError("profile has no BEV grid configured")

    def fingerprint(self) -> str:
        """Short hash of the geometry-relevant content (not timestamps or free-form notes)."""
        d = self.to_dict()
        for key in ("created_utc", "software", "mounting"):
            d.pop(key)
        return hashlib.sha256(json.dumps(d, sort_keys=True, allow_nan=False).encode()).hexdigest()[:16]

    # -- derived profiles -----------------------------------------------------------------
    def with_road_plane(self, road_plane: RoadPlaneCalibration) -> "CalibrationProfile":
        return dataclasses.replace(self, road_plane=road_plane, created_utc=_now_utc())

    def with_bev(self, bev: BEVConfig) -> "CalibrationProfile":
        return dataclasses.replace(self, bev=bev)

    def with_source_ids(self, *source_ids: str) -> "CalibrationProfile":
        return dataclasses.replace(self, source_ids=tuple(source_ids))

    def with_mounting(self, **notes: Any) -> "CalibrationProfile":
        return dataclasses.replace(self, mounting={**self.mounting, **notes})

    def with_undistortion_alpha(self, alpha: float | None) -> "CalibrationProfile":
        """Change the undistortion crop. Invalidates (and therefore refuses) an existing road-plane calibration."""
        if self.road_plane is not None:
            raise ProfileError("changing the undistortion alpha invalidates the road-plane homography; "
                               "choose alpha before calibrating the road plane")
        return dataclasses.replace(self, undistortion=UndistortionSpec.from_intrinsics(self.intrinsics, alpha))

    def scaled_to(self, width: int, height: int) -> "CalibrationProfile":
        """Profile for a uniformly rescaled image. Only valid if the camera delivers the same view at the new size."""
        s_intr = self.intrinsics.scaled_to(width, height)
        spec = self.undistortion.scaled(width, height)
        rp = None
        if self.road_plane is not None:
            old = self.road_plane
            sx, sy = width / old.image_size[0], height / old.image_size[1]
            S = np.array([[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1.0]])
            H = S @ old.homography
            H = H / np.linalg.norm(H)

            def px(p):
                return None if p is None else p * [sx, sy] + [(sx - 1) / 2, (sy - 1) / 2]
            metrics = dict(old.metrics)
            for key in ("rms_reprojection_px", "max_reprojection_px"):
                if key in metrics:
                    metrics[key] *= (sx + sy) / 2
            rp = RoadPlaneCalibration(H, spec.camera_matrix, (width, height), old.ground_points_m, px(old.image_points_undistorted_px),
                                      px(old.image_points_raw_px), old.inlier_mask, old.method, old.camera_facing,
                                      old.origin_description, old.calibrated_hull_m, metrics, old.exact_fit,
                                      old.warnings + (f"rescaled from {old.image_size[0]}x{old.image_size[1]}; valid only if the camera "
                                                      f"delivers the same field of view at {width}x{height}",))
        return dataclasses.replace(self, intrinsics=s_intr, undistortion=spec, road_plane=rp)

    # -- compatibility policy --------------------------------------------------------------
    def check_compatibility(self, width: int, height: int, source_id: str | None = None, *, strict_source: bool = True) -> None:
        if (width, height) != self.image_size:
            raise ResolutionMismatchError(
                f"frame is {width}x{height} but the profile {self.camera_id!r} was calibrated at "
                f"{self.image_size[0]}x{self.image_size[1]}. Use the calibrated resolution, or recalibrate "
                f"(profile.scaled_to() is only valid if the camera delivers the same view at the new size).")
        if strict_source and self.source_ids and source_id not in self.source_ids:
            raise SourceMismatchError(f"frame source {source_id!r} is not one of the sources this profile is bound to: "
                                      f"{list(self.source_ids)}")

    # -- serialisation ---------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "format": FORMAT_ID, "schema_version": SCHEMA_VERSION, "camera_id": self.camera_id,
            "created_utc": self.created_utc, "software": dict(self.software), "source_ids": list(self.source_ids),
            "mounting": copy.deepcopy(self.mounting), "intrinsics": self.intrinsics.to_dict(),
            "undistortion": self.undistortion.to_dict(),
            "road_plane": None if self.road_plane is None else self.road_plane.to_dict(),
            "bev": None if self.bev is None else self.bev.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Any) -> "CalibrationProfile":
        if not isinstance(data, dict):
            raise ProfileError("profile must be a JSON object")
        if data.get("format") != FORMAT_ID:
            raise ProfileError(f"not an M02 calibration profile (format={data.get('format')!r})")
        version = data.get("schema_version")
        if version != SCHEMA_VERSION or isinstance(version, bool):
            raise ProfileError(f"unsupported schema_version {version!r}; this software reads version {SCHEMA_VERSION}")
        missing = [k for k in ("camera_id", "intrinsics", "undistortion") if k not in data]
        if missing:
            raise ProfileError(f"profile missing fields: {missing}")
        if not isinstance(data.get("source_ids", []), list) or not isinstance(data.get("mounting", {}), dict) \
                or not isinstance(data.get("software", {}), dict):
            raise ProfileError("source_ids must be a list and mounting / software objects")
        try:
            return cls(
                camera_id=data["camera_id"],
                intrinsics=IntrinsicParameters.from_dict(data["intrinsics"]),
                undistortion=UndistortionSpec.from_dict(data["undistortion"]),
                road_plane=None if data.get("road_plane") is None else RoadPlaneCalibration.from_dict(data["road_plane"]),
                bev=None if data.get("bev") is None else BEVConfig.from_dict(data["bev"]),
                source_ids=tuple(data.get("source_ids", [])), mounting=data.get("mounting", {}),
                created_utc=data.get("created_utc", _now_utc()), software=data.get("software", {}))
        except ProfileError:
            raise
        except (CalibrationError, ValueError, TypeError, KeyError, IndexError) as exc:
            raise ProfileError(f"invalid profile content: {exc}") from exc

    def save(self, path: str | Path) -> Path:
        """Atomically write the profile as JSON. Existing files are replaced only after a complete write."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(self.to_dict(), indent=2, allow_nan=False)
        fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=target.name + ".", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(text + "\n")
            os.replace(tmp, target)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        return target

    @classmethod
    def load(cls, path: str | Path) -> "CalibrationProfile":
        p = Path(path)
        try:
            if not p.is_file():
                raise ProfileError(f"calibration profile not found: {p}")
            size = p.stat().st_size
            if size > MAX_PROFILE_BYTES:
                raise ProfileError(f"profile is {size} bytes, above the {MAX_PROFILE_BYTES} byte limit")
            raw = p.read_bytes()
        except OSError as exc:
            raise ProfileError(f"cannot read calibration profile {p}: {exc}") from exc
        try:
            text = raw.decode("utf-8")
            data = json.loads(text, parse_constant=_reject_constant, object_pairs_hook=_no_duplicate_keys)
        except UnicodeDecodeError as exc:
            raise ProfileError(f"profile {p} is not valid UTF-8: {exc}") from None
        except ValueError as exc:
            raise ProfileError(f"profile {p} is not valid JSON: {exc}") from None
        except RecursionError:
            raise ProfileError(f"profile {p} is nested too deeply to be a calibration profile") from None
        return cls.from_dict(data)


def _reject_constant(name: str):
    raise ValueError(f"non-finite constant {name} is not allowed")


def _no_duplicate_keys(pairs: list) -> dict:
    out: dict = {}
    for k, v in pairs:
        if k in out:
            raise ValueError(f"duplicate key {k!r}")
        out[k] = v
    return out


def new_profile(intrinsics: IntrinsicParameters, camera_id: str, *, alpha: float | None = 0.0,
                source_ids: tuple[str, ...] = (), **mounting: Any) -> CalibrationProfile:
    """Start a profile from an intrinsic calibration (road plane and BEV are added later)."""
    return CalibrationProfile(camera_id, intrinsics, UndistortionSpec.from_intrinsics(intrinsics, alpha),
                              source_ids=tuple(source_ids), mounting=mounting)
