"""M03 CLI: calibrated images, recorded video and M01 camera/stream acquisition."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import time
import uuid
from typing import Iterator

import cv2

from camera_calibration import (BEVFrame, BEVProcessor, BEVWorker, CalibrationError, CalibrationProfile,
                                CalibrationValidator, DistanceMeasurement, MeasurementProvenance, ProcessorConfig, ValidationReport)
from camera_input import CameraConfig, CameraManager, FramePacket
from camera_input.core import redact_source
from lane_detection.lane_detection import classify_source
from .config import LaneConfig
from .pipeline import LaneDetector
from .visualization import draw_observation


def _frames(source: str, profile: CalibrationProfile, max_frame_age_s: float | None = .5,
            validation_report: ValidationReport | None = None) -> Iterator[BEVFrame]:
    kind, value = classify_source(source)
    processor = BEVProcessor(profile, ProcessorConfig(max_frame_age_s=None if kind in {"image", "video"} else max_frame_age_s),
                             validation_report=validation_report)
    if kind == "image":
        image = cv2.imread(str(value))
        if image is None:
            raise ValueError(f"cannot decode image: {value}")
        h, w = image.shape[:2]
        image.flags.writeable = False
        yield processor.process(FramePacket(image, 1, redact_source(value), time.time_ns(), time.monotonic_ns(),
                                             w, h, session_id=uuid.uuid4().hex))
        return
    width, height = profile.image_size
    camera_config = CameraConfig(source=value, kind="file" if kind == "video" else "live",
                                 width=width if kind != "video" else None,
                                 height=height if kind != "video" else None)
    with CameraManager(camera_config) as camera, BEVWorker(processor, camera) as worker:
        sequence = 0
        while True:
            try:
                frame = worker.wait_for_bev(sequence, timeout=5)
            except EOFError:
                return
            sequence = frame.bev_sequence
            yield frame


def _validation_from_measurements(path: Path, profile: CalibrationProfile) -> ValidationReport:
    """Re-run M02 validation; persisted summaries cannot grant runtime trust."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("profile_fingerprint") != profile.fingerprint():
        raise ValueError("validation measurements must name this exact profile fingerprint")
    try:
        provenance = MeasurementProvenance(**data["provenance"])
        distances = [DistanceMeasurement(**record) for record in data["distances"]]
        report = CalibrationValidator(profile).check_distances(distances, data["tolerance_m"], provenance=provenance)
    except (KeyError, TypeError) as exc:
        raise ValueError(f"malformed validation measurements: {exc}") from exc
    if not report.trusted_for(profile):
        raise ValueError("validation measurements do not meet M02 trust requirements")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="image, video, camera index or stream URL")
    parser.add_argument("--profile", required=True, help="M02 profile; legacy IPM JSON is incompatible")
    parser.add_argument("--config", type=Path, help="JSON object of LaneConfig overrides")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="0 processes to EOF")
    parser.add_argument("--output", type=Path, help="observation JSONL and sampled debug PNG directory")
    parser.add_argument("--save-every", type=int, default=30)
    parser.add_argument("--validation-measurements", type=Path, help="attested independent distance measurements; revalidated by M02")
    args = parser.parse_args(argv)
    if args.limit < 0 or args.save_every < 1:
        parser.error("limit must be >= 0; save-every must be >= 1")
    try:
        overrides = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
        if not isinstance(overrides, dict):
            raise ValueError("config must be a JSON object")
        for name in ("lane_width_m", "vehicle_reference_bev_px", "assumed_lane_width_m"):
            if isinstance(overrides.get(name), list):
                overrides[name] = tuple(overrides[name])
        if classify_source(args.source)[0] in {"image", "video"}:
            overrides.setdefault("max_frame_age_s", None)
        detector = LaneDetector(LaneConfig(**overrides))
        profile = CalibrationProfile.load(args.profile)
        profile.require_metric()
        validation = _validation_from_measurements(args.validation_measurements, profile) if args.validation_measurements else None
    except (OSError, ValueError, TypeError, CalibrationError) as exc:
        parser.error(str(exc))
    frames = _frames(args.source, profile, detector.config.max_frame_age_s, validation)
    summary = {"source": redact_source(args.source), "profile_fingerprint": profile.fingerprint(),
               "profile_origin": profile.road_plane.origin_description,
               "processed_observations": 0, "first_source_frame_id": None, "last_source_frame_id": None,
               "source_frames_skipped_by_consumer": 0, "end_reason": "error",
               "latest_only_delivery": True, "completed_to_eof": False}
    try:
        with ExitStack() as stack:
            log = None
            if args.output:
                args.output.mkdir(parents=True, exist_ok=True)
                log = stack.enter_context((args.output / "observations.jsonl").open("w", encoding="utf-8"))
            previous_id = None
            previous_session = None
            for index, frame in enumerate(frames):
                save = args.output is not None and index % args.save_every == 0
                observation = detector.process(frame, debug=save or not args.headless)
                record = observation.to_dict()
                record["source_frames_skipped_by_consumer"] = (frame.source_frame_id - previous_id - 1
                    if previous_session == frame.source_session_id and previous_id is not None else None)
                record["partial_export_requested"] = args.limit > 0
                previous_id, previous_session = frame.source_frame_id, frame.source_session_id
                summary["processed_observations"] += 1
                if summary["first_source_frame_id"] is None:
                    summary["first_source_frame_id"] = frame.source_frame_id
                summary["last_source_frame_id"] = frame.source_frame_id
                summary["source_frames_skipped_by_consumer"] += record["source_frames_skipped_by_consumer"] or 0
                encoded = json.dumps(record, allow_nan=False)
                print(encoded)
                if log:
                    log.write(encoded + "\n")
                if save or not args.headless:
                    debug = detector.last_debug
                    overlay = draw_observation(frame.image, observation, debug)
                    if save:
                        for name, image in (("overlay", overlay), ("candidates", debug.candidate_mask),
                                            ("accepted", debug.accepted_mask), ("rejected", debug.rejected_mask)):
                            path = args.output / f"{index:06d}_{name}.png"
                            if not cv2.imwrite(str(path), image):
                                raise OSError(f"could not write {path}")
                    if not args.headless:
                        cv2.imshow("M03 BEV", overlay)
                        cv2.imshow("M03 marking evidence", debug.candidate_mask)
                        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                            summary["end_reason"] = "user_quit"
                            break
                if args.limit and index + 1 >= args.limit:
                    summary["end_reason"] = "limit"
                    break
            else:
                summary["end_reason"] = "eof"
                summary["completed_to_eof"] = True
    except (OSError, ValueError, CalibrationError, cv2.error) as exc:
        parser.exit(2, f"M03 error: {exc}\n")
    finally:
        frames.close()
        if args.output and args.output.is_dir():
            (args.output / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        if not args.headless:
            cv2.destroyAllWindows()
    return 0
