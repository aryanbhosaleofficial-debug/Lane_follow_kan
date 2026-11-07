import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from camera_calibration.pose_diagnostics import diagnose_video


class PoseDiagnosticTests(unittest.TestCase):
    def test_moving_scene_is_inconclusive_and_stationary_references_are_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); video = root / "frames.mp4"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
            if not writer.isOpened(): self.skipTest("mp4v unavailable")
            for _ in range(2): writer.write(np.zeros((48, 64, 3), np.uint8))
            writer.release()
            report = diagnose_video(video, [1, 2])
            self.assertEqual(report["conclusion"], "INCONCLUSIVE_FOR_MOUNTING_STABILITY")
            self.assertIsNone(report["comparisons"][0]["median_displacement_px"])
            base = [[5, 5], [50, 5], [50, 40], [5, 40]]
            references = root / "points.json"
            data = {"target_stationary": True, "frames": [
                {"frame_id": 1, "points_px": base}, {"frame_id": 2, "points_px": [p.copy() for p in base]}]}
            references.write_text(json.dumps(data), encoding="utf-8")
            self.assertEqual(diagnose_video(video, [1, 2], stationary_points=references)["conclusion"],
                             "STABLE_RELATIVE_TO_STATIONARY_TARGET")
            data["frames"][1]["points_px"][0][0] += 10
            references.write_text(json.dumps(data), encoding="utf-8")
            changed = diagnose_video(video, [1, 2], stationary_points=references)
            self.assertEqual(changed["conclusion"], "VIEWPOINT_OR_TARGET_CHANGED")
            self.assertEqual(changed["stationary_target"]["max_point_shift_px_by_frame"][2], 10)
            data["target_stationary"] = False
            references.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(ValueError): diagnose_video(video, [1, 2], stationary_points=references)


if __name__ == "__main__": unittest.main()
