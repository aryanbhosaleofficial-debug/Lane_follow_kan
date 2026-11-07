# M01 and M02 final corrective verification

## Repository and scope

Local `main` and GitHub `main` both resolved to
`32ac5c4672d6de5e224a8a8cc5e59c0d1f5af6ec` during this review.
The 20 pre-existing modified M02 files contained the reviewed corrections. They were preserved.
No commit, push, reset or cleanup was performed. Original recordings and calibration profiles
were preserved. This report covers software readiness, not physical metric certification.

## Reviewed findings

| Finding | Verified disposition |
|---|---|
| REV-001 | Synthetic-origin normalization and marker detection retained and tested. |
| REV-002 | Attested, signed runtime reports retained; configurable TrustPolicy, relative precision, distinct references and spatial coverage added. |
| REV-003 | Direct geometry remains mathematical and potentially unverified; validated_geometry is the documented trusted path. |
| REV-005 | CLI JSON explicitly states that it cannot grant trusted geometry and records profile provenance. |
| REV-006 | Validity/support masks documented; neither proves an occlusion-free drivable area. |
| REV-007 | Small numerical tolerance at calibrated hull boundaries retained and tested. |
| REV-008 | Undistortion documentation retained. |
| REV-009 | Overfit intrinsic solves produce calibration-quality errors; malformed input regressions pass. |
| REV-015 | Comparison exports show unverified status and synthetic profile provenance. |
| REV-016 | Deliberately limited exports carry partial_export and frame_limit metadata. |
| REV-018 | Backward frame IDs rejected within a session; new sessions remain valid. |
| REV-019 | Excessively nested profile JSON produces an actionable ProfileError. |
| REV-020 | Ground/BEV markers retained; missing homography-direction validation added with regression. |
| REV-022 | Frame processing uses the mapper's cached fingerprint. |

The supplied finding list contains fourteen identifiers, despite the earlier summary saying thirteen fixes.

## Additional corrections

The old evidence floor could be met by repeating one measurement three times. Distance duplicates
(including reversed endpoints) and repeated held-out positions now fail. Trusted references must
span two dimensions and a configurable share of each calibrated axis range.

TrustPolicy defaults are three references, 0.05 m maximum absolute tolerance, 10% relative error
and tolerance, and 10% span along each calibrated axis. These are configurable project screening
criteria, not universal physical acceptance limits. Distance relative scale is measured segment
length; held-out position relative scale is calibrated hull diameter. Policies and evidence geometry
are included in the runtime report integrity signature. A huge tolerance cannot grant trust simply
by raising the absolute policy limit. Software cannot authenticate an operator's measurements.

M01 documentation now explains that some OpenCV 4.x drawing operations ignore NumPy read-only
flags. Consumers must copy shared frames before drawing. No M01 runtime change was justified.
Read-completion timestamps are software timestamps, not guaranteed exposure timestamps. A blocking
backend can outlive stop's timeout; the existing API reports that limitation.

## Tests and environment

Windows 11, Intel64 Family 6 Model 183, Python 3.14.10, NumPy 2.5.4, OpenCV 5.0.0.

| Command | Result |
|---|---|
| python -m unittest discover -q | 328 passed; zero failed/skipped |
| python -m unittest camera_input.test_camera -v | 45 passed; zero failed/skipped |
| python -m unittest discover -s camera_calibration -t . -p "test_*.py" -v | 279 passed before the final spatial-coverage regression; final root run includes that additional test |
| python -m unittest camera_calibration.video_asset_tests -v | 2 passed; zero failed/skipped |
| Isolated OpenCV 4.13.0.92 root discovery | 327 passed before final spatial-coverage change |
| OpenCV 4.13 final status/validation selection | 30 passed; zero failed/skipped |

Root discovery includes M01, M02 and three legacy lane IPM tests. The historical video_asset_tests.py
filename requires its explicit command. Expected failure-path logs are emitted during successful tests.

## Actual video and output integrity

`videos/test.mp4`: 864x496, 23.998470605941044 nominal FPS, 590 decoded frames.
Two complete runs exported to ignored `outputs/m01_m02_final_validation/pixel/` and
`outputs/m01_m02_final_validation/metric_synthetic/`. Each processed IDs 1..590 with zero skipped
IDs, one session and clean EOF. All four videos in each run reopened with 590 frames and expected
dimensions. Diagnostic samples were generated; the metric comparison was visually inspected.
Metadata retains source identity, timestamps, calibration state and fingerprint. Real-video asset
tests check another simultaneous consumer, byte-identical source frames and output correspondence.

Pixel mode is a PIXEL-SPACE PERSPECTIVE DEMONSTRATION. M02 mode uses the existing compatible
synthetic profile, labels UNVERIFIED METRIC BEV and SYNTHETIC GEOMETRY, and remains METRIC_UNVERIFIED.
The raw BEV and mask videos are numerical artifacts: interpret them with their accompanying
processing_metrics.json and frame_metadata.jsonl. A white mask is not a measured road label.

## Performance measurements

Recorded-video M02 run, 350x500 BEV, 16 OpenCV threads:

| Quantity | Measurement |
|---|---:|
| BEV map initialization | 235.432 ms |
| Transform P50 / P95 / P99 / max | 0.409 / 0.500 / 0.551 / 4.090 ms |
| Four-output encoding total | 1.948 s |
| Paced end-to-end throughput | 23.972 FPS |
| Python allocation peak | 7.284 MB |
| Direct unpaced decode | 590 frames in 0.4074 s; 1448 FPS |
| M01 read-completion to consumer receipt, 120 frames, P50/P95/P99/max | 0.235 / 0.363 / 0.406 / 0.412 ms |
| Separate undistortion-map setup | 8.385 ms |

`python -m camera_calibration.benchmark --frames 100` also completed: default synthetic 640x480
input/350x500 output averaged 0.13 ms; net Python allocation growth was 0.529 MB, approximately one
retained BEV. This is a short bounded-memory check, not an extended native-memory soak.
Native resident-memory growth, exposure latency, live-camera FPS and Raspberry Pi results are unmeasured.
Nominal-paced export throughput includes waiting and must not be presented as maximum processing speed.

## CI and physical readiness

`.github/workflows/m01_m02_tests.yml` runs on push, pull request and manual dispatch with Python 3.12.
It installs requirements, runs root discovery once and explicitly runs real-video tests. YAML and
expected triggers/commands were checked locally. This uncommitted workflow has not run on GitHub;
no hosted CI success is claimed. Ubuntu/Python 3.12 execution awaits the first hosted run.

Follow camera_calibration/README.md: rigid final mounting; fixed resolution/focus; measured checkerboard
and varied captures; intrinsic calibration; measured ground marks; documented X-forward/Y-left origin;
homography; independent held-out references and selected acceptance policy; live-camera restart and
source-switch checks; Pi 5 latency, CPU, resident memory, FPS, temperature, throttling and extended run.
The existing recording has no documented physical references and cannot certify metric road geometry.

## Files requiring commit

Preserved and reviewed existing modifications:
camera_calibration/README.md, __init__.py, bev.py, cli.py, intrinsics.py, mapper.py, pipeline.py,
plane_calibration.py, profile.py, roadplane.py, test_intrinsics.py, test_pipeline.py,
test_plane_calibration.py, test_profile.py, test_status.py, test_validation.py,
test_video_validation.py, undistort.py, validation.py, video_validation.py.

Additional files: .gitignore, camera_input/README.md, .github/workflows/m01_m02_tests.yml,
M01_M02_VERIFICATION.md. Generated outputs and the isolated environment are ignored.

Verdict: READY FOR REAL-CAMERA CALIBRATION. Software completion is separate from physical calibration.
