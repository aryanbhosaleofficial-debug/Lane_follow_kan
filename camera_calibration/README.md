# M02 - Camera calibration and bird's-eye view

Turns the frames M01 publishes into a **metric bird's-eye view (BEV)** of the road plane: lens undistortion, a road-plane
homography calibrated from your own measurements, and a precomputed one-pass remap for runtime. It does **no** lane detection,
obstacle detection or control.

> **Status: software complete and verified on synthetic data only.** Nothing here has been run against a real camera, a real
> track, or a Raspberry Pi 5. Real-camera accuracy and Pi 5 speed need separate hardware validation (checklist at the end).
> The `lane_detection/ipm_*` prototype and `calibration/ipm_calibration.json` are untouched; that JSON is pixel-space only
> (`physical_scale: null`) and cannot be converted into an M02 profile, because metric scale needs measured ground points.

## Contents
1. [Quick start](#quick-start) - 2. [Coordinate conventions](#coordinate-conventions) - 3. [Calibrating the real camera](#calibrating-the-real-camera)
4. [File formats](#file-formats) - 5. [Runtime API and M01 integration](#runtime-api-and-m01-integration) - 6. [Profile compatibility policy](#profile-compatibility-policy)
7. [Errors](#errors) - 8. [Performance](#performance) - 9. [Limitations](#limitations) - 10. [Tests](#tests) - 11. [Hardware validation checklist](#hardware-validation-checklist-not-run)

## Quick start

```text
pip install -r requirements.txt                       # opencv-python, numpy (no other dependency)
python -m camera_calibration synthetic-demo --out demo_out   # whole workflow on SIMULATED data; writes profile, overlays, report
python -m camera_calibration --help                   # all commands
python -m unittest discover -s camera_calibration -t . -p "test_*.py"
python -m camera_calibration.benchmark                # measure latency/memory on this machine
```

`synthetic-demo` calibrates nothing real. Every artefact it writes says SYNTHETIC; do not use its profile for a vehicle.

## Coordinate conventions

All of M02 uses these, everywhere (code, JSON, overlays):

| Frame | Definition |
|---|---|
| **Ground** (metres) | **X forward, Y left, Z up** (right-handed). Origin: a reference point **you** choose and record (`origin_description`), e.g. the floor point under the lens. |
| **Raw image** | The frame as M01 delivers it (lens-distorted). OpenCV pixels: x right, y down, the *centre* of the top-left pixel is (0, 0). |
| **Undistorted image** | Ideal pinhole pixels in the `K_new` system of the profile's `undistortion` (see `alpha`). The homography maps ground -> these pixels. |
| **BEV image** | u right, v down, pixel centres at integers. `u = origin_u - y/mpp`, `v = origin_v - x/mpp`: **forward is UP, vehicle-left is image-LEFT**. The ground origin sits at BEV pixel (`origin_u_px`, `origin_v_px`), possibly outside the image. |

`mpp` = `meters_per_pixel`. Extents follow exactly: `x_max = (origin_v + 0.5)*mpp` (top edge), `y_max = (origin_u + 0.5)*mpp` (left edge).

The homography is valid **only for points on the calibrated road plane**. A cone, sign, speed breaker or another vehicle is not on it: its pixels
are mapped to the ground point *behind* it along the viewing ray, so it appears stretched away from the camera. The BEV also cannot show road hidden
behind an obstacle. Metric scale comes **only** from the ground measurements you supply; no lane width or other dimension is ever assumed.

## Calibrating the real camera

Two independent steps. Do them with the camera **rigidly mounted in its final position** and at the **resolution, zoom and focus it will run with**.
Changing any of those, or bumping the camera, invalidates the road-plane calibration (and a different resolution or focus invalidates the intrinsics).

### Step 1 - Intrinsics and lens distortion (chessboard)

Collecting images:
1. Print a chessboard on a **rigid, flat** surface (glue to a board; paper curls). Count its **inner corners** (where four squares meet), not squares: a
   board of 10 x 7 squares has **9 x 6** inner corners. Prefer one even and one odd count (e.g. 9x6): it makes the board's orientation unambiguous.
2. Measure the printed **square side** with a ruler. Note the unit (`mm`, `cm` or `m`). This is required and has no default.
3. Disable auto-focus if the camera allows (a changing focus changes the intrinsics). Use even lighting; avoid glare on the board.
4. Capture **at least 10, preferably 15-25** images at the runtime resolution. Vary: **tilt** (up to ~45 degrees, in different directions), **position**
   (centre, all four corners and edges of the frame - distortion is strongest at the edges), and **distance**. Do not take all images fronto-parallel.
5. Put them in one folder (PNG/JPG; the folder is not searched recursively).

Run:
```text
python -m camera_calibration intrinsic --images calib_images --pattern 9x6 --square 25 --unit mm \
    --camera-id vehicle-cam --out vehicle_cam.json --overlay-dir overlays --expected-size 1280x720
```
Read the report. It is not a pass/fail oracle for geometric accuracy:
* **RMS reprojection error** is a screening figure (default limit 1 px, `--max-rms`). A low RMS can hide poor view diversity, so the tool also checks
  **board-orientation spread**, **image coverage** and that the distortion is **invertible over the whole image**. If diversity is insufficient it
  **refuses** (exit 2, nothing written) and tells you what to add. `--no-enforce-quality` writes the profile anyway but the issues remain in it.
* Look at `overlays/*_corners.png`: **corner 0** must be the same physical corner in every image (a flipped detection ruins the calibration).
  `*_reprojection.png` shows residuals (x20); systematic patterns mean a model problem.
* **Models:** `standard5` (default, recommended), `radial2` (milder lenses, fewer parameters), `fisheye` (wide-angle lenses; needs more views - with
  only 16 the 4-term model can over-fit and be refused as non-invertible), `rational8` (**not recommended** without a demonstrated need: on mild lenses it
  returned large, cancelling coefficients).
* **`--alpha`** (default 0) sets the undistortion crop: 0 = no black border (some field of view is discarded), 1 = keep every source pixel (black borders),
  `None` in code = no zoom. **Choose it now**: changing it after the road-plane step invalidates the homography (the profile refuses).

### Step 2 - Road plane (metric)

1. **Pick the ground frame.** Choose an origin on the floor (e.g. directly under the lens, or the centre of the front axle), an X axis pointing **forward**
   (away from the vehicle) and Y pointing to the vehicle's **left**. Write down the origin in words.
2. **Place reference marks on the road plane** and measure each one's (x, y) in **metres** from the origin with a tape measure/ruler. Use **at least 6,
   spread over the area you care about** (a rectangular grid of marks is ideal; a calibration chessboard laid flat works). The minimum is 4, but a 4-point fit
   has zero residual by construction and tells you nothing about accuracy (the tool flags it). The hull of your points is the **calibrated region**:
   outside it the homography extrapolates and error grows (synthetic tests: ~2 mm inside, 12-43 mm outside).
3. **Photograph** the marks with the mounted camera (a raw frame, full resolution) and get each mark's **pixel** position in the **raw** image:
   ```text
   python -m camera_calibration pick-points --image road.png --out points.json
   ```
   This writes a template with `ground_m: null`. Fill in each measured `ground_m` and the `origin_description`. The calibrator **refuses** a file with nulls.
4. Optionally add `held_out` correspondences (marks you did NOT use for the fit) and `mounting.camera_height_m` (lens height, measured).
5. Choose the BEV grid (extents in metres, and `--mpp`), then run:
   ```text
   python -m camera_calibration road-plane --profile vehicle_cam.json --points points.json --out vehicle_metric.json \
       --bev-extents 0.3 2.3 -0.7 0.7 --mpp 0.004 --image road.png --overlay-dir overlays
   ```
6. Read the result:
   * `rms_metric_error_m` / `max_metric_error_m` and **`loo_rms_metric_error_m`** (leave-one-out: the honest generalisation figure when you have >= 5 points).
   * **camera height** / pitch / roll recovered from the homography: compare with a tape measure (`--camera-height`). A large disagreement means wrong
     intrinsics, mis-picked points, or points that are not on one plane.
   * Errors you will see instead of a silent bad result: collinear / repeated points, points straddling the horizon, **mirrored or swapped axes**
     (handedness check - always on), a **rotated axis labelling** (e.g. X measured to the side; the recovered camera heading must match `--facing`,
     default `forward`; use `--facing unchecked` only for a deliberately side-looking camera), ill-conditioned layouts.
   * Outliers: `--method ransac` or `lmeds` (needs > 4 points) rejects gross mis-picks and reports which.
7. **Verify with independent measurements** (the only evidence of real-world accuracy):
   ```text
   python -m camera_calibration check-distances --profile vehicle_metric.json --measurements dist.json --tolerance-m 0.01
   ```
   `dist.json` lists pixel pairs and tape-measured distances (below). **You** choose `--tolerance-m`. Also inspect `overlays/ground_grid.png`: the projected metric
   grid should follow real road features and match a tape laid on the road. `python -m camera_calibration info --profile vehicle_metric.json` runs the
   self-consistency checks (these prove the *software* is coherent, not that the calibration matches reality).
8. **BEV grid choice.** Each BEV pixel is sampled without an anti-aliasing filter. The tool reports how many source pixels one BEV pixel spans; if a
   large share exceeds ~2, fine detail there can alias (use a smaller `mpp`, or a farther range). Pixels never seen by the camera are masked, not guessed.

Never reuse a profile after moving, re-aiming, re-focusing, zooming or changing the resolution/crop of the camera. A session ID change alone does
**not** invalidate a profile; physical changes do, and cannot be detected from metadata.

## File formats

**Points file** (`road-plane`; `held_out` and `mounting` optional):
```json
{ "space": "raw",
  "origin_description": "floor point directly below the lens",
  "correspondences": [ {"image_px": [412.5, 388.0], "ground_m": [0.50, -0.35]}, ... ],
  "held_out":        [ {"image_px": [640.0, 300.2], "ground_m": [1.10, 0.00]} ],
  "mounting": {"camera_height_m": 0.25} }
```
`space` is `"raw"` (picked on the frame M01 delivers) or `"undistorted"` (picked on an image undistorted with this profile). Mislabelling costs ~40 mm in
the synthetic tests, so state it correctly.

**Distance file** (`check-distances`): `{"distances": [{"a_px": [x, y], "b_px": [x, y], "distance_m": 0.50, "label": "lane width"}]}` - both points on the road plane.

**Profile** (`*.json`, schema 1): validated plain JSON, no pickle. Contains intrinsics + quality report, undistortion (`alpha`, `K_new`), the road plane (homography,
the correspondences and fit metrics it came from, calibrated hull, recovered pose), the BEV grid, optional `source_ids`, free-form `mounting` notes (documentation only,
never used in the geometry), creation time and library versions. Loading rejects malformed JSON, duplicate keys, NaN/Infinity, wrong types/shapes, inconsistent
parts, an edited homography whose stored residual no longer reproduces, and URL credentials in identifiers. Saves are atomic.

## Runtime API and M01 integration

M02 never opens a camera. It reads M01's `FramePacket`s through M01's non-consuming interface, so it cannot take frames from other consumers.

```python
from camera_input import CameraConfig, CameraManager
from camera_calibration import CalibrationProfile, BEVProcessor, BEVWorker

profile = CalibrationProfile.load("vehicle_metric.json")
with CameraManager(CameraConfig(source=0, width=1280, height=720)) as camera, BEVWorker(BEVProcessor(profile), camera) as worker:
    after = 0
    while True:
        bev = worker.wait_for_bev(after, timeout=1.0)       # each consumer keeps its own `after`
        after = bev.bev_sequence
        lane_xy_m = bev.geometry.bev_to_ground([[175.0, 250.0]])     # BEV pixel -> ground metres (x forward, y left)
```

`BEVProcessor.process(packet)` (synchronous, thread-safe) and `process_latest(camera)` do the same without a thread. `ManagerFrameSource` is the per-consumer
reader (`wait_for_frame`-based) used by the worker; it retries through M01's brief "Capture stopped" error during a source switch without spinning, and reports a
terminal camera error. M01 absent? Any object with the documented `FramePacket` attributes works.

**`BEVFrame` contract for downstream modules**
* `image` - `(H, W, 3)` uint8 BGR, **read-only**. Meaningful only where `valid_mask == 255`; other pixels hold `border_value` and are **not road**.
* `valid_mask` - 255 where the camera observed that ground point. `support_mask` - observed **and** inside the calibrated region (+ `support_margin_m`);
  pixels observed but not supported are extrapolated and less accurate.
* Identity: `key == (source_session_id, source_frame_id)` is unambiguous (frame ids are monotonic across sessions in M01, but use both). `source_id`,
  `acquired_unix_ns`, `acquired_monotonic_ns`, `source_width/height`, `source_presentation_ms` are M01's values **unchanged**.
* `bev_sequence` increments only when a **new** source frame was processed; processing the same `(session, frame)` again returns the cached object. Equal
  sequence means the same BEV.
* `source_frames_skipped` - frames M01 published between the previous processed frame and this one (same session; `None` across sessions). Use it for dt-aware filters.
* `age_seconds()` - seconds since M01 **acquired** the frame (software age). It does not include sensor exposure, USB or driver latency.
* `processed_monotonic_ns`, `processing_seconds` - M02's own timing. `profile_fingerprint` - which calibration produced it. `notes` - e.g. "profile rescaled".
* `geometry` - the conversions below, bound to this frame's grid. `undistorted_image` only with `ProcessorConfig(include_undistorted=True)`.
* Arrays are **shared and read-only**; copy (`.copy()`) before modifying. M02 never modifies M01's arrays.
  **Warning - OpenCV 4.x:** the read-only flag stops NumPy writes on every version, but on OpenCV **4.8** `cv2.line`, `cv2.rectangle` and `cv2.putText` **silently draw
  into read-only arrays** (verified; OpenCV 5.0 raises instead). The same applies to M01's `FramePacket.image`. Any module that draws overlays must draw on a copy
  (`img = packet.image.copy()`), or it will corrupt the frame for every other consumer without any error.

**Conversions** (`CameraGeometry`, batches of N x 2, explicit units): `image_to_ground`, `ground_to_image`, `ground_to_bev`, `bev_to_ground`, `image_to_bev`,
`bev_to_image`, `undistort_points`, `distort_points`, `ground_distance`, `check_ground_points`. Invalid inputs return **NaN rows** (or raise with `on_invalid="raise"`):
pixels at/above the horizon, points behind the camera, points beyond where the lens model is invertible (a lens polynomial can fold back far outside the field of
view, which would otherwise produce meaningless "ghost" pixels), and non-finite inputs. Remember: valid for road-plane points only.

**Freshness and errors.** `ProcessorConfig.max_frame_age_s` (default 0.5 s) refuses stale frames; a worker that cannot process a frame withdraws the previous BEV so it never
passes for current, reports the error to waiting consumers, and keeps going (a persistent fault is logged once per type, then every 100th time). Waiters never block past
`stop()`. The latest-frame design (no queue) keeps age bounded for a controller; a consumer that needs every frame must use M01 directly.

## Profile compatibility policy

* Resolution must match the calibrated resolution **exactly**; otherwise `ResolutionMismatchError`. There is no silent rescaling. `ProcessorConfig(allow_uniform_rescale=True)`
  (or `profile.scaled_to(w, h)`) rescales only for the **same aspect ratio** and only if you know the camera delivers the same field of view at that size (many cameras crop
  in other modes); aspect changes are always refused.
* If a profile lists `source_ids` (M01's `FramePacket.source_id`), other sources raise `SourceMismatchError`; an empty list means unbound.
* A capture `session_id` change does not invalidate a profile. Credentials in URLs are redacted from source ids, logs, errors and profiles.

## Errors

All derive from `CalibrationError`: `InvalidCalibrationInputError`, `InsufficientCalibrationDataError`, `CalibrationQualityError` (carries the rejected result),
`DegenerateGeometryError`, `ProfileError`, `ResolutionMismatchError`, `SourceMismatchError`, `InvalidBEVRegionError`, `InvalidFrameError`, `StaleFrameError`,
`FrameUnavailableError` (`.reason`: `timeout`, `no_frame`, `camera_error`, `stopped`). M02 never returns a BEV from an uncalibrated, mismatched or partially invalid
input. The CLI turns them into `error: ...` on stderr with exit status 2.

## Performance

Measured with `python -m camera_calibration.benchmark` on one development machine (Windows 11, Intel CPU reported as Family 6 Model 183 with 16 logical cores, Python 3.14, OpenCV 5.0.0,
SYNTHETIC frames, no camera). **These are not Raspberry Pi 5 results and exclude all sensor/USB/driver time.** Example run:

| Source | BEV | Setup (once, at construction) | Warm per-frame (mean / p95) |
|---|---|---|---|
| 640x480 | 350x500 @ 4 mm/px | 230 ms | 0.15 / 0.19 ms |
| 1280x720 | 350x500 | 230 ms | 0.18 / 0.22 ms |
| 1920x1080 | 350x500 | 230 ms | 0.18 / 0.23 ms |
| 640x480, single-threaded OpenCV | 350x500 | - | 0.89 / 0.99 ms |

* Cost is driven by the **BEV size**, not the source resolution (700x1000: ~0.8 ms; 175x250: ~0.06 ms); setup scales with BEV pixels (55 ms / 230 ms / 900 ms).
* Memory: net growth over 400 frames was one BEV (0.53 MB), not 400.
* End-to-end through the real M01 manager + `BEVWorker` with a synthetic 30 fps source: frame age at the consumer p50 0.5 ms, p95 0.8 ms (0 frames skipped).
* **float32 remap tables are the default (`fixed_point_maps=False`) because they measured ~3.7x faster than int16 fixed-point tables here** (0.14 vs 0.50 ms). That was
  measured on x86 only; the opposite may hold on ARM. Run the benchmark on the Pi 5 (it prints both) and set `BEVConfig(fixed_point_maps=...)` accordingly.
* Do not read "fps-equivalent" columns as camera FPS: M02's throughput is far above any camera rate here; the camera and M01 set the real rate.

## Limitations

* Planar road assumption (above); no handling of slopes, speed breakers as part of the surface, or hanging objects. Very non-planar floors will not calibrate well.
* Intrinsic and road-plane accuracy on the **real** camera is unmeasured. Synthetic results (exact ground truth) give ~1-3 mm RMS inside the calibrated hull with 0.5 px
  of picking noise; real error depends on your picking and tape-measure accuracy, lens quality and rolling shutter/motion blur, none of which are simulated.
* A lens polynomial is invertible only up to some radius; beyond it BEV pixels are masked as unobserved. Strong distortion plus a wide BEV region loses the corners.
* No anti-aliasing prefilter, no exposure/colour correction, no rolling-shutter or vehicle-motion compensation, no temperature/focus drift model.
* The `live` command's window and `pick-points` need a GUI-enabled OpenCV build (they are not unit-tested; headless `live --frames N --no-gui` is).
* Declared dependency `opencv-python>=4.8`; developed and tested on **OpenCV 5.0.0 / NumPy 2.x / Python 3.14**. OpenCV 5 moved the fisheye flags and changed argument
  orders, which the code handles; older versions are expected to work but see the test section for what was actually run.

## Tests

```text
python -m unittest discover -s camera_calibration -t . -p "test_*.py"      # all M02 tests (synthetic, deterministic seeds, no hardware)
python -m unittest camera_input.test_camera                                  # M01 regression
```
`test_intrinsics`, `test_undistort`, `test_profile`, `test_plane_calibration`, `test_bev`, `test_validation`, `test_pipeline` (against the real M01 `CameraManager` with
fake captures), `test_cli`, `test_performance`. Synthetic tests use clearly named SYNTHETIC cameras/roads (`synthetic.py`); none of their numbers describe a real
camera or track.

## Hardware validation checklist (NOT RUN)

On the vehicle / Raspberry Pi 5, camera mounted, vehicle safely stationary:
1. Intrinsics from >= 15 varied chessboard images at the runtime resolution; confirm RMS, coverage, spread and the corner-0 overlays.
2. Road-plane calibration with >= 6 measured marks plus >= 4 **independent** held-out marks; confirm `check-distances` within a tolerance you choose, and the camera-height cross-check.
3. Check straight real road features are straight and parallel in the BEV and curved ones keep their measured radius.
4. Run `live` / `BEVWorker` for an extended period: measure BEV latency, `age_seconds()` p95, CPU, memory and temperature on the Pi 5; compare `fixed_point_maps` True/False.
5. Exercise source switches, unplug/replug and a camera restart; confirm no stale BEV is ever used.
6. Re-verify after any vibration test or re-mounting; define the recalibration trigger for your vehicle.

## Recorded project video validation (pixel demonstration)

The project recording is `videos/test.mp4` (864 x 496, nominal 23.998 FPS, 590 frames). The existing
`calibration/ipm_calibration.json` expects 1920 x 1080, has no intrinsic coefficients, and has
`physical_scale: null`. It cannot calibrate this video metrically. The following command processes
the entire recording through M01 and produces an explicitly uncalibrated pixel-only comparison:

```text
python -m camera_calibration.video_validation --video videos/test.mp4 --out outputs/m02_video_validation --quad 365 20 700 20 863 495 0 495
python -m unittest camera_calibration.test_video_validation -v
python -m unittest camera_calibration.video_asset_tests -v
```

The quadrilateral is a manually chosen visualization on frame 1 in TL, TR, BR, BL order. It is
not a measured road rectangle. Adjust it for other videos; do not interpret output pixel distance
as metres. `--limit N` provides a short trial. The command writes synchronized comparison, BEV and
validity-mask MP4 files, sample JPEGs (spread over the actual length), per-frame M01 metadata in
JSONL, and aggregate metrics. Every saved MP4 is reopened and decoded after writing; a skipped
source frame ID, **including frames lost before the first read**, fails the run. While a run is in
progress, and after any failure, the folder holds `RUN_INCOMPLETE.txt` (with the error) and no
`processing_metrics.json`, so stale metrics can never vouch for partial videos. The original video
is never modified. A compatible M02 metric profile can be supplied with `--profile PATH` instead of
`--quad`, subject to exact resolution and source checks; only that mode runs M02's runtime
(`transform_engine` in the metrics says which engine produced the frames).

**What the pixel demonstration is not (measured in the independent review).**

* **Not a bird's-eye view, even on frame 1.** Lane boundaries that are parallel on the floor stay 46-69 degrees apart after
  this warp (54 degrees on frame 1): the quad's top edge (y = 20) lies above the floor, in the wall and door reflections.
  A quad whose left and right edges lie *on* the two lane boundaries makes them parallel on its own frame.
* **No fixed homography fits this recording.** The camera's attitude to the floor changes (strong roll at the start,
  near-vertical view at the end; the lane vanishing point moves about 100 px within the first 6 s). Even a lane-aligned
  quad from frame 1 leaves 12-20 degrees of non-parallelism by frames 61-139. A BEV needs footage from a camera rigidly
  mounted on the vehicle, which this recording is not.
* **No calibration content.** The recording has no chessboard and no measured ground references, so neither intrinsic
  nor metric calibration can be derived from it. Physical metric accuracy cannot be verified from this video.

The validity mask shows pixels sampled from the input image; it is not a calibrated-region or
occlusion mask. With a quad that lies inside the image (as above) every output pixel is sampled, so
the mask is uniformly white and carries no information. No undistortion occurs in pixel mode because
intrinsics for this video are absent. The real-video tests also drive the first 120 frames through
M02's actual runtime (`BEVWorker`) with a clearly labelled PLACEHOLDER-geometry profile: that checks
frame identity, completeness, bit-exact output and non-mutation of M01's frames, not geometry.
M01's legacy `read()` mailbox replacement counter may rise while `wait_for_frame()` still processes
every source frame. Use the explicit skipped-ID count to assess this pipeline's completeness.
The nominal-FPS-paced pipeline is for complete export; elapsed time includes pacing and should
not be called camera FPS. Run `python -m camera_calibration.benchmark` separately for synthetic
metric-remap performance on the current computer.


## Corrective review: trust states and real-video limits

`BEVFrame.calibration_status` is `METRIC_UNVERIFIED` by default even when a profile contains metres.
It becomes `VALID_METRIC` only when `BEVProcessor(profile, validation_report=report)` receives a
passing **measured** `CalibrationValidator` report bound to that exact profile fingerprint.
`bev.validated_geometry()` raises `ProfileError` otherwise. The older `bev.geometry` attribute
remains for compatibility; any coordinates it returns are profile coordinates and must not be
used as physical truth without independent validation. The video exporter reports `PIXEL_ONLY`
for `--quad`. Missing or incompatible metric calibration raises rather than producing a metric
output. M01 session changes alone do not invalidate calibration, but a changed camera/lens,
focus, crop, zoom, mounting pose or effective resolution does.

The mounted-camera workflow is: fix the camera rigidly, lock resolution and lens/focus settings,
record camera height and origin, then capture varied checkerboard views at that resolution.
Measure the checkerboard square side; run the `intrinsic` command above. Place at least four,
preferably six or more, non-collinear ground marks and measure their X-forward/Y-left positions
in metres from the documented vehicle origin. Keep independent held-out marks and distances.
Run `pick-points`, `road-plane`, `info`, and finally:

```text
python -m camera_calibration check-distances --profile vehicle_metric.json --measurements dist.json --tolerance-m 0.01 --report-out measured_validation.json
python -m camera_calibration.pose_diagnostics --video videos/test.mp4 --out outputs/m02_video_review/viewpoint_diagnostic.json
```

Inspect the report, reference-point overlay and metric grid against independent ruler
measurements. The tolerance is an operator decision. A saved report records a profile fingerprint
and errors but cannot prove the operator truly measured independent points. Once valid measured
references exist, create a report in Python using `CalibrationValidator(profile).check_distances(...)`
and pass it to `BEVProcessor`. A synthetic or self-consistency report is not physical evidence.

The pose diagnostic samples frames and reports ORB match counts. On moving-vehicle footage,
feature motion is ambiguous and the conclusion is `INCONCLUSIVE_FOR_MOUNTING_STABILITY`. With a
stationary camera/target recording, `--stationary-points points.json --max-shift-px 3` can compare
four or more corresponding target pixels per selected frame. The file must contain
`{"target_stationary": true, "frames": [{"frame_id": 1, "points_px": [[x,y], ...]}, ...]}`.
This checks target/camera repeatability during that recording; it cannot certify later mounting
stability while driving. Never derive a frame-wise metric homography from moving-scene feature
matches alone.

`python -m camera_calibration.video_validation` produces a labeled perspective demonstration in
pixel mode; it draws the chosen quad on a copy of the original frame. The raw validity mask is
unchanged (it may be all white for an in-frame quad). `mask_diagnostic.mp4` displays its coverage
percentage and the unknown ground support. Metric mode draws measured reference points and a
grid on copies; shared M01 source arrays are not used as drawing canvases. Run the real-video
asset tests explicitly with `python -m unittest camera_calibration.video_asset_tests -v`.
Repository-wide discovery now includes the three lane IPM tests; run
`python -m unittest discover -q` from the repository root.

Hardware checks remain **NOT RUN**: measure real-camera frame age/FPS and M02 p50/p95/p99 latency,
CPU, native/Python memory and temperature on the Raspberry Pi 5; verify resolution changes,
source switches, restart, vibration/re-mounting, and an extended run before vehicle use.
