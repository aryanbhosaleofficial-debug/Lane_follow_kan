# M03 delivery and validation report

Historical initial delivery, superseded by [M03_CORRECTIVE_REPORT.md](M03_CORRECTIVE_REPORT.md).
The results and limitations below describe the pre-review implementation, including
its former lateral-width geometry and confidence policy.

Date: 2026-11-08. Repository base: `5dc8def`. This report describes local software
changes, not a deployed vehicle or a new Git commit. No commit/push was requested.

## A. Implementation summary

Added `lane_detection.advanced` as an isolated M02 BEVFrame consumer. It validates
geometry/provenance, intersects supported/visible ROI, extracts configurable HSV
white/yellow evidence, supports local-contrast preprocessing, and applies
vertical-safe Canny/Hough analysis. Bounded robust row association retains multiple
marking chains; normalized robust linear/quadratic fitting produces boundaries,
support intervals and diagnostics. Pair checks reject incompatible width,
intersections, convergence, unsupported ROI and ambiguity. Missing-boundary
inference is optional and always uncertain. No control/navigation/tracking was added.

## B. Changed files

Modified:

| File | Reason |
|---|---|
| `.gitignore` | Ignore generated `outputs/m03/` artifacts |
| `README.md` | Correct current M01/M02/M03 status and link usage/validation |
| `lane_detection/README.md` | Explicitly distinguish legacy raw-image compatibility |

Created:

| File(s) under `lane_detection/` | Reason |
|---|---|
| `M03_DESIGN.md` | Repository audit, baseline and architecture |
| `M03_VALIDATION.md` | This delivery/evidence/limitations report |
| `M03_BENCHMARK.json` | Actual per-case hardware/stage/wall timing distributions |
| `M03_SYNTHETIC_EVALUATION.json` | Actual independent synthetic geometry errors and states |
| `advanced/__init__.py`, `advanced/__main__.py` | Public exports and module CLI entry point |
| `advanced/config.py` | Validated immutable screening configuration |
| `advanced/models.py` | Immutable JSON-safe output/diagnostic contracts |
| `advanced/validation.py` | Strict M02 input/geometry/mask checks |
| `advanced/preprocessing.py` | Color/local-contrast evidence and cached supported ROI |
| `advanced/extraction.py` | Hough filtering and multi-chain row association |
| `advanced/fitting.py` | Normalized deterministic robust fits, linear fallback and diagnostics |
| `advanced/inference.py` | Explicit optional single-boundary width-prior inference |
| `advanced/association.py` | Adjacent pairs, analytic width checks and ambiguity rejection |
| `advanced/pipeline.py` | End-to-end perception, repeat cache, timings and debug evidence |
| `advanced/visualization.py` | Copy-safe overlays distinguishing observation and inference |
| `advanced/cli.py` | M01/M02 image/video/camera/stream input, headless exports and run summaries |
| `advanced/synthetic.py` | Clearly labelled synthetic profile and independent pixel scenes |
| `advanced/evaluation.py` | Reproducible geometry evaluation and debug montage |
| `advanced/benchmark.py` | CPU-only M03 and M02+M03 measurement harness |
| `advanced/test_contracts.py` | Configuration, metadata and invalid-input tests |
| `advanced/test_preprocessing.py` | Color, shadow, border, gap and cache tests |
| `advanced/test_extraction.py` | Vertical/straight/multi-candidate/Hough and legacy comparison tests |
| `advanced/test_fitting.py` | Curvature, robustness, fallback, conditioning and evidence-loss tests |
| `advanced/test_discontinuities.py` | Dashed/broken/adjacent gaps and inference tests |
| `advanced/test_association.py` | Width, extrema, crossings, ROI and metric screening tests |
| `advanced/test_pipeline.py` | M01/M02 integration, trust, repeats, uncertainty, debug and CLI tests |
| `advanced/test_scenarios.py` | End-to-end independently known geometry and rejection scenarios |
| `advanced/README.md` | API, policies, configuration, commands and limitations |

Removed: none. M01/M02 runtime, existing tests, legacy detector/IPM, calibration
assets and dependencies were preserved. Generated logs/images/JSONL live in
ignored paths and are not source changes.

## C. Features

Implemented and tested: vertical/near-vertical and displaced straight pairs;
gentle/tight curves within configured limits; disconnected dashed and broken
observations; short same-frame gap association without filling evidence masks;
multiple boundary/pair hypotheses; strict supported ROI; robust fitting and
linear fallback; width/ordering/intersection/convergence/curvature screening;
conservative validity and heuristic confidence with near/far evidence scores;
explicit optional inference; pixel-only unverified output; gated ground samples;
frame identity/timestamps; headless and GUI-capable CLI; stage benchmarking.

Partial by design: complex junctions/sharp turns are detected as ambiguous or lose
support rather than selecting a branch. Each boundary must be a single-valued
quadratic/linear u(v). Inference is a lateral-width approximation. Illumination
robustness has synthetic evidence only, and thresholds require track tuning.

Unsupported/out of scope: route/roundabout choice, tracking/persistent prediction,
waypoints, obstacle reasoning, IMU/ROS integration, steering/throttle/PID.

## D. Testing report

Executed commands (Python 3.14.10, NumPy 2.5.4, OpenCV 5.0.0):

| Command | Result |
|---|---|
| `python -m unittest discover -v` before production changes | 333 passed; 0 failed/skipped, 33.002 s |
| `python -m unittest discover -s lane_detection/advanced -t . -q` final package | 54 passed; 0 failed/skipped |
| `python -m unittest discover -v` final integrated validation | 387 passed; 0 failed/skipped, 35.870 s |
| `python -m unittest lane_detection.advanced.test_contracts -v` after final configuration type guards | 4 passed; 0 failed/skipped |
| `python -m unittest camera_calibration.video_asset_tests -v` | 2 passed; 0 failed/skipped, 32.316 s |
| `python -m lane_detection.advanced.evaluation` | 13 scenes passed; 0 failed |
| `python -m compileall -q lane_detection/advanced` | Exit 0 |
| `git diff --check` | Exit 0 |

Full root discovery includes all pre-existing M01/M02/legacy IPM tests and the
54 added tests. Explicit video asset tests are outside default discovery because
of their historical filename. Expected M01/M02 failure-path logs and malformed
video warnings are generated by passing negative tests, not unresolved failures.

Requested scenario coverage:

| Scenario | Evidence |
|---|---|
| 1 Straight solid | `test_scenarios`, `test_pipeline`: boundary/centre errors, side identity |
| 2 Gentle curves | `test_scenarios`, `test_fitting`: polynomial error/residual |
| 3 Tight curves within limits | Same, plus explicit excessive-curvature rejection |
| 4 Dashed | `test_discontinuities`, `test_scenarios`: correct association and gap intervals |
| 5 Broken | Same: short gap recovery and no long-gap bridging |
| 6 One visible boundary | `test_pipeline`: uncertain default; explicit prior inference remains uncertain |
| 7 No visible boundaries | `test_pipeline`, `test_scenarios`: invalid, no centreline |
| 8 Shadows/illumination | `test_preprocessing`, `test_scenarios`: local contrast improves support/score |
| 9 Noise/spurious fragments | `test_scenarios`: geometry retained; noise-only false-positive rejection |
| 10 Multiple adjacent lanes | `test_association`, `test_scenarios`: candidates/pairs retained, no selection |
| 11 Ambiguous junction | Crossing and converging fork tests; no selected centreline |
| 12 Vertical/near-vertical | `test_extraction`, `test_scenarios`; legacy dx=0 rejection compared directly |
| 13 BEV borders/support | `test_preprocessing`, `test_association`, `test_scenarios` |
| 14 Invalid sizes/inputs | `test_contracts`, `test_scenarios`: dtype/mask/grid/provenance rejection |
| 15 Insufficient evidence | `test_fitting`, `test_scenarios`: no unstable/unsupported fit |
| 16 Repeated source frame | `test_pipeline`: same object returned, conflict/backward IDs rejected |

Additional tests cover decreasing scores as solid -> dashed -> single -> empty
evidence disappears; near/far confidence; no previous-frame prediction; capacity
caps forcing uncertainty; Hough failure retaining curve detection; byte-identical
source preservation; headless exports; simulated trusted/untrusted/forged-status
metric gates. Trust fixtures simulate measurements solely to test software gates;
they are not physical calibration evidence.

The standalone 13-scene evaluation uses 100 samples over each accepted domain.
Eight positive scenes all satisfy a 2.5 px maximum-error threshold. Observed worst
boundary error was **0.404 px**; worst centreline error **0.329 px**. All five negative
or ambiguous scenes have no selected valid lane. This is a small deterministic
synthetic evaluation, not a claim about real-track accuracy or general false-positive rate.

Two defects discovered during expansion were fixed: a fork stopping just before
intersection escaped the first crossing check, and an isolated fragment in a dash
gap could redirect a chain. Regression tests cover both. No unresolved automated
test failures remain.

## E. Performance report

Machine: Windows 11 build 26300, **Intel Core i5-13450HX**, 10 cores / 16 logical
processors. Python 3.14.10, OpenCV 5.0.0, NumPy 2.5.4. OpenCV was explicitly limited
to one thread in the benchmark. These are desktop results, **not Raspberry Pi 5**.

Command:

```powershell
python -m lane_detection.advanced.benchmark --frames 200 --warmup 20 --threads 1 --output outputs/m03/benchmark.json
```

Each case contains 20 untimed warmups followed by 200 measured frames with unique
source identities. No repeat-cache hits are timed. M03-only fixtures already have
BEV images. Combined cases include real M02 remapping of synthetic source images
at the listed source and BEV resolution. Setup/map-build time is recorded separately.
No camera acquisition, file decoding, drawing, display or encoding is included.
Throughput is `1000 / mean(latency_ms)`, not inverse median.

| Workload | Source / BEV | Median ms | p95 ms | Worst ms | Throughput FPS |
|---|---|---:|---:|---:|---:|
| M03 solid | BEV 320x240 | 5.710 | 6.510 | 8.151 | 173.2 |
| M03 curved + dashed + noise | BEV 320x240 | 8.101 | 13.193 | 19.471 | 114.3 |
| M02 + M03, multi-marking scene | 320x240 / 320x240 | 8.446 | 15.910 | 19.035 | 102.8 |
| M03 solid | BEV 640x480 | 11.625 | 21.990 | 27.697 | 75.2 |
| M03 curved + dashed + noise | BEV 640x480 | 14.222 | 22.515 | 32.049 | 63.2 |
| M02 + M03, multi-marking scene | 640x480 / 640x480 | 14.597 | 24.719 | 32.371 | 61.4 |

Median internal M03 stage milliseconds (full median/p95/worst in JSON):

| Workload / BEV | Preprocess + validation | Extraction | Fit + inference | Association | M03 total |
|---|---:|---:|---:|---:|---:|
| Solid 320x240 | 0.856 | 3.463 | 1.118 | 0.138 | 5.682 |
| Curved/dashed/noise 320x240 | 0.877 | 4.419 | 2.564 | 0.141 | 8.058 |
| Solid 640x480 | 2.664 | 7.495 | 1.152 | 0.152 | 11.587 |
| Curved/dashed/noise 640x480 | 2.655 | 7.891 | 3.105 | 0.152 | 14.050 |

Stage medians need not sum to total median. Debug assembly, when enabled outside
this benchmark, is charged to association/total. CPU scheduling affects tails;
these are observed samples, not guaranteed deadlines. Each M03-only case returned
200/200 valid observations. Combined M02 scenes contain an additional centre
dashed marking and returned 200/200 uncertain observations, deliberately selecting
no driving lane. Their throughput is not a detection-success measurement.

Optimizations include a shared immutable ROI-mask cache, sparse row-centre votes,
bounded tracks/runs, no full-frame coordinate mesh, clean-fit fast paths, batched
robust-model solves, small-list local prediction, optional debug work and a bounded
last-frame cache. No acquisition-blocking work was added to M01/M02.

## F. Integration verification

M01 CameraManager publishes read-only BGR FramePackets. M02 consumes them through
its non-consuming interface and produces BEVFrame. M03 consumes that object using
its original masks, grid, coordinate conversions and provenance. Unit integration
uses the actual M01 manager with simulated capture and real BEVProcessor. Existing
video asset tests verify independent simultaneous consumers and input integrity.

Full recorded-video CLI check:

```powershell
python -m lane_detection.advanced --source videos/test.mp4 --profile calibration/test_video_metric_profile.json --headless --save-every 100 --output outputs/m03/video_full
```

Observed EOF summary: **578 M03 observations**, source IDs **1..590**, **12 source
frames skipped** by the latest-only consumer. States: **288 uncertain, 290 invalid,
0 valid**; **0 validated metric exports**. Run summary explicitly records EOF and
profile origin `SYNTHETIC`. The run overlapped regression activity; it is an I/O/
provenance smoke check, not the isolated performance benchmark.

Exported source/BEV samples and the synthetic montage were visually inspected.
The bundled profile does not describe measured camera/road geometry: the BEV
strongly distorts this recording and projects wall-like scene regions into the
grid. This run **does not demonstrate useful real-video lane-pair detection**.
No labels exist here to establish real-video geometry error or false-positive rate.
Do not tune or certify physical dimensions from this synthetic profile.

## G. Known limitations

Implementation: fixed HSV/color assumptions; grid-dependent thresholds; bounded
association can lose evidence in clutter; quadratic u(v) cannot represent arbitrary
topology or sideways/hairpin geometry; fork recognition is heuristic, not a complete
junction classifier; conservative gates can reject usable lanes; dashed gaps are
bounded rather than temporally predicted. Width is lateral separation, and inference
is a lateral offset, not a normal-width geometric reconstruction. There is no
occlusion/obstacle model. Timing under sustained Pi load is unknown. Confidence
is an explainable screening heuristic, not calibrated reliability.

Missing evidence: measured camera intrinsics/mounting/road-plane validation for
the actual competition setup, measured lane-width configuration, labelled track
images under representative conditions, physical camera operation, Raspberry Pi 5
latency/throughput/memory/thermal runs, alternate supported OpenCV/NumPy-version
validation, and sustained deployment tests. A GUI implementation exists; automated
validation exercised headless drawing/exports, not interactive camera/GUI use.

## H. Final verdict

**CONDITIONAL PASS.** Core M03 software is implemented with comprehensive synthetic
geometry/uncertainty tests, passing M01/M02 regressions, reproducible headless
integration and measured desktop performance. Physical calibration, useful labelled
real-track detection and Raspberry Pi validation remain outstanding. Zero valid
pairs on the bundled synthetic-profile video run is explicitly disclosed; no
deployment reliability claim is justified.

## I. Next-module interface

Import `LaneDetector`, `LaneConfig`, `LaneObservation`, `Boundary`, `Polynomial`,
`LanePair`, `FitDiagnostics`, `StageTimings` from `lane_detection.advanced`.
`process(BEVFrame, debug=False) -> LaneObservation`; `to_dict()` serializes to JSON.
Future tracking/waypoint consumers should retain:

* Session/frame identity, BEV sequence, source ID, acquisition Unix/monotonic times
  and calibration profile fingerprint. Compute frame age from acquisition time.
* `validity`, `valid`, reason codes and evidence score. No selected pair on ambiguous
  or unsupported observed geometry. Inferred pairs remain uncertain even when a
  mathematical centreline is available.
* All boundary IDs, observed/inferred flags, `(a,b,c)` pixel coefficients,
  accepted v-domain, observed/unsupported intervals and fit diagnostics.
* `compatible_pairs` as unselected hypotheses, plus explicit selected pair left/right
  IDs, centreline, width consistency and inference flag when supported.
* Missing-boundary information; a lone unassigned line uses `unresolved_side`.
* Pixel coordinate frame/units and calibration status. Ground sample tuples are
  X-forward/Y-left metres only when `metric_validated` is true. Metric widths are
  otherwise absent. Never infer trust from profile units or a status string alone.
* Per-stage timings and independently computed frame age; timings are not sensor
  exposure-to-output latency.

Tracking must manage temporal association/freshness itself. Waypoints must respect
curve domains, gaps, uncertainty and metric trust; M03 does not choose an intended
lane or export vehicle commands.
