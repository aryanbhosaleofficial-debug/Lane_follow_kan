# M03 architecture and audit

Initial audit/design record. Corrective geometry, contracts and verification are
documented in [M03_CORRECTIVE_REPORT.md](M03_CORRECTIVE_REPORT.md).

Audit baseline: local commit `5dc8def`, clean tracked working tree. `python -m unittest
discover -v`: 333 tests passed in 33.002 s on 2026-11-08. Documentation's earlier
328 count is historical; it is not the current baseline. Dependencies are NumPy
and OpenCV only. Tests use unittest. No AGENTS.md was found in the workspace or
its immediate parent.

## Existing contracts

M01 `CameraManager` publishes shared read-only BGR `FramePacket`s. Identity is
session_id + frame_id, with Unix and monotonic acquisition timestamps. `latest`
and `wait_for_frame` are non-consuming; legacy `read` consumes a separate mailbox.
M02 `BEVProcessor.process(packet)` performs existing fused undistortion/remapping;
`BEVWorker` provides latest-only asynchronous delivery. Neither needs replacement.
`BEVFrame` carries image, valid_mask, support_mask, CameraGeometry, BEVConfig,
source identity/timestamps, profile fingerprint and CalibrationStatus. Its
`validated_geometry()` enforces held-out validation; `geometry` alone does not.
Masks describe projection/calibration support, not occlusion or free road.

The legacy `lane_detection.py` uses raw-image trapezoids, grayscale/Canny/Hough,
image-half grouping and first-order endpoint fits. It discards dx == 0, cannot
represent curves/multiple lanes or evidence gaps, and checks ordering only at
one row. `ipm_transform.py` loads a different four-point pixel homography format;
it is not an M02 calibration profile. Both compatibility programs are preserved.
Root README roadmap is stale relative to working M01/M02 code. Existing sample
metric video profile is synthetic/unverified, not physical calibration evidence.

## New package and decisions

Add `lane_detection.advanced`, isolated from legacy code:

* `config`, `models`, `validation`: validated immutable configuration, immutable
  serializable observation records and strict BEV input checks.
* `preprocessing`: HSV white/yellow evidence with optional local contrast/CLAHE,
  optional filtering/morphology; intersect both M02 masks before extraction.
  Morphology may reject noise but never adds pixels to published evidence.
* `extraction`: Canny/Hough diagnostic segments with vertical-safe orientation
  and deduplication; row-run centres and bounded same-frame association retain
  multiple curves. Hough is supporting evidence, never the sole curve source.
* `fitting`: normalized polynomial least squares, deterministic robust rejection,
  linear fallback, support intervals/gaps and residual/coverage diagnostics.
  Intended quadratic is `u(v) = a*v^2 + b*v + c` (prompt has a repeated term).
* `association`: adjacent-pair checks over their shared supported interval,
  ordering, width/variation, curvature, ROI coverage and unique-pair policy.
  Multiple plausible pairs yield ambiguity, without choosing a driving lane.
* `pipeline`: synchronous pure perception over BEVFrame, bounded last-frame
  cache, timings, provenance and optional debug arrays. No temporal prediction.
* `visualization`, `cli`, `benchmark`, synthetic fixtures and unittest suites.

Outputs always contain pixel curves and evidence diagnostics. Ground sample
exports and metre-based constraints require `validated_geometry()` success.
Unverified calibration retains useful pixel observations with explicit limits.
No physical nominal width is provided by default. Pixel width bounds are
screening configuration, not measured track dimensions. Missing-boundary
inference additionally requires a configured prior AND an explicit observed
boundary role; a single line cannot establish its lane side on its own.

Validity policy: a valid observation requires one uniquely compatible pair of
independently observed supported boundaries, adequate coverage and confidence,
and no detected branch ambiguity. Missing/inferred/ambiguous evidence is uncertain
or invalid. Confidence is a heuristic evidence score, never a probability.
Curves are limited to observed longitudinal extent (no endpoint extrapolation);
internal gaps remain labelled and bounded. Inferred pair output is uncertain.

## Verification plan

Independent synthetic pixel polynomials with known errors cover the sixteen
requested scenarios. Also test real M01->M02 integration, trust restrictions,
immutability, bounded repeat caching, CLI headless exports, legacy vertical-line
comparison and deterministic processing. Run full discovery and explicit video
asset regressions. Benchmark 320x240 and 640x480 BEV, plus actual M02 conversions,
reporting warm median/p95/max and throughput on this desktop; no Pi claim.
Real-track correctness, physical calibration and Pi thermal/latency validation
remain separate outstanding evidence.
