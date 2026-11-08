# M03 advanced lane detection

M03 consumes **M02 BEVFrame objects**, detects per-frame marking boundaries and
returns conservative lane observations. It implements no navigation, persistent
tracking, waypoint generation or vehicle control. NumPy and OpenCV are the only
external dependencies. The legacy raw-image detector and four-point IPM tools
remain separate and unchanged.

Current corrective evidence: [R-09 report](../M03_R09_FINAL_REPORT.md),
[reviewer handoff](../M03_R09_REVIEW_HANDOFF.md). Earlier final reports describe
the preceding implementation and do not establish original-protocol R-09 closure.

## Pipeline

1. Validate the BGR frame, grid, binary masks, geometry and profile identity.
2. Intersect `valid_mask` and `support_mask`, with an eroded safety margin.
3. Extract white/yellow HSV evidence. Optional local contrast helps dim paint;
   optional CLAHE, blur and opening are independently configured. Opening only
   removes evidence. Closing is deliberately omitted: gaps must remain visible.
4. Run Canny on evidence and optionally Hough (`use_hough=True`). Retain vertical segments, reject transverse/short
   strokes, suppress overlapping votes. Hough supports the confidence of curve
   candidates; curves do not require Hough success.
5. Gather marking-run centres at sampled rows. Associate bottom to top using
   bounded robust local predictions and one-to-one assignments, retaining multiple
   chains. Runs wider than the marking-width limit are not lane evidence. Retire
   chains after their allowable gap: provisional one/two-row chains tolerate only
   two sampled steps, established chains tolerate `max_gap_px`. Only live chains
   consume `max_tracks`; eligible retired chains consume `max_fit_candidates`.
   Short retired clutter is discarded. Branch votes from discarded clutter do not
   establish a spatial fork. Row widths screen sustained tapered streaks using
   normal-width p90/p10, excluding dash caps and fit outliers.
6. Fit normalized linear/quadratic models with deterministic robust sampling and
   inlier refitting. A clean-fit fast path and batched small solves reduce CPU
   cost. Prefer a linear fit when quadratic improvement is insufficient. Reject
   insufficient span/coverage, excessive gaps/residuals, slope and curvature.
7. Check adjacent-pair ordering, analytic ordering extrema, normal-distance width consistency,
   convergence/intersections and supported curve/centreline pixels. Preserve
   competing pairs; do not select an intended lane by image half or vehicle position.
8. Emit provenance, evidence diagnostics, confidence, validity and stage timings.

The polynomial is **`u(v) = a*v² + b*v + c`** in BEV pixels. `v_min..v_max` is the
accepted longitudinal domain. Consumers must respect it. There is **zero endpoint
extrapolation in exported samples**. Limited-support polynomials return NaN outside
their domain; existing unrestricted polynomial evaluation remains for other
callers. The
polynomial also describes internal gaps mathematically, but each gap is listed in
`unsupported_intervals`. `observed_intervals` describe associated sampled rows,
not a dense claim that every pixel between row centres was painted. Debug
`accepted_mask` is always a subset of actual thresholded candidate pixels.

For downstream queries use `observation.evaluate_supported_centreline(rows)`.
It returns NaN unless the observation is valid and the row is inside both the
reported usable interval and the centreline domain, and outside reported paint
gaps. This conservative helper also guards ordinary full-domain polynomials.
`.valid` alone does not verify a planner's requested horizon. A limited pair can
remain valid on its near interval with `limited_support=True`; far-field errors
are reported separately and do not measure the accuracy of that near interval.

## API and M01/M02 integration

```python
from camera_input import CameraConfig, CameraManager
from camera_calibration import CalibrationProfile, BEVProcessor, BEVWorker
from lane_detection.advanced import LaneConfig, LaneDetector

# Supply your own camera profile and measured grid-specific screening bounds.
profile = CalibrationProfile.load("vehicle_metric.json")
detector = LaneDetector(LaneConfig())
width, height = profile.image_size
with CameraManager(CameraConfig(source=0, width=width, height=height)) as camera:
    with BEVWorker(BEVProcessor(profile), camera) as worker:
        sequence = 0
        while True:
            frame = worker.wait_for_bev(sequence, timeout=1.0)
            sequence = frame.bev_sequence
            observation = detector.process(frame)  # no camera I/O or GUI here
            # A future consumer must inspect validity AND coordinate/trust fields.
```

M01 timestamps and `(source_session_id, source_frame_id)` are preserved. The
detector holds one frame/result; processing the exact latest object again returns
the same observation. Different objects claiming that identity, and backward IDs
within a session, are rejected. A new session is independent. No previous curve
is reused when current-frame evidence disappears. Calls are serialized with a
lock. M02's latest-only worker keeps acquisition independent of M03; slow consumers
can skip frames. This is bounded latency delivery, not guaranteed video decoding
of every frame. CLI JSONL and run summaries report consumer skips and end reason.
M03 does not provide a separate tracking or asynchronous prediction worker.

Freshness is checked on every call, including cached repeats. The default
`max_frame_age_s=0.5` uses acquisition monotonic time and raises M02's
`StaleFrameError`. Invalid/future timestamps are refused. Configure
`max_frame_age_s=None` explicitly for offline replay. The CLI selects offline
semantics for images/recordings and retains checks for cameras/streams.
`processing_frame_age_s` describes the original processing instant, including on
cached results; consumers must recompute current age from acquisition time.

## Validity, confidence and calibration

`validity == "valid"` requires exactly one compatible, independently observed
pair, sufficient boundary evidence, nonzero shared observed paint by default,
no detected branch/capacity ambiguity and a
pair confidence at least `min_pair_confidence`. With ambiguous or missing evidence,
output is `uncertain`; no supported boundaries produces `invalid`. Ambiguous and
low-score observed pairs have **no selected centreline**. `compatible_pairs` are
hypotheses for inspection, not instructions to select a route.

An explicitly configured width prior plus `single_boundary_role` may produce an
inferred pair; it **always remains uncertain**, with `inferred=True`, a lower score,
an unobserved boundary and missing-boundary metadata. Its centreline is an uncertain
estimate and must not be treated as a valid observation. Inference uses a normal
offset and refuses folded offsets or inaccurate polynomial approximations.

Scores are engineering evidence scores, **not calibrated probabilities**. Boundary
score uses 55% evidence density within the accepted domain, 25% full-frame
longitudinal span and 20% residual quality, penalized
by rejected-point fraction. A multiplicative `residual_quality_factor` further
reduces the whole score when inlier p95 exceeds `max_residual_px`, or sustained
signed bias exceeds `near_field_max_error_px`. Sustained bias uses the existing
coherent-misfit interval test; isolated dash caps do not establish it. The factor
is the minimum of one and those tolerance/error ratios. Full-observation RMS/p95,
inlier RMS/p95/max and coherent signed bias remain separately visible. Hough
support, when enabled, contributes a factor in 0.95..1.0.
Pixel evidence confidence is independent of metric trust. Pair score uses the weaker boundary
and a width-variation penalty. Inference applies a 0.45 factor. Unselected uncertain
observations use a 0.35 factor. These are documented screening heuristics to tune
against labelled track data, not a statistical reliability claim. Near/far coverage,
RMS/p95 residual, inlier counts, gaps, curvature and normalized conditioning remain
available separately in diagnostics. Near/far confidence fields are local evidence
scores: half-frame row coverage times residual quality for that half. They do not
incorporate physical calibration trust; inspect `metric_validated` separately.

Output curves always use pixels. **Ground samples and metric widths are absent
unless `frame.validated_geometry()` succeeds**. A status string alone cannot grant
trust. Metre width/curvature constraints and metre inference priors also require
that gate; otherwise pair validity is refused, with pixel candidates retained.
Ground sample axes use M02 conversions: X forward metres, Y left metres; BEV u
right, v down. Forward is upward. Curvature is `|u''|/(1+u'^2)^1.5` in inverse pixels;
only validated geometry permits inverse-metre screening (`curvature_px / mpp`).
No unverified metric number is exported. Neither M02 mask detects object occlusion,
raised objects, road semantics or free space.

## Configuration

`LaneConfig` is frozen and validates types, ranges and related bounds. See
[config.py](config.py) for every parameter. Important groups are:

| Parameters | Meaning |
|---|---|
| `preprocessing`, `white_*`, `yellow_*` | Fixed HSV evidence or selectable local-contrast white evidence |
| `blur_size`, `opening_size`, `clahe` | Optional image operations; sizes of 1 disable blur/opening |
| `roi_margin_px` | Erode visibility/support intersection to suppress border evidence |
| `canny_*`, `hough_*` | Supporting edge/segment extraction |
| `row_step`, `max_marking_width_px` | Row sampling and acceptable paint-run width |
| `association_margin_px`, `max_gap_px` | Bounded lateral association and internal evidence gap |
| `min_points`, `min_span_px`, `min_coverage` | Independent observed support floor |
| `max_residual_px`, `inlier_threshold_px`, `max_outlier_fraction` | Robust fit acceptance |
| `max_abs_du_dv`, `max_curvature_px_inv` | BEV orientation/curve screening |
| `min_lane_width_px`, `max_lane_width_px`, `max_width_variation` | Lateral separation screening, not measured physical track dimensions |
| `lane_width_m`, `max_curvature_m_inv` | Optional trusted metric constraints |
| `nominal_lane_width_px` or `nominal_lane_width_m`, `single_boundary_role` | Explicit optional inference assumptions |
| `max_tracks`, `max_row_runs` | Work caps; exceeding them forces uncertainty |

Default pixel bounds are broad screening values for small BEV grids. Tune them
for your calibrated output resolution, actual paint thickness, measured track
widths and intended forward extent. There is **no default nominal physical width**.
Widths use directed normal intersections in both directions. Ambiguous or absent
supported intersections refuse a pair. Correspondence midpoints define
`centreline_points_px`; the centreline polynomial must approximate them within
`centreline_fit_tolerance_px` (default 1 px). Widths are sampled on fitted geometry.
Signed reference curvature is positive for a vehicle-left turn as ground X
increases. Absolute curvature is exact for the fitted quadratic, not a claim that
the real road is quadratic.
Models with overhangs/multiple u values at the same v cannot be represented by one
quadratic; this detector conservatively loses support or reports ambiguity.

CLI configuration is a JSON object of `LaneConfig` overrides. For example, the
synthetic 320x240 road with independently specified 100-pixel lane separation can
use `{"min_lane_width_px": 85, "max_lane_width_px": 115}`. This example is not a
measurement of the competition track. Threshold defaults do not scale themselves
silently when changing grid resolution.

For equivalent geometry on another BEV scale, use
`LaneConfig.for_bev(bev, lane_width_m=(measured_min, measured_max))`. Supply your own
width configuration. Algorithm lengths scale from a reference nominal grid
(default 0.004 m/px); counts/work limits remain separate. Unverified scale produces
nominal pixel bounds plus `physical_configuration_unverified`.
`require_validated_metric=True` refuses pairs until M02 grants metric trust.
Scale ratios outside 0.25..4 of the reference are refused; beyond this range,
provide explicit grid-specific pixel thresholds and validate them independently.
Original positional `LaneConfig` arguments retain their order; later additions
are keyword-only. Configuration is bound to that m/px; rebuild it when the scale changes. The factory
does not grant trust.

`max_tracks`, `max_row_runs`, `max_association_comparisons`, `max_fit_candidates`
bound work. Exhaustion stops fitting/pairing and returns uncertain with
`processing_budget_exceeded`, never a partial selected centreline. These are
operation limits, not a guaranteed wall-clock deadline. Hough is disabled by
default after the 896-scene ablation showed identical validity and supported
geometry with/without it. Enable `use_hough=True` for optional segment/vote
diagnostics; segments remain capped by `max_hough_segments`. These synthetic
results do not establish that Hough is never useful on real camera data.

Association uses the already sorted row runs to binary-search the intersection
of prediction and lateral gates before counting/scoring candidates. The 4,000
comparison budget is global per frame and counts only spatially admissible
track/run pairs. Index lookup remains bounded by sampled rows, `max_tracks` and
`max_row_runs`; the cap is still enforced before candidate scoring. This fixes
budget exhaustion, but cannot make competing fragment geometry unambiguous.

Local contrast honors `white_min_value`; lower it explicitly for dim paint (the
shadow demo uses 90). It no longer borrows the yellow brightness threshold.
Default yellow bounds exclude the tested orange and muted-tan swatches; color
alone cannot distinguish every material. `evidence_fraction` and
`saturated_marking_evidence` diagnose overly broad masks; configure
`max_evidence_fraction` for the refusal threshold.

## Pair observability and vehicle reference

`observed_fraction` counts observed endpoints divided by twice the number of
uniform centre-v bins (`support_samples`). Endpoint rows are interpolated from
normal correspondences, rather than assumed equal. `shared_observed_fraction`
requires both endpoints; `shared_observed_span_px` sums bin widths satisfying that
condition. `inferred_fraction` counts endpoints on an inferred boundary.
`unsupported_intervals` includes bins where either endpoint lacks observed paint.
All denominators use the accepted centreline v-domain. These are sampled evidence
diagnostics, not dense semantic labels. Sparse dashes can remain geometrically
valid with lower evidence fractions and confidence. Zero shared observed paint
refuses selection by default (`no_shared_observed_support`). An explicit
`require_shared_observation=False` permits association-only staggered dashes,
retaining zero-shared diagnostics and lower evidence confidence. This is a
configuration policy choice, not proof of painted continuity.

A valid geometric pair is not automatically an ego lane. Configure
`vehicle_reference_bev_px=(u,v)`, or explicitly declare
`ground_origin_is_vehicle=True` to use the actual BEV origin. Otherwise reference
status is unknown. No forward extrapolation is made outside supported domains.
The ground origin may lie outside the raster (the real M02 test profile has
origin v=574.5 for a 500-row BEV). Declaring that origin as the vehicle does not
move it into the image. Such queries return `vehicle_reference_outside_bev` with
no offset. Supply an explicitly measured/projected in-domain vehicle reference
and inspect `lane_reference_valid`/`ego_lane_valid`; choosing the image centre
as a substitute would change the physical reference and is not automatic.
A reference just beyond the near endpoint can be evaluated within
`max_reference_extension_px` (default 6): both predicted boundary pixels must
contain unblurred paint and all three reference pixels must have valid calibration
support. `reference_extrapolated` and `reference_extension_px` report this query.
This does not extend the output curve domain or invent an observed mask.
`brackets_origin`, `lane_reference_valid` and `ego_lane_valid` remain separate from
lane geometry validity. `ego_offset_px` is lateral, positive vehicle-left of the
centre at the reference row. `ego_offset_m` has the same sign and requires trusted
scale. `BEVMetadata` carries the grid and documented origin; its m/px is nominal
metadata, not evidence that calibration is physically verified.

Coherent evidence rejected by a global fit is exposed in
`excluded_observation_intervals`. `all_observation_rms_px`,
`all_observation_p95_px`, `max_inlier_error_px` and `coherent_bias_max_px`
distinguish full-observation misfit from retained inlier residuals. Sustained signed
bias beyond `near_field_max_error_px` is screened over at least twice
`max_unmodeled_run_px`, to avoid treating short raster/stroke caps as a new curve.

When a global model is truncated or coherently biased, M03 attempts the simplest
scoped recovery: an anchored near-field line, independently verified for BOTH
boundaries. It stops before unsupported curved evidence using
`near_field_guard_px`; it never projects that line through a turn. A unique,
supported near pair can be `valid` with `limited_support=True`,
`support_status="limited_near_field"`, `usable_support_v_px` and
`unsupported_far_intervals`. Pair-level `valid_support_v_px` and `observed_span_px`
describe usable geometry and shared painted span respectively. Far-unsupported
intervals extend from the farthest associated input evidence to the usable model
start. All rows outside `usable_support_v_px` remain unusable, including any
unobserved border rows outside those intervals. A downstream client must inspect these fields AND the
polynomial domain; `valid` alone never promises full-road coverage. Limited curves
return NaN outside their domains. Coherent far-field misfit remains diagnostic.
Forks and competing pairs remain uncertain. Disjoint support caused by a long
occlusion uses `disjoint_lane_support`, rather than a spatial-fork reason.

`fit_endpoint_margin_px` excludes round paint/stroke tips from the centre-curve
domain and refits without them. Diagnostics distinguish final fit inliers from
`endpoint_trimmed_points`. The minimum observed-row floor is applied before this
conservative endpoint margin; final-domain coverage is separately reported. A
smaller median residual alone cannot justify a linear model whose endpoint error
exceeds the configured inlier screening scale.

## Reproducible commands

Run from repository root with the existing `requirements.txt` installed:

```powershell
python -m unittest discover -s lane_detection/advanced -t . -v
python -m unittest discover -v
python -m unittest camera_calibration.video_asset_tests -v
python -m lane_detection.advanced.evaluation --output outputs/m03/synthetic
python -m lane_detection.advanced.benchmark --frames 200 --warmup 20 --threads 1 --output outputs/m03/benchmark.json
python -m lane_detection.advanced --source videos/test.mp4 --profile calibration/test_video_metric_profile.json --headless --output outputs/m03/video_full
```

The bundled video profile has **synthetic, unverified geometry**. The video command
is an integration demonstration, not measured lane-width or track accuracy evidence.
Use `--limit 12` for a partial smoke run. `run_summary.json` distinguishes EOF,
limit, quit and errors, and includes profile origin and frame skips. JSONL preserves
the original IDs; it does not renumber skipped frames. Sampled overlays and candidate,
accepted and rejected PNGs use `--save-every` (default 30). `--headless` makes no GUI
calls. Omit it for windows. Green is observed fit, orange dashes are gaps/inference,
cyan is the selected centreline; masks separately show evidence. Drawing always
copies the shared image. Diagnostic time is included when debug is requested.

For your own image, video, camera (`--source 0`) or stream, replace `--source` and
`--profile` with compatible inputs. Images are processed through M02; **raw-image
pixels are never relabelled as BEV**. Profile source bindings and exact acquisition
resolution remain enforced. Legacy `calibration/ipm_calibration.json` cannot be
used as an M02 profile. The old `python lane_detection/lane_detection.py ...`
remains the separately documented raw-image compatibility path.

`--validation-measurements path.json` re-runs the existing M02 validator. Required
fields: `profile_fingerprint`, `tolerance_m`, `provenance` (operator, method,
reference, independent_from_fit=true), and `distances` containing M02
DistanceMeasurement fields. The fingerprint must match and all M02 evidence/trust
requirements apply. A saved report summary cannot grant trust and is not accepted
in place of independent attested measurements.

## Validation and limits

See [M03_FINAL_REPORT.md](../M03_FINAL_REPORT.md) and
[M03_FINAL_REVIEW_HANDOFF.md](../M03_FINAL_REVIEW_HANDOFF.md) for the latest repairs.
[M03_CORRECTIVE_REPORT.md](../M03_CORRECTIVE_REPORT.md) is historical evidence.
Historical initial evidence is in [M03_VALIDATION.md](../M03_VALIDATION.md), [M03_BENCHMARK.json](../M03_BENCHMARK.json)
and [M03_SYNTHETIC_EVALUATION.json](../M03_SYNTHETIC_EVALUATION.json).
Synthetic pass results exercise software geometry; they do not establish real-track
reliability. Strong reflections, unusual paint colors, severe shadows, occluders,
large gaps, very tight or sideways turns and complex topology remain challenges.
Conservative gates may reject usable lanes; ambiguous pair output intentionally
does not choose a driving lane. Raspberry Pi latency, camera mounting, physical
metric calibration, thermal performance and labelled real-track accuracy remain
unvalidated. There is no ROS, IMU compensation, route/roundabout selection or control.

The marking-width screen rejects tested tapered white/yellow streaks, and saturated
floor evidence is refused. Uniform streaks that reproduce paint color, width and
geometry are still indistinguishable from real paint using this input alone. M02
visibility/calibration support is not a road-surface or obstacle mask. No confidence
threshold establishes semantic certainty; labelled real-track data and an
independent road/free-space source are required to resolve that limitation.

There is no minimum supported hairpin radius expressed solely in pixels. The
linear/quadratic `u(v)` model must be single-valued, remain inside slope/curvature
limits and fit its reported domain without sustained signed bias. At 0.004 m/px,
a 160 px centre radius would nominally be 0.64 m, but that conversion grants no
physical validation. Tight arcs over short visible spans may fit accurately;
long arcs, sideways turns, or turns that double back in v are outside this model's
verified operating range. `coherent_model_misfit` must not be disabled to accept
them. Near straight support can be recovered only with explicit limited scope.
