# M03 corrective implementation

This is the corrective record following the supplied independent review. Initial
M03 source was already uncommitted/untracked; it was preserved under ignored
`outputs/m03/corrective/baseline/`, with SHA-256 hashes. No commit/push/reset was
performed. M01/M02 production code is unchanged. The final evidence and reviewer
handoff below describe the delivered working tree. Only the supplied grouped
review findings were available; no separate Claude report/scripts were present.

## A. Fix summary

| Finding | Reproduction and correction |
|---|---|
| M03-01 | Side-clipped row centroids biased fits. Discard clipped runs; conservatively trim/refit stroke caps. Check ROI only over endpoint spans participating in shared normal correspondences. Unsupported candidate sections outside that domain cannot reject a supported pair. Score conditional density while retaining the observed-row support floor. Of 30 mirrored slope/exit combinations, 28 have enough common support; the two slope=.15/exit=140 cases lack the configured 65px span and remain invalid. |
| M03-02 | Laterally shifted quadratics did not test normal width. Solve directed normal intersections in both directions, refuse ambiguous roots/domain failures, measure Euclidean normal widths, and fit centreline to actual correspondence midpoints with bounded approximation error. |
| M03-03 | Narrow rejected pairs erased alternatives. Preserve close-marking ambiguity and check intervening markings across the shared domain. All 24 inner/outer companion configurations are uncertain rather than silently selecting a different edge. |
| M03-04 | Added explicit pixel reference or explicit ground-origin-as-vehicle declaration. Geometry validity, bracketing and ego-lane validity are independent; unsupported reference queries do not extrapolate. |
| M03-05 | Added endpoint observed/inferred fractions, shared support, gaps and reason codes using uniform centre-domain bins and normal endpoint correspondences. |
| M03-06 | Default .5s monotonic freshness checked before work/cache reuse and before output; invalid/future stamps rejected. Explicit offline replay bypasses age limits, not timestamp integrity. CLI keeps live upstream freshness. |
| M03-07 | Added independent straight/physical-circle ground textures rendered through actual M01/M02; assert both boundaries, centreline, status, confidence, trust, coordinates and input integrity. |
| M03-08 | M02 straight-to-arc scene reproduced silently discarded coherent evidence. Expose excluded observations and retained fraction; coherent unsupported sections force uncertainty without a selected centreline. No expensive global model expansion was needed. |
| M03-09 | Vectorized all row runs; cached row indices; bounded association comparisons and fit candidates. Incomplete budget-limited frames are uncertain and never fitted/paired as complete observations. |
| M03-10 | Added backward-compatible `LaneConfig.for_bev`; scale algorithm lengths and curvature screening, record nominal physical assumptions, bind config to scale, and optionally require validated metric geometry. |

### P3 disposition

| Concern | Disposition |
|---|---|
| Local-contrast white threshold ignored | Fixed. The white brightness floor is honored; shadow demos explicitly configure 90 rather than silently borrowing yellow_min_value. |
| Saturated masks lack diagnostics | Fixed. Fraction of supported ROI containing evidence is exported; saturated masks refuse extraction with a reason. |
| Orange/tan admitted as yellow | Confirmed for two independent swatches. Tightened default hue/saturation and added rejection tests; arbitrary materials with indistinguishable colors remain unverified. Thresholds stay configurable. |
| Hough cost versus limited contribution | Bounded, optional, bypassed on incomplete/saturated frames; geometry remains available without Hough. A measured on/off ablation is recorded in final results. |
| 0.85 calibration confidence penalty | Removed. Pixel evidence score is independent of metric trust. Metric exports/constraints still require validated_geometry; regression now verifies this separation. |
| Lateral midpoint approximation | Fixed using normal-correspondence midpoints. Exact samples accompany the bounded-error polynomial. Prior inference also uses a normal offset and refuses folds/inaccurate approximations. |
| Tight-arc curvature overestimation | Rejected as an analytic-formula defect: the formula is correct for the fitted quadratic, not an exact circle estimate. Model error was confirmed and addressed with bounded fitting/domain checks and explicit unsupported status. Added signed convention and a physical-radius matrix. Exact road curvature beyond the model remains unsupported. |
| Far-field forks | Full shared-domain alternative checks plus a far-field fork regression; no near-field-only pair selection. This is not a complete semantic junction classifier. |
| BEV metadata/signs/config/report support | Added immutable grid/origin metadata and signed curvature. CLI validates configuration with actionable parse errors and can regenerate trusted M02 runtime reports from independent attested measurements. Persisted report summaries are intentionally not accepted as trust. |

### API and policy changes

All original constructors/exports remain, with defaulted additional fields.
Required corrections change width semantics to normal distance; evidence confidence
no longer discounts unverified scale; local-contrast white thresholds now mean what
they say; and default freshness is .5s. Offline callers must explicitly select
`max_frame_age_s=None`. `Polynomial` remains quadratic/linear, with exact normal
centre samples and unsupported intervals supplied separately. No route selection
or persistent tracking was introduced.

Pair observability denominators and reference sign conventions are specified in
[advanced/README.md](advanced/README.md). Physical configuration does not confer
calibration trust. The signed curvature is positive for a vehicle-left turn as X
increases; absolute curvature refers to fitted geometry only.

## B. Changed files

Relative to the initial saved M03 source, modified production files under
`lane_detection/advanced/`:

| Files | Purpose |
|---|---|
| `association.py`, `models.py` | Normal geometry, alternatives, pair evidence and downstream contracts |
| `extraction.py`, `fitting.py` | Clipped/cap support, bounded work and explicit model exclusions |
| `preprocessing.py`, `config.py` | Honest threshold semantics, saturation and scale/freshness/reference policy |
| `pipeline.py`, `inference.py` | Safe output policy, reference query and normal-prior inference |
| `cli.py`, `__init__.py` | Live/offline behavior, attested validation input and metadata export |
| `benchmark.py`, `evaluation.py` | Explicit offline benchmark policy and corrected shadow configuration |

New production/helpers: `geometry.py` (normal intersections and centre model),
`observability.py` (pair support), `adversarial.py` (independent geometry fixtures),
`review_benchmark.py` (comparable saved-source benchmark), `review_evaluation.py`
(physical acceptance matrix).

New tests: `test_review_p1.py`, `test_review_contracts.py`,
`test_review_integration.py`, `test_review_scaling.py`, `test_review_cleanup.py`.
Modified tests: `test_fitting.py`, `test_pipeline.py`, `test_preprocessing.py`,
`test_scenarios.py`.

Documentation: updated `advanced/README.md`, root `README.md`, and historical
`M03_DESIGN.md`/`M03_VALIDATION.md` supersession notices. Added this report,
`M03_CORRECTIVE_RESULTS.json` (raw evidence), and `M03_REVIEWER_HANDOFF.md`.
No files were removed. No dependencies or M01/M02/legacy production code changed.

Test expectation changes correct explicit contract errors rather than hide
failures: shadow tests now set their white floor; calibration no longer changes an
identical pixel evidence score; known robust inliers are accounted for as final
inliers plus exactly two conservative endpoint trims. Geometry tolerances were
not relaxed. Sparse-dash validity still has an explicit regression. The matrix
requires moderate supported arcs to succeed, so rejecting every arc cannot pass.

## C. Before/after P1 results

The initial test-first reproduction had 63 failing subcases: 30 edge cases,
10 anchored normal-width cases and 23 companion cases. Independent support
analysis subsequently identified the two shallow/late edge cases as correctly
unsupported, as required by the acceptance policy.

| Acceptance set | Initial source | Corrected source |
|---|---|---|
| 30 mirrored edge cases | 0 valid; only 2 expected-invalid controls passed | 28 valid, 2 correctly unsupported; maximum boundary error 0.120 px |
| 90 physical circle cases | 50/90 policy/error checks passed; 9 valid | 90/90 checks passed; 73 valid, 17 explicitly unsupported |
| 24 companion configurations | 23 returned valid; 12 selected the wrong centre (>0.5 px error), worst 22.5 px | 24 uncertain, zero selected centrelines |

Circles use radii 220/300/400/700/1200 px, widths 60/100/140 px, orientations
-0.35/0/+0.35 and row spans 35..205 or 65..175. Width bounds are truth +/-3 px;
the accuracy matrix configures 0.75 px RMS and 1 px inlier limits to support its
1.5 px geometry target, rather than using broad default screening tolerances.
All valid cases satisfy that target: worst boundary/centre/mean-width errors are
0.948 / 0.858 / 0.201 px respectively. Twelve tight/steep model cases are explicitly
unsupported; five short normal-overlap cases lack sufficient shared domain.
Expected normal overlap is derived independently from concentric-circle angular
overlap, including the two 3px cap margins and row-sampling allowance. Moderate
cases with enough support are required to return valid. The JSON records every
combination and rejection reason; all 144 adversarial checks passed.

Additional shared-ROI and stroke-cap regressions were added after the wider matrix
and final diff audit exposed failures. No P1 is left unresolved in the exercised
software acceptance conditions.

## D. Testing report

Executed from repository root, zero failures/skips:

| Command | Passed | Duration |
|---|---:|---:|
| `python -m unittest discover -s lane_detection/advanced -t . -v` | 85 | 1.813 s |
| `python -m unittest camera_input.test_camera -v` | 46 | 5.835 s |
| `python -m unittest discover -s camera_calibration -t . -p "test_*.py" -v` | 284 | 43.115 s |
| `python -m unittest discover -v` final | 418 | 37.283 s |
| `python -m unittest camera_calibration.video_asset_tests -v` | 2 | 32.999 s |
| `python -m lane_detection.advanced.review_evaluation --output outputs/m03/corrective/evaluation_after.json` | 144 cases | recorded in JSON |
| `python -m lane_detection.advanced.evaluation --output outputs/m03/corrective/synthetic` | 13 scenes | recorded in JSON |

The root count includes the module suites; those repeated commands are not
additional unique tests. M03 increased from 54 to 85 methods. New tests exercise
mirrored side exits, normal arcs, ambiguous/no normal roots, companions, stale
cache hits, invalid timestamps, explicit references, sparse pair evidence,
same-geometry scaling, physical M02 warps, transition exclusions, work budgets,
color/mask failures, CLI validation, and actual trust-gate behavior. Existing
negative-path warning logs are expected. Compilation, JSON and whitespace/source
integrity checks are included in final verification.

## E. Performance report

Measured desktop CPU results are recorded in `M03_CORRECTIVE_RESULTS.json`,
including median/p95/worst latency, throughput, stage medians, image hashes and
Hough ablation. Hardware: Intel Core i5-13450HX (10 cores, 16 logical processors),
Windows 11 build 26300, Python 3.14.10, OpenCV 5.0.0, NumPy 2.5.4. One OpenCV
thread; other backend thread settings and CPU frequency were not locked. Each
case uses 10 warmups and 100 timed unique-source frames. Acquisition, rendering,
GUI, serialization and encoding are excluded. Debug is disabled. These are
sampled observations, not guaranteed deadlines or Raspberry Pi 5 measurements.

| Workload | BEV | Before median / p95 / worst ms | After median / p95 / worst ms | After FPS |
|---|---|---:|---:|---:|
| clean | 320x240 | 5.363 / 5.858 / 7.112 | 4.801 / 5.278 / 5.500 | 205.0 |
| clutter | 320x240 | 32.340 / 49.286 / 70.983 | 5.116 / 5.779 / 6.077 | 192.7 |
| clean | 640x480 | 11.341 / 23.739 / 26.106 | 9.426 / 10.061 / 10.416 | 105.6 |
| clutter | 640x480 | 32.170 / 54.237 / 66.667 | 9.778 / 17.869 / 23.160 | 91.3 |

Final corrected internal stage medians (ms):

| Workload / BEV | Preprocess + validation | Extraction | Fitting | Association + contracts |
|---|---:|---:|---:|---:|
| clean 320x240 | 0.840 | 2.107 | 1.069 | 0.688 |
| clutter 320x240 | 0.955 | 4.038 | 0.001 | 0.020 |
| clean 640x480 | 2.595 | 4.483 | 1.337 | 0.727 |
| clutter 640x480 | 2.957 | 6.600 | 0.001 | 0.030 |

Hough clean-frame on/off median ms:
* 320x240: 4.801 / 4.555; on/off p95 5.278 / 5.037.
* 640x480: 9.426 / 8.138; on/off p95 10.061 / 8.792.

The same seeded 160-fragment images have matching SHA-256 hashes in all runs.
Runs were sequential without other validation jobs, but turbo/frequency/background
scheduling were not controlled. Initial repetitions varied substantially; these
final distributions are reproducible workloads, not a universal speedup guarantee.
The deterministic work limits explain the reduced clutter work; a wall-clock
deadline is not asserted. Stage medians need not sum to the total median.

Clean cases remain 100/100 valid; baseline and corrected clutter cases remain
100/100 uncertain. Corrected incomplete frames deliberately return no partial
geometry. The cost reduction does not claim improved clutter recall. Hough ablation
keeps geometry/validity behavior and measures its limited clean-frame contribution;
clutter already bypasses it, so clutter timing differences are scheduling noise.

## F. M01/M02 compatibility and end-to-end behavior

All M01/M02 source and requirements are unchanged. Identity, acquisition timestamps,
read-only input bytes, actual coordinate conversion and trust restrictions are
asserted through actual M01/M02 processing. Frames remain independent; no temporal
prediction, navigation or vehicle control was added. CLI metric reports must be
freshly issued by M02 from independent attested evidence; JSON summaries cannot
grant trust. Live freshness remains enforced upstream and in M03; replay policy
is explicit.

The final headless recorded-video check reaches EOF at IDs 1..590, with 590 M03
observations and zero skips in that run. States: 311 uncertain, 278 invalid and one
geometrically valid pixel pair (source frame 18). All are METRIC_UNVERIFIED;
zero metric exports and zero ego-valid lanes. The valid pixel pair was independently
reprocessed and visually inspected. Its physical correctness cannot be established
from the synthetic, mismatched profile and unlabelled recording. This is an I/O/
provenance check, not evidence of real-track accuracy. Earlier loaded runs skipped
frames; latest-only delivery is still not a per-frame delivery guarantee.

## G. Outstanding issues and validation limits

No physical camera, competition-track ground truth, sustained Raspberry Pi 5 run,
interactive GUI or alternate library-version validation was performed. The bundled
synthetic video profile is not a measured physical calibration. Complex transitions
can be explicitly uncertain instead of fitted; operation limits bound work rather
than guarantee a wall-clock deadline. Color thresholds and statistical confidence
still need labelled real-track evaluation. These are not vehicle-readiness claims.

No confirmed P1 or P2 blocker remains in the stated software acceptance conditions.
Broader material/color discrimination, exact physical curvature beyond the fitted
model, complete junction semantics and a richer transition model remain limited
or unverified, as listed in the P3 dispositions. Conservatism can reduce recall.

## H. Final status

**CONDITIONAL PASS**: all confirmed blocking software findings are repaired and
the relevant regression/adversarial/integration checks pass. The working tree is
ready for independent software re-review. Physical calibration, labelled
real-track correctness and Raspberry Pi 5 deployment evidence remain absent.
The unverified recording's lone pixel-valid hypothesis is explicitly disclosed,
not presented as a verified driving lane.

## I. Reviewer handoff

See [M03_REVIEWER_HANDOFF.md](M03_REVIEWER_HANDOFF.md) for exact reproduction
commands, changed-file inventory, policy changes and focused review questions.
[M03_CORRECTIVE_RESULTS.json](M03_CORRECTIVE_RESULTS.json) contains before/after
matrices, raw timings, video outcomes, source hashes and test counts. Ignored
`outputs/m03/corrective/` contains the initial source snapshot, source patch,
logs' supporting exports, masks and visually inspected artifacts. Initial failing
test logs are `m03_p1_before.log`, `m03_contracts_before.log`,
`m03_integration_before.log`, `m03_scaling_before.log`, `m03_cleanup_before.log`;
final suite logs use `m03_corrective_*.log`.
