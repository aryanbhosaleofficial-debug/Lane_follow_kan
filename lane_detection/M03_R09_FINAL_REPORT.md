# M03 final R-09 correction

Verdict: **CONDITIONAL PASS for the bounded software correction; original 8/8
recall acceptance is NOT met and R-09 closure requires independent review.**
The comparison-budget defect is corrected. No claim of unconditional M03 approval,
semantic lane recognition, calibrated confidence, Pi performance or track
readiness is made. Historical M03 reports remain intact and are superseded by
this report for this correction.

## Implementation

`extraction.py` intersects the existing predicted-position and lateral-distance
gates using binary search on sorted row-run centres before counting/scoring
candidates. It preserves one-to-one greedy ordering and tie breaks, dashed-gap
policy, track retirement, all existing capacity caps and the global 4,000 eligible
comparison budget. Neither threshold expansion nor candidate pruning was used
to force recall. Exhaustion produces uncertainty with no selected pair.

For R sampled rows, T=`max_tracks`, J=`max_row_runs`, index work is bounded by
O(R T log(J+1)); scored choices total at most 4,000 per frame. Row/run, active
track and fit-candidate caps still independently apply. Stored observed points
are bounded by (T + `max_fit_candidates` + 1) R, including the candidate-overload
sentinel. Input/preprocessing memory scales with raster size. Caps bound work,
not wall-clock time. No claim of a CPU deadline is made.

`pipeline.py` explicitly refuses references outside the BEV raster before any
reference calculation. `models.py` adds `evaluate_supported_centreline(rows)`:
NaN for invalid/uncertain observations, outside the declared and polynomial
domains, and within reported shared-paint gaps. Existing `.valid` semantics and
unrestricted low-level polynomial APIs remain compatible. Limited polynomials
retain their NaN extrapolation guard. Reference extension never enlarges a curve.

`fitting.py` exposes a multiplicative `residual_quality_factor`, based on existing
configured p95 and sustained signed-bias error tolerances. It penalizes the whole
confidence score when those tolerances are exceeded. The first trial penalized
isolated signed-bias windows and reduced an existing dashed recall check; that
trial was corrected to require the existing sustained-misfit criterion. All
original tests now pass unchanged in purpose. Scores are engineering evidence
scores, not calibrated probabilities. Full-observation errors outside a recovered
near interval remain visible without being misrepresented as its local error.

Hough is now off by default, with `use_hough=True` retained. An identical-input
896-scene ablation yielded identical validity and support, with no demonstrated
detection benefit. Existing Hough-specific tests explicitly enable it and retain
every assertion; a new test checks default-off and optional-on geometry equality.
Canny/debug edge output is retained.

## Original R-09 protocol and recall exceptions

Claude's original `p9_r09a.py` was found in its temporary scratchpad. Its exact
functions are preserved in the evidence bundle. An AST-based identity check
verified all 64 requested 320x240 images byte-for-byte. The reviewer uses longer
fragments (4..15 row displacement, nominal 2 px thickness), different placement
bounds and different intensities than the earlier author's fixture (3..9 rows).
That difference explains why the previous 107-test suite missed the failure.

`r09_cases.py` preserves the original 320x240 straight protocol. Curved, dashed,
multi-lane, companion, no-lane and Gaussian-noise scenes are explicitly additional
fixtures using the same clutter. The 640x480 variant resizes the whole raster by
nearest neighbour and scales algorithm lengths through `LaneConfig.for_bev`;
it is not claimed to be an original reviewer protocol. Pixel width limits are
the reviewer's 45..260 base pixels, rather than the author's narrower bounds.

Before: 60 fragments, 0/8 valid at 320x240; seven budget events. After: **6/8
valid and geometrically correct inside their reported domains**, zero budget
events. At 640x480: **7/8**, zero budget events. The remaining 320 seeds 5 and 7
retain competing fragment evidence:

* Seed 5: an additional supported chain near u=107..118, between true u=70/200
  lanes. Its raw observation intervals include 56..68, 104..113, 143..173 and
  179..185. It passes the existing fragmented-boundary contract and triggers
  `competing_close_markings`/`intervening_marking`.
* Seed 7: another supported chain near u=215..225, beside the true right lane,
  triggers companion-marking ambiguity. At doubled resolution, supported chains
  also trigger the retained branch safeguard.

The ground-truth lane strokes are present. The detector has no semantic label
telling it to ignore those other paint-like chains. Under the current conservative
fragmented/companion contract, **6/8 is the defensible selected-pair maximum at
320x240 without changing ambiguity policy**. This is a contract-based exception
for reviewer assessment, not a proof that a richer detector cannot disambiguate
them. The new regression records these refusals explicitly; no previous test
was removed or weakened to obtain that result. Unconditional 8/8 remains open.

Seed 0 is a correct limited pair on approximately rows 155..233. The original
script tests row 120, obtains NaN and labels the valid result wrong. We report
that raw midpoint result separately: 5 midpoint passes versus 6 scoped-correct
valids. NaN outside a limited domain is the required behavior. No new unsupported
extrapolation was introduced to satisfy the midpoint check.

## Findings

| Finding | Status | Evidence / remaining limit |
|---|---|---|
| R-09 | PARTIAL | Budget defect fixed algorithmically; original unconditional 8/8 not achieved; conservative exceptions above require review. |
| F-01 | FIXED | Actual M02 profile origin v=574.5 exceeds its 500 rows. Out-of-grid references have no ego offset. Explicit in-domain queries tested; no image-centre substitution. |
| O-1 | FIXED | Full-data/inlier error diagnostics retained; new tolerance-normalized confidence factor and monotonic-error tests. No calibration claim. |
| V-1 | PARTIAL | Broad colored floors, tapered streaks and horizontal glare rejected; constant-width paint-like wall/glare remains accepted in semantic audit. |
| D-1 | FIXED (delivery) | New non-ignored report/results/handoff/archive; complete overlay, preserved entry baseline, raw evidence, diff and SHA256 manifest. M03 remains uncommitted by instruction. |
| P3-a | FIXED | Explicit support metadata retained; additive guarded planner query; limited/full/gap/invalid regression cases. |
| P3-b | FIXED (scope) | Geometry limits documented by single-valued model, support span, slope, curvature and misfit; no safeguard relaxation. |
| P3-c | FIXED | Exact 6 px, just below/above, zero, far-side and outside-grid tests. Above-cap queries remain inside the raster; permissive evidence cannot bypass the cap. An isolated in-memory cap-removal mutation is caught by this test. |
| C-1 | PARTIAL | Default Hough disabled after ablation; opt-in retained. Khaki swatch rejected; cream/gold streak confusion and confidence calibration require real labelled data. |

## False-valid rates and geometry limits

The identical before/after 896-scene breadth matrix has zero incorrect-valid
outputs in either engine. This includes 512 ambiguous/no-lane/noise scenes with
zero valid selections. Straight/curve/dashed correct-valid totals improved from
72/70/74 to 105/99/98 out of 128 each. All 399 original breadth capacity refusals
disappeared within these requested clutter levels. Extreme stress still returns
explicit uncertainty when caps are hit.

This is **not** a zero semantic false-positive claim. A separate 20-scene
color/material audit has 15 negative wall/glare/floor fixtures; **8/15 are accepted
before and after**. White, yellow, cream and gold wall/glare streaks reproduce
paint-like geometry; the tested khaki swatch is excluded by saturation bounds.
Broad floors are refused. Identical lane/wall/glare pixels cannot be distinguished
using the available inputs. M02 support masks verify remap visibility, not road
surface or obstacles. Resolving this needs independent road semantics and labelled
physical data; raising a global confidence threshold is not a semantic solution.

The 21-case physical-circle audit accepts 16, maximum centre radial error 0.78 px.
Some tight radii fit short supported spans; R140/R160 with width 60 can be refused
for coherent model misfit. A universal pixel-radius cutoff would misdescribe
this behavior. Sideways/doubling-back turns and long arcs that exceed the
linear/quadratic approximation remain unsupported.

## Protected behavior and integration

The prior P1 evaluation passes 144/144 checks: edge exits 30/30, normal-width
arcs 90/90, companion markings 24/24. The existing transition evaluation passes
72/72 with 60/60 near controls recovered. The original reviewer R-08 140-scene
matrix retains 62/70 solid and 58/70 dashed valid cases in both engines, with zero
limited-domain NaN failures. Four existing solid cases exceed a stricter 2.5 px
maximum boundary-error audit (worst 2.745 px); neither count nor errors increase.
Their centre errors remain at most 2.269 px. These are exposed rather than
described as universally subpixel geometry. The original source functions are
executed with portable imports/adapters; raster and geometry equations are intact.

M01/M02 production files are untouched. Tests preserve calibration trust through
`validated_geometry()`; status alone does not grant trust. Unverified fixtures
remain pixel-only. Source/session/frame/profile identity and acquisition timestamps
are preserved; masks/images are read-only; repeats return the cached observation
only after freshness checks, identity collisions/backward IDs/future/stale stamps
remain refused. Ground coordinates remain X forward/Y left, with BEV u right/v
down. Actual M01 -> M02 -> M03 image-warp and simulated metric-attestation tests
pass. Synthetic attestation is software-gate testing, not physical certification.

## Tests and reproduction

Run at repository root in PowerShell. Raw logs/results and saved reviewer scripts
are in the archive under `evidence/`; restore them to `outputs/m03/r09_final` to
reuse these paths. Baseline is the entry **107-test** engine from this round,
not the older 85-test package. Never put the saved package at repository root,
where it could participate in unittest discovery.

```powershell
python -m unittest discover -s camera_input -t . -v
python -m unittest discover -s camera_calibration -t . -v
python -m unittest discover -s lane_detection/advanced -t . -v
python -m unittest discover -v
python -m unittest camera_calibration.video_asset_tests -v
python -m lane_detection.advanced.review_evaluation --output outputs/m03/r09_final/p1.json
python -m lane_detection.advanced.transition_evaluation --output outputs/m03/r09_final/transitions.json
python -m lane_detection.advanced.reviewer_reproduction --reviewer-dir outputs/m03/r09_final --baseline-dir outputs/m03/r09_final/baseline --output outputs/m03/r09_final/reviewer_before.json
python -m lane_detection.advanced.reviewer_reproduction --reviewer-dir outputs/m03/r09_final --output outputs/m03/r09_final/reviewer_after.json
python -m lane_detection.advanced.r09_evaluation --baseline-dir outputs/m03/r09_final/baseline --repeats 20 --output outputs/m03/r09_final/performance_before.json
python -m lane_detection.advanced.r09_evaluation --repeats 20 --output outputs/m03/r09_final/performance_after_on.json
python -m lane_detection.advanced.r09_evaluation --repeats 20 --no-hough --output outputs/m03/r09_final/performance_after_default.json
python -m lane_detection.advanced.r09_evaluation --baseline-dir outputs/m03/r09_final/baseline --repeats 1 --breadth --output outputs/m03/r09_final/breadth_before.json
python -m lane_detection.advanced.r09_evaluation --repeats 1 --breadth --output outputs/m03/r09_final/ablation_on.json
python -m lane_detection.advanced.r09_evaluation --repeats 1 --breadth --no-hough --output outputs/m03/r09_final/ablation_off.json
python -m lane_detection.advanced.r09_evaluation --repeats 2 --stress --no-hough --output outputs/m03/r09_final/stress.json
python -m lane_detection.advanced.semantic_evaluation --baseline-dir outputs/m03/r09_final/baseline --output outputs/m03/r09_final/semantics_before.json
python -m lane_detection.advanced.semantic_evaluation --output outputs/m03/r09_final/semantics_after.json
git diff --check
```

Final counts, raw latency samples, stage/budget/track diagnostics, memory results,
image hashes and every changed-file path are in
[M03_R09_FINAL_RESULTS.json](M03_R09_FINAL_RESULTS.json). Performance tables and
final artifact checks are appended below after measurement.

<!-- GENERATED EVIDENCE -->

## Final measured performance

Intel Core i5-13450HX, Windows 11 build 26300, Python 3.14.10, OpenCV 5.0.0, NumPy 2.5.4; one OpenCV thread. Eight seeded images/group, two warmups and 20 timed repeats per image (160 samples/group). Timings exclude acquisition, drawing, encoding and memory tracing. The three runs used identical image hashes. Hough-on compares like-for-like; default-off measures the final configuration. Background OS scheduling remains uncontrolled. The 320 clean maximum increased despite lower median/p95; no hard worst-case deadline is claimed.

Latency cells are **median / p95 / max ms**. Recall is scoped-correct valid scenes out of eight.

| Size / case | Entry, Hough on | Corrected, Hough on | Corrected default off | Recall entry→corrected | Budget events entry→corrected |
|---|---|---|---|---|---|
| 320×240 / fragments 0 | 6.03 / 7.02 / 8.37 | 5.58 / 7.39 / 11.08 | 4.76 / 5.58 / 10.18 | 8→8 | 0→0 |
| 320×240 / fragments 40 | 11.41 / 12.82 / 17.52 | 9.40 / 10.69 / 16.45 | 8.11 / 9.55 / 15.00 | 8→8 | 0→0 |
| 320×240 / fragments 60 | 8.74 / 17.30 / 19.10 | 11.92 / 16.16 / 22.43 | 10.34 / 14.16 / 19.15 | 0→6 | 7→0 |
| 320×240 / fragments 80 | 7.29 / 8.05 / 11.79 | 15.10 / 18.42 / 21.05 | 12.99 / 16.76 / 20.59 | 0→4 | 8→0 |
| 320×240 / fragments 100 | 6.55 / 7.08 / 7.49 | 19.46 / 25.54 / 34.82 | 16.99 / 21.82 / 27.51 | 0→1 | 8→0 |
| 320×240 / specks 100 | 7.56 / 9.07 / 15.93 | 6.77 / 7.68 / 14.03 | 5.71 / 6.77 / 9.91 | 8→8 | 0→0 |
| 320×240 / specks 200 | 9.94 / 11.62 / 14.82 | 8.45 / 9.23 / 12.04 | 6.79 / 9.72 / 12.90 | 8→8 | 0→0 |
| 320×240 / specks 300 | 12.34 / 14.08 / 22.20 | 10.51 / 11.46 / 12.99 | 7.77 / 9.12 / 20.86 | 8→8 | 0→0 |
| 640×480 / fragments 0 | 8.80 / 9.65 / 12.70 | 8.25 / 9.53 / 13.15 | 7.13 / 8.32 / 10.25 | 8→8 | 0→0 |
| 640×480 / fragments 40 | 14.52 / 16.51 / 19.34 | 12.58 / 13.71 / 20.81 | 10.40 / 12.27 / 18.60 | 8→8 | 0→0 |
| 640×480 / fragments 60 | 10.03 / 11.22 / 13.04 | 14.94 / 18.21 / 26.83 | 11.97 / 14.88 / 16.82 | 0→7 | 8→0 |
| 640×480 / fragments 80 | 8.78 / 9.86 / 11.76 | 18.07 / 20.77 / 23.16 | 14.76 / 17.59 / 26.29 | 0→5 | 8→0 |
| 640×480 / fragments 100 | 8.03 / 9.18 / 11.05 | 22.82 / 28.80 / 37.69 | 18.68 / 24.07 / 36.35 | 0→2 | 8→0 |
| 640×480 / specks 100 | 11.03 / 12.15 / 13.66 | 9.87 / 11.18 / 16.53 | 8.55 / 9.74 / 13.31 | 8→8 | 0→0 |
| 640×480 / specks 200 | 15.14 / 17.35 / 29.78 | 12.97 / 14.50 / 20.64 | 10.10 / 11.54 / 16.81 | 8→8 | 0→0 |
| 640×480 / specks 300 | 9.08 / 10.07 / 12.55 | 15.94 / 19.92 / 29.43 | 12.08 / 16.05 / 21.71 | 0→8 | 8→0 |

All requested straight-scene groups have zero incorrect-valid outputs before/after. At 60 fragments, maximum counted comparisons are 4,000→601 (320) and 4,000→634 (640); peak active tracks remain 19 and 20 respectively. Comparison counts have different accounting semantics: entry counts every active-track/run pair; corrected counts only spatially eligible pairs. The 240-scene increased stress matrix has zero incorrect-valid results and 76 conservative capacity exits; peak active tracks 48, maximum scored comparisons 3,562. All capacity exits expose uncertainty, not partial selected geometry.

### Memory

The separate existing author 200-speck fixture uses ten warmups plus 100 new frames at each resolution. Python tracemalloc excludes native OpenCV allocations; process working-set data are also saved. RSS peaks are cumulative for the process.

| Mode / size | Python peak bytes | Live growth bytes | Process peak working set bytes |
|---|---:|---:|---:|
| before / 320×240 | 2355814 | 7935 | 54743040 |
| before / 640×480 | 5029619 | 10661 | 59088896 |
| after_default / 320×240 | 1329501 | 9389 | 59088896 |
| after_default / 640×480 | 5004759 | 8002 | 59457536 |

No frame-accumulating track store was observed. Bounds and repeated-frame tests provide stronger evidence than interpreting a short memory trend as a proof of no possible leak.

## Final regression counts

| Suite | Passed | Failed | Skipped |
|---|---:|---:|---:|
| M01 | 46 | 0 | 0 |
| M02 | 284 | 0 | 0 |
| M03 | 121 | 0 | 0 |
| full_discovery | 454 | 0 | 0 |
| video_assets | 2 | 0 | 0 |
| Fresh HEAD plus packaged overlay | 454 | 0 | 0 |

Fourteen new test methods supplement the entry 107 M03 tests; original assertions are retained. CLI reaches EOF at 590 observations, source IDs 1..590, no skipped source frames; profile remains SYNTHETIC / METRIC_UNVERIFIED. Negative-input tests intentionally emit camera/decode diagnostics; suite status comes from unittest summaries.

Additional executed commands:

```powershell
python -m unittest lane_detection.advanced.test_r09_spatial -v
python -m unittest lane_detection.advanced.test_final_support lane_detection.advanced.test_final_interfaces -v
python -m unittest lane_detection.advanced.test_final_quality -v
python outputs/m03/r09_final/association_equivalence.py
python outputs/m03/r09_final/verify_cap_mutation.py
python -m lane_detection.advanced --source videos/test.mp4 --profile calibration/test_video_metric_profile.json --headless --output outputs/m03/r09_final/video
python outputs/m03/r09_final/finalize_evidence.py
```

Memory was measured by `availability_evaluation.memory_report(Path("outputs/m03/r09_final/baseline"), 100)` and `memory_report(None, 100)` in a separate one-thread Python process. All intermediate trial logs, including failed new-fixture assumptions and the corrected isolated-bias experiment, are retained in the bundle.

## Changed-file manifest for this round

Compare against the entry snapshot, not Git HEAD, which has no M03 package.

| File | Kind | Change |
|---|---|---|
| `lane_detection/advanced/config.py` | production | modified |
| `lane_detection/advanced/extraction.py` | production | modified |
| `lane_detection/advanced/fitting.py` | production | modified |
| `lane_detection/advanced/models.py` | production | modified |
| `lane_detection/advanced/pipeline.py` | production | modified |
| `lane_detection/advanced/r09_cases.py` | evaluation | added |
| `lane_detection/advanced/r09_evaluation.py` | evaluation | added |
| `lane_detection/advanced/README.md` | documentation | modified |
| `lane_detection/advanced/reviewer_reproduction.py` | evaluation | added |
| `lane_detection/advanced/semantic_evaluation.py` | evaluation | added |
| `lane_detection/advanced/test_extraction.py` | test | modified |
| `lane_detection/advanced/test_final_quality.py` | test | added |
| `lane_detection/advanced/test_final_support.py` | test | added |
| `lane_detection/advanced/test_pipeline.py` | test | modified |
| `lane_detection/advanced/test_r09_spatial.py` | test | added |

Delivery artifacts: `M03_R09_FINAL_REPORT.md`, `M03_R09_FINAL_RESULTS.json`, `M03_R09_REVIEW_HANDOFF.md`, `M03_R09_REVIEW_BUNDLE.zip`, `M03_R09_REVIEW_BUNDLE.sha256`, all under `lane_detection/`. They are not Git-ignored. `.gitignore`, root README and lane_detection README already differed at entry and were preserved. Other M03 files were already untracked and remain present. No M01/M02 production or legacy detector file changed. The archive includes the whole current M03 overlay, entry snapshot, `source.diff`, evidence and a per-member SHA256 manifest; its sidecar hashes the archive itself.

## Remaining risks and approval boundary

Software: original unconditional 8/8 recall is unmet; conservative ambiguity causes refusals. Constant-width non-road materials remain semantic false positives. Confidence is heuristic, and four original transition cases exceed a 2.5 px boundary audit. Low-level polynomial evaluation remains unrestricted for backward compatibility; consumers must respect domains or use the new guarded query. Measured desktop maxima vary with scheduling.

Missing physical evidence: vehicle reference/calibration provenance, labelled real-track recognition and score calibration, Pi 5 CPU/thermal latency and physical vehicle behavior. These cannot be established by synthetic fixtures or desktop execution. No newly introduced P0/P1 was identified. R-09 budget correction is ready for independent inspection, but unconditional M03 software approval should not be asserted until the documented recall exception is accepted or resolved.
