# M03 independent re-review handoff

Review the current **working tree**, not only Git HEAD. Initial M03 was already
uncommitted/untracked. Nothing was committed/pushed; do not reset/clean the work.
Read [M03_CORRECTIVE_REPORT.md](M03_CORRECTIVE_REPORT.md) and
[M03_CORRECTIVE_RESULTS.json](M03_CORRECTIVE_RESULTS.json).

## Reproduce software checks

From repository root with the existing requirements installed:

```powershell
python -m unittest discover -s lane_detection/advanced -t . -p "test_review_*.py" -v
python -m unittest discover -s lane_detection/advanced -t . -v
python -m unittest camera_input.test_camera -v
python -m unittest discover -s camera_calibration -t . -p "test_*.py" -v
python -m unittest discover -v
python -m unittest camera_calibration.video_asset_tests -v
python -m lane_detection.advanced.review_evaluation --output outputs/m03/corrective/reviewer_evaluation.json
```

Expected final counts: 31 corrective methods, 85 total M03 methods, 46 M01,
284 M02, 418 root-discovery tests, two explicit video-asset tests. All pass with
zero skips. The 144-case matrix passes with 28 valid side-exit cases, two expected
unsupported exits, 73 valid physical-circle cases, 17 unsupported circle cases,
and no valid selections in 24 companion cases. Counts from repeated module/root
runs should not be summed as distinct tests.

The saved initial source exists locally at `outputs/m03/corrective/baseline/`.
Its hashes and the before/after results are preserved. To independently rerun the
initial failures or comparable timing on this workspace:

```powershell
python -m lane_detection.advanced.review_evaluation --baseline-dir outputs/m03/corrective/baseline --output outputs/m03/corrective/reviewer_before.json
python -m lane_detection.advanced.review_benchmark --baseline-dir outputs/m03/corrective/baseline --frames 100 --warmup 10 --output outputs/m03/corrective/reviewer_perf_before.json
python -m lane_detection.advanced.review_benchmark --frames 100 --warmup 10 --output outputs/m03/corrective/reviewer_perf_after.json
python -m lane_detection.advanced.review_benchmark --frames 100 --warmup 10 --disable-hough --output outputs/m03/corrective/reviewer_no_hough.json
```

The baseline snapshot is ignored and may be absent from a fresh transfer. In that
case the corrected acceptance checks still run independently; consult the saved
before records and obtain the snapshot/source patch for a live before comparison.
Benchmarks set one OpenCV thread, use identical image hashes, exclude acquisition/
GUI/I/O, and label budget-exhausted frames uncertain. Repeat locally rather than
treating these Windows desktop numbers as Pi performance.

```powershell
python -m lane_detection.advanced --source videos/test.mp4 --profile calibration/test_video_metric_profile.json --headless --save-every 100 --output outputs/m03/corrective/reviewer_video
```

This is unverified synthetic-profile geometry. The final inspected run processed
all 590 source frames, returned 311 uncertain/278 invalid/one pixel-valid pair,
and exported no trusted metric or ego-valid lane. Source frame 18's physical
correctness is unproven. Do not use this run to approve vehicle operation.

## Changed-file inventory

Under `lane_detection/advanced/`, modified:

`__init__.py`, `association.py`, `benchmark.py`, `cli.py`, `config.py`,
`evaluation.py`, `extraction.py`, `fitting.py`, `inference.py`, `models.py`,
`pipeline.py`, `preprocessing.py`, `README.md`, `test_fitting.py`,
`test_pipeline.py`, `test_preprocessing.py`, `test_scenarios.py`.

Added: `adversarial.py`, `geometry.py`, `observability.py`, `review_benchmark.py`,
`review_evaluation.py`, `test_review_cleanup.py`, `test_review_contracts.py`,
`test_review_integration.py`, `test_review_p1.py`, `test_review_scaling.py`.

Also updated root `README.md`, `lane_detection/M03_DESIGN.md`,
`lane_detection/M03_VALIDATION.md`; added `lane_detection/M03_CORRECTIVE_REPORT.md`,
`lane_detection/M03_CORRECTIVE_RESULTS.json`, this handoff. Removed none.
M01/M02 production, legacy detector/IPM and requirements are unchanged.

## Focus the review

1. Normal intersections, multiple-root refusal, midpoint approximation bounds,
   analytic ordering, shared-domain ROI checks and cap/clipped-run support.
2. Alternatives retained even when narrow pairs are rejected; no selected
   centreline under companion/fork/model/capacity ambiguity.
3. Pair support denominators use uniform centre-v bins with actual normal endpoint
   rows. Evidence gaps and inferred boundaries remain distinguishable.
4. Geometry validity versus explicit ego reference; unknown reference never becomes
   an ego-valid lane. Signs are vehicle-left-positive for offsets/curvature.
5. Default .5s freshness, repeated-frame checks, positive/future timestamps and
   explicit offline replay; live M02 freshness is preserved.
6. No trust from nominal BEV scale or imported summaries. CLI measurements are
   revalidated by M02 with exact profile binding and its unchanged TrustPolicy.
7. Budget exhaustion returns uncertain with no partial geometry. Evaluate the
   recall/performance tradeoff rather than assuming faster clutter means detection.

Known limits: quadratic u(v), conservative transition refusal, sampled normal
geometry, heuristic confidence and fork detection, configurable color assumptions,
unverified real-track accuracy and Pi thermal/latency behavior. Original report
scripts were not provided; the acceptance scenes were independently constructed.
