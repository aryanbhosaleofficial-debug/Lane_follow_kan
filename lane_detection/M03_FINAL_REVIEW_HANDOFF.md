# Claude Sonnet 5.5: M03 final review handoff

Review against Git HEAD `5dc8defbe291f0c3882dc955527503a96dc892e9` plus this
uncommitted M03 overlay. No commit/push was authorized. The archive's `baseline/`
is the immutable initial 85-test package; `overlay/` is current source and docs;
`evidence/` contains raw results/logs; `source.diff` compares initial/current M03.
Verify `manifest.json` hashes before review. Earlier 54-test evidence is historical.

Read [M03_FINAL_REPORT.md](M03_FINAL_REPORT.md) and the raw
[M03_FINAL_RESULTS.json](M03_FINAL_RESULTS.json). Inspect:

1. Active-track retirement and all budget exits in `advanced/extraction.py`.
2. Near-only recovery in `fitting.py` and `pipeline.py`; independently verify both
   edge fits, explicit valid domain, unsupported far evidence and NaN outside.
3. Prior edge-exit/normal-width/companion ambiguity checks remain unchanged in
   purpose and ground-truth tolerances. The arc acceptance test now requires an
   actual supported success; the transition matrix requires 60/60 near controls.
4. Keyword-only configuration additions retain the original 45 positional fields.
5. Zero shared painted coverage defaults to uncertainty, with explicit opt-in.
6. Tapered paint-width rejection does not establish road semantics. Constant-width
   paint-like glare/walls and floor materials remain a documented P3 limitation.

Changed-file inventory and reasons are in the report and hash manifest. No
M01/M02 production modules, legacy detector or requirements were edited.

## Exact commands (repository root, PowerShell)

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s camera_input -t . -v
python -m unittest discover -s camera_calibration -t . -v
python -m unittest discover -s lane_detection/advanced -t . -v
python -m unittest discover -v
python -m unittest camera_calibration.video_asset_tests -v
python -m unittest lane_detection.advanced.test_final_availability lane_detection.advanced.test_final_interfaces lane_detection.advanced.test_final_evidence -v
python -m lane_detection.advanced.review_evaluation --output outputs/m03/final_review_p1.json
python -m lane_detection.advanced.transition_evaluation --output outputs/m03/final_transitions.json
python -m lane_detection.advanced.transition_evaluation --baseline-dir outputs/m03/final_correction/baseline --output outputs/m03/before_transitions.json
python -m lane_detection.advanced.availability_evaluation --repeats 20 --memory-frames 100 --output outputs/m03/after.json
python -m lane_detection.advanced.availability_evaluation --baseline-dir outputs/m03/final_correction/baseline --repeats 20 --memory-frames 100 --output outputs/m03/before.json
python -m lane_detection.advanced.availability_evaluation --repeats 20 --no-hough --output outputs/m03/no_hough.json
python -m lane_detection.advanced.availability_evaluation --repeats 10 --stress --output outputs/m03/stress.json
python -m lane_detection.advanced.availability_evaluation --geometry curve --repeats 5 --output outputs/m03/curve_clutter.json
python -m lane_detection.advanced.availability_evaluation --geometry dashed --repeats 5 --output outputs/m03/dashed_clutter.json
python -m lane_detection.advanced.availability_evaluation --geometry competing --repeats 5 --output outputs/m03/competing.json
python -m lane_detection.advanced.benchmark --frames 100 --warmup 10 --threads 1 --output outputs/m03/stages.json
python -m lane_detection.advanced --source videos/test.mp4 --profile calibration/test_video_metric_profile.json --headless --output outputs/m03/final_video
git diff --check
```

For an extracted bundle, copy the contents of `overlay/` into a fresh checkout
at the specified HEAD and copy `baseline/` to
`outputs/m03/final_correction/baseline`. Keep that directory below `outputs/`:
putting the saved Python package at repository root would cause full unittest
discovery to execute its 85 historical tests in addition to the current suite.
Baseline package filenames/hashes remain untouched.
Image hashes must match between baseline/current runs; timing variability is
expected. Acquisition, visualization, encoding and tracemalloc are excluded from
latency measurements. Memory is a separate run; Windows RSS is process-wide.
The video profile is synthetic/unverified, so the video run is integration
evidence, not lane accuracy or metre validation.

## Findings requiring independent attention

V-1/C-1 remain PARTIAL: no road-semantics input can distinguish otherwise identical
streaks from paint. R-08 scoped validity is intentionally usable only inside its
reported domain, including for future tracking/waypoint consumers. Stress-budget
refusals are deliberate and must never expose a selected partial pair. Review
confidence as an engineering evidence score, not a probability. No Pi or vehicle
readiness claim is made.

## Recorded comparison

Final suites: M01 46, M02 284, M03 107, full discovery 440, video assets 2;
all pass, no skips. Original M03 suite had 85 methods; 22 added. Prior P1 matrix:
144/144 checks. Physical transition matrix: 72/72, supported controls recovered
40/60→60/60; maximum accepted-domain centre error 2.611→1.218 base px.
Primary clutter recall 7/32→32/32 with zero wrong-valid. Curved clutter 32/32,
dashed 31/32; one case conservatively uncertain. Competing scenes 0/32 selected.
Extreme stress budget exits remain uncertain. Full video CLI: 590 frames to EOF.

Desktop Hough-on median / p95 ms, baseline→corrected:

| Case | Baseline | Corrected |
|---|---|---|
| 320 clean | 5.282 / 6.028 | 5.877 / 6.712 |
| 320 fragments | 12.060 / 13.397 | 10.944 / 12.185 |
| 320 specks | 8.794 / 9.688 | 9.742 / 10.987 |
| 640 clean | 7.951 / 8.725 | 8.617 / 9.586 |
| 640 fragments | 11.244 / 15.581 | 14.624 / 15.878 |
| 640 specks | 9.268 / 10.213 | 15.616 / 17.211 |

Intel i5-13450HX, Windows 11, Python 3.14.10, OpenCV 5.0.0, NumPy 2.5.4,
one OpenCV thread; 160 samples/group. Baseline early-abort latency must be assessed
with its lower recall. Raw maxima, stages, counts, memory and image hashes are
included. See the report for exact changed paths and the full defect matrix.
