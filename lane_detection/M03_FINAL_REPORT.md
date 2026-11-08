# M03 final corrective validation

This report supersedes the earlier M03 corrective report for current software
behavior. Earlier artifacts are retained as historical measurements. No score is
assigned. Desktop synthetic validation is separate from vehicle readiness.

## Scope and evidence provenance

Base Git HEAD: `5dc8defbe291f0c3882dc955527503a96dc892e9` (M01/M02).
M03 was already uncommitted at entry. The initial 35-file M03 package, including
the 85-test baseline, was copied and hashed before editing. No commit, push, merge,
reset, baseline deletion or M01/M02 production change was performed.

The second review's scripts were unavailable. The seeded clutter cases here are
independently specified in `advanced/availability_cases.py`, not claimed to be
byte-identical review reproductions. Baseline and corrected engines receive
identical images/configurations; per-image SHA256 and raw timings are exported.
The earlier 54-test engine was also evaluated as a separate historical reference.

## Defect status

| Finding | Status | Evidence / limitation |
|---|---|---|
| R-09 obsolete tracks exhaust capacity | FIXED | Retire obsolete chains; discard short clutter; independently bound active chains, completed candidates, row runs and comparisons. Eligible chains alone establish branch ambiguity. Seeded recall/stress and repeated-frame memory checks. |
| R-08 transition loses usable near lane | FIXED | Scoped near-line recovery independently verifies both edges. Accepted domain, excluded observations and unsupported far interval are explicit. No extension through the turn. Physical transition matrix and actual M02 reference test. |
| F-01 bottom reference excluded | FIXED | Bounded near-only reference query requires raw paint at both edges and valid/support masks; separate extension metadata, no curve-domain enlargement. |
| F-02 unreasonable scale ratios | FIXED | Factory refuses scale ratios outside 0.25..4; wider grids require independently validated explicit pixel configuration. |
| F-03 bad image traceback | FIXED | Decode/runtime input errors produce concise CLI error with exit status 2 and cleanup. |
| F-04 positional config compatibility | FIXED | Original 45 positional fields retain order; subsequent fields keyword-only; golden positional API regression. |
| V-1 wall/glare false positives | PARTIAL | Tapered streaks refused using paint-width consistency. Uniform paint-like wall/glare patterns remain semantically indistinguishable without road-surface evidence. |
| O-1 fit error hidden by inliers | FIXED | Full-observation RMS/p95, max inlier error, sustained signed bias and excluded intervals; coherent misfit triggers scoped recovery or refusal. |
| O-2 staggered dashes zero shared evidence | FIXED | Explicit zero-shared reason and default refusal; opt-in association-only policy preserves diagnostics and lower evidence score. |
| O-3 occlusion misreported as fork | FIXED | Disjoint lane support and processing-budget reasons distinguished from spatial alternatives; no forced lane selection. |
| T-1 vacuous acceptance test | FIXED | Requires successful supported arc control and transition recovery counts, independently checks geometry. |
| D-1 ignored/uncommitted evidence | FIXED | Portable archive includes source, immutable baseline, raw results, logs, source diff and hashes; no Git commit authorized or made. |
| C-1 Hough, yellow floors, confidence | PARTIAL | Hough ablation measured; optional disable retained. Broad yellow floors/tapered bands refused, narrow yellow paint retained. Uniform paint-like materials remain ambiguous. Evidence scores are not calibrated probabilities; metric trust remains gated by M02. |

## Corrections and downstream contract

Extraction retires established chains after `max_gap_px`, and immature one/two-row
chains after two sampled row steps. Only active chains count against `max_tracks`.
Eligible retired candidates are retained within `max_fit_candidates`; short chains
are discarded. Caps remain unchanged. A budget failure returns uncertain with no
selected pair. Diagnostics expose creation, retirement, stored samples, candidate
peaks and comparison counts. Limits are operation caps, not a CPU deadline.

R-08 uses a verified near-field **line**, avoiding a new piecewise public model.
An anchored near fit stops before model disagreement; both partners are verified
within a shared scoped domain. Original fork/companion ambiguity remains a veto.
Output can be `valid` only over the exported domain, with `limited_support=True`,
`support_status="limited_near_field"`, `usable_support_v_px`, pair
`valid_support_v_px`, `observed_span_px` and `unsupported_far_intervals`. Limited
polynomials evaluate to NaN outside their domains; JSON uses finite coefficients
and explicit domains. Full-observation errors remain visible even after recovery.
No artificial paint is added. Consumers must inspect support, not just `valid`.

The existing 0.004 m/px reference remains nominal until M02 validates it. Raw BEV
pixels are not relabelled as trusted metres. Ego membership remains distinct from
geometric lane validity. Freshness checks, source session/frame identity, immutable
images/masks and acquisition timestamps retain their previous contracts.

## Changed files in this round

Paths below are relative to `lane_detection/advanced/` unless stated otherwise.
These are compared against the preserved 85-test package, not Git HEAD, which
contains no M03 package.

| Files | Purpose |
|---|---|
| `extraction.py` | Active/retired lifecycle, bounded candidates, eligible branch votes, run widths and extraction diagnostics |
| `fitting.py` | Full-data misfit diagnostics, coherent-bias check and scoped near recovery |
| `pipeline.py` | Recovery selection, marking-width screen, support policy, bounded reference and accurate refusal reasons |
| `association.py`, `observability.py` | Limited pair scope, disjoint-support reasons, invalid marking exclusion and shared-paint policy |
| `models.py` | Additive diagnostics/support/reference fields; limited polynomial evaluation guard |
| `config.py` | Keyword-only additions, original positional compatibility, reference/fit/evidence policies, scale-ratio validation |
| `preprocessing.py` | Unblurred reference evidence query |
| `inference.py` | Preserve polynomial extrapolation policy |
| `cli.py` | Clean runtime input errors and exit status |
| `visualization.py` | Explicit LIMITED label, rejected marking appearance, fractional-domain-safe curve sampling and correct candidate labels |
| `availability_cases.py`, `availability_evaluation.py` (new) | Independent seeded clutter/transition scenes, baseline comparison, latency/stage/count/memory evaluation |
| `transition_evaluation.py` (new) | Independent physical straight-to-arc matrix and near-only controls |
| `test_final_availability.py` (new) | Recall, bounded memory/work, transition limits, forks and missing edges |
| `test_final_interfaces.py` (new) | F-01 through F-04 regressions |
| `test_final_evidence.py` (new) | Glare/yellow floor, shared support, reasons and misfit regressions |
| `test_review_integration.py`, `test_review_p1.py` | Strong actual-M02 scoped geometry assertion; non-vacuous supported arc control |
| `README.md` | Revised API, support, lifecycle and limitation documentation |
| Repository `README.md`, `lane_detection/README.md` | Link current validation from existing M03 entry points |
| `lane_detection/M03_FINAL_REPORT.md`, `M03_FINAL_REVIEW_HANDOFF.md`, `M03_FINAL_RESULTS.json`, `M03_FINAL_REVIEW_BUNDLE.zip` (new) | Durable report, raw evidence and portable independent-review package |

No production files were removed. Earlier untracked M03 files remain present.
The pre-existing `.gitignore` change was preserved, not altered this round.

## Reproduction

Run the commands in [M03_FINAL_REVIEW_HANDOFF.md](M03_FINAL_REVIEW_HANDOFF.md).
Machine-readable results are in [M03_FINAL_RESULTS.json](M03_FINAL_RESULTS.json).
The archive includes the initial baseline separately for before/after runs, logs,
SHA256 manifest and a source diff. A fresh checkout at the stated HEAD plus the
archive's `overlay/` directory reconstructs this review state without a commit.

## Validation results

Initial baseline: **85 M03 tests passed**. Current package: **107 M03 tests passed**
(22 added methods, including parameterized matrices). Two earlier tests were
strengthened: the actual M02 transition now requires a valid scoped result with
ground geometry accuracy; the tight-arc test requires a supported successful
control instead of permitting every case to be refused. Neither ground-truth
tolerance nor prior P1 acceptance matrix was weakened.

| Executed command | Passed | Failed | Skipped |
|---|---:|---:|---:|
| `python -m unittest discover -s camera_input -t . -v` | 46 | 0 | 0 |
| `python -m unittest discover -s camera_calibration -t . -v` | 284 | 0 | 0 |
| `python -m unittest discover -s lane_detection/advanced -t . -v` | 107 | 0 | 0 |
| `python -m unittest discover -v` | 440 | 0 | 0 |
| `python -m unittest camera_calibration.video_asset_tests -v` | 2 | 0 | 0 |

The full count includes the component suites; do not add these rows together.
M01's injected device/worker failures and the corrupt-video fixture log expected
errors while their assertions pass. Archived test-first logs intentionally contain
the reproduced pre-fix failures; they are distinct from final passing logs.
The three new regression modules also passed as a separate targeted command:
`python -m unittest lane_detection.advanced.test_final_availability lane_detection.advanced.test_final_interfaces lane_detection.advanced.test_final_evidence -v`
(22 methods). `python -m compileall -q lane_detection/advanced` and
`git diff --check` completed successfully.

The archive was reconstructed onto a clean `git archive HEAD` tree and full
discovery independently passed **440 tests** in 41.424 seconds. Source hashes,
the archive manifest and all 35 initial baseline file hashes were verified. An
earlier packaging rehearsal put the historical baseline at repository root,
also discovering its 85 tests; both logs are retained. The final handoff places
the baseline under `outputs/`, keeping historical tests out of current discovery.

### Availability and geometry before/after

Each clutter cell uses 8 independent RNG seeds. Valid geometry must stay within
2.5 base pixels of the independent centreline over its entire output domain.
The target corresponds to half the nominal 5-pixel marking thickness and is not
a metre accuracy claim. The restored primary set is substantially tighter in
practice: worst centre error **0.360 base px**.

| Scene | Initial 85-test package | Corrected | Wrong-valid after |
|---|---:|---:|---:|
| 320x240, 60 fragments | 5/8 | 8/8 | 0/8 |
| 320x240, 200 specks | 0/8 | 8/8 | 0/8 |
| 640x480, 60 fragments | 2/8 | 8/8 | 0/8 |
| 640x480, 200 specks | 0/8 | 8/8 | 0/8 |

The separate earlier 54-test engine achieved 8/8, 8/8, 6/8 and 5/8 respectively
on these same independently constructed scenes. The corrected primary recall
therefore recovers or exceeds that reproducible earlier reference. Review-reported
counts are not substituted for measured counts.

Additional clutter sets: polynomial curves **32/32**, dashed lanes **31/32**;
competing three-boundary scenes **0/32 valid selections**. All accepted geometry
was within the 2.5-pixel tolerance; no wrong-valid output was measured. The one
640x480 dashed/fragment case (seed 5) remains uncertain because the normal-midpoint
centreline cannot be approximated within the existing configured tolerance.

Stress: 144 scenes, fragments 0/30/60/120/240 and specks 100/200/400/800 at both
sizes. At fragments <=60 or specks <=200: **80/80 correct**, no budget exhaustion.
At fragments >=120 or specks >=400: **64/64 uncertain budget exits**, no selected
partial pair. **0/144 wrong-valid**. These counts establish bounded refusal under
this workload, not guaranteed detection under arbitrary clutter.

R-08 independent physical transitions use concentric normal-offset arcs, radii
220/300/700 base px, width 130 px, transition rows 90/120/150, both turn directions,
solid/dashed paint and both resolutions: **72 cases**. The near-only controls
establish 60 supported cases. Initial recovery was **40/60**, corrected **60/60**.
All 72 final checks pass: 66 valid (50 limited), six unsupported cases remain
uncertain. Initial accepted geometry exceeded the 2-pixel domain error tolerance
in 14 cases (maximum 2.611 px); corrected maximum is **1.218 px**, with none above
tolerance. The 2-pixel transition tolerance was selected against the independent
5-pixel raster truth, not fitted detector coefficients; earlier P1 tests retain
their stricter 1.5-pixel threshold. Additional tests cover arc visibility only at
rows 200–240 in the 640 image, missing/occluded edges and a fork that stays uncertain.

Prior P1 matrix: **144/144 checks pass**: 30 edge exits (28 valid, two legitimately
insufficient), 90 physical arcs (73 valid, 17 unsupported) and 24 companion-marking
ambiguity cases (none selected valid). Actual M01→M02→M03 tests independently
verify straight edge/centre geometry, a normal-width arc, trust gating, timestamps,
session identity and unchanged read-only arrays. The M02 curved reference now
returns a scoped valid near lane; ground-truth lateral centre error is <0.008 m
in the synthetic known-geometry coordinate comparison, with no unverified metre
measurement exported.

Full headless CLI replay processed **590/590 video frames to EOF**, IDs 1–590,
zero consumer skips. The profile remains `SYNTHETIC`/unverified. Video asset tests
also passed independently; no lane accuracy claim is made for the unlabelled clip.

### Measured CPU performance

Machine: **13th Gen Intel Core i5-13450HX**, 10 cores/16 logical processors,
16,849,256,448 physical RAM bytes; Windows 11 build 26300. Python **3.14.10**,
OpenCV **5.0.0**, NumPy **2.5.4**, OpenCV threads **1**. Each table cell pools
20 measured repetitions ×8 seeds (160 samples), one warmup per scene. Clean
images repeat across seeds; clutter images differ. Frame construction, acquisition,
visualization, encoding and memory tracing are excluded. Baseline/current image
hash equality was asserted. Values are **median / p95 / maximum milliseconds**.

| Size / scene | Initial | Corrected | Recall before → after |
|---|---|---|---|
| 320x240 clean | 5.282 / 6.028 / 6.650 | 5.877 / 6.712 / 8.254 | 8/8 → 8/8 |
| 320x240 fragments | 12.060 / 13.397 / 21.222 | 10.944 / 12.185 / 16.290 | 5/8 → 8/8 |
| 320x240 specks | 8.794 / 9.688 / 11.095 | 9.742 / 10.987 / 13.989 | 0/8 → 8/8 |
| 640x480 clean | 7.951 / 8.725 / 10.187 | 8.617 / 9.586 / 10.316 | 8/8 → 8/8 |
| 640x480 fragments | 11.244 / 15.581 / 23.482 | 14.624 / 15.878 / 24.621 | 2/8 → 8/8 |
| 640x480 specks | 9.268 / 10.213 / 11.006 | 15.616 / 17.211 / 22.053 | 0/8 → 8/8 |

Restoring useful detection costs work compared with early abandonment. Clean
median overhead is approximately 0.60/0.67 ms at the two sizes. These are desktop
measurements, not Pi 5 performance or guaranteed deadlines. Throughput and stage
medians, full raw latency samples, workload hashes and counts are in the results.
Corrected throughput (1000 divided by pooled mean latency) is 166.9/90.0/101.9 fps
for 320 clean/fragments/specks and 115.5/67.9/63.4 fps for 640 respectively.
The separate stage benchmark includes M02+M03 workloads; multiple boundaries in
those workloads intentionally remain uncertain.

Hough ablation (separate same-scene runs, 160 samples/group): both configurations
correct **48/48** including clean scenes, zero wrong-valid. Hough-on → off median
ms: 320 clean 5.863→5.231, fragments 11.303→9.738, specks 9.872→8.287;
640 clean 8.864→8.323, fragments 15.136→12.711, specks 15.768→12.213.
This set shows modest evidence-score support without a recall benefit. Hough is
still enabled by default to preserve caller behavior; disable explicitly when
the intended workload validates that choice. No universal contribution is claimed.

Primary workload peaks: **19 active tracks**, **2 completed candidates**, **247
stored row observations**, **3,572 comparisons**, and **103 total tracks created
within a frame**. Limits remain 48 active, eight fit candidates and 4,000
comparisons. All primary budget-exhaustion frequencies dropped to **0/32** from
**25/32**. Total created tracks can exceed the active limit because obsolete
short tracks are retired; no frame history is retained.

Memory was measured separately with tracemalloc, 10 warmup +100 frames of the
same speck workload and unique source IDs (no cache hits):

| Size | Initial Python live growth / peak | Corrected growth / peak | Corrected process RSS before → after / peak |
|---|---|---|---|
| 320x240 | 4,295 / 1,279,457 bytes | 20,113 / 1,320,681 bytes | 56,184,832 → 56,291,328 / 60,919,808 bytes |
| 640x480 | 5,503 / 4,965,889 bytes | 9,109 / 5,026,960 bytes | 58,449,920 → 57,536,512 / 64,045,056 bytes |

RSS is process-wide, allocator-dependent and cumulative, not per-call allocation.
The repeated-frame regression asserts live growth and operation/storage bounds;
the measurements do not prove absence of every possible leak.

## Final software assessment

**CONDITIONAL PASS.** Both P2 availability defects are resolved on the independent
reproductions; all final software regressions and prior P1 evaluations pass.
V-1/C-1 remain partial semantic/calibration concerns, not silently reported as
solved. The module is ready for independent software re-review with explicit
remaining limits. This verdict does not approve physical vehicle operation.

## Remaining limits

No real track labels, Raspberry Pi 5 measurements, physical metric calibration,
camera-mount validation or on-vehicle testing were available. Paint-like walls,
glare and floor patterns can be visually indistinguishable from lane paint. M02
support masks describe visibility/calibration, not road semantics or free space.
These limitations require independent evidence, not arbitrary confidence changes.

Transitions without sufficient near support can still be uncertain. Scoped line
recovery does not represent arbitrary sharp curves, overhangs or route branches.
High clutter deliberately forces uncertainty once bounded operation limits are
exceeded. Hough/HSV and evidence scores require labelled track evaluation before
making application-specific reliability claims.
