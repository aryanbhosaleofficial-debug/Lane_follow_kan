# Claude Sonnet 5.5 reviewer handoff

Review base Git HEAD `5dc8defbe291f0c3882dc955527503a96dc892e9` plus the complete
uncommitted overlay in `M03_R09_REVIEW_BUNDLE.zip`. M03 was untracked at entry;
no commit, staging, push, merge, reset or deletion was performed. Earlier work and
historical review artifacts remain present.

Start with [M03_R09_FINAL_REPORT.md](M03_R09_FINAL_REPORT.md) and the machine-readable
[M03_R09_FINAL_RESULTS.json](M03_R09_FINAL_RESULTS.json). The verdict is
**CONDITIONAL PASS for the correction, with original unconditional R-09 8/8
acceptance explicitly unmet**. Do not convert passing unit tests into R-09 closure.

## Corrections to inspect

1. `advanced/extraction.py`: binary-search the intersection of existing spatial
   gates before counting candidates. The global 4,000 cap remains; matching order
   and retirement are unchanged. A high diagnostic-budget comparison found exact
   track/width/branch equality with the entry engine across 896 scenes.
2. `advanced/models.py` / `pipeline.py`: guarded downstream centreline query;
   explicit reference-outside-BEV refusal; no image-centre substitution. Exact
   6 px reference-extension cap and raw paint/support requirements remain intact.
3. `advanced/fitting.py`: tolerance-normalized residual-quality factor. Sustained
   bias uses the existing coherent-misfit policy; isolated dash caps must not
   reduce recall as systematic model error. Diagnostics are not probabilities.
4. `advanced/config.py`: Hough default off after 896-scene on/off ablation gave
   identical validity and support. Optional Hough/vote tests remain active.

## Remaining issues requiring judgment

* Original 60-fragment results improve 0/8 -> 6/8 scoped-correct at 320x240, and
  0/8 -> 7/8 at 640x480, with zero remaining budget events. Seeds 5/7 at 320 retain
  competing supported fragment chains under the existing companion contract.
  Assess the documented ambiguity exception independently. The code does not
  discard those chains simply to force 8/8.
* Original midpoint checker counts only 5/8 at 320 because one correct limited
  pair excludes row 120. It returns NaN there as required; its domain is roughly
  155..233. Compare reported-domain ground truth separately from that raw checker.
* Paint-like wall/glare/cream/gold inputs remain semantically ambiguous. Separate
  semantic negative audit: 8/15 false-valid before and after. The random-clutter
  breadth audit is 0/896 incorrect-valid before and after; these are different
  denominators and must not be conflated.
* Original R-08 remains 62/70 solid, 58/70 dashed. Four pre-existing cases have
  maximum boundary error 2.70..2.75 px under a stricter 2.5 px audit, unchanged
  before/after. Zero limited-domain NaN failures. The original functions/raster
  equations are preserved; portable adapters replace machine-specific imports.
* Tight arc support depends on model span, slope, curvature and signed bias;
  radius alone is not a supported operating-range claim. Road semantics, physical
  calibration, track accuracy, Pi/thermal performance and confidence calibration
  still require labelled real-world/hardware validation.

## Reproduce

Verify the bundle SHA256 using its `.sha256` sidecar and verify every archive member
against `manifest.json`. Extract `overlay/` into a fresh checkout at the stated
HEAD. Restore `baseline/` and `evidence/` to `outputs/m03/r09_final/` (the baseline
must remain below `outputs/`). The bundle baseline is the entry **107-test engine**,
not the previous 85-test baseline. `evidence/p9_r09a.py` and `p10_r08.py` are the
preserved reviewer sources. Do not run their original absolute-path rv imports
on another machine; use `advanced.reviewer_reproduction` as shown in the report.

Exact executed commands, current changed-file manifest, per-image hashes, every
latency sample, capacity/track diagnostics, memory data and test logs accompany
the report/results. `evidence/association_equivalence.py` reruns the strict
entry/current extraction comparison. `evidence/finalize_evidence.py` builds the
results and portable package. Memory tracing is separate from timing. Windows RSS
is process-wide and its peak is cumulative; do not attribute it to one stage.

Final tests: M01 46, M02 284, M03 121, full discovery 454, video assets 2, all pass,
no skips. A fresh Git HEAD plus the packaged overlay also passes all 454 tests,
with imports verified to come from the reconstructed tree. Prior P1 matrix
144/144; author transition matrix 72/72, required near
controls 60/60. Full video CLI reaches EOF with 590/590 source frames, no consumer
skips, on synthetic/unverified geometry. No real-track accuracy follows from it.

Final performance tables are in the report: i5-13450HX / Windows 11 / Python
3.14.10 / OpenCV 5.0.0 / NumPy 2.5.4, one OpenCV thread, 20 timed repeats plus two
warmups per scene. Baseline/current images must hash identically. Assess lower
baseline early-abort times alongside zero recall rather than as equivalent work.
