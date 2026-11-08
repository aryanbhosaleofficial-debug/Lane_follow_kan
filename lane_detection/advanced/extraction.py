"""Vertical-safe Hough evidence and bounded multi-boundary row-run association."""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median
from functools import lru_cache
from bisect import bisect_left, bisect_right
import cv2
import numpy as np

from .config import LaneConfig
from .preprocessing import Preprocessed
from .models import ExtractionDiagnostics


@dataclass(frozen=True, eq=False)
class Extraction:
    tracks: tuple[np.ndarray, ...]         # each Nx2: observed u,v row-run centres
    hough_segments: tuple[tuple[int, int, int, int], ...]
    overloaded: bool
    branch_detected: bool
    diagnostics: ExtractionDiagnostics = ExtractionDiagnostics()
    track_widths: tuple[np.ndarray, ...] = ()


@dataclass
class _Track:
    points: list[tuple[float, float]]
    widths: list[float] = field(default_factory=list)
    slope: float = 0.0
    intercept: float = 0.0
    track_id: int = 0

    def append(self, u: float, v: float, width: float) -> None:
        self.points.append((u, v))
        self.widths.append(width)
        if len(self.points) >= 3:
            recent = self.points[-8:]
            self.slope = median((b[0] - a[0]) / (b[1] - a[1]) for a, b in zip(recent, recent[1:]))
            self.intercept = median(x - self.slope * y for x, y in recent)
        else:
            self.intercept = u


def hough_segments(edges: np.ndarray, config: LaneConfig) -> tuple[tuple[int, int, int, int], ...]:
    """Keep vertical/curved tangents; reject horizontal strokes and overlapping votes."""
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, config.hough_threshold,
                            minLineLength=config.hough_min_length, maxLineGap=config.hough_max_gap)
    if lines is None:
        return ()
    candidates = []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        if y1 > y2:
            x1, y1, x2, y2 = x2, y2, x1, y1
        dx, dy = int(x2 - x1), int(y2 - y1)
        length = float(np.hypot(dx, dy))
        if dy == 0 or abs(dx) > config.max_abs_du_dv * dy or length < config.hough_min_length:
            continue
        candidates.append((length, tuple(map(int, (x1, y1, x2, y2)))))
    kept: list[tuple[int, int, int, int]] = []
    for _, seg in sorted(candidates, key=lambda item: (-item[0], item[1])):
        x1, y1, x2, y2 = seg
        duplicate = False
        for p1, q1, p2, q2 in kept:
            lo, hi = max(y1, q1), min(y2, q2)
            if hi <= lo:
                continue
            v = (lo + hi) / 2
            u = x1 + (v - y1) * (x2 - x1) / (y2 - y1)
            p = p1 + (v - q1) * (p2 - p1) / (q2 - q1)
            overlap = (hi - lo) / min(y2 - y1, q2 - q1)
            slope_difference = abs((x2 - x1) / (y2 - y1) - (p2 - p1) / (q2 - q1))
            if abs(u - p) <= config.segment_duplicate_px and overlap > 0.5 and slope_difference < 0.15:
                duplicate = True
                break
        if not duplicate:
            kept.append(seg)
        if len(kept) >= config.max_hough_segments:
            break
    return tuple(kept)


def _runs(row: np.ndarray, max_width: int, roi: np.ndarray | None = None) -> np.ndarray:
    padded = np.pad((row != 0).astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts, ends = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)
    widths = ends - starts
    keep = widths <= max_width
    if roi is not None:
        # A run cut by either side of the supported ROI has an inward-biased
        # centroid. It is not a full marking-width observation; discard that row.
        interior = (starts > 0) & (ends < len(row))
        keep &= interior
        indices = np.flatnonzero(interior)
        keep[indices] &= (roi[starts[indices] - 1] != 0) & (roi[ends[indices]] != 0)
    return (starts[keep] + ends[keep] - 1) / 2.0


@lru_cache(maxsize=8)
def _sample_rows(height: int, step: int) -> np.ndarray:
    rows = np.arange(height - 1, -1, -step)
    rows.flags.writeable = False
    return rows


def _all_runs(pre: Preprocessed, config: LaneConfig):
    rows = _sample_rows(pre.marking_mask.shape[0], config.row_step)
    sampled = pre.marking_mask[::-config.row_step]
    changes = np.diff(np.pad((sampled != 0).astype(np.int8), ((0, 0), (1, 1))), axis=1)
    row, starts = np.nonzero(changes == 1)
    _, ends = np.nonzero(changes == -1)
    keep = (ends - starts <= config.max_marking_width_px) & (starts > 0) & (ends < sampled.shape[1])
    ix = np.flatnonzero(keep)
    keep[ix] &= (pre.roi[rows[row[ix]], starts[ix] - 1] != 0) & (pre.roi[rows[row[ix]], ends[ix]] != 0)
    centres = (starts[keep] + ends[keep] - 1) / 2.0
    counts = np.bincount(row[keep], minlength=len(rows))
    return rows, centres, np.r_[0, np.cumsum(counts)], (ends - starts)[keep]


def extract(pre: Preprocessed, config: LaneConfig) -> Extraction:
    """Associate row-run centres bottom to top; prediction never adds evidence.

    Finite gap and lateral gates prevent arbitrary merging across adjacent lanes.
    Run/track caps bound work on clutter and force an uncertain observation.
    """
    tracks: list[_Track] = []
    completed: list[_Track] = []
    overloaded = False
    branch = False
    branch_events: set[tuple[int, int]] = set()
    work = 0
    created = retired = discarded = peak_active = peak_completed = peak_points = 0
    budget_reason = None
    height = pre.marking_mask.shape[0]

    def eligible(track: _Track) -> bool:
        pts = track.points
        return (len(pts) >= config.min_points and pts[0][1] - pts[-1][1] >= config.min_span_px
                and len(pts) * config.row_step / height >= config.min_coverage)

    rows, centres, offsets, widths = _all_runs(pre, config)
    for k, v in enumerate(rows):
        active = []
        for track in tracks:
            # Two isolated row samples do not establish continuity across a dash
            # gap. Provisional tracks tolerate one missing sampled row; established
            # chains retain the configured max_gap_px for dashed/broken paint.
            allowed_gap = config.max_gap_px if len(track.points) >= 3 else min(config.max_gap_px, 2 * config.row_step)
            if track.points[-1][1] - v > allowed_gap:
                retired += 1
                if eligible(track):
                    completed.append(track)
                else:
                    discarded += 1
            else:
                active.append(track)
        tracks = active
        peak_completed = max(peak_completed, len(completed))
        if len(completed) > config.max_fit_candidates:
            overloaded, budget_reason = True, "completed_candidate_limit"
            break
        runs = centres[offsets[k]:offsets[k + 1]]
        row_widths = widths[offsets[k]:offsets[k + 1]]
        if len(runs) > config.max_row_runs:
            overloaded, budget_reason = True, "row_run_limit"
            break
        choices: list[tuple[float, int, int]] = []
        for i, track in enumerate(tracks):
            pts = track.points
            last_u, last_v = pts[-1]
            gap = last_v - v
            if gap > config.max_gap_px:
                continue
            predicted = track.slope * v + track.intercept
            gate = config.association_margin_px + 0.12 * gap
            lateral_gate = config.max_abs_du_dv * gap + config.association_margin_px
            # _all_runs emits ascending centres within each sampled row. Intersect
            # both existing spatial gates before scoring/counting candidates;
            # distant clutter is not a matching comparison. Inclusive endpoints
            # and (distance, track index, run index) ties preserve determinism.
            start = bisect_left(runs, max(predicted - gate, last_u - lateral_gate))
            stop = bisect_right(runs, min(predicted + gate, last_u + lateral_gate))
            comparisons = max(0, stop - start)
            if work + comparisons > config.max_association_comparisons:
                overloaded, budget_reason = True, "association_comparison_limit"
                break
            work += comparisons
            maturity_bonus = config.association_margin_px * .25 if len(pts) >= config.min_points else 0.0
            for j in range(start, stop):
                choices.append((abs(float(runs[j]) - predicted) - maturity_bonus, i, j))
        if overloaded:
            break
        used_tracks: set[int] = set()
        used_runs: set[int] = set()
        # Multiple established chains converging onto one observation signal a merge.
        for j in range(len(runs)):
            established = [i for _, i, k in choices if k == j and len(tracks[i].points) >= config.min_points]
            if len(established) > 1:
                for a in range(len(established)):
                    for b in range(a + 1, len(established)):
                        branch_events.add((tracks[established[a]].track_id, tracks[established[b]].track_id))
        for _, i, j in sorted(choices):
            if i not in used_tracks and j not in used_runs:
                tracks[i].append(float(runs[j]), float(v), float(row_widths[j]))
                used_tracks.add(i)
                used_runs.add(j)
        for j, u in enumerate(runs):
            if j not in used_runs:
                if len(tracks) >= config.max_tracks:
                    overloaded, budget_reason = True, "active_track_limit"
                    break
                else:
                    tracks.append(_Track([(float(u), float(v))], widths=[float(row_widths[j])], intercept=float(u), track_id=created))
                    created += 1
        peak_active = max(peak_active, len(tracks))
        peak_points = max(peak_points, sum(len(t.points) for t in tracks) + sum(len(t.points) for t in completed))
        if overloaded:
            break
    # Reject impossible support before allocating arrays or running robust fits.
    candidates = completed + [track for track in tracks if eligible(track)]
    supported_ids = {track.track_id for track in candidates}
    branch = any(a in supported_ids and b in supported_ids for a, b in branch_events)
    peak_completed = max(peak_completed, len(candidates))
    if len(candidates) > config.max_fit_candidates:
        overloaded, budget_reason = True, "fit_candidate_limit"
    arrays = tuple(np.asarray(track.points, dtype=float) for track in candidates[:config.max_fit_candidates])
    diagnostics = ExtractionDiagnostics(created, retired, discarded, peak_active, peak_completed, peak_points, work, budget_reason)
    candidate_widths = tuple(np.asarray(track.widths, float) for track in candidates[:config.max_fit_candidates])
    return Extraction(arrays, () if overloaded or not config.use_hough else hough_segments(pre.edges, config), overloaded, branch, diagnostics, candidate_widths)
