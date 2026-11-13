"""
analysis/yaseen_duplicate_groups.py

Yaseen 2018 is not 1000 independent recordings. The files are short
(1.2-4 s) crops, and many are overlapping crops of the same source
recording: a normalised cross-correlation between two files that peaks
above 0.95 means the waveforms are the same audio at a time offset, not
two hearts that happen to sound alike. Found 2026-09-07 when two MS files
produced identical segmentations frame for frame.

Why it matters: a recording-level train/test split (what the sibling repo
used, and what experiments/yaseen_three_arms.py does by default) puts crops
of the same source on both sides of the split. That inflates accuracy in
the same way a recording-level CirCor split would (CirCor is split by
patient for exactly this reason) and it overstates the effective n of
every class-level fidelity number.

What this does:
  1. For every within-class pair of files, computes the maximum absolute
     normalised cross-correlation (both signals mean-removed and unit-norm,
     full-lag FFT correlation). Pairs whose lengths differ by more than 50%
     are skipped as non-comparable.
  2. Links pairs above XCORR_THRESHOLD into connected components (a
     source group). Transitive linking is intended: if A overlaps B and B
     overlaps C, all three come from one source.
  3. Prints the number of groups per class at three thresholds so the
     choice of 0.95 can be defended, then writes one row per file with its
     group id and group size at XCORR_THRESHOLD to
     paths.YASEEN_DUPLICATE_GROUPS_CSV. experiments/yaseen_three_arms.py
     uses that file for a group-level split when asked.

Limitations, stated: (a) cross-class duplicates are not checked, since the
labels are per file and a cross-class match would be a labelling error
rather than a split problem; (b) a threshold on raw-waveform correlation
can miss crops that were re-gained or filtered differently, so the group
count is an upper bound on the number of independent sources, not an exact
count. Both belong in the paper's Yaseen limitations.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repo root: makes `paths` and the packages importable from any cwd
import paths


import itertools

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import fftconvolve

XCORR_THRESHOLD = 0.95
REPORT_THRESHOLDS = (0.90, 0.95, 0.98)
MAX_LENGTH_RATIO_DIFF = 0.5
OUTPUT_CSV = paths.YASEEN_DUPLICATE_GROUPS_CSV


def load_unit_signals(cls):
    files = sorted((paths.YASEEN_RAW_DIR / cls).glob("*.wav"))
    sigs = []
    for f in files:
        y, _ = sf.read(f, dtype="float64")
        if y.ndim > 1:
            y = y.mean(axis=1)
        y = y - y.mean()
        sigs.append(y / (np.linalg.norm(y) + 1e-12))
    return [f.stem for f in files], sigs


def pairwise_max_xcorr(sigs):
    n = len(sigs)
    m = np.zeros((n, n))
    for i, j in itertools.combinations(range(n), 2):
        a, b = sigs[i], sigs[j]
        if abs(len(a) - len(b)) > MAX_LENGTH_RATIO_DIFF * max(len(a), len(b)):
            continue
        m[i, j] = m[j, i] = float(np.max(np.abs(fftconvolve(a, b[::-1], mode="full"))))
    return m


def components(adjacency):
    n = adjacency.shape[0]
    label = -np.ones(n, dtype=int)
    current = 0
    for start in range(n):
        if label[start] >= 0:
            continue
        stack = [start]
        label[start] = current
        while stack:
            u = stack.pop()
            for v in np.nonzero(adjacency[u])[0]:
                if label[v] < 0:
                    label[v] = current
                    stack.append(v)
        current += 1
    return label


def main():
    rows = []
    print(f"{'class':>4} " + " ".join(f"groups@{t:.2f}" for t in REPORT_THRESHOLDS) + "   largest@0.95")
    for cls in paths.YASEEN_CLASSES:
        names, sigs = load_unit_signals(cls)
        m = pairwise_max_xcorr(sigs)
        counts = []
        for t in REPORT_THRESHOLDS:
            lab = components(m > t)
            counts.append(len(set(lab)))
        lab = components(m > XCORR_THRESHOLD)
        sizes = pd.Series(lab).value_counts()
        print(f"{cls:>4} " + " ".join(f"{c:11d}" for c in counts) + f"   {sizes.max():6d}   (n files {len(names)})")
        for name, g in zip(names, lab):
            rows.append({"recording_id": name, "label": cls, "group_id": f"{cls}_{g:03d}", "group_size": int(sizes[g])})
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nTotal source groups at {XCORR_THRESHOLD}: {df['group_id'].nunique()} for {len(df)} files")
    print(f"Wrote {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
