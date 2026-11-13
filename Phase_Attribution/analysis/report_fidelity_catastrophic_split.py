"""
analysis/report_fidelity_catastrophic_split.py

Finishes the outline's "segmentation-error accounting" step: reports
phase-attribution fidelity for the predicted arm with the known
catastrophic-segmentation recordings excluded and reported separately,
instead of blending them into one number.

Input: phase_diagnostics_per_recording.csv (already generated).

CATASTROPHIC_THRESHOLD matches the earlier confound-check script
(segmenter_accuracy < 0.05). Change here if you want a different cut.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repo root: makes `paths` and the packages importable from any cwd
import paths


import pandas as pd

PER_RECORDING_CSV = paths.PHASE_DIAGNOSTICS_PER_RECORDING_CSV
CATASTROPHIC_THRESHOLD = 0.05


def naive_baseline(expected_phases_series):
    all_phases = set()
    for s in expected_phases_series:
        all_phases.update(str(s).split(","))
    all_phases.discard("")
    best_rate, best_phase = 0.0, None
    for phase in sorted(all_phases):
        rate = expected_phases_series.apply(lambda s: phase in str(s).split(",")).mean()
        if rate > best_rate:
            best_rate, best_phase = rate, phase
    return best_phase, best_rate


def report(label, sub_df):
    n = len(sub_df)
    if n == 0:
        print(f"{label}: n=0, skipping")
        return
    fidelity_rate = sub_df["fidelity_match_raw"].mean()
    best_phase, baseline_rate = naive_baseline(sub_df["expected_phases"])
    print(f"\n--- {label} (n={n}) ---")
    print(f"  Actual model fidelity: {fidelity_rate * 100:.1f}%")
    print(f"  Naive baseline (always guess {best_phase}): {baseline_rate * 100:.1f}%")
    print(f"  Model vs baseline: {(fidelity_rate - baseline_rate) * 100:+.1f} points")
    print(f"  Classification accuracy (correct) on this subset: {sub_df['correct'].mean() * 100:.1f}%")


def main():
    df = pd.read_csv(PER_RECORDING_CSV)
    scoped = df[df["has_expected_phase"] == True].copy()

    print("=" * 70)
    print("GROUND TRUTH ARM (no segmentation error by definition -- reference point)")
    print("=" * 70)
    gt = scoped[scoped["boundary_source"] == "ground_truth"]
    report("ground_truth (all)", gt)

    print("\n" + "=" * 70)
    print("PREDICTED ARM -- split by segmenter accuracy")
    print("=" * 70)
    pred = scoped[scoped["boundary_source"] == "predicted"].copy()
    catastrophic = pred[pred["segmenter_accuracy"] < CATASTROPHIC_THRESHOLD]
    clean = pred[pred["segmenter_accuracy"] >= CATASTROPHIC_THRESHOLD]

    report(f"predicted -- catastrophic (segmenter_accuracy < {CATASTROPHIC_THRESHOLD})", catastrophic)
    report(f"predicted -- clean (segmenter_accuracy >= {CATASTROPHIC_THRESHOLD})", clean)
    report("predicted -- all (unsplit, for comparison)", pred)


if __name__ == "__main__":
    main()