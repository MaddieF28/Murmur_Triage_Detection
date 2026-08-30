"""
check_phase_labeling.py

Loads segmented_data.pkl (produced by run_Springer.py) and visually/
numerically sanity-checks one recording's phase segmentation -- to
confirm S1/Systole/S2/Diastole boundaries and labels are correct
before trusting fidelity numbers built on top of them.

This inspects the EXACT same phase_segments structure that (presumably)
feeds your feature extraction step, so if something looks wrong here,
it's wrong in the actual pipeline, not just in a reimplementation.
"""

from pathlib import Path
import pickle

import numpy as np
import matplotlib.pyplot as plt

# --- config -----------------------------------------------------------

THIS_DIR = Path(__file__).resolve().parent
SEGMENTED_DATA_PATH = THIS_DIR / "segmented_data.pkl"
FEATURE_TABLE_PATH = THIS_DIR / "pcg_feature_table.csv"

OUTPUT_DIR = THIS_DIR / "phase_labeling_check"
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET_FILENAME = "New_AS_184.wav"  # the file flagged earlier as attributed to S2 instead of Systole

STATE_COLORS = {"S1": "tab:red", "Systole": "tab:blue", "S2": "tab:green", "Diastole": "tab:orange"}
EXPECTED_CYCLE = ["S1", "Systole", "S2", "Diastole"]


# --- loading ------------------------------------------------------------

def load_recording(filename: str) -> dict:
    with open(SEGMENTED_DATA_PATH, "rb") as f:
        data = pickle.load(f)

    for rec in data["results"]:
        if rec["filename"] == filename:
            return rec

    # Also check failures, in case the target file didn't segment successfully
    for f in data["failures"]:
        if f["filename"] == filename:
            raise ValueError(f"{filename} FAILED segmentation: {f['error']}")

    raise ValueError(f"{filename} not found in results or failures.")


# --- plotting -----------------------------------------------------------

def plot_labeled_waveform(rec: dict):
    signal = rec["signal"]
    fs = rec["fs"]
    segments = rec["phase_segments"]
    t = np.arange(len(signal)) / fs

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(t, signal, color="black", linewidth=0.5)

    seen_in_legend = set()
    for label, start, end in segments:
        color = STATE_COLORS.get(label, "gray")
        legend_label = label if label not in seen_in_legend else None
        seen_in_legend.add(label)
        ax.axvspan(t[start], t[min(end, len(t) - 1)], color=color, alpha=0.25, label=legend_label)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Phase labeling check: {rec['filename']} (true label: {rec['label']})")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"labeled_waveform_{rec['filename'].replace('.wav', '')}.png", dpi=150)
    plt.close(fig)


def print_segment_sequence(rec: dict):
    """
    Prints every detected segment in order with duration, and flags any
    transition that violates the expected S1 -> Systole -> S2 ->
    Diastole -> S1 -> ... cycle. Also flags segments where Systole is
    shorter than S2/S1 on average -- physiologically, systole is
    usually the LONGER interval (roughly 1/3 of a cardiac cycle vs S1/
    S2's short transient bursts), so if your Systole segments are
    consistently as short as S1/S2, that's a strong sign of mislabeling
    even without an outright order violation.
    """
    fs = rec["fs"]
    segments = rec["phase_segments"]

    print(f"\nSegment sequence for {rec['filename']} (label: {rec['label']}, fs: {fs}):")
    print(f"{'#':<4}{'phase':<10}{'start (s)':<12}{'end (s)':<12}{'duration (s)':<12}")
    for i, (label, start, end) in enumerate(segments):
        dur = (end - start) / fs
        print(f"{i:<4}{label:<10}{start/fs:<12.3f}{end/fs:<12.3f}{dur:<12.3f}")

    # Order check
    print("\nChecking cycle order (expected S1 -> Systole -> S2 -> Diastole -> S1...)...")
    problems = []
    for i in range(1, len(segments)):
        prev_label = segments[i - 1][0]
        curr_label = segments[i][0]
        if prev_label in EXPECTED_CYCLE:
            expected_next = EXPECTED_CYCLE[(EXPECTED_CYCLE.index(prev_label) + 1) % 4]
            if curr_label != expected_next:
                problems.append((i, prev_label, curr_label, expected_next))
    if problems:
        print(f"  FOUND {len(problems)} out-of-order transitions:")
        for i, prev, curr, expected in problems[:10]:
            print(f"    segment {i}: {prev} -> {curr}  (expected {expected})")
    else:
        print("  Order looks correct.")

    # Duration sanity check
    print("\nMean duration per phase (this recording):")
    durations = {}
    for label, start, end in segments:
        durations.setdefault(label, []).append((end - start) / fs)
    for label in EXPECTED_CYCLE:
        if label in durations:
            vals = durations[label]
            print(f"  {label:<10} mean={np.mean(vals):.3f}s  n={len(vals)}")
    if "Systole" in durations and "S1" in durations:
        if np.mean(durations["Systole"]) <= np.mean(durations["S1"]) * 1.5:
            print("  NOTE: Systole segments are not much longer than S1 -- physiologically "
                  "unusual (systole is normally the longer interval). Worth checking closely "
                  "against the waveform plot.")


# --- feature table cross-check -------------------------------------------

def check_feature_column_prefixes(filename: str):
    """
    Confirms the phase label strings used here ('S1', 'Systole', 'S2',
    'Diastole') match the column prefixes actually used in
    pcg_feature_table.csv. A silent mismatch here (e.g. this pickle
    says 'Systole' but feature columns are prefixed 'systole_' or
    'SYS_') wouldn't raise an error anywhere downstream -- it would
    just silently produce empty/misaligned lookups in phase_cols,
    which is exactly the kind of bug that produces plausible-looking
    but wrong fidelity numbers.
    """
    import pandas as pd
    if not FEATURE_TABLE_PATH.exists():
        print(f"\n(skipping feature table check -- {FEATURE_TABLE_PATH} not found)")
        return

    df = pd.read_csv(FEATURE_TABLE_PATH)
    print(f"\nFeature table column prefixes found:")
    for prefix in EXPECTED_CYCLE:
        cols = [c for c in df.columns if c.startswith(f"{prefix}_")]
        status = "OK" if cols else "MISSING -- CHECK NAMING MISMATCH"
        print(f"  {prefix}_*  ->  {len(cols)} columns [{status}]  e.g. {cols[:3]}")

    recording_id = filename  # confirm this matches however recording_id is stored in the table
    row = df[df["recording_id"] == recording_id]
    if row.empty:
        print(f"\n  NOTE: '{recording_id}' not found in feature table's recording_id column -- "
              f"check whether filenames there include/exclude the .wav extension, "
              f"or a different id scheme is used.")
    else:
        print(f"\n  Found matching row for {recording_id} in feature table.")


# --- main -------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Loading {TARGET_FILENAME} from {SEGMENTED_DATA_PATH}...")
    rec = load_recording(TARGET_FILENAME)

    print(f"  {len(rec['signal'])} samples @ {rec['fs']} Hz "
          f"({len(rec['signal'])/rec['fs']:.2f}s), true label: {rec['label']}")
    print(f"  {len(rec['phase_segments'])} phase segments detected")

    plot_labeled_waveform(rec)
    print_segment_sequence(rec)
    check_feature_column_prefixes(TARGET_FILENAME)

    print(f"\nSaved labeled waveform plot to {OUTPUT_DIR}")