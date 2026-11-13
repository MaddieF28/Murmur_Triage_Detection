"""
segmenter_eval/identify_catastrophic_recordings.py

Reads the FULL circor_segmenter_accuracy.csv (all ~3156 recordings,
produced by segmenter_eval/segmenter_accuracy_report.py) and writes out the list of
recording_ids that fall into the catastrophic (<5%) and poor (5-50%)
segmenter-accuracy buckets.

This is deliberately separate from the GMM comparison script so the
catastrophic/poor list is defined ONCE, off the full dataset, rather
than off whatever 606-recording test split a single GMM run happened
to use. That list can then be:
  (a) fed into analysis/fidelity_with_without_catastrophic.py, and
  (b) reused later as an exclusion filter before training the
      class-conditional phase model (Milestone 3), if you decide to
      drop these recordings rather than fix Problem 2 first.

Output: catastrophic_and_poor_recordings.csv with columns
  recording_id, accuracy, bucket
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repo root: makes `paths` and the packages importable from any cwd
import paths


import pandas as pd

SEGMENTER_ACCURACY_CSV = paths.CIRCOR_SEGMENTER_ACCURACY_CSV
OUTPUT_CSV = paths.CATASTROPHIC_AND_POOR_CSV

CATASTROPHIC_THRESHOLD = 0.05
POOR_THRESHOLD = 0.50


def bucket(acc):
    if acc < CATASTROPHIC_THRESHOLD:
        return "catastrophic (<5%)"
    elif acc < POOR_THRESHOLD:
        return "poor (5-50%)"
    else:
        return "ok (>=50%)"


def main():
    df = pd.read_csv(SEGMENTER_ACCURACY_CSV)
    print(f"Loaded {len(df)} total recordings from {SEGMENTER_ACCURACY_CSV}")

    df["bucket"] = df["accuracy"].apply(bucket)

    counts = df["bucket"].value_counts().reindex(
        ["catastrophic (<5%)", "poor (5-50%)", "ok (>=50%)"]
    )
    print("\nFull-dataset bucket counts:")
    print(counts)

    flagged = df[df["bucket"] != "ok (>=50%)"].copy()
    flagged = flagged.sort_values("accuracy")

    print(f"\n{len(flagged)} recordings flagged as catastrophic or poor "
          f"out of {len(df)} total ({100 * len(flagged) / len(df):.2f}%)")

    print("\nCatastrophic recording_ids (<5% accuracy):")
    cat_ids = flagged[flagged["bucket"] == "catastrophic (<5%)"]["recording_id"].tolist()
    for rid in cat_ids:
        print(f"  {rid}")
    print(f"Total catastrophic: {len(cat_ids)}")

    flagged[["recording_id", "accuracy", "bucket"]].to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved full flagged list to {OUTPUT_CSV}")
    print("Use this file's recording_id column as your exclusion/inclusion filter "
          "in the fidelity comparison and (later) in phase-model training.")


if __name__ == "__main__":
    main()