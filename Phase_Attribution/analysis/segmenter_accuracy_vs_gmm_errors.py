"""


Merges circor_segmenter_accuracy.csv (per-recording segmenter frame
accuracy, from segmenter_eval/segmenter_accuracy_report.py) with
gmm_murmur_per_recording_comparison.csv (per-recording GMM correctness
for both boundary sources, from experiments/arm_a_gmm.py) to test
directly: do the GMM's predicted-boundary errors concentrate in the
recordings where the segmenter itself did badly?

Key design point: this checks BOTH arms against segmenter accuracy, not
just the predicted arm. The ground_truth arm never sees predicted
boundaries at all, so its correctness should be roughly INDEPENDENT of
segmenter accuracy on that same recording. If gt_correct also drops in
the badly-segmented bucket, that's a sign these recordings are just
generally hard (noisy audio, ambiguous murmurs, etc.) rather than the
segmenter's errors specifically causing the GMM's errors. If only
pred_correct drops there, that's a clean causal signal: bad boundaries
-> bad predicted-boundary features -> bad classification, isolated from
"this recording is hard" confounds.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repo root: makes `paths` and the packages importable from any cwd
import paths


import numpy as np
import pandas as pd

SEGMENTER_ACCURACY_CSV = paths.CIRCOR_SEGMENTER_ACCURACY_CSV
GMM_COMPARISON_CSV = paths.GMM_PER_RECORDING_CSV
OUTPUT_MERGED_CSV = paths.ANALYSIS_RESULTS_DIR / "segmenter_accuracy_vs_gmm_errors.csv"

# Two cut points: CATASTROPHIC isolates the exact-zero rotation/flip
# failure signature we already know about; POOR is a softer boundary to
# see if the effect extends beyond the catastrophic cases.
CATASTROPHIC_THRESHOLD = 0.05
POOR_THRESHOLD = 0.50


def bucket(acc):
    if acc < CATASTROPHIC_THRESHOLD:
        return "catastrophic (<5%)"
    elif acc < POOR_THRESHOLD:
        return "poor (5-50%)"
    else:
        return "ok (>=50%)"


def print_bucket_table(df, bucket_col, value_cols, label):
    print(f"\n{label}:")
    grouped = df.groupby(bucket_col)[value_cols].mean()
    counts = df.groupby(bucket_col).size().rename("n")
    table = grouped.join(counts)
    # order buckets sensibly rather than alphabetically
    order = ["catastrophic (<5%)", "poor (5-50%)", "ok (>=50%)"]
    table = table.reindex([b for b in order if b in table.index])
    print(table.to_string(float_format=lambda x: f"{x:.3f}"))


def main():
    seg_df = pd.read_csv(SEGMENTER_ACCURACY_CSV)
    gmm_df = pd.read_csv(GMM_COMPARISON_CSV)

    print(f"Segmenter accuracy: {len(seg_df)} recordings")
    print(f"GMM comparison: {len(gmm_df)} recordings (test set)")

    merged = gmm_df.merge(seg_df[["recording_id", "accuracy"]], on="recording_id", how="left")
    merged = merged.rename(columns={"accuracy": "segmenter_accuracy"})

    n_unmatched = merged["segmenter_accuracy"].isna().sum()
    if n_unmatched:
        print(f"WARNING: {n_unmatched} GMM test recording(s) had no matching segmenter accuracy "
              f"row -- dropping them from this analysis.")
    merged = merged.dropna(subset=["segmenter_accuracy"])

    merged["segmenter_bucket"] = merged["segmenter_accuracy"].apply(bucket)

    print("\n" + "=" * 70)
    print(f"Merged: {len(merged)} test recordings with both segmenter accuracy and GMM results")
    print("=" * 70)

    print("\nRecordings per segmenter-accuracy bucket:")
    print(merged["segmenter_bucket"].value_counts().reindex(
        ["catastrophic (<5%)", "poor (5-50%)", "ok (>=50%)"]
    ))

    print_bucket_table(
        merged, "segmenter_bucket", ["gt_correct", "pred_correct"],
        "GMM correctness rate by segmenter-accuracy bucket (gt_correct vs pred_correct)"
    )

    print_bucket_table(
        merged, "segmenter_bucket", ["gt_pred_agree"],
        "GT-vs-predicted prediction AGREEMENT rate by segmenter-accuracy bucket"
    )

    # Isolate the causal signal: pred_correct minus gt_correct per bucket.
    # Near 0 in the "ok" bucket, meaningfully negative in "catastrophic"
    # would be the clean confirmation.
    print("\npred_correct - gt_correct per bucket (negative = predicted-boundary arm doing "
          "worse specifically where segmentation is bad):")
    diff = merged.groupby("segmenter_bucket").apply(
        lambda g: g["pred_correct"].mean() - g["gt_correct"].mean()
    )
    order = ["catastrophic (<5%)", "poor (5-50%)", "ok (>=50%)"]
    diff = diff.reindex([b for b in order if b in diff.index])
    print(diff.to_string(float_format=lambda x: f"{x:+.3f}"))

    # Simple correlation as a continuous-variable sanity check alongside
    # the bucketed view.
    corr_pred = merged["segmenter_accuracy"].corr(merged["pred_correct"].astype(float))
    corr_gt = merged["segmenter_accuracy"].corr(merged["gt_correct"].astype(float))
    print(f"\nCorrelation(segmenter_accuracy, pred_correct): {corr_pred:.3f}")
    print(f"Correlation(segmenter_accuracy, gt_correct):   {corr_gt:.3f}")
    print("(if pred's correlation is clearly higher than gt's, that supports a real link "
          "between boundary quality and predicted-arm errors, not just 'hard recording' confounds)")

    merged.to_csv(OUTPUT_MERGED_CSV, index=False)
    print(f"\nSaved merged per-recording table to {OUTPUT_MERGED_CSV}")


if __name__ == "__main__":
    main()