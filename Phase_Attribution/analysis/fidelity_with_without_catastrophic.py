"""
analysis/fidelity_with_without_catastrophic.py

Answers a specific question: does GMM correctness / gt-vs-predicted
fidelity change materially once known catastrophically-mis-segmented
recordings are removed from the evaluation set?

Inputs:
  - gmm_murmur_per_recording_comparison.csv  (606-recording GMM test results)
  - circor_segmenter_accuracy.csv            (FULL 3156-recording accuracy
        file -- used for correlation, so every recording has a real
        accuracy value, not just the flagged ones)
  - catastrophic_and_poor_recordings.csv     (output of
        segmenter_eval/identify_catastrophic_recordings.py -- used ONLY to define which
        recording_ids to exclude for the with/without comparison)

IMPORTANT: earlier versions of this script merged segmenter_accuracy in
from catastrophic_and_poor_recordings.csv, which only contains flagged
(non-"ok") rows. That silently NaN'd out every "ok" recording's
accuracy, so the printed correlations were computed on ~30 recordings
instead of 606/600 -- noise, not signal. Fixed by merging accuracy from
the full accuracy file instead.

This does NOT re-run cross-validation, so it can't give you a bigger n
in the catastrophic bucket than whatever catastrophic recordings
happened to land in this particular 606-recording test split (still
likely to be single digits — see the printed overlap count). To get a
real bigger-n answer you'd need to re-run GMM train/test across
multiple folds so every recording appears in some test set; this
script tells you clearly how many catastrophic recordings ARE in this
test split so you know how much to trust the "with" vs "without"
comparison here versus needing that bigger re-run.

Output: two full metrics reports printed to stdout, plus
  fidelity_with_catastrophic.csv
  fidelity_without_catastrophic.csv
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repo root: makes `paths` and the packages importable from any cwd
import paths


import pandas as pd

GMM_COMPARISON_CSV = paths.GMM_PER_RECORDING_CSV
FULL_ACCURACY_CSV = paths.CIRCOR_SEGMENTER_ACCURACY_CSV
FLAGGED_CSV = paths.CATASTROPHIC_AND_POOR_CSV


def report(df, label):
    print(f"\n{'=' * 70}")
    print(f"{label}  (n={len(df)})")
    print("=" * 70)

    gt_rate = df["gt_correct"].mean()
    pred_rate = df["pred_correct"].mean()
    agree_rate = df["gt_pred_agree"].mean()

    print(f"gt_correct rate:    {gt_rate:.3f}")
    print(f"pred_correct rate:  {pred_rate:.3f}")
    print(f"pred - gt:          {pred_rate - gt_rate:+.3f}")
    print(f"gt_pred_agree rate: {agree_rate:.3f}")

    if "segmenter_accuracy" in df.columns and df["segmenter_accuracy"].notna().any():
        corr_pred = df["segmenter_accuracy"].corr(df["pred_correct"].astype(float))
        corr_gt = df["segmenter_accuracy"].corr(df["gt_correct"].astype(float))
        print(f"corr(segmenter_accuracy, pred_correct): {corr_pred:.3f}")
        print(f"corr(segmenter_accuracy, gt_correct):   {corr_gt:.3f}")

    return {
        "label": label,
        "n": len(df),
        "gt_correct_rate": gt_rate,
        "pred_correct_rate": pred_rate,
        "pred_minus_gt": pred_rate - gt_rate,
        "gt_pred_agree_rate": agree_rate,
    }


def main():
    gmm_df = pd.read_csv(GMM_COMPARISON_CSV)
    full_acc_df = pd.read_csv(FULL_ACCURACY_CSV)
    flagged_df = pd.read_csv(FLAGGED_CSV)

    print(f"GMM comparison: {len(gmm_df)} recordings")
    print(f"Full segmenter accuracy file: {len(full_acc_df)} recordings")
    print(f"Flagged (catastrophic+poor) list: {len(flagged_df)} recordings total, from full CirCor")

    catastrophic_ids = set(
        flagged_df.loc[flagged_df["bucket"] == "catastrophic (<5%)", "recording_id"]
    )
    poor_ids = set(
        flagged_df.loc[flagged_df["bucket"] == "poor (5-50%)", "recording_id"]
    )

    # Bring segmenter_accuracy back in for correlation checks -- from the
    # FULL accuracy file, so every recording (including "ok" ones) has a
    # real value instead of NaN.
    merged = gmm_df.merge(
        full_acc_df[["recording_id", "accuracy"]].rename(columns={"accuracy": "segmenter_accuracy"}),
        on="recording_id",
        how="left",
    )
    n_missing = merged["segmenter_accuracy"].isna().sum()
    if n_missing:
        print(f"WARNING: {n_missing} GMM recording(s) had no match in the full accuracy file "
              f"-- correlation will silently drop these, same NaN-handling caveat applies to "
              f"whatever remains unmatched.")

    n_cat_in_test = merged["recording_id"].isin(catastrophic_ids).sum()
    n_poor_in_test = merged["recording_id"].isin(poor_ids).sum()
    print(f"\nOf the full catastrophic list, {n_cat_in_test} recording(s) appear in this "
          f"{len(gmm_df)}-recording test split.")
    print(f"Of the full poor list, {n_poor_in_test} recording(s) appear in this test split.")
    if n_cat_in_test < 15:
        print("NOTE: this is a small overlap. The 'without catastrophic' comparison below is "
              "still only removing a handful of recordings from this split -- it does not give "
              "you the full-dataset statistical power. For that you need every catastrophic "
              "recording to have appeared in a test fold at some point (multi-fold re-run).")

    with_all = merged.copy()
    without_catastrophic = merged[~merged["recording_id"].isin(catastrophic_ids)].copy()
    without_catastrophic_and_poor = merged[
        ~merged["recording_id"].isin(catastrophic_ids | poor_ids)
    ].copy()

    rows = []
    rows.append(report(with_all, "WITH catastrophic + poor recordings (current, as-is)"))
    rows.append(report(without_catastrophic, "WITHOUT catastrophic recordings only"))
    rows.append(report(without_catastrophic_and_poor, "WITHOUT catastrophic AND poor recordings"))

    summary = pd.DataFrame(rows)
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print("=" * 70)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}" if isinstance(x, float) else str(x)))

    with_all.to_csv(paths.ANALYSIS_RESULTS_DIR / "fidelity_with_catastrophic.csv", index=False)
    without_catastrophic.to_csv(paths.ANALYSIS_RESULTS_DIR / "fidelity_without_catastrophic.csv", index=False)
    print("\nSaved fidelity_with_catastrophic.csv and fidelity_without_catastrophic.csv")


if __name__ == "__main__":
    main()