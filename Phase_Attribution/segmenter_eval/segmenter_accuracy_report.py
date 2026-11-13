"""
segmenter_eval/segmenter_accuracy_report.py

Reports per-recording frame-level segmenter accuracy for the predicted
boundaries sitting in circor_predicted_frames_checkpoint.pkl (produced
by features/build_circor_phase_features.py's 5-fold cross-fit). Doesn't retrain
anything -- just reloads CirCor + rebuilds clips (to get ground-truth
frames back, which weren't saved in the checkpoint) and compares them
against the already-computed predictions.

Output: circor_segmenter_accuracy.csv (recording_id, fold, n_clips,
n_frames, accuracy), plus a console summary matching the same shape as
segmenter_eval/evaluate_segmenter_all_datasets.py's report (overall accuracy, low-accuracy flags).
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repo root: makes `paths` and the packages importable from any cwd
import paths


import pickle

import numpy as np
import pandas as pd

from segmenter_eval.evaluate_segmenter_all_datasets import (
    load_dataset, build_clips, CIRCOR_DIR, BROKEN_RECORDINGS, STATE_LABELS,
)

CHECKPOINT_PATH = paths.CIRCOR_PREDICTED_FRAMES_CHECKPOINT
OUTPUT_CSV = paths.CIRCOR_SEGMENTER_ACCURACY_CSV
LOW_ACCURACY_THRESHOLD = 0.05  # flag recordings below this, same convention as segmenter_eval/evaluate_segmenter_all_datasets.py


def main():
    print(f"Loading checkpoint: {CHECKPOINT_PATH}")
    with open(CHECKPOINT_PATH, "rb") as f:
        checkpoint = pickle.load(f)
    predicted_frames_by_name = checkpoint["predicted_frames_by_name"]
    fold_of_name = checkpoint["fold_of_name"]
    print(f"Checkpoint has predicted frames for {len(predicted_frames_by_name)} recordings")

    print("\nReloading CirCor and rebuilding clips (ground truth only, no training)...")
    circor_recs, circor_segs, circor_names = load_dataset(CIRCOR_DIR, "CirCor", exclude=BROKEN_RECORDINGS)
    clips_flat, ann_flat, names_flat, _ = build_clips(circor_recs, circor_segs, circor_names, "CirCor")

    circor_by_name = {}
    for clip, ann, name in zip(clips_flat, ann_flat, names_flat):
        circor_by_name.setdefault(name, []).append(ann)  # only need ground-truth frames here

    rows = []
    mismatched_clip_counts = []

    for name, gt_frames_list in circor_by_name.items():
        if name not in predicted_frames_by_name:
            print(f"  WARNING: {name} has ground truth but no entry in checkpoint -- skipping")
            continue
        pred_frames_list = predicted_frames_by_name[name]

        if len(gt_frames_list) != len(pred_frames_list):
            mismatched_clip_counts.append(name)
            continue

        correct = 0
        total = 0
        for gt, pred in zip(gt_frames_list, pred_frames_list):
            min_len = min(len(gt), len(pred))
            correct += int(np.sum(np.asarray(gt[:min_len]) == np.asarray(pred[:min_len])))
            total += min_len

        rows.append({
            "recording_id": name,
            "fold": fold_of_name.get(name, -1),
            "n_clips": len(gt_frames_list),
            "n_frames": total,
            "accuracy": correct / total if total > 0 else float("nan"),
        })

    if mismatched_clip_counts:
        print(f"\nWARNING: {len(mismatched_clip_counts)} recording(s) had a different clip count "
              f"between ground truth and the checkpoint's predictions -- skipped, likely means "
              f"clip order/count drifted since the preprocessing run "
              f"(e.g. code changed between runs). First few: {mismatched_clip_counts[:10]}")

    df = pd.DataFrame(rows).sort_values("accuracy")
    df.to_csv(OUTPUT_CSV, index=False)

    print("\n" + "=" * 60)
    print("SEGMENTER ACCURACY ON PREDICTED-BOUNDARY RECORDINGS (5-fold cross-fit)")
    print("=" * 60)
    print(f"Recordings scored: {len(df)}")
    print(f"Mean accuracy: {df['accuracy'].mean():.4f}")
    print(f"Median accuracy: {df['accuracy'].median():.4f}")
    weighted = np.average(df["accuracy"], weights=df["n_frames"])
    print(f"Frame-weighted mean accuracy: {weighted:.4f}")

    flagged = df[df["accuracy"] < LOW_ACCURACY_THRESHOLD]
    print(f"\nRecordings below {LOW_ACCURACY_THRESHOLD:.0%} accuracy: {len(flagged)}")
    if len(flagged) > 0:
        print(flagged[["recording_id", "fold", "accuracy"]].to_string(index=False))

    print(f"\nAccuracy distribution:")
    print(df["accuracy"].describe())

    print(f"\nSaved full per-recording accuracy to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()