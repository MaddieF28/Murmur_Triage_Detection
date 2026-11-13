"""
analysis/fidelity_baseline_and_catastrophic.py

Two questions the raw 56.6%/41.0% fidelity numbers can't answer on
their own:

1. BASELINE: is 56.6% (ground_truth arm) or 41.0% (predicted arm)
   actually better than a trivial rule that ignores the model
   entirely and always guesses whichever phase is most common in
   CirCor's expected-phase labels? If CirCor's murmur-timing labels
   are heavily skewed toward Systolic, a naive "always guess Systole"
   rule could match a large fraction of recordings for free, and the
   model needs to clear that bar before either number means anything.

2. CATASTROPHIC CONFOUND: is the predicted-arm's fidelity drop
   (56.6% -> 41.0%) spread evenly across all segmentation quality
   levels, or concentrated in the small set of recordings with known
   catastrophic (near-zero-accuracy) segmentation? These are different
   claims -- "phase attribution degrades gracefully with any
   segmentation noise" vs "phase attribution is fine except when the
   known Problem 2 bug hits a recording" -- and only one of them is
   currently supported by anything you've measured.

Inputs:
  - phase_fidelity_per_recording.tsv   (from experiments/arm_a_gmm.py)
  - circor_segmenter_accuracy.csv      (FULL 3156-recording accuracy file,
        from segmenter_eval/segmenter_accuracy_report.py)

Outputs (printed + saved):
  - naive baseline match rate, per candidate phase
  - fidelity vs segmenter-accuracy bucket table (predicted arm only --
    the ground_truth arm doesn't have a meaningful segmenter-accuracy
    value since it uses the true boundaries by definition)
  - list of recordings that matched under ground_truth boundaries but
    stopped matching under predicted boundaries ("flips"), with their
    segmenter accuracy, so you can see directly whether they cluster
    at the low end
  - phase_fidelity_with_segmenter_accuracy.csv
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repo root: makes `paths` and the packages importable from any cwd
import paths


import pandas as pd

FIDELITY_TSV = paths.GMM_PHASE_FIDELITY_TSV
FULL_ACCURACY_CSV = paths.CIRCOR_SEGMENTER_ACCURACY_CSV
OUTPUT_CSV = paths.ANALYSIS_RESULTS_DIR / "phase_fidelity_with_segmenter_accuracy.csv"

CATASTROPHIC_THRESHOLD = 0.05
POOR_THRESHOLD = 0.50


def bucket(acc):
    if pd.isna(acc):
        return "unknown"
    elif acc < CATASTROPHIC_THRESHOLD:
        return "catastrophic (<5%)"
    elif acc < POOR_THRESHOLD:
        return "poor (5-50%)"
    else:
        return "ok (>=50%)"


def compute_naive_baseline(scoped_df):
    """For each phase that ever appears in an expected_phases set, what
    fraction of recordings would a trivial 'always guess this phase'
    rule match? Reports all candidates so you can see which one (if
    any) the actual model needs to beat."""
    all_phases = set()
    for s in scoped_df["expected_phases"]:
        all_phases.update(s.split(","))

    print("\nNaive baseline -- 'always guess phase X' match rate, per candidate phase:")
    results = {}
    for phase in sorted(all_phases):
        matches = scoped_df["expected_phases"].apply(lambda s: phase in s.split(","))
        rate = matches.mean()
        results[phase] = rate
        print(f"  always guess {phase:>10}: {rate * 100:.1f}% ({matches.sum()}/{len(scoped_df)})")

    best_phase = max(results, key=lambda p: results[p])
    print(f"  -> best naive baseline: always guess {best_phase} = {results[best_phase] * 100:.1f}%")
    return results, best_phase


def main():
    fidelity_df = pd.read_csv(FIDELITY_TSV, sep="\t")
    full_acc_df = pd.read_csv(FULL_ACCURACY_CSV)

    scoped = fidelity_df[fidelity_df["has_expected_phase"]].copy()
    print(f"Loaded {len(fidelity_df)} rows from {FIDELITY_TSV} "
          f"({len(scoped)} with a known expected phase)")

    print("\nExpected-phase set distribution (across both arms -- same underlying "
          "recordings, so this is double-counted per arm; halve for per-recording counts):")
    print(scoped["expected_phases"].value_counts())

    # --- 1. Naive baseline ---
    print("\n" + "=" * 70)
    print("BASELINE CHECK")
    print("=" * 70)
    for arm in ("ground_truth", "predicted"):
        arm_scoped = scoped[scoped["boundary_source"] == arm]
        actual_rate = arm_scoped["fidelity_match"].mean()
        print(f"\n--- {arm} arm (n={len(arm_scoped)}) ---")
        print(f"Actual model fidelity: {actual_rate * 100:.1f}%")
        baseline_results, best_phase = compute_naive_baseline(arm_scoped)
        delta = actual_rate - baseline_results[best_phase]
        print(f"Model vs best naive baseline: {delta * 100:+.1f} points")

    # --- 2. Catastrophic-segmentation confound (predicted arm only) ---
    print("\n" + "=" * 70)
    print("CATASTROPHIC-SEGMENTATION CONFOUND (predicted arm)")
    print("=" * 70)

    pred_scoped = scoped[scoped["boundary_source"] == "predicted"].copy()
    merged = pred_scoped.merge(
        full_acc_df[["recording_id", "accuracy"]].rename(columns={"accuracy": "segmenter_accuracy"}),
        on="recording_id",
        how="left",
    )
    n_missing = merged["segmenter_accuracy"].isna().sum()
    if n_missing:
        print(f"WARNING: {n_missing} recording(s) had no match in {FULL_ACCURACY_CSV} -- "
              f"excluded from the bucket breakdown below.")

    merged["segmenter_bucket"] = merged["segmenter_accuracy"].apply(bucket)

    print("\nFidelity match rate by segmenter-accuracy bucket (predicted arm):")
    bucket_table = merged.groupby("segmenter_bucket").agg(
        n=("fidelity_match", "size"),
        fidelity_match_rate=("fidelity_match", "mean"),
    )
    order = ["catastrophic (<5%)", "poor (5-50%)", "ok (>=50%)", "unknown"]
    bucket_table = bucket_table.reindex([b for b in order if b in bucket_table.index])
    print(bucket_table.to_string(float_format=lambda x: f"{x:.3f}"))

    # --- 3. Recordings that flipped from match (gt) to non-match (pred) ---
    print("\n" + "=" * 70)
    print("RECORDINGS THAT FLIPPED FROM MATCH (gt) TO NON-MATCH (predicted)")
    print("=" * 70)

    gt_scoped = scoped[scoped["boundary_source"] == "ground_truth"][
        ["recording_id", "fidelity_match", "dominant_phase"]
    ].rename(columns={"fidelity_match": "gt_fidelity_match", "dominant_phase": "gt_dominant_phase"})

    flip_check = merged.merge(gt_scoped, on="recording_id", how="left")
    flipped = flip_check[
        (flip_check["gt_fidelity_match"] == True) & (flip_check["fidelity_match"] == False)
    ].copy()
    flipped = flipped.sort_values("segmenter_accuracy")

    print(f"\n{len(flipped)} recording(s) flipped from match to non-match:")
    if len(flipped) > 0:
        print(flipped[[
            "recording_id", "segmenter_accuracy", "segmenter_bucket",
            "gt_dominant_phase", "dominant_phase", "expected_phases",
        ]].to_string(index=False))

        flip_bucket_counts = flipped["segmenter_bucket"].value_counts()
        print(f"\nFlip count by segmenter bucket:\n{flip_bucket_counts}")
    else:
        print("(none)")

    merged.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved predicted-arm fidelity + segmenter accuracy to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()