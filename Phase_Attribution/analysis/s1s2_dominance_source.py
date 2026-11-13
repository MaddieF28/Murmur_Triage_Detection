"""
analysis/s1s2_dominance_source.py

Extends check_s1s2_dominance_source.py with three things that were
missing for a real conclusion about whether S1/S2 phase dominance
reflects genuine diagnostic content:

1. Per-cell counts for the murmur grade/pitch/quality crosstabs (Test 2
   in the original script reported rates with no n -- a 100% cell on
   n=1 is meaningless).

2. Duration confound check: GMM log-likelihood magnitude is sensitive
   to segment length/variance, independent of real discriminative
   content. S1/S2 segments are typically much shorter than
   Systole/Diastole segments. If a duration-like feature exists per
   phase, this checks whether phase duration correlates with margin
   magnitude -- a correlation would mean Test 3's "confident, not
   marginal" margins could be a length artifact, not evidence of real
   signal.
   ASSUMPTION FLAGGED, NOT VERIFIED: this looks for feature columns
   matching "{phase}_*duration*" (case-insensitive) in
   circor_binary_features.tsv. If no such column exists, it says so
   explicitly rather than silently skipping -- tell me where duration
   lives (if anywhere) if this comes up empty and it matters.

3. Direct, GMM-independent feature-separation test: for each phase,
   computes Cohen's d (standardized Present-vs-Absent mean difference)
   across that phase's raw feature columns, averaged into one
   per-phase separation score, for both boundary_source arms. This is
   the test that actually answers "does this phase's feature space
   separate the two classes" -- it doesn't go through GMM likelihoods
   at all, so it isn't subject to the same scale/duration confound as
   the margin-based metric.

Inputs:
  - phase_fidelity_per_recording.tsv
  - circor_binary_features.tsv (FEATURE_TABLE_PATH from
    experiments/arm_a_gmm.py)
  - CirCor metadata CSV (CIRCOR_METADATA_PATH from
    experiments/arm_a_gmm.py)

Reuses load_murmur_lookup, get_patient_id, get_feature_columns,
phase_columns, PHASES, CLASSES, LABEL_COLUMN, FEATURE_TABLE_PATH,
CIRCOR_METADATA_PATH directly from experiments/arm_a_gmm.py rather
than re-deriving them, since those are already the pipeline's source
of truth. Importing that module only runs its function/constant
definitions (main() is guarded by __name__ == "__main__"), so this is
safe.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repo root: makes `paths` and the packages importable from any cwd
import paths


import numpy as np
import pandas as pd

from segmenter_eval.evaluate_segmenter_all_datasets import CIRCOR_DIR
from experiments.arm_a_gmm import (
    CIRCOR_METADATA_PATH,
    CLASSES,
    FEATURE_TABLE_PATH,
    LABEL_COLUMN,
    PHASES,
    get_feature_columns,
    get_patient_id,
    load_murmur_lookup,
    phase_columns,
)

FIDELITY_TSV = paths.GMM_PHASE_FIDELITY_TSV

SYSTOLIC_TIMING_COLUMN = "Systolic murmur timing"
DIASTOLIC_TIMING_COLUMN = "Diastolic murmur timing"
SYSTOLIC_GRADE_COLUMN = "Systolic murmur grading"
SYSTOLIC_PITCH_COLUMN = "Systolic murmur pitch"
SYSTOLIC_QUALITY_COLUMN = "Systolic murmur quality"
DIASTOLIC_GRADE_COLUMN = "Diastolic murmur grading"
DIASTOLIC_PITCH_COLUMN = "Diastolic murmur pitch"
DIASTOLIC_QUALITY_COLUMN = "Diastolic murmur quality"

S1S2 = {"S1", "S2"}
SYSDIA = {"Systole", "Diastole"}


def load_metadata():
    if not CIRCOR_METADATA_PATH.exists():
        raise FileNotFoundError(f"CirCor metadata CSV not found at {CIRCOR_METADATA_PATH}.")
    df = pd.read_csv(CIRCOR_METADATA_PATH)
    df["patient_id"] = df["Patient ID"].astype(str).str.strip()
    return df.set_index("patient_id")


def cohens_d(present_vals, absent_vals):
    present_vals = present_vals[~np.isnan(present_vals)]
    absent_vals = absent_vals[~np.isnan(absent_vals)]
    if len(present_vals) < 2 or len(absent_vals) < 2:
        return np.nan
    pooled_std = np.sqrt(
        ((len(present_vals) - 1) * np.var(present_vals, ddof=1)
         + (len(absent_vals) - 1) * np.var(absent_vals, ddof=1))
        / (len(present_vals) + len(absent_vals) - 2)
    )
    if pooled_std == 0 or np.isnan(pooled_std):
        return np.nan
    return (np.mean(present_vals) - np.mean(absent_vals)) / pooled_std


def main():
    fidelity_df = pd.read_csv(FIDELITY_TSV, sep="\t")
    scoped = fidelity_df[fidelity_df["has_expected_phase"]].copy()
    scoped["dominant_group"] = scoped["dominant_phase"].apply(
        lambda p: "S1/S2" if p in S1S2 else ("Systole/Diastole" if p in SYSDIA else "other")
    )
    scoped["patient_id"] = scoped["recording_id"].apply(get_patient_id)

    meta = load_metadata()
    joined = scoped.join(meta, on="patient_id", how="left", rsuffix="_meta")

    def pick_field(row, sys_col, dia_col):
        vals = []
        if pd.notna(row.get(SYSTOLIC_TIMING_COLUMN)):
            vals.append(row.get(sys_col))
        if pd.notna(row.get(DIASTOLIC_TIMING_COLUMN)):
            vals.append(row.get(dia_col))
        vals = [v for v in vals if pd.notna(v)]
        return vals[0] if vals else None

    joined["murmur_grade"] = joined.apply(lambda r: pick_field(r, SYSTOLIC_GRADE_COLUMN, DIASTOLIC_GRADE_COLUMN), axis=1)
    joined["murmur_pitch"] = joined.apply(lambda r: pick_field(r, SYSTOLIC_PITCH_COLUMN, DIASTOLIC_PITCH_COLUMN), axis=1)
    joined["murmur_quality"] = joined.apply(lambda r: pick_field(r, SYSTOLIC_QUALITY_COLUMN, DIASTOLIC_QUALITY_COLUMN), axis=1)

    print("=" * 70)
    print("TEST 2 (fixed): murmur grade/pitch/quality vs dominant_group, WITH COUNTS")
    print("=" * 70)
    pred_only = joined[joined["boundary_source"] == "predicted"]
    for field in ("murmur_grade", "murmur_pitch", "murmur_quality"):
        print(f"\n--- {field} (predicted arm) ---")
        counts = pd.crosstab(pred_only[field], pred_only["dominant_group"])
        rates = pd.crosstab(pred_only[field], pred_only["dominant_group"], normalize="index")
        combined = counts.astype(str) + "  (" + (rates * 100).round(1).astype(str) + "%)"
        print(combined.to_string())
        print(f"  row totals: {counts.sum(axis=1).to_dict()}")

    print("\n" + "=" * 70)
    print("TEST 4: duration confound -- does phase duration correlate with margin size?")
    print("=" * 70)
    feat_df = pd.read_csv(FEATURE_TABLE_PATH, sep="\t")
    duration_cols = {}
    for phase in PHASES:
        candidates = [c for c in feat_df.columns if c.startswith(f"{phase}_") and "duration" in c.lower()]
        if candidates:
            duration_cols[phase] = candidates[0]
    if not duration_cols:
        print("No column matching '{phase}_*duration*' found in circor_binary_features.tsv for any phase. "
              "No duration data available in this feature table to check the length confound -- "
              "if duration is tracked elsewhere (e.g. raw segmenter boundary output), point me at it "
              "and I'll redo this check against that instead of guessing a column name.")
    else:
        print(f"Found duration columns: {duration_cols}")
        feat_df["patient_id"] = feat_df["recording_id"].apply(get_patient_id)
        merged_dur = fidelity_df.merge(
            feat_df[["recording_id", "boundary_source"] + list(duration_cols.values())],
            on=["recording_id", "boundary_source"], how="left",
        )
        for phase, col in duration_cols.items():
            margin_col = f"margin_{phase}"
            if margin_col not in merged_dur.columns:
                continue
            sub = merged_dur[["boundary_source", col, margin_col]].dropna()
            for arm in ("ground_truth", "predicted"):
                arm_sub = sub[sub["boundary_source"] == arm]
                if len(arm_sub) > 2:
                    corr = arm_sub[col].corr(arm_sub[margin_col].abs())
                    print(f"  [{arm}] corr(duration, |margin|) for {phase}: {corr:.3f} (n={len(arm_sub)})")

    print("\n" + "=" * 70)
    print("TEST 5: direct per-phase feature separation (Cohen's d), GMM-independent")
    print("=" * 70)
    murmur_lookup = load_murmur_lookup(CIRCOR_METADATA_PATH)
    feat_df["murmur"] = feat_df["patient_id"].apply(murmur_lookup)
    feat_df = feat_df[feat_df["murmur"].isin(CLASSES)].copy()
    feature_cols = get_feature_columns(feat_df)

    for arm in ("ground_truth", "predicted"):
        print(f"\n--- {arm} arm ---")
        arm_df = feat_df[feat_df["boundary_source"] == arm]
        present = arm_df[arm_df[LABEL_COLUMN] == "Present"]
        absent = arm_df[arm_df[LABEL_COLUMN] == "Absent"]
        for phase in PHASES:
            cols = phase_columns(feature_cols, phase)
            if not cols:
                continue
            ds = [
                cohens_d(present[c].values.astype(float), absent[c].values.astype(float))
                for c in cols
            ]
            ds = [d for d in ds if not np.isnan(d)]
            mean_abs_d = np.mean(np.abs(ds)) if ds else float("nan")
            print(f"  {phase:>10}: mean |Cohen's d| across {len(ds)} feature(s) = {mean_abs_d:.3f}")


if __name__ == "__main__":
    main()