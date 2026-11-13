"""
analysis/compute_training_calibrated_margins.py

Executes the specific, already-decided plan from earlier in this project:
calibrate each phase's margin using scale derived from the GMM's OWN
TRAINING data -- not test-set spread (already shown to actively destroy
fidelity: 56.6% -> 10.7%) and not the outlier-contaminated test-set std
used in the near-tie check (also shown to be unreliable: a handful of
degenerate-margin recordings inflate the yardstick).

WHAT THIS DOES:
1. Rebuilds each arm's training set exactly as analysis/gather_phase_diagnostics.py
   does for its GMM-fit-diagnostics output (same recording_id filter,
   same scaler.transform).
2. For every TRAINING recording (not test), scores it under both class
   GMMs for every phase and records margin_{phase} = loglik_Present -
   loglik_Absent. This is the thing that was missing: we had the mean
   training log-likelihood (self_score) but never the per-recording
   training margin DISTRIBUTION, which is what you need to build an
   actual scale.
3. Computes, per phase, per arm, from TRAINING margins only:
     - mean, std  (parametric)
     - median, MAD  (robust; MAD scaled by 1.4826 to approximate std
       under normality, so it's comparable in magnitude to std)
4. Loads phase_diagnostics_per_recording.csv (test-set margins, already
   computed) and calibrates each test margin against the TRAINING scale:
     calibrated_std_margin_{phase}  = (margin - train_mean) / train_std
     calibrated_mad_margin_{phase}  = (margin - train_median) / (train_mad * 1.4826)
5. Recomputes dominant_phase and fidelity_match under both calibration
   schemes, and prints a comparison table against raw and against the
   already-known-bad test-set z-score, so all four can be judged side
   by side rather than one at a time.

This does NOT touch or re-litigate the known-catastrophic-clip list;
those recordings are assumed already handled/excluded upstream.

k-fold (updated 2026-09-07)
---------------------------
experiments/arm_a_gmm.py now trains one set of GMMs per fold and the
pooled results[arm] dict carries no models. This script therefore reads
results["gt_fold_results"] / ["pred_fold_results"], computes the training
margin distribution per fold (that fold's GMMs on that fold's training
recordings), and calibrates each test recording against the scale of the
fold in which it was HELD OUT. Using a single pooled scale would leak a
recording's own training-side margin into its yardstick. The "56.6% ->
10.7%" figure above is from the earlier single-split run; the k-fold
equivalent is printed by STEP 4.

ASSUMPTIONS (flagged per project convention -- tell me if any are wrong):
  - gmms[class][phase] is a fitted sklearn GaussianMixture with
    .score_samples(X) available (same object used in analysis/gather_phase_diagnostics.py)
  - CLASSES contains literal strings "Present" and "Absent" (matches the
    literal keys already used in phase_diagnostics_per_recording.csv:
    loglik_Present_{phase}, loglik_Absent_{phase})
  - every per-fold result dict carries gmms / scaler / train_recording_ids /
    test_recording_ids, as written by experiments/arm_a_gmm.py's run_arm
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repo root: makes `paths` and the packages importable from any cwd
import paths


import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from experiments.arm_a_gmm import (
    CLASSES,
    FEATURE_TABLE_PATH,
    PHASES,
    RESULTS_PATH,
    get_feature_columns,
    get_patient_id,
    load_murmur_lookup,
    CIRCOR_METADATA_PATH,
    phase_columns,
)

PER_RECORDING_CSV = paths.PHASE_DIAGNOSTICS_PER_RECORDING_CSV
TRAIN_MARGINS_OUT = paths.ANALYSIS_RESULTS_DIR / "training_margin_distributions.csv"
CALIBRATED_OUT = paths.ANALYSIS_RESULTS_DIR / "phase_diagnostics_calibrated.csv"


def build_train_df(feat_df, arm, train_ids, scaler, feature_cols):
    train_df = feat_df[
        (feat_df["boundary_source"] == arm) & (feat_df["recording_id"].isin(train_ids))
    ].copy()
    murmur_lookup = load_murmur_lookup(CIRCOR_METADATA_PATH)
    train_df["murmur"] = train_df["patient_id"].apply(murmur_lookup)
    train_df = train_df.dropna(subset=feature_cols)
    train_df[feature_cols] = scaler.transform(train_df[feature_cols])
    return train_df


FOLD_KEYS = {"ground_truth": "gt_fold_results", "predicted": "pred_fold_results"}


def compute_training_margins(results):
    """Returns a long DataFrame: recording_id, boundary_source, fold, phase, margin.

    k-fold: each fold has its own GMMs + scaler + training set (stored under
    results["gt_fold_results"] / ["pred_fold_results"] by the k-fold version
    of experiments/arm_a_gmm.py). Training margins are computed per fold with
    that fold's models on that fold's training recordings, so a recording
    contributes one training margin per fold in which it was NOT held out.
    """
    feat_df = pd.read_csv(FEATURE_TABLE_PATH, sep="\t")
    feat_df["patient_id"] = feat_df["recording_id"].apply(get_patient_id)
    feature_cols = get_feature_columns(feat_df.drop(columns=["patient_id"]))

    rows = []
    for arm, fold_key in FOLD_KEYS.items():
        fold_results = results.get(fold_key)
        if fold_results is None:
            raise KeyError(
                f"{RESULTS_PATH} has no '{fold_key}' key -- re-run experiments/arm_a_gmm.py "
                f"(k-fold version) so per-fold GMMs and scalers are saved."
            )
        for fold_idx, fold_result in enumerate(fold_results):
            gmms = fold_result["gmms"]
            scaler: StandardScaler = fold_result["scaler"]
            train_ids = set(fold_result["train_recording_ids"])

            train_df = build_train_df(feat_df, arm, train_ids, scaler, feature_cols)
            if train_df.empty:
                print(f"WARNING: empty training set reconstructed for arm={arm} fold={fold_idx}. "
                      f"Check boundary_source/recording_id filtering assumptions.")
                continue

            for phase in PHASES:
                cols = phase_columns(feature_cols, phase)
                if not cols:
                    continue
                if "Present" not in gmms or "Absent" not in gmms:
                    print(f"WARNING: gmms dict for arm={arm} fold={fold_idx} missing 'Present'/'Absent' "
                          f"keys; found keys: {list(gmms.keys())}. Fix CLASSES assumption.")
                    continue
                gmm_present = gmms["Present"].get(phase)
                gmm_absent = gmms["Absent"].get(phase)
                if gmm_present is None or gmm_absent is None:
                    continue

                X = train_df[cols].values
                ll_present = gmm_present.score_samples(X)
                ll_absent = gmm_absent.score_samples(X)
                margin = ll_present - ll_absent

                for rid, m in zip(train_df["recording_id"].values, margin):
                    rows.append({
                        "recording_id": rid,
                        "boundary_source": arm,
                        "fold": fold_idx,
                        "phase": phase,
                        "margin": m,
                    })

    return pd.DataFrame(rows)


def test_fold_lookup(results):
    """recording_id -> fold index in which that recording was HELD OUT, per
    arm. A test recording must be calibrated against the scale of the fold
    whose training set excluded it -- using a pooled all-fold scale would
    leak that recording's own training-side margin into its yardstick."""
    lookup = {}
    for arm, fold_key in FOLD_KEYS.items():
        for fold_idx, fold_result in enumerate(results[fold_key]):
            for rid in fold_result["test_recording_ids"]:
                lookup[(arm, rid)] = fold_idx
    return lookup


def summarize_training_scale(train_margins_df):
    summary_rows = []
    for (arm, fold, phase), group in train_margins_df.groupby(["boundary_source", "fold", "phase"]):
        vals = group["margin"].values
        mean = np.mean(vals)
        std = np.std(vals, ddof=0)
        median = np.median(vals)
        mad = np.median(np.abs(vals - median))
        summary_rows.append({
            "boundary_source": arm,
            "fold": fold,
            "phase": phase,
            "n_train": len(vals),
            "train_margin_mean": mean,
            "train_margin_std": std,
            "train_margin_median": median,
            "train_margin_mad_scaled": mad * 1.4826,
            "train_margin_min": np.min(vals),
            "train_margin_max": np.max(vals),
        })
    return pd.DataFrame(summary_rows)


def calibrate_test_margins(per_rec_df, scale_df, fold_lookup):
    scale_lookup = {
        (r["boundary_source"], r["fold"], r["phase"]): r
        for _, r in scale_df.iterrows()
    }

    out = per_rec_df.copy()
    out["fold"] = [fold_lookup.get((arm, rid)) for arm, rid in zip(out["boundary_source"], out["recording_id"])]
    n_missing = out["fold"].isna().sum()
    if n_missing:
        print(f"WARNING: {n_missing} test row(s) could not be mapped to a held-out fold -- "
              f"their calibrated margins will be NaN.")

    for phase in PHASES:
        margin_col = f"margin_{phase}"
        if margin_col not in out.columns:
            continue
        std_calib = []
        mad_calib = []
        for _, row in out.iterrows():
            key = (row["boundary_source"], row["fold"], phase)
            scale = scale_lookup.get(key)
            if scale is None:
                std_calib.append(np.nan)
                mad_calib.append(np.nan)
                continue
            m = row[margin_col]
            std_calib.append((m - scale["train_margin_mean"]) / scale["train_margin_std"] if scale["train_margin_std"] else np.nan)
            mad_denom = scale["train_margin_mad_scaled"]
            mad_calib.append((m - scale["train_margin_median"]) / mad_denom if mad_denom else np.nan)
        out[f"calib_std_margin_{phase}"] = std_calib
        out[f"calib_mad_margin_{phase}"] = mad_calib

    def dominant(row, prefix):
        best_phase, best_val = None, -np.inf
        for phase in PHASES:
            col = f"{prefix}_{phase}"
            if col not in row or pd.isna(row[col]):
                continue
            v = abs(row[col])
            if v > best_val:
                best_val, best_phase = v, phase
        return best_phase

    out["dominant_phase_calib_std"] = out.apply(lambda r: dominant(r, "calib_std_margin"), axis=1)
    out["dominant_phase_calib_mad"] = out.apply(lambda r: dominant(r, "calib_mad_margin"), axis=1)

    def matches(row, col):
        expected = row["expected_phases"]
        if not isinstance(expected, str) or not expected:
            return None
        return row[col] in expected

    out["fidelity_match_calib_std"] = out.apply(lambda r: matches(r, "dominant_phase_calib_std"), axis=1)
    out["fidelity_match_calib_mad"] = out.apply(lambda r: matches(r, "dominant_phase_calib_mad"), axis=1)
    return out


def main():
    with open(RESULTS_PATH, "rb") as f:
        results = pickle.load(f)

    print("=" * 70)
    print("STEP 1: computing per-recording TRAINING margins (this may take a moment)")
    print("=" * 70)
    train_margins_df = compute_training_margins(results)
    if train_margins_df.empty:
        print("No training margins computed -- check warnings above before proceeding.")
        sys.exit(1)
    train_margins_df.to_csv(TRAIN_MARGINS_OUT, index=False)
    print(f"Saved {len(train_margins_df)} rows to {TRAIN_MARGINS_OUT}")
    print()

    print("=" * 70)
    print("STEP 2: summarizing training-derived scale per (arm, fold, phase)")
    print("=" * 70)
    scale_df = summarize_training_scale(train_margins_df)
    print(scale_df.round(3).to_string(index=False))
    print()
    print("Fold-averaged scale per (arm, phase):")
    print(scale_df.groupby(["boundary_source", "phase"], sort=False)
          [["n_train", "train_margin_mean", "train_margin_std", "train_margin_median", "train_margin_mad_scaled"]]
          .mean().round(3).to_string())
    print()
    print("NOTE: compare train_margin_std vs train_margin_mad_scaled per row.")
    print("If they're wildly different, the training margins ALSO have outliers,")
    print("and the MAD-based calibration is the one to trust, not the std one.")
    print()

    print("=" * 70)
    print("STEP 3: calibrating test-set margins against training scale")
    print("=" * 70)
    per_rec_path = Path(PER_RECORDING_CSV)
    if not per_rec_path.exists():
        print(f"Could not find {PER_RECORDING_CSV} -- run analysis/gather_phase_diagnostics.py first.")
        sys.exit(1)
    per_rec_df = pd.read_csv(per_rec_path)
    calibrated_df = calibrate_test_margins(per_rec_df, scale_df, test_fold_lookup(results))
    calibrated_df.to_csv(CALIBRATED_OUT, index=False)
    print(f"Saved calibrated per-recording table to {CALIBRATED_OUT}")
    print()

    print("=" * 70)
    print("STEP 4: fidelity comparison -- raw vs test-set-zscore vs BOTH training calibrations")
    print("=" * 70)
    eligible = calibrated_df[calibrated_df["has_expected_phase"] == True].copy()  # noqa: E712
    for arm in eligible["boundary_source"].unique():
        sub = eligible[eligible["boundary_source"] == arm]
        n = len(sub)
        raw = sub["fidelity_match_raw"].mean()
        old_z = sub["fidelity_match_zscored"].mean() if "fidelity_match_zscored" in sub.columns else np.nan
        calib_std = sub["fidelity_match_calib_std"].mean()
        calib_mad = sub["fidelity_match_calib_mad"].mean()
        print(f"[{arm}] n={n}")
        print(f"    raw (existing)             : {raw:.1%}")
        print(f"    test-set z-score (known bad): {old_z:.1%}")
        print(f"    training-calib (std)       : {calib_std:.1%}")
        print(f"    training-calib (MAD, robust): {calib_mad:.1%}")
        print()

    print("Read this plainly: if training-calib doesn't clearly beat raw here,")
    print("the scale-conflation hypothesis is not just 'implemented wrong twice' --")
    print("it's looking less likely to be the actual driver of the fidelity gap,")
    print("and the next step should be investigating something other than margin scale.")


if __name__ == "__main__":
    main()