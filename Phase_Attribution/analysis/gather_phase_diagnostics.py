"""
analysis/gather_phase_diagnostics.py

Pulls together everything potentially relevant to "why does the GMM's
phase-attribution margin sometimes disagree with which phase actually
separates Present/Absent" into two CSVs, so multiple candidate causes
can be checked at once instead of one script per hypothesis:

  - scale/magnitude artifacts in the raw margin (addressed via
    z-scored margins, computed both ways)
  - GMM fit degeneracy (near-zero component weight, tiny/huge
    covariance) that could produce unstable or inflated
    log-likelihoods independent of real signal
  - phase duration as a confound (already partly checked, included
    again here alongside everything else so it can be cross-referenced)
  - segmenter accuracy (predicted arm only)
  - murmur grade/pitch/quality metadata

Output 1: phase_diagnostics_per_recording.csv
  One row per (recording_id, boundary_source). Columns:
    recording_id, boundary_source, patient_id, true_label,
    predicted_label, correct,
    margin_{phase}, loglik_Present_{phase}, loglik_Absent_{phase},
    duration_{phase} (if a duration column exists in the feature table),
    z_margin_{phase}  -- margin_{phase} z-scored against the
        distribution of margin_{phase} within this arm's test set
        (mean/std computed over ALL test recordings in that arm, not
        per-recording, so this is a fair per-phase rescaling)
    dominant_phase_raw   -- argmax(|margin_phase|), same as the
        existing pipeline's dominant_phase
    dominant_phase_zscored -- argmax(|z_margin_phase|)
    expected_phases, has_expected_phase,
    fidelity_match_raw, fidelity_match_zscored
    segmenter_accuracy (predicted arm only; NaN for ground_truth)
    murmur_grade, murmur_pitch, murmur_quality

Output 2: phase_diagnostics_gmm_fit.csv
  One row per (boundary_source, fold, class, phase) -- one set of GMMs is
  fitted per k-fold by experiments/arm_a_gmm.py, read from
  results["gt_fold_results"] / ["pred_fold_results"]; a fold-averaged
  summary is printed. Columns:
    boundary_source, fold, class, phase, n_train_samples, feature_count,
    component_weights (list, as string), min_weight, max_weight,
    mean_cov_diag, min_cov_diag, max_cov_diag,
    self_score  -- gmm.score() on its OWN training data (average
        log-likelihood per sample) -- a very tight/overfit fit here
        would show as an unusually large-magnitude self_score relative
        to other phases, which is one way a phase could win the raw
        margin comparison for reasons unrelated to real class
        separation.

ASSUMPTIONS FLAGGED (matching prior scripts in this session):
  - patient_id = recording_id.split("_", 1)[0]
  - duration columns, if present, match "{phase}_*duration*"
    (case-insensitive) in circor_binary_features.tsv
  - CIRCOR_METADATA_PATH / SYSTOLIC_TIMING_COLUMN etc. as defined in
    experiments/arm_a_gmm.py
If any of these are wrong for your setup, the output will say so
(missing columns) rather than silently produce wrong numbers -- flag
it back to me rather than me guessing further.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repo root: makes `paths` and the packages importable from any cwd
import paths


import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from experiments.arm_a_gmm import (
    CIRCOR_METADATA_PATH,
    CLASSES,
    FEATURE_TABLE_PATH,
    LABEL_COLUMN,
    PHASES,
    RESULTS_PATH,
    get_feature_columns,
    get_patient_id,
    load_murmur_lookup,
    load_murmur_timing_lookup,
    phase_columns,
)

SEGMENTER_ACCURACY_CSV = paths.CIRCOR_SEGMENTER_ACCURACY_CSV  # full 3156-recording file, as used before

SYSTOLIC_TIMING_COLUMN = "Systolic murmur timing"
DIASTOLIC_TIMING_COLUMN = "Diastolic murmur timing"
SYSTOLIC_GRADE_COLUMN = "Systolic murmur grading"
SYSTOLIC_PITCH_COLUMN = "Systolic murmur pitch"
SYSTOLIC_QUALITY_COLUMN = "Systolic murmur quality"
DIASTOLIC_GRADE_COLUMN = "Diastolic murmur grading"
DIASTOLIC_PITCH_COLUMN = "Diastolic murmur pitch"
DIASTOLIC_QUALITY_COLUMN = "Diastolic murmur quality"

PER_RECORDING_OUT = paths.PHASE_DIAGNOSTICS_PER_RECORDING_CSV
GMM_FIT_OUT = paths.PHASE_DIAGNOSTICS_GMM_FIT_CSV


def load_metadata():
    df = pd.read_csv(CIRCOR_METADATA_PATH)
    df["patient_id"] = df["Patient ID"].astype(str).str.strip()
    return df.set_index("patient_id")


def pick_field(row, timing_sys_present, timing_dia_present, sys_col, dia_col, meta_row):
    vals = []
    if timing_sys_present:
        vals.append(meta_row.get(sys_col))
    if timing_dia_present:
        vals.append(meta_row.get(dia_col))
    vals = [v for v in vals if pd.notna(v)]
    return vals[0] if vals else None


def main():
    with open(RESULTS_PATH, "rb") as f:
        results = pickle.load(f)

    feat_df = pd.read_csv(FEATURE_TABLE_PATH, sep="\t")
    feat_df["patient_id"] = feat_df["recording_id"].apply(get_patient_id)

    duration_cols = {}
    for phase in PHASES:
        candidates = [c for c in feat_df.columns if c.startswith(f"{phase}_") and "duration" in c.lower()]
        if candidates:
            duration_cols[phase] = candidates[0]
    if not duration_cols:
        print("NOTE: no '{phase}_*duration*' columns found in feature table -- "
              "duration_{phase} columns will be omitted from output. Tell me if duration "
              "data lives elsewhere and matters.")

    meta = load_metadata()
    timing_lookup = load_murmur_timing_lookup(CIRCOR_METADATA_PATH)

    seg_acc_path = Path(SEGMENTER_ACCURACY_CSV)
    seg_acc_df = None
    if seg_acc_path.exists():
        seg_acc_df = pd.read_csv(seg_acc_path)[["recording_id", "accuracy"]].rename(
            columns={"accuracy": "segmenter_accuracy"}
        )
    else:
        print(f"NOTE: {SEGMENTER_ACCURACY_CSV} not found -- segmenter_accuracy column will be omitted.")

    # ---------------- Output 1: per-recording diagnostics ----------------
    per_recording_rows = []
    for arm in ("ground_truth", "predicted"):
        arm_result = results[arm]
        recs = arm_result["per_recording_results"]

        # z-score reference: mean/std of margin_{phase} across this arm's
        # full test set, computed once per phase.
        margin_arrays = {phase: np.array([r["phase_margins"][phase] for r in recs]) for phase in PHASES}
        margin_mean = {phase: margin_arrays[phase].mean() for phase in PHASES}
        margin_std = {phase: margin_arrays[phase].std(ddof=0) or 1.0 for phase in PHASES}

        for r in recs:
            pid = get_patient_id(r["recording_id"])
            expected = timing_lookup.get(pid, set())
            meta_row = meta.loc[pid] if pid in meta.index else {}
            timing_sys = pd.notna(meta_row.get(SYSTOLIC_TIMING_COLUMN)) if len(meta_row) else False
            timing_dia = pd.notna(meta_row.get(DIASTOLIC_TIMING_COLUMN)) if len(meta_row) else False

            row = {
                "recording_id": r["recording_id"],
                "boundary_source": arm,
                "patient_id": pid,
                "true_label": r["true_label"],
                "predicted_label": r["predicted_label"],
                "correct": r["correct"],
                "expected_phases": ",".join(sorted(expected)) if expected else "",
                "has_expected_phase": len(expected) > 0,
                "murmur_grade": pick_field(r, timing_sys, timing_dia, SYSTOLIC_GRADE_COLUMN, DIASTOLIC_GRADE_COLUMN, meta_row) if len(meta_row) else None,
                "murmur_pitch": pick_field(r, timing_sys, timing_dia, SYSTOLIC_PITCH_COLUMN, DIASTOLIC_PITCH_COLUMN, meta_row) if len(meta_row) else None,
                "murmur_quality": pick_field(r, timing_sys, timing_dia, SYSTOLIC_QUALITY_COLUMN, DIASTOLIC_QUALITY_COLUMN, meta_row) if len(meta_row) else None,
            }

            z_margins = {}
            for phase in PHASES:
                m = r["phase_margins"][phase]
                row[f"margin_{phase}"] = m
                row[f"loglik_Present_{phase}"] = r["per_class_per_phase_loglik"]["Present"][phase]
                row[f"loglik_Absent_{phase}"] = r["per_class_per_phase_loglik"]["Absent"][phase]
                z = (m - margin_mean[phase]) / margin_std[phase]
                z_margins[phase] = z
                row[f"z_margin_{phase}"] = z

            dom_raw = max(PHASES, key=lambda p: abs(r["phase_margins"][p]))
            dom_z = max(PHASES, key=lambda p: abs(z_margins[p]))
            row["dominant_phase_raw"] = dom_raw
            row["dominant_phase_zscored"] = dom_z
            row["fidelity_match_raw"] = (dom_raw in expected) if expected else None
            row["fidelity_match_zscored"] = (dom_z in expected) if expected else None

            for phase, col in duration_cols.items():
                feat_row = feat_df[(feat_df["recording_id"] == r["recording_id"]) & (feat_df["boundary_source"] == arm)]
                row[f"duration_{phase}"] = feat_row[col].iloc[0] if len(feat_row) else None

            per_recording_rows.append(row)

    per_recording_df = pd.DataFrame(per_recording_rows)
    if seg_acc_df is not None:
        per_recording_df = per_recording_df.merge(seg_acc_df, on="recording_id", how="left")
    per_recording_df.to_csv(PER_RECORDING_OUT, index=False)
    print(f"Saved {len(per_recording_df)} rows to {PER_RECORDING_OUT}")

    # ---------------- Output 2: GMM fit diagnostics (per fold) ----------------
    # Since the k-fold conversion of experiments/arm_a_gmm.py the pooled
    # results[arm] dict has no fitted models -- each fold trains its own
    # GMMs + scaler, stored under results["gt_fold_results"] /
    # results["pred_fold_results"]. One row per (arm, fold, class, phase),
    # then a fold-averaged summary printed for the write-up.
    fit_rows = []
    feature_cols = get_feature_columns(feat_df.drop(columns=["patient_id"]))
    murmur_lookup = load_murmur_lookup(CIRCOR_METADATA_PATH)
    for arm, fold_key in (("ground_truth", "gt_fold_results"), ("predicted", "pred_fold_results")):
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

            train_df = feat_df[(feat_df["boundary_source"] == arm) & (feat_df["recording_id"].isin(train_ids))].copy()
            train_df["murmur"] = train_df["patient_id"].apply(murmur_lookup)
            train_df = train_df.dropna(subset=feature_cols)
            train_df[feature_cols] = scaler.transform(train_df[feature_cols])

            for cls in CLASSES:
                class_rows = train_df[train_df["murmur"] == cls]
                for phase in PHASES:
                    cols = phase_columns(feature_cols, phase)
                    if not cols or phase not in gmms.get(cls, {}):
                        continue
                    gmm = gmms[cls][phase]
                    X = class_rows[cols].values
                    self_score = gmm.score(X) if len(X) else float("nan")
                    cov = gmm.covariances_  # shape (n_components, n_features) for diag
                    weights = gmm.weights_
                    fit_rows.append({
                        "boundary_source": arm,
                        "fold": fold_idx,
                        "class": cls,
                        "phase": phase,
                        "n_train_samples": len(X),
                        "feature_count": len(cols),
                        "component_weights": ",".join(f"{w:.4f}" for w in weights),
                        "min_weight": float(np.min(weights)),
                        "max_weight": float(np.max(weights)),
                        "mean_cov_diag": float(np.mean(cov)),
                        "min_cov_diag": float(np.min(cov)),
                        "max_cov_diag": float(np.max(cov)),
                        "self_score": float(self_score),
                    })

    fit_df = pd.DataFrame(fit_rows)
    fit_df.to_csv(GMM_FIT_OUT, index=False)
    print(f"Saved {len(fit_df)} rows to {GMM_FIT_OUT} (one per arm x fold x class x phase)")

    print("\nGMM fit diagnostics averaged over folds (mean of per-fold values):")
    summary = (fit_df.groupby(["boundary_source", "class", "phase"], sort=False)
               [["n_train_samples", "min_weight", "mean_cov_diag", "min_cov_diag", "max_cov_diag", "self_score"]]
               .mean().round(4))
    print(summary.to_string())


if __name__ == "__main__":
    main()