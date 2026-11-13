"""
analysis/margin_scaling_sensitivity.py

Does the GMM's phase attribution depend on how each phase's margin is
scaled?

Why
---
Camera-ready, reviewer 1, major 3. Section V says "rescaling margins by
training-fold standard deviation or MAD lowered agreement, so margins are
reported unscaled". The reviewer asks whether unscaled margins were
prespecified and suggests reporting both. Two things are checked here
instead of taken from the pre-submission log:

  1. What was computed before submission
     (analysis/compute_training_calibrated_margins.py) was STANDARDISATION,
     (m - train mean) / train SD and (m - train median) / train MAD, not
     the "rescaling" the paper describes. Subtracting a centre moves each
     phase's zero point, so a margin of 0 no longer means "no evidence
     either way" and |standardised margin| measures distance from a typical
     (mostly Absent) training recording rather than evidence for or
     against Present. Pure rescaling, m / scale, keeps zero and sign and
     only changes how much each phase counts, which is the question the
     paper's sentence raises (per-phase mixtures with different covariance
     scales).
  2. Whether the training SD is a fair scale for a phase or is set by a
     few extreme margins. The pre-submission log asserted the latter
     without showing it.

What
----
GMM only, because the sentence is about the GMM. Per boundary source, fold
and phase, the scale comes from that fold's TRAINING margins: the stored
fold GMMs re-score their own training recordings, and each test recording
is scaled with the fold in which it was held out. This is the same leak-free
construction as compute_training_calibrated_margins.py, whose functions are
reused. Variants:
  raw                        m
  rescaled by SD             m / SD
  rescaled by MAD            m / MAD
  standardised by mean, SD   (m - mean) / SD      (run before submission)
  standardised by median, MAD (m - median) / MAD  (run before submission)
Reported for each variant:
  - mean systole weight-share and agreement on the 616 Present recordings
    with an annotated phase (the paper's primary and discrete metrics);
  - mean systole share on the 2385 Absent recordings (Section V's 36.3%).
Checks: the raw row must reproduce Table I, and the standardised rows must
reproduce the pre-submission log.

It also prints each phase's training-margin distribution (quantiles, SD,
MAD, and SD with the most extreme 1% trimmed) so the SD-versus-MAD question
is answered from the data.

Part 2: all three classifiers. Reviewer 2 (limitation 3) asks whether the
between-classifier comparisons survive a normalisation. The per-phase MLP
and MLP+SHAP fold models were not stored, so their training-fold scales
cannot be rebuilt without retraining. Part 2 therefore uses one rule for
all three: divide each phase's margins by that phase's MAD over all 3001
out-of-fold recordings. The rule uses no labels and needs no retraining.
For the GMM it is shown next to the training-fold MAD of part 1, so the
substitution is visible. It reports per-phase shares (is systole still
the largest?), agreement, and the paired systole-share differences
between classifiers with patient-clustered 95% CIs, the same procedure as
analysis/patient_clustered_paired_tests.py.

Part 3: the paper's central expert-to-predicted boundary shift, re-measured
under ONE common per-phase scale. Rescaling each boundary source by its
own MAD would partly divide away the change being measured, so a single
MAD, taken from one boundary source, is applied to both.

Nothing is retrained.
Outputs: paths.MARGIN_SCALING_SENSITIVITY_CSV, paths.MARGIN_SCALE_DIAGNOSTICS_CSV,
paths.MARGIN_SCALING_CROSS_CLASSIFIER_CSV.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repo root: makes `paths` and the packages importable from any cwd
import paths

import pickle

import numpy as np
import pandas as pd

from experiments.arm_a_gmm import RESULTS_PATH
from experiments.compare_arms_fidelity import load_model_fidelity
from analysis.compute_training_calibrated_margins import (
    compute_training_margins,
    summarize_training_scale,
    test_fold_lookup,
)
from analysis.patient_clustered_paired_tests import patient_clustered, patient_ids

PHASES = ("S1", "Systole", "S2", "Diastole")
SYS = PHASES.index("Systole")
CLASSIFIER_TSVS = {
    "GMM": paths.GMM_PHASE_FIDELITY_TSV,
    "per-phase MLP": paths.NN_PHASE_FIDELITY_TSV,
    "MLP+SHAP": paths.WHOLECLIP_PHASE_FIDELITY_TSV,
}
PAIRS = (("GMM", "per-phase MLP"), ("MLP+SHAP", "GMM"), ("MLP+SHAP", "per-phase MLP"))


def training_scale_diagnostics(train_margins):
    rows = []
    for (arm, phase), g in train_margins.groupby(["boundary_source", "phase"]):
        v = g["margin"].to_numpy()
        med = np.median(v)
        lo, hi = np.percentile(v, [0.5, 99.5])
        rows.append(dict(
            boundary=arm, phase=phase, n_pooled_over_folds=len(v),
            min=v.min(), p1=np.percentile(v, 1), p25=np.percentile(v, 25), median=med,
            p75=np.percentile(v, 75), p99=np.percentile(v, 99), max=v.max(),
            sd=v.std(), sd_trim_1pct=v[(v >= lo) & (v <= hi)].std(),
            mad_scaled=1.4826 * np.median(np.abs(v - med)),
            n_abs_over_100=int((np.abs(v) > 100).sum()), n_abs_over_1000=int((np.abs(v) > 1000).sum()),
        ))
    return pd.DataFrame(rows)


def fold_scales(sub, arm, scale_df, fold_lookup):
    """(n x 4) training mean / SD / median / MAD of the fold that held each recording out."""
    folds = [fold_lookup[(arm, rid)] for rid in sub["recording_id"]]
    s = scale_df[scale_df["boundary_source"] == arm].set_index(["fold", "phase"])

    def get(col):
        return np.array([[s.loc[(f, p), col] for p in PHASES] for f in folds])

    return dict(mean=get("train_margin_mean"), sd=get("train_margin_std"),
                median=get("train_margin_median"), mad=get("train_margin_mad_scaled"))


def variants(m, sc):
    return {
        "raw": m,
        "rescaled by SD": m / sc["sd"],
        "rescaled by MAD": m / sc["mad"],
        "standardised by mean, SD (pre-submission)": (m - sc["mean"]) / sc["sd"],
        "standardised by median, MAD (pre-submission)": (m - sc["median"]) / sc["mad"],
    }


def systole_share_and_agreement(values, expected):
    a = np.abs(values)
    systole_share = a[:, SYS] / a.sum(axis=1)
    dominant = np.array(PHASES)[a.argmax(axis=1)]
    agree = None if expected is None else np.array([d in e for d, e in zip(dominant, expected)])
    return systole_share, agree


def out_of_fold_mad(df):
    """Per-phase MAD of the out-of-fold margins of ALL recordings (no labels used)."""
    m = df[[f"margin_{p}" for p in PHASES]].to_numpy(dtype=float)
    return 1.4826 * np.median(np.abs(m - np.median(m, axis=0)), axis=0)


def cross_classifier_rows():
    """Part 2: every classifier rescaled by its own out-of-fold per-phase MAD."""
    data = {name: dict(zip(("ground_truth", "predicted"), load_model_fidelity(path)))
            for name, path in CLASSIFIER_TSVS.items()}
    summary, pairs = [], []
    for arm in ("ground_truth", "predicted"):
        present_shares = {}
        for name in CLASSIFIER_TSVS:
            df = data[name][arm]
            mad = out_of_fold_mad(df)
            present = df[df["has_expected_phase"]].reset_index(drop=True)
            m = present[[f"margin_{p}" for p in PHASES]].to_numpy(dtype=float)
            expected = present["expected_phases"].fillna("").astype(str).str.split(",").tolist()
            for variant, v in (("raw", m), ("rescaled by out-of-fold MAD", m / mad)):
                a = np.abs(v)
                shares = a / a.sum(axis=1, keepdims=True)
                dominant = np.array(PHASES)[a.argmax(axis=1)]
                agree = np.array([d in e for d, e in zip(dominant, expected)])
                row = dict(boundary=arm, classifier=name, variant=variant, n=len(present),
                           agreement_pct=100 * agree.mean())
                row.update({f"share_{p}_pct": 100 * shares[:, i].mean() for i, p in enumerate(PHASES)})
                row.update({f"mad_{p}": mad[i] for i, p in enumerate(PHASES)})
                summary.append(row)
                present_shares[(name, variant)] = pd.Series(shares[:, SYS], index=present["recording_id"])
        for variant in ("raw", "rescaled by out-of-fold MAD"):
            for a_name, b_name in PAIRS:
                sa, sb = present_shares[(a_name, variant)], present_shares[(b_name, variant)]
                common = sa.index.intersection(sb.index)
                diffs = (sa[common] - sb[common]).to_numpy()
                mean, lo, hi, p, k = patient_clustered(diffs, patient_ids(common))
                pairs.append(dict(boundary=arm, variant=variant, comparison=f"{a_name} - {b_name}",
                                  n=len(common), n_patients=k, mean_diff_pts=100 * mean,
                                  patient_ci_lo=100 * lo, patient_ci_hi=100 * hi, patient_p=p))
    return pd.DataFrame(summary), pd.DataFrame(pairs)


def boundary_shift_rows():
    """Part 3: the paper's expert -> predicted boundary shift under ONE common
    per-phase scale. Rescaling each boundary source by its own MAD would
    partly divide away the very change being measured (predicted boundaries
    enlarge some phases' margins), so the same MAD -- taken from one
    boundary source -- is applied to both."""
    rows = []
    for name, path in CLASSIFIER_TSVS.items():
        dfs = dict(zip(("ground_truth", "predicted"), load_model_fidelity(path)))
        for scale_source in ("raw", "ground_truth", "predicted"):
            mad = np.ones(len(PHASES)) if scale_source == "raw" else out_of_fold_mad(dfs[scale_source])
            row = dict(classifier=name, scale="none" if scale_source == "raw" else f"MAD from {scale_source}")
            for arm, df in dfs.items():
                present = df[df["has_expected_phase"]]
                a = np.abs(present[[f"margin_{p}" for p in PHASES]].to_numpy(dtype=float) / mad)
                expected = present["expected_phases"].fillna("").astype(str).str.split(",").tolist()
                dominant = np.array(PHASES)[a.argmax(axis=1)]
                row[f"systole_share_{arm}_pct"] = 100 * (a[:, SYS] / a.sum(axis=1)).mean()
                row[f"agreement_{arm}_pct"] = 100 * np.mean([d in e for d, e in zip(dominant, expected)])
            row["systole_share_change_pts"] = row["systole_share_predicted_pct"] - row["systole_share_ground_truth_pct"]
            row["agreement_change_pts"] = row["agreement_predicted_pct"] - row["agreement_ground_truth_pct"]
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    with open(RESULTS_PATH, "rb") as f:
        results = pickle.load(f)
    train_margins = compute_training_margins(results)
    scale_df = summarize_training_scale(train_margins)
    fold_lookup = test_fold_lookup(results)

    diagnostics = training_scale_diagnostics(train_margins)
    diagnostics.to_csv(paths.MARGIN_SCALE_DIAGNOSTICS_CSV, index=False)

    rows = []
    fidelity = dict(zip(("ground_truth", "predicted"), load_model_fidelity(paths.GMM_PHASE_FIDELITY_TSV)))
    for arm, df in fidelity.items():
        for subset, sub in (("Present", df[df["has_expected_phase"]]), ("Absent", df[df["true_label"] == "Absent"])):
            m = sub[[f"margin_{p}" for p in PHASES]].to_numpy(dtype=float)
            expected = (sub["expected_phases"].fillna("").astype(str).str.split(",").tolist()
                        if subset == "Present" else None)
            sc = fold_scales(sub, arm, scale_df, fold_lookup)
            for name, v in variants(m, sc).items():
                share, agree = systole_share_and_agreement(v, expected)
                rows.append(dict(boundary=arm, subset=subset, variant=name, n=len(sub),
                                 systole_share_pct=100 * share.mean(),
                                 agreement_pct=np.nan if agree is None else 100 * agree.mean()))
            sys_abs = np.abs(m[:, SYS])
            rows.append(dict(boundary=arm, subset=subset, variant="(count: test |m_Systole| > 1000)",
                             n=len(sub), systole_share_pct=np.nan, agreement_pct=float((sys_abs > 1000).sum())))
    out = pd.DataFrame(rows)
    out.to_csv(paths.MARGIN_SCALING_SENSITIVITY_CSV, index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print("Training-margin distribution per phase (pooled over the 5 folds' training sets)\n")
    print(diagnostics.round(2).to_string(index=False))
    print("\nFold-averaged scales actually used (SD vs MAD x 1.4826):")
    print(scale_df.groupby(["boundary_source", "phase"])[["train_margin_std", "train_margin_mad_scaled"]]
          .mean().round(2).to_string())
    print("\nGMM attribution under each scaling (Present: n=616; Absent: n=2385)\n")
    print(out.round(1).to_string(index=False))
    summary, pairs = cross_classifier_rows()
    print("\nPart 2 -- all three classifiers, raw vs rescaled by out-of-fold per-phase MAD (n=616 Present)\n")
    cols = ["boundary", "classifier", "variant"] + [f"share_{p}_pct" for p in PHASES] + ["agreement_pct"]
    print(summary[cols].round(1).to_string(index=False))
    print("\nOut-of-fold per-phase MAD used:")
    print(summary[summary["variant"] == "raw"][["boundary", "classifier"] + [f"mad_{p}" for p in PHASES]]
          .round(2).to_string(index=False))
    print("\nPaired systole-share differences, patient-clustered 95% CI\n")
    print(pairs.round(4).to_string(index=False))
    shift = boundary_shift_rows()
    print("\nPart 3 -- expert -> predicted boundary shift with ONE common per-phase scale (n=616 Present)\n")
    print(shift.round(1).to_string(index=False))
    pd.concat([summary.assign(table="per-classifier"), pairs.assign(table="paired systole share"),
               shift.assign(table="boundary shift, common scale")],
              ignore_index=True).to_csv(paths.MARGIN_SCALING_CROSS_CLASSIFIER_CSV, index=False)
    print(f"\nWrote {paths.MARGIN_SCALING_SENSITIVITY_CSV}\nWrote {paths.MARGIN_SCALE_DIAGNOSTICS_CSV}"
          f"\nWrote {paths.MARGIN_SCALING_CROSS_CLASSIFIER_CSV}")


if __name__ == "__main__":
    main()
