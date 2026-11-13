"""
experiments/arm_c_wholeclip_nn_shap.py

Arm C: the literature-standard post-hoc baseline. ONE discriminative NN
trained on all four phases' features at once (no phase silo, no additive
structure imposed), explained after the fact with model-agnostic SHAP
(KernelExplainer), and the per-feature SHAP values summed into per-phase
contributions so the result can sit in the same fidelity tables as Arm A
(phase-siloed GMM) and Arm B (phase-siloed NN).

Why this arm exists
-------------------
Arms A and B both make phase importance an identity of the model: the
decision is a sum of per-phase terms, so "which phase drove the decision"
is read off the model with no explainer. The paper's motivating claim is
that this is preferable to explaining an unconstrained model post hoc.
That claim needs the unconstrained-plus-post-hoc cell actually run, on the
same folds, features and fidelity metric, or it is an assertion. This
script is that cell.

What is held fixed against Arms A and B
---------------------------------------
- Same feature table, same 176 usable features, same training-fold
  StandardScaler (prepare_arm from Arm A).
- Same patient-level folds, loaded from Arm A's saved pickle (never
  re-derived), so every recording is scored out-of-fold exactly once and
  paired comparisons by recording_id are valid.
- Same boundary_source pair (ground_truth / predicted), always both.
- Same fidelity code path: build_phase_fidelity_table / compute_weight_share
  from Arm A, fed a per-recording dict with the same keys Arms A and B
  produce (phase_margins, dominant_phase, ...).
- Capacity: HIDDEN_LAYER_SIZES = (48,) = 4 x Arm B's 12 hidden units per
  phase, so the whole-clip model gets the same total hidden budget as the
  four silos combined. Everything else (Adam, early stopping, max_iter,
  seed) matches Arm B.

How SHAP becomes a per-phase margin
-----------------------------------
KernelExplainer is run on f(x) = logit P(Present | x) (the same scale as
Arm B's per-phase log-odds), with a background of SHAP_BACKGROUND_SIZE
training rows per fold. SHAP's efficiency property gives, per recording,

    f(x) - E_background[f]  =  sum_j phi_j          (exactly, up to sampling)

so summing phi_j over the features belonging to phase p yields a signed
per-phase term m_p with sum_p m_p = f(x) - E[f]. That is the exact
post-hoc analogue of the structural arms' additive decomposition, and it
is what goes into phase_margins / dominant_phase / weight-share (primary).

A second, more conventional aggregation is also stored: the absolute mass
sum_j |phi_j| per phase (columns share_abs_<phase>, dominant_phase_abs,
fidelity_match_abs). It is not additive to the decision and can't cancel
within a phase, so it tends to look more diffuse. Report the signed version
as primary (like-for-like with Arms A and B) and the absolute version as a
robustness check.

What is different, and must be said in the paper
------------------------------------------------
- Structural margins are relative to the OTHER CLASS (likelihood ratio /
  log-odds). SHAP margins are relative to a BACKGROUND EXPECTATION. Same
  additive shape, different reference point.
- SHAP is an estimate: it depends on the background sample and on the
  sampling budget. To quantify that, fold STABILITY_FOLD (ground_truth arm)
  is explained twice with two background seeds and the per-recording
  change in phase weight-share is written to WHOLECLIP_SHAP_STABILITY_CSV.
  Structural attribution has no such degree of freedom.
- SHAP has a cost. Wall-clock per explained recording is logged and saved
  in the pickle's config; the structural arms' attribution costs one
  forward pass.

Outputs:
  - wholeclip_nn_shap_comparison_results.pkl
  - wholeclip_nn_shap_per_recording_comparison.csv
  - phase_fidelity_per_recording_wholeclip.tsv  (read by
    experiments/compare_arms_fidelity.py for the three-way table)
  - wholeclip_shap_background_stability.csv
  - Console: per-fold accuracy, pooled accuracy + confusion, SHAP
    efficiency check, runtime, phase-attribution fidelity (signed and
    absolute aggregations), background-stability summary
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repo root: makes `paths` and the packages importable from any cwd
import paths


import pickle
import time

import numpy as np
import pandas as pd
import shap
from sklearn.neural_network import MLPClassifier

from experiments.arm_a_gmm import (
    CIRCOR_METADATA_PATH,
    CLASSES,
    FEATURE_TABLE_PATH,
    LABEL_COLUMN,
    PHASES,
    build_phase_fidelity_table,
    compute_hard_subset_fidelity,
    compute_majority_baseline,
    compute_phase_precision_recall,
    compute_weight_share,
    get_feature_columns,
    load_and_label_table,
    load_murmur_timing_lookup,
    phase_columns,
    prepare_arm,
    print_fidelity_summary,
    print_weight_share_table,
    print_win_rate_table,
)
from experiments.arm_b_nn_shap import (
    build_per_recording_comparison,
    load_arm_a_folds,
    pool_fold_results,
)

RESULTS_PATH = paths.WHOLECLIP_RESULTS_PKL
PER_RECORDING_CSV_PATH = paths.WHOLECLIP_PER_RECORDING_CSV
PHASE_FIDELITY_TSV_PATH = paths.WHOLECLIP_PHASE_FIDELITY_TSV
STABILITY_CSV_PATH = paths.WHOLECLIP_SHAP_STABILITY_CSV

RANDOM_STATE = 42
HIDDEN_LAYER_SIZES = (48,)   # 4 phases x Arm B's 12 units: same total hidden budget as the silos combined
MAX_ITER = 2000
EPS = 1e-6

SHAP_BACKGROUND_SIZE = 50    # training rows per fold used as KernelExplainer background (same as Arm B)
SHAP_NSAMPLES = "auto"       # KernelExplainer budget; "auto" = 2 * n_features + 2048 model evaluations
STABILITY_FOLD = 0           # fold whose ground_truth arm is re-explained with a second background
STABILITY_BACKGROUND_SEED = 43


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------

def fit_wholeclip_nn(train_df, feature_cols):
    X = train_df[feature_cols].values
    y = (train_df[LABEL_COLUMN].values == "Present").astype(int)
    clf = MLPClassifier(
        hidden_layer_sizes=HIDDEN_LAYER_SIZES,
        max_iter=MAX_ITER,
        random_state=RANDOM_STATE,
        early_stopping=True,
    )
    clf.fit(X, y)
    return clf


def make_logit_fn(clf):
    """f(x) = logit P(Present | x). Explaining the logit rather than the
    probability keeps SHAP on the same scale as Arm B's summed log-odds and
    avoids the saturation that squashes probability-scale attributions
    for confident predictions."""
    present_idx = clf.classes_.tolist().index(1)

    def f(X):
        p = np.clip(clf.predict_proba(np.asarray(X))[:, present_idx], EPS, 1 - EPS)
        return np.log(p / (1 - p))

    return f


# ---------------------------------------------------------------------
# SHAP -> per-phase decomposition
# ---------------------------------------------------------------------

def explain_rows(clf, train_df, X_test, feature_cols, background_seed):
    background = shap.sample(train_df[feature_cols], min(SHAP_BACKGROUND_SIZE, len(train_df)),
                             random_state=background_seed)
    explainer = shap.KernelExplainer(make_logit_fn(clf), background)
    t0 = time.time()
    shap_values = explainer.shap_values(X_test, nsamples=SHAP_NSAMPLES, silent=True)
    elapsed = time.time() - t0
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3:  # (n, n_features, n_outputs) on some shap versions
        shap_values = shap_values[:, :, 0]
    expected_value = float(np.ravel(explainer.expected_value)[0])
    return shap_values, expected_value, elapsed


def phase_index_map(feature_cols):
    return {phase: [i for i, c in enumerate(feature_cols) if c.startswith(f"{phase}_")] for phase in PHASES}


def decompose(shap_row, phase_idx):
    signed = {p: float(shap_row[idx].sum()) for p, idx in phase_idx.items()}
    absolute = {p: float(np.abs(shap_row[idx]).sum()) for p, idx in phase_idx.items()}
    return signed, absolute


# ---------------------------------------------------------------------
# One fold, one boundary source
# ---------------------------------------------------------------------

def run_arm(df, boundary_source, feature_cols, train_pids, test_pids, fold_idx):
    train_df, test_df, scaler = prepare_arm(df, boundary_source, feature_cols, train_pids, test_pids)
    clf = fit_wholeclip_nn(train_df, feature_cols)
    f = make_logit_fn(clf)

    X_test = test_df[feature_cols].values.astype(float)
    total_logit = f(X_test)
    predicted = np.where(total_logit >= 0, "Present", "Absent")

    shap_values, expected_value, elapsed = explain_rows(clf, train_df, X_test, feature_cols, RANDOM_STATE)
    phase_idx = phase_index_map(feature_cols)

    per_recording_results = []
    for i, (_, row) in enumerate(test_df.iterrows()):
        signed, absolute = decompose(shap_values[i], phase_idx)
        per_recording_results.append({
            "recording_id": row["recording_id"],
            "true_label": row[LABEL_COLUMN],
            "predicted_label": str(predicted[i]),
            "correct": str(predicted[i]) == row[LABEL_COLUMN],
            "total_logit": float(total_logit[i]),
            "shap_expected_value": expected_value,
            "shap_sum": float(shap_values[i].sum()),
            # signed per-phase SHAP sums: additive, sum_p == total_logit - expected_value
            "phase_margins": signed,
            "dominant_phase": max(signed, key=lambda p: abs(signed[p])),
            # absolute per-phase SHAP mass: conventional importance, not additive
            "phase_abs_mass": absolute,
            "dominant_phase_abs": max(absolute, key=lambda p: absolute[p]),
        })

    # Efficiency check: SHAP values must reconstruct the logit up to sampling error.
    recon_err = np.abs(shap_values.sum(axis=1) + expected_value - total_logit)
    accuracy = float(np.mean([r["correct"] for r in per_recording_results]))
    print(f"  [{boundary_source}] fold accuracy: {accuracy * 100:.1f}%  (n={len(per_recording_results)})  "
          f"SHAP {elapsed / len(X_test):.2f} s/recording, max reconstruction error {recon_err.max():.4f}")

    stability_df = None
    if fold_idx == STABILITY_FOLD and boundary_source == "ground_truth":
        alt_values, alt_expected, alt_elapsed = explain_rows(
            clf, train_df, X_test, feature_cols, STABILITY_BACKGROUND_SEED)
        rows = []
        for i, r in enumerate(per_recording_results):
            alt_signed, _ = decompose(alt_values[i], phase_idx)
            share_a = compute_weight_share(r["phase_margins"])
            share_b = compute_weight_share(alt_signed)
            row = {
                "recording_id": r["recording_id"],
                "boundary_source": boundary_source,
                "fold": fold_idx,
                "true_label": r["true_label"],
                "dominant_seed42": r["dominant_phase"],
                "dominant_seed43": max(alt_signed, key=lambda p: abs(alt_signed[p])),
                "expected_value_seed42": expected_value,
                "expected_value_seed43": alt_expected,
            }
            for p in PHASES:
                row[f"share_{p}_seed42"] = share_a[p]
                row[f"share_{p}_seed43"] = share_b[p]
                row[f"share_{p}_absdiff"] = abs(share_a[p] - share_b[p])
            rows.append(row)
        stability_df = pd.DataFrame(rows)

    per_class_accuracy = {}
    for cls in CLASSES:
        subset = [r for r in per_recording_results if r["true_label"] == cls]
        per_class_accuracy[cls] = np.mean([r["correct"] for r in subset]) if subset else float("nan")
    confusion = pd.crosstab(
        pd.Series([r["true_label"] for r in per_recording_results], name="true"),
        pd.Series([r["predicted_label"] for r in per_recording_results], name="predicted"),
    )

    return {
        "boundary_source": boundary_source,
        "fold": fold_idx,
        "per_recording_results": per_recording_results,
        "accuracy": accuracy,
        "per_class_accuracy": per_class_accuracy,
        "confusion_matrix": confusion,
        "clf": clf,
        "scaler": scaler,
        "shap_expected_value": expected_value,
        "shap_seconds_per_recording": elapsed / len(X_test),
        "max_reconstruction_error": float(recon_err.max()),
        "train_recording_ids": train_df["recording_id"].tolist(),
        "test_recording_ids": test_df["recording_id"].tolist(),
        "stability_df": stability_df,
    }


# ---------------------------------------------------------------------
# Reporting helpers specific to the absolute-mass aggregation
# ---------------------------------------------------------------------

def add_abs_mass_columns(fidelity_df, result, timing_lookup):
    """Adds share_abs_<phase>, dominant_phase_abs, fidelity_match_abs to the
    fidelity table produced by Arm A's build_phase_fidelity_table (which
    only knows about phase_margins)."""
    by_id = {r["recording_id"]: r for r in result["per_recording_results"]}
    out = fidelity_df.copy()
    for phase in PHASES:
        out[f"share_abs_{phase}"] = [
            compute_weight_share(by_id[rid]["phase_abs_mass"])[phase] for rid in out["recording_id"]
        ]
    out["dominant_phase_abs"] = [by_id[rid]["dominant_phase_abs"] for rid in out["recording_id"]]
    out["fidelity_match_abs"] = [
        (dom in exp) if exp else None
        for dom, exp in zip(out["dominant_phase_abs"],
                            [set(e.split(",")) if e else set() for e in out["expected_phases"]])
    ]
    return out


def print_abs_mass_summary(fdf, label):
    scoped = fdf[fdf["has_expected_phase"]]
    if len(scoped) == 0:
        return
    match = scoped["fidelity_match_abs"].astype(bool).mean()
    shares = "  ".join(f"{p}={scoped[f'share_abs_{p}'].mean() * 100:.1f}%" for p in PHASES)
    print(f"  [{label}] absolute-mass aggregation (sum |phi| per phase), n={len(scoped)}: "
          f"dominant matches expected {match * 100:.1f}%; mean share {shares}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    print(f"Loading {FEATURE_TABLE_PATH}")
    df = load_and_label_table()
    feature_cols = get_feature_columns(df)
    print(f"Using {len(feature_cols)} feature columns "
          f"({', '.join(f'{p}: {len(phase_columns(feature_cols, p))}' for p in PHASES)})")

    folds = load_arm_a_folds()
    print(f"Reused Arm A folds: {len(folds)} folds")

    print("\n" + "=" * 60)
    print(f"FITTING WHOLE-CLIP NN + KERNEL SHAP ACROSS {len(folds)} FOLDS "
          f"(hidden={HIDDEN_LAYER_SIZES}, background={SHAP_BACKGROUND_SIZE}, nsamples={SHAP_NSAMPLES})")
    print("=" * 60)
    gt_fold_results, pred_fold_results = [], []
    for i, (train_pids, test_pids) in enumerate(folds):
        print(f"\n--- Fold {i + 1}/{len(folds)} ---")
        gt_fold_results.append(run_arm(df, "ground_truth", feature_cols, train_pids, test_pids, i))
        pred_fold_results.append(run_arm(df, "predicted", feature_cols, train_pids, test_pids, i))

    gt_result = pool_fold_results(gt_fold_results)
    pred_result = pool_fold_results(pred_fold_results)

    print("\n" + "=" * 60)
    print("POOLED OUT-OF-FOLD COMPARISON")
    print("=" * 60)
    for label, res in (("ground_truth", gt_result), ("predicted", pred_result)):
        print(f"{label:>12} accuracy: {res['accuracy'] * 100:.1f}%  (n={len(res['per_recording_results'])})  "
              f"per-class: " + ", ".join(f"{c} {a * 100:.1f}%" for c, a in res["per_class_accuracy"].items()))
    print(f"delta (predicted - ground_truth): {(pred_result['accuracy'] - gt_result['accuracy']) * 100:+.1f} points")
    print(f"\nground_truth confusion matrix:\n{gt_result['confusion_matrix']}")
    print(f"\npredicted confusion matrix:\n{pred_result['confusion_matrix']}")

    all_folds = gt_fold_results + pred_fold_results
    sec = np.mean([fr["shap_seconds_per_recording"] for fr in all_folds])
    err = max(fr["max_reconstruction_error"] for fr in all_folds)
    n_explained = sum(len(fr["per_recording_results"]) for fr in all_folds)
    print(f"\nSHAP cost: {sec:.2f} s per recording ({n_explained} recordings explained, "
          f"{sec * n_explained / 60:.1f} min total); max |sum(phi) + E[f] - f(x)| = {err:.4f}")

    comparison_df = build_per_recording_comparison(gt_result, pred_result)
    print(f"\nGT-vs-predicted prediction agreement rate: {comparison_df['gt_pred_agree'].mean() * 100:.1f}% "
          f"({comparison_df['gt_pred_agree'].sum()}/{len(comparison_df)})")
    comparison_df.to_csv(PER_RECORDING_CSV_PATH, index=False)
    print(f"Saved per-recording comparison to {PER_RECORDING_CSV_PATH}")

    print("\n" + "=" * 60)
    print("PHASE-ATTRIBUTION FIDELITY (pooled out-of-fold; signed per-phase SHAP sums = primary)")
    print("=" * 60)
    timing_lookup = load_murmur_timing_lookup(CIRCOR_METADATA_PATH)
    gt_fidelity_df = add_abs_mass_columns(
        build_phase_fidelity_table(gt_result, "ground_truth", timing_lookup), gt_result, timing_lookup)
    pred_fidelity_df = add_abs_mass_columns(
        build_phase_fidelity_table(pred_result, "predicted", timing_lookup), pred_result, timing_lookup)
    print_fidelity_summary(gt_fidelity_df, "ground_truth")
    print_fidelity_summary(pred_fidelity_df, "predicted")

    print("\n--- Absolute-mass aggregation (robustness; not additive to the decision) ---")
    print_abs_mass_summary(gt_fidelity_df, "ground_truth")
    print_abs_mass_summary(pred_fidelity_df, "predicted")

    fidelity_df = pd.concat([gt_fidelity_df, pred_fidelity_df], ignore_index=True)
    fidelity_df.to_csv(PHASE_FIDELITY_TSV_PATH, sep="\t", index=False)
    print(f"\nSaved per-recording phase fidelity to {PHASE_FIDELITY_TSV_PATH}")

    print("\n--- Discrete: dominant-phase win-rate (GT vs. predicted x phase) ---")
    win_rate_table = print_win_rate_table(gt_fidelity_df, pred_fidelity_df)
    print("\n--- Continuous: mean |margin|-normalized weight-share (GT vs. predicted x phase) ---")
    weight_share_table = print_weight_share_table(gt_fidelity_df, pred_fidelity_df)

    fidelity_summary = {}
    for boundary_source, fdf in (("ground_truth", gt_fidelity_df), ("predicted", pred_fidelity_df)):
        scoped = fdf[fdf["has_expected_phase"]]
        majority_phase, baseline_rate = compute_majority_baseline(fdf)
        hard_df, hard_rate = compute_hard_subset_fidelity(fdf, majority_phase)
        fidelity_summary[boundary_source] = {
            "n_scoped": len(scoped),
            "raw_fidelity": scoped["fidelity_match"].mean() if len(scoped) > 0 else float("nan"),
            "raw_fidelity_abs_mass": scoped["fidelity_match_abs"].astype(bool).mean() if len(scoped) > 0 else float("nan"),
            "majority_phase": majority_phase,
            "majority_baseline": baseline_rate,
            "hard_subset_n": len(hard_df),
            "hard_subset_fidelity": hard_rate,
            "precision_recall": compute_phase_precision_recall(fdf),
        }

    print("\n" + "=" * 60)
    print(f"SHAP BACKGROUND STABILITY (fold {STABILITY_FOLD + 1}, ground_truth arm, "
          f"background seeds {RANDOM_STATE} vs {STABILITY_BACKGROUND_SEED})")
    print("=" * 60)
    stability_df = next((fr["stability_df"] for fr in gt_fold_results if fr["stability_df"] is not None), None)
    if stability_df is not None:
        agree = (stability_df["dominant_seed42"] == stability_df["dominant_seed43"]).mean()
        print(f"  n={len(stability_df)} recordings; dominant phase unchanged in {agree * 100:.1f}%")
        for p in PHASES:
            d = stability_df[f"share_{p}_absdiff"]
            print(f"  {p:>9}: mean share {stability_df[f'share_{p}_seed42'].mean() * 100:.1f}% vs "
                  f"{stability_df[f'share_{p}_seed43'].mean() * 100:.1f}%; per-recording |diff| "
                  f"mean {d.mean() * 100:.1f} pts, 90th pct {d.quantile(0.9) * 100:.1f} pts")
        stability_df.to_csv(STABILITY_CSV_PATH, index=False)
        print(f"  Saved to {STABILITY_CSV_PATH}")
    else:
        print("  (no stability run recorded)")

    output = {
        "ground_truth": gt_result,
        "predicted": pred_result,
        "gt_fold_results": [{k: v for k, v in fr.items() if k != "stability_df"} for fr in gt_fold_results],
        "pred_fold_results": [{k: v for k, v in fr.items() if k != "stability_df"} for fr in pred_fold_results],
        "feature_cols": feature_cols,
        "folds": [(list(tr), list(te)) for tr, te in folds],
        "fidelity_summary": fidelity_summary,
        "win_rate_table": win_rate_table,
        "weight_share_table": weight_share_table,
        "config": {
            "label_column": LABEL_COLUMN,
            "hidden_layer_sizes": HIDDEN_LAYER_SIZES,
            "max_iter": MAX_ITER,
            "random_state": RANDOM_STATE,
            "n_folds": len(folds),
            "shap_background_size": SHAP_BACKGROUND_SIZE,
            "shap_nsamples": SHAP_NSAMPLES,
            "shap_seconds_per_recording": float(sec),
            "shap_max_reconstruction_error": float(err),
            "stability_fold": STABILITY_FOLD,
            "stability_background_seed": STABILITY_BACKGROUND_SEED,
            "classes": CLASSES,
            "phases": PHASES,
        },
    }
    with open(RESULTS_PATH, "wb") as f:
        pickle.dump(output, f)
    print(f"Saved full results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
