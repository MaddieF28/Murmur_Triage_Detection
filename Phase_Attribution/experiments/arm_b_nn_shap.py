"""
experiments/arm_b_nn_shap.py

Arm B (discriminative NN + SHAP) for the CirCor Murmur Present-vs-Absent
binary task -- the direct counterpart to experiments/arm_a_gmm.py's
Arm A (generative GMM). Same feature table, same patient-level K-FOLD
split (reused from Arm A's saved results, not re-derived -- see below),
same boundary_source comparison (ground_truth vs. predicted phase
boundaries), same phase-attribution-fidelity evaluation shape.

Why reuse Arm A's folds instead of re-splitting (UPDATED for k-fold)
------------------------------------------------------------------------
Arm A now saves the full list of (train_pids, test_pids) fold splits
into gmm_murmur_comparison_results.pkl (key: "folds") instead of a
single train/test split. Loading that list directly -- rather than
re-deriving folds with the same RANDOM_STATE and N_FOLDS -- removes the
assumption that two separate files' fold logic stays in sync forever.
If Arm A's fold count, stratification, or random_state ever changes,
Arm B follows automatically instead of silently diverging. This is what
makes the GT-vs-predicted-arm delta and the generative-vs-discriminative
delta comparisons actually mean what they claim to mean.

Per-phase discriminative NNs + log-odds combination
------------------------------------------------------
One small MLPClassifier per phase (S1/Systole/S2/Diastole), trained
only on that phase's feature columns -> P(Present | phase). Phases are
combined by SUMMING LOG-ODDS across phases (not averaging
probabilities). This is the discriminative analog of Arm A's summed
per-phase log-likelihoods:

    Arm A: total_ll[cls]     = sum_phase  loglik(phase | cls)
    Arm B: total_logit       = logit(pi) + sum_phase [ logit(P(Present | phase)) - logit(pi) ]

where pi is the training-fold Present prior. Both are naive combinations
that assume phase-conditional independence given the class -- same
assumption, same shape, so the comparison isn't confounded by one arm
combining phases in a fundamentally different way than the other.

PRIOR CORRECTION (2026-09-07; changes the numbers, see below)
------------------------------------------------------
The earlier version summed the raw per-phase log-odds:
total_logit = sum_phase logit(P(Present | phase)). That is wrong under
the stated independence assumption. Bayes' rule per phase gives

    logit P(Present | x_p) = log [p(x_p|Present) / p(x_p|Absent)] + logit(pi)

so each silo's log-odds carries the class prior, and summing four silos
counts logit(pi) FOUR times. With pi ~ 0.205 (logit ~ -1.35) that put a
constant -5.4 on every decision, which (i) biased the classifier toward
Absent (Present recall 29%, specificity 99%) and (ii) added a common
|1.35| offset to every phase's margin, flattening weight-share toward
uniform regardless of evidence. Arm A's margin is a log-likelihood ratio
with no prior term, so the two arms were not on the same footing.

The fix: per-phase margin m_p = logit P(Present | x_p) - logit(pi_fold),
which IS the per-phase log-likelihood ratio implied by the silo, and the
decision total_logit = logit(pi_fold) + sum_p m_p. The prior is counted
once. m_p now sums exactly to total_logit - logit(pi), the same shape as
Arm A's margins summing to LL(Present) - LL(Absent). AUC is unchanged by
this (constant shift); accuracy, recall, weight-share and fidelity all
change. Before/after (pooled 5-fold, ground_truth / predicted):
accuracy 84.7 / 85.3% -> ~75.5 / 78.1%; Present recall 29.4 / 32.8% ->
~62.7 / 63.6%; Systole weight-share 30.1 / 28.0% -> ~36.7 / 32.8%;
argmax fidelity 36.7 / 33.4% -> ~51.3 / 43.3% (from the offline check in
results/analysis/nn_prior_corrected_margins.tsv; the numbers this script
prints are authoritative).

Because the combination is additive, each phase's evidence term m_p IS
its contribution to the final decision (mirrors Arm A's phase_margins).
So dominant_phase = argmax(|m_p|) here means exactly what it means in
Arm A: the phase that actually moved the decision, not just the phase
the model "fits" best.

K-fold (NEW)
------------------------------------------------------
Same rationale as Arm A: every patient's recordings are held out and
scored exactly once across the N_FOLDS folds, pooled before computing
accuracy / phase-fidelity / weight-share, instead of one ~20% test
slice. This is what grows the known-timing fidelity subset (previously
stuck around n=122 / n=8 hard subset) toward the dataset's full count.
Retrains N_FOLDS separate sets of per-phase MLPClassifiers -- accepted
cost, same as Arm A refitting its GMMs per fold.

SHAP (Arm B only, per the project doc)
------------------------------------------------
Model-agnostic shap.KernelExplainer on each phase's
MLPClassifier.predict_proba. Run on a SAMPLE of correctly-classified
test recordings PER FOLD (not all of them, and not just one fold) --
KernelExplainer is slow and the doc is explicit this is a qualitative
sanity check ("are the SHAP-identified features ones with known
acoustic relevance"), not a formal metric folded into the fidelity
number. N_SHAP_SAMPLES and SHAP_BACKGROUND_SIZE below control the
speed/coverage tradeoff per fold.

Outputs:
  - nn_murmur_comparison_results.pkl
  - nn_murmur_per_recording_comparison.csv
  - phase_fidelity_per_recording_nn.tsv
  - shap_top_features_sample.tsv
  - Console summary: per-fold accuracy, pooled accuracy, per-class
    accuracy, per-phase reliance, GT-vs-predicted agreement rate,
    phase-attribution fidelity, SHAP top-feature sample (all computed
    over pooled out-of-fold predictions except SHAP, which is sampled
    per fold and concatenated)
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repo root: makes `paths` and the packages importable from any cwd
import paths


from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import shap
from sklearn.neural_network import MLPClassifier

from experiments.arm_a_gmm import (
    CIRCOR_METADATA_PATH,
    CLASSES,
    FEATURE_TABLE_PATH,
    LABEL_COLUMN,
    N_FOLDS,
    PHASES,
    RESULTS_PATH as GMM_RESULTS_PATH,
    build_phase_fidelity_table,
    compute_hard_subset_fidelity,
    compute_majority_baseline,
    compute_phase_precision_recall,
    compute_weight_share,
    get_patient_id,
    load_and_label_table,
    load_murmur_timing_lookup,
    get_feature_columns,
    phase_columns,
    prepare_arm,
    print_fidelity_summary,
    print_weight_share_table,
    print_win_rate_table,
)

RESULTS_PATH = paths.NN_RESULTS_PKL
PER_RECORDING_CSV_PATH = paths.NN_PER_RECORDING_CSV
PHASE_FIDELITY_TSV_PATH = paths.NN_PHASE_FIDELITY_TSV
SHAP_SAMPLE_TSV_PATH = paths.NN_SHAP_SAMPLE_TSV

RANDOM_STATE = 42
HIDDEN_LAYER_SIZES = (12,)
MAX_ITER = 2000
N_SHAP_SAMPLES = 15          # correctly-classified test recordings per fold per arm to explain
SHAP_BACKGROUND_SIZE = 50    # training rows used as SHAP background per phase
SHAP_NSAMPLES = 100          # KernelExplainer's internal sampling budget (speed knob)
TOP_K_FEATURES = 5
EPS = 1e-6


# ---------------------------------------------------------------------
# Reused folds (from Arm A's saved results, not re-derived) -- UPDATED
# ---------------------------------------------------------------------

def load_arm_a_folds(gmm_results_path=GMM_RESULTS_PATH):
    """Loads the list of (train_pids, test_pids) fold splits Arm A saved,
    so Arm B is evaluated on exactly the same folds. Raises with a clear
    message if Arm A's pickle predates the k-fold conversion (old
    train_patient_ids/test_patient_ids-only format), since that shape is
    no longer compatible with this script.
    """
    if not gmm_results_path.exists():
        raise FileNotFoundError(
            f"Arm A results not found at {gmm_results_path}. Run experiments/arm_a_gmm.py "
            f"first -- Arm B reuses its patient-level folds so the two arms are directly comparable."
        )
    with open(gmm_results_path, "rb") as f:
        gmm_output = pickle.load(f)
    if "folds" not in gmm_output:
        raise ValueError(
            f"{gmm_results_path} has no 'folds' key -- it looks like it was saved by an older, "
            f"pre-k-fold version of experiments/arm_a_gmm.py. Re-run that script (current "
            f"version) to regenerate it before running this script."
        )
    return [(set(train_pids), set(test_pids)) for train_pids, test_pids in gmm_output["folds"]]


# ---------------------------------------------------------------------
# Fitting / classification
# ---------------------------------------------------------------------

def fit_all_nns(train_df, feature_cols):
    """Fits on integer-encoded labels (Present=1, Absent=0), not the
    raw strings. sklearn's MLPClassifier with early_stopping=True runs
    an internal validation-score check that, on some sklearn versions,
    breaks on string class labels (np.isnan() on a string array ->
    TypeError). Integer labels sidestep it entirely; classify_recording
    below maps back via clf.classes_.
    """
    nns = {}
    for phase in PHASES:
        cols = phase_columns(feature_cols, phase)
        X = train_df[cols].values
        y = (train_df[LABEL_COLUMN].values == "Present").astype(int)
        clf = MLPClassifier(
            hidden_layer_sizes=HIDDEN_LAYER_SIZES,
            max_iter=MAX_ITER,
            random_state=RANDOM_STATE,
            early_stopping=True,
        )
        clf.fit(X, y)
        nns[phase] = clf
    return nns


def training_prior_logit(train_df):
    """logit of the Present prior on this fold's training rows. Subtracted
    from every silo's log-odds and added back once in classify_recording,
    so the prior is counted once, not once per phase (see module docstring,
    PRIOR CORRECTION)."""
    pi = float((train_df[LABEL_COLUMN].values == "Present").mean())
    pi = min(max(pi, EPS), 1 - EPS)
    return float(np.log(pi / (1 - pi)))


def classify_recording(row, nns, feature_cols, prior_logit):
    per_phase_prob_present = {}
    per_phase_logodds = {}
    per_phase_evidence = {}

    for phase in PHASES:
        cols = phase_columns(feature_cols, phase)
        x = row[cols].values.astype(float).reshape(1, -1)
        proba = nns[phase].predict_proba(x)[0]
        class_order = nns[phase].classes_.tolist()  # ints: [0, 1] == [Absent, Present]
        p_present = float(proba[class_order.index(1)])
        p_present = min(max(p_present, EPS), 1 - EPS)
        per_phase_prob_present[phase] = p_present
        logodds = float(np.log(p_present / (1 - p_present)))
        per_phase_logodds[phase] = logodds
        # Evidence term: the silo's log-odds minus the training prior it
        # carries. This is the per-phase log-likelihood ratio implied by
        # the silo, the direct analogue of Arm A's margin.
        per_phase_evidence[phase] = logodds - prior_logit

    # Prior counted once, evidence summed across phases (see PRIOR
    # CORRECTION in the module docstring).
    total_logit = prior_logit + sum(per_phase_evidence.values())
    predicted_label = "Present" if total_logit >= 0 else "Absent"

    # Additive by construction -- each phase's evidence term IS its
    # contribution to (total_logit - prior_logit), same role as Arm A's
    # phase_margins.
    phase_margins = per_phase_evidence
    dominant_phase = max(phase_margins, key=lambda p: abs(phase_margins[p]))

    return {
        "recording_id": row["recording_id"],
        "true_label": row[LABEL_COLUMN],
        "predicted_label": predicted_label,
        "correct": predicted_label == row[LABEL_COLUMN],
        "per_phase_prob_present": per_phase_prob_present,
        "per_phase_raw_logodds": per_phase_logodds,
        "prior_logit": prior_logit,
        "total_logit": total_logit,
        "phase_margins": phase_margins,
        "dominant_phase": dominant_phase,
    }


def run_arm(df, boundary_source, feature_cols, train_pids, test_pids, verbose=True):
    if verbose:
        print(f"\n--- Arm: {boundary_source} ---")
    train_df, test_df, scaler = prepare_arm(df, boundary_source, feature_cols, train_pids, test_pids)
    if verbose:
        print(f"  train: {len(train_df)}  test: {len(test_df)}")

    nns = fit_all_nns(train_df, feature_cols)
    prior_logit = training_prior_logit(train_df)
    if verbose:
        print(f"  training Present prior logit: {prior_logit:.3f}")

    per_recording_results = [classify_recording(row, nns, feature_cols, prior_logit)
                             for _, row in test_df.iterrows()]
    accuracy = np.mean([r["correct"] for r in per_recording_results]) if per_recording_results else float("nan")

    per_class_accuracy = {}
    for cls in CLASSES:
        subset = [r for r in per_recording_results if r["true_label"] == cls]
        per_class_accuracy[cls] = np.mean([r["correct"] for r in subset]) if subset else float("nan")

    confusion = pd.crosstab(
        pd.Series([r["true_label"] for r in per_recording_results], name="true"),
        pd.Series([r["predicted_label"] for r in per_recording_results], name="predicted"),
    )

    if verbose:
        print(f"  accuracy: {accuracy * 100:.1f}%")
        for cls, acc in per_class_accuracy.items():
            print(f"    {cls}: {acc * 100:.1f}%")
        print(f"  confusion matrix:\n{confusion}")
    else:
        print(f"  [{boundary_source}] fold accuracy: {accuracy * 100:.1f}%  (n={len(per_recording_results)})")

    return {
        "boundary_source": boundary_source,
        "per_recording_results": per_recording_results,
        "accuracy": accuracy,
        "per_class_accuracy": per_class_accuracy,
        "confusion_matrix": confusion,
        "nns": nns,
        "prior_logit": prior_logit,
        "scaler": scaler,
        "train_df": train_df,
        "test_df": test_df,
        "train_recording_ids": train_df["recording_id"].tolist(),
        "test_recording_ids": test_df["recording_id"].tolist(),
    }


def pool_fold_results(fold_results):
    """Same pooling logic as Arm A's -- concatenates per_recording_results
    across folds (each recording appears in exactly one fold's test set),
    recomputes accuracy / per-class accuracy / confusion over the pool.
    Keeps train_df/test_df out of the pooled dict (they're per-fold and
    not meaningful pooled); SHAP sampling below uses the per-fold results
    directly, before pooling.
    """
    all_results = []
    for fr in fold_results:
        all_results.extend(fr["per_recording_results"])

    accuracy = np.mean([r["correct"] for r in all_results]) if all_results else float("nan")
    per_class_accuracy = {}
    for cls in CLASSES:
        subset = [r for r in all_results if r["true_label"] == cls]
        per_class_accuracy[cls] = np.mean([r["correct"] for r in subset]) if subset else float("nan")

    confusion = pd.crosstab(
        pd.Series([r["true_label"] for r in all_results], name="true"),
        pd.Series([r["predicted_label"] for r in all_results], name="predicted"),
    )

    return {
        "boundary_source": fold_results[0]["boundary_source"],
        "per_recording_results": all_results,
        "accuracy": accuracy,
        "per_class_accuracy": per_class_accuracy,
        "confusion_matrix": confusion,
        "test_recording_ids": [r["recording_id"] for r in all_results],
    }


# ---------------------------------------------------------------------
# Cross-arm (GT vs predicted boundary) comparison
# ---------------------------------------------------------------------

def build_per_recording_comparison(gt_result, pred_result):
    gt_by_id = {r["recording_id"]: r for r in gt_result["per_recording_results"]}
    pred_by_id = {r["recording_id"]: r for r in pred_result["per_recording_results"]}
    common_ids = sorted(set(gt_by_id) & set(pred_by_id))
    if set(gt_by_id) != set(pred_by_id):
        print(f"  NOTE: ground_truth arm has {len(gt_by_id)} test recordings, predicted arm has "
              f"{len(pred_by_id)} -- comparison covers the {len(common_ids)} present in both.")

    rows = []
    for rid in common_ids:
        gt_r, pred_r = gt_by_id[rid], pred_by_id[rid]
        row = {
            "recording_id": rid,
            "true_label": gt_r["true_label"],
            "gt_predicted": gt_r["predicted_label"],
            "gt_correct": gt_r["correct"],
            "pred_predicted": pred_r["predicted_label"],
            "pred_correct": pred_r["correct"],
            "gt_pred_agree": gt_r["predicted_label"] == pred_r["predicted_label"],
        }
        for phase in PHASES:
            row[f"gt_{phase}_logodds"] = gt_r["phase_margins"][phase]
            row[f"pred_{phase}_logodds"] = pred_r["phase_margins"][phase]
        rows.append(row)
    return pd.DataFrame(rows)


def print_phase_reliance_summary(result, label):
    correct_results = [r for r in result["per_recording_results"] if r["correct"]]
    if not correct_results:
        print(f"  [{label}] no correctly-classified test recordings")
        return
    print(f"  [{label}] mean per-phase log-odds contribution on {len(correct_results)} "
          f"correctly-classified test recordings (toward predicted class):")
    for phase in PHASES:
        vals = [r["phase_margins"][phase] for r in correct_results]
        print(f"    {phase:>10}: {np.mean(vals):.3f}")


# ---------------------------------------------------------------------
# Phase-attribution fidelity: build_phase_fidelity_table, compute_weight_share
# imported from run_gmm_binary_comparison -- same functions scoring both
# arms, not two implementations that happen to agree.
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# SHAP -- qualitative feature-level check on a sample (fast mode)
# ---------------------------------------------------------------------

def sample_correct_results(result, n_samples, random_state=RANDOM_STATE):
    correct_results = [r for r in result["per_recording_results"] if r["correct"]]
    if len(correct_results) <= n_samples:
        return correct_results
    rng = np.random.RandomState(random_state)
    idx = rng.choice(len(correct_results), size=n_samples, replace=False)
    return [correct_results[i] for i in sorted(idx)]


def explain_sample(result, feature_cols, boundary_source):
    """Runs KernelExplainer only on the sampled recordings' dominant
    phase, lazily building one explainer per phase actually needed
    (not all four) and reusing it across recordings that share a
    dominant phase -- keeps this to a handful of slow calls, not one
    per recording per phase. Operates on a single fold's result dict
    (needs train_df/test_df/nns, which only per-fold results carry).
    """
    train_df, test_df, nns = result["train_df"], result["test_df"], result["nns"]
    sample = sample_correct_results(result, N_SHAP_SAMPLES)
    test_by_id = {rid: idx for idx, rid in enumerate(test_df["recording_id"].values)}

    explainers = {}
    rows = []
    for r in sample:
        phase = r["dominant_phase"]
        cols = phase_columns(feature_cols, phase)

        if phase not in explainers:
            background = shap.sample(train_df[cols], min(SHAP_BACKGROUND_SIZE, len(train_df)),
                                      random_state=RANDOM_STATE)
            explainers[phase] = shap.KernelExplainer(nns[phase].predict_proba, background)

        test_idx = test_by_id[r["recording_id"]]
        x = test_df.iloc[[test_idx]][cols]

        raw_shap = explainers[phase].shap_values(x, nsamples=SHAP_NSAMPLES, silent=True)
        # shap's return shape differs by version: list-of-arrays (one per
        # class, older API) vs. a single (1, n_features, n_classes) array
        # (newer API). Normalize to a flat (n_features,) vector for the
        # "Present" class either way.
        present_idx = nns[phase].classes_.tolist().index(1)  # ints: [0, 1] == [Absent, Present]
        if isinstance(raw_shap, list):
            present_shap = np.asarray(raw_shap[present_idx])[0]
        else:
            arr = np.asarray(raw_shap)
            present_shap = arr[0, :, present_idx] if arr.ndim == 3 else arr[0]

        ranked = sorted(zip(cols, present_shap), key=lambda kv: abs(kv[1]), reverse=True)
        top = ranked[:TOP_K_FEATURES]

        for rank, (feat, val) in enumerate(top, start=1):
            rows.append({
                "boundary_source": boundary_source,
                "recording_id": r["recording_id"],
                "true_label": r["true_label"],
                "dominant_phase": phase,
                "rank": rank,
                "feature": feat,
                "shap_value_toward_present": float(val),
            })

    return pd.DataFrame(rows)


def print_shap_summary(shap_df, label):
    print(f"\n  [{label}] SHAP top-{TOP_K_FEATURES} features on {shap_df['recording_id'].nunique()} "
          f"sampled correctly-classified recordings (pooled across folds):")
    if len(shap_df) == 0:
        print("    no sampled recordings to explain")
        return
    top1 = shap_df[shap_df["rank"] == 1]
    print(f"    most-frequent #1 feature: {top1['feature'].value_counts().head(5).to_dict()}")
    # Looped rather than groupby(...).apply(dict) -- on recent pandas,
    # apply() flattens a dict-returning function into a (phase, feature)
    # MultiIndex instead of a Series of dicts keyed by phase, which
    # prints as one misleading line per feature instead of one per phase.
    for phase in PHASES:
        phase_df = shap_df[shap_df["dominant_phase"] == phase]
        if len(phase_df) == 0:
            continue
        counts = phase_df["feature"].value_counts().head(3).to_dict()
        print(f"    [{phase}] top features across sample: {counts}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    print(f"Loading {FEATURE_TABLE_PATH}")
    df = load_and_label_table()

    feature_cols = get_feature_columns(df)
    print(f"Using {len(feature_cols)} feature columns")

    folds = load_arm_a_folds()
    print(f"Reused Arm A folds: {len(folds)} folds "
          f"(~{len(folds[0][0])} train / ~{len(folds[0][1])} test patients per fold)")

    print("\n" + "=" * 60)
    print(f"FITTING BOTH ARMS ACROSS {len(folds)} FOLDS (NN + SHAP)")
    print("=" * 60)
    gt_fold_results, pred_fold_results = [], []
    gt_shap_parts, pred_shap_parts = [], []
    for i, (train_pids, test_pids) in enumerate(folds):
        print(f"\n--- Fold {i + 1}/{len(folds)} ---")
        gt_fr = run_arm(df, "ground_truth", feature_cols, train_pids, test_pids, verbose=False)
        pred_fr = run_arm(df, "predicted", feature_cols, train_pids, test_pids, verbose=False)

        # SHAP sampled per fold (needs each fold's own train_df/test_df/nns),
        # then concatenated below -- not run on the pooled result, which has
        # no single scaler/model to explain with.
        gt_shap_parts.append(explain_sample(gt_fr, feature_cols, "ground_truth"))
        pred_shap_parts.append(explain_sample(pred_fr, feature_cols, "predicted"))

        gt_fold_results.append(gt_fr)
        pred_fold_results.append(pred_fr)

    gt_result = pool_fold_results(gt_fold_results)
    pred_result = pool_fold_results(pred_fold_results)

    print("\n" + "=" * 60)
    print("POOLED OUT-OF-FOLD COMPARISON")
    print("=" * 60)
    print(f"ground_truth accuracy: {gt_result['accuracy'] * 100:.1f}%  "
          f"(n={len(gt_result['per_recording_results'])})")
    print(f"predicted   accuracy: {pred_result['accuracy'] * 100:.1f}%  "
          f"(n={len(pred_result['per_recording_results'])})")
    print(f"delta (predicted - ground_truth): "
          f"{(pred_result['accuracy'] - gt_result['accuracy']) * 100:+.1f} points")
    print(f"\nground_truth confusion matrix:\n{gt_result['confusion_matrix']}")
    print(f"\npredicted confusion matrix:\n{pred_result['confusion_matrix']}")

    print("\nPer-phase reliance (log-odds contribution -- NOT a fidelity metric, see phase "
          "fidelity section below for that):")
    print_phase_reliance_summary(gt_result, "ground_truth")
    print_phase_reliance_summary(pred_result, "predicted")

    comparison_df = build_per_recording_comparison(gt_result, pred_result)
    agreement_rate = comparison_df["gt_pred_agree"].mean()
    print(f"\nGT-vs-predicted prediction agreement rate: {agreement_rate * 100:.1f}% "
          f"({comparison_df['gt_pred_agree'].sum()}/{len(comparison_df)} pooled test recordings)")

    comparison_df.to_csv(PER_RECORDING_CSV_PATH, index=False)
    print(f"\nSaved per-recording comparison to {PER_RECORDING_CSV_PATH}")

    print("\n" + "=" * 60)
    print("PHASE-ATTRIBUTION FIDELITY (pooled out-of-fold)")
    print("=" * 60)
    timing_lookup = load_murmur_timing_lookup(CIRCOR_METADATA_PATH)
    gt_fidelity_df = build_phase_fidelity_table(gt_result, "ground_truth", timing_lookup)
    pred_fidelity_df = build_phase_fidelity_table(pred_result, "predicted", timing_lookup)
    print_fidelity_summary(gt_fidelity_df, "ground_truth")
    print_fidelity_summary(pred_fidelity_df, "predicted")

    fidelity_df = pd.concat([gt_fidelity_df, pred_fidelity_df], ignore_index=True)
    fidelity_df.to_csv(PHASE_FIDELITY_TSV_PATH, sep="\t", index=False)
    print(f"\nSaved per-recording phase fidelity to {PHASE_FIDELITY_TSV_PATH}")

    print("\n--- Discrete: dominant-phase win-rate (GT vs. predicted x phase) ---")
    win_rate_table = print_win_rate_table(gt_fidelity_df, pred_fidelity_df)

    print("\n--- Continuous: mean |margin|-normalized weight-share (GT vs. predicted x phase) ---")
    weight_share_table = print_weight_share_table(gt_fidelity_df, pred_fidelity_df)

    # Baseline-corrected fidelity summary, saved (not just printed) -- same
    # shape as Arm A's, computed with the identical shared functions, so the
    # two arms' fidelity numbers are comparable on the same terms.
    fidelity_summary = {}
    for boundary_source, fdf in (("ground_truth", gt_fidelity_df), ("predicted", pred_fidelity_df)):
        scoped = fdf[fdf["has_expected_phase"]]
        majority_phase, baseline_rate = compute_majority_baseline(fdf)
        hard_df, hard_rate = compute_hard_subset_fidelity(fdf, majority_phase)
        fidelity_summary[boundary_source] = {
            "n_scoped": len(scoped),
            "raw_fidelity": scoped["fidelity_match"].mean() if len(scoped) > 0 else float("nan"),
            "majority_phase": majority_phase,
            "majority_baseline": baseline_rate,
            "lift_over_baseline": (scoped["fidelity_match"].mean() - baseline_rate) if len(scoped) > 0 else float("nan"),
            "hard_subset_n": len(hard_df),
            "hard_subset_fidelity": hard_rate,
            "precision_recall": compute_phase_precision_recall(fdf),
        }

    print("\n" + "=" * 60)
    print(f"SHAP (sampled per fold, n<={N_SHAP_SAMPLES} correctly-classified recordings per fold per arm)")
    print("=" * 60)
    gt_shap_df = pd.concat(gt_shap_parts, ignore_index=True) if gt_shap_parts else pd.DataFrame()
    pred_shap_df = pd.concat(pred_shap_parts, ignore_index=True) if pred_shap_parts else pd.DataFrame()
    print_shap_summary(gt_shap_df, "ground_truth")
    print_shap_summary(pred_shap_df, "predicted")

    shap_df = pd.concat([gt_shap_df, pred_shap_df], ignore_index=True)
    shap_df.to_csv(SHAP_SAMPLE_TSV_PATH, sep="\t", index=False)
    print(f"\nSaved SHAP sample to {SHAP_SAMPLE_TSV_PATH}")

    output = {
        "ground_truth": gt_result,
        "predicted": pred_result,
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
            "n_folds": N_FOLDS,
            "n_shap_samples": N_SHAP_SAMPLES,
            "shap_background_size": SHAP_BACKGROUND_SIZE,
            "shap_nsamples": SHAP_NSAMPLES,
            "classes": CLASSES,
            "phases": PHASES,
        },
    }
    with open(RESULTS_PATH, "wb") as f:
        pickle.dump(output, f)
    print(f"Saved full results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()