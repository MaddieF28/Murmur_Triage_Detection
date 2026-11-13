"""
experiments/arm_a_gmm.py

Arm A (generative GMM) for the CirCor Murmur Present-vs-Absent binary
task (switched from Outcome -- see rationale below). Trains and
evaluates TWICE on the feature table produced by
features/build_circor_phase_features.py -- once using ground_truth-boundary
features, once using predicted-boundary features -- using patient-level
K-FOLD cross-validation (not a single train_test_split), so the two
results are directly comparable AND every patient's recording is
eventually held out and scored exactly once.

Why Murmur instead of Outcome
------------------------------
Outcome (Normal/Abnormal) is a clinical/echocardiogram-based diagnosis
label, not purely an acoustic one -- CirCor's own metadata has rows
where Murmur=Absent but Outcome=Abnormal, meaning a real fraction of
the "Abnormal" class may sound acoustically normal by design. That's a
property of the label, not something any feature set can fix, and it
was very likely a major contributor to a Yaseen-vs-CirCor accuracy gap
that looked like a bug but wasn't. Murmur Present/Absent is a directly
audible property, so it's the fairer test of whether this pipeline
(segmenter -> features -> GMM) can pick up real acoustic signal at all.
Unknown-murmur recordings are dropped -- not a clean binary answer.

Why patient-level (not recording-level) splits
------------------------------------------------
CirCor has up to 4 recordings per patient (one per valve site:
AV/PV/TV/MV), and Murmur (like Outcome) is a PATIENT-level label --
identical across all of a patient's recordings. Splitting by
recording_id let one patient's sites land in both train and test,
which is leakage (the model could partly be recognizing a patient it
already trained on via a different site, not generalizing). Splitting
by patient_id (the numeric filename prefix) fixes this.

Why K-FOLD instead of a single train_test_split (NEW)
------------------------------------------------------
The old single 80/20 split only ever tested ~20% of patients once. The
phase-attribution-fidelity evaluation (the whole point of this
project) is scoped further to recordings with a known Systolic/
Diastolic murmur timing -- with a single split that subset was stuck
around n=122 (and an n=8 "hard subset"), which is too small to trust
for the concentration-vs-uniform claim this project rests on.

With patient-level StratifiedKFold (N_FOLDS folds), every patient is
in the test fold exactly once across the full run. Pooling all folds'
out-of-fold predictions before computing accuracy / phase-fidelity /
weight-share means the fidelity tables get built over ~100% of CirCor's
known-timing recordings instead of one fold's ~20% slice -- without
ever training and predicting on the same patient (each patient's test
predictions always come from a model that never saw that patient during
that fold's training). This costs N_FOLDS separate GMM fits per arm
instead of one, which is the accepted tradeoff.

Phase-attribution fidelity
------------------------------------------------
Per recording, per boundary_source arm, computes a discriminative
margin per phase:

    margin_phase = loglik_Present_phase - loglik_Absent_phase

This is NOT the same as the raw per-class log-likelihood printed by
print_phase_reliance_summary -- a phase can have a large raw
log-likelihood under the winning class while still not discriminating
Present from Absent at all (it just fits that class's distribution
well in general). The margin is what actually decomposes the total
decision (per_class_total["Present"] - per_class_total["Absent"] is
exactly the sum of the four phase margins), so argmax(|margin_phase|)
is the phase that actually drove the decision.

That dominant phase is then compared against CirCor's own per-patient
Systolic/Diastolic murmur timing fields (ground truth for WHERE the
murmur actually is) to produce a fidelity rate: does the phase the
model relied on match where the murmur clinically is.

ASSUMPTION FLAGGED, NOT VERIFIED: SYSTOLIC_TIMING_COLUMN and
DIASTOLIC_TIMING_COLUMN below are the standard published
CirCor/PhysioNet-2022 metadata column names. They have not been
checked against your actual training_data.csv. If they're wrong, this
script raises immediately (with your file's actual column list printed)
rather than silently computing a wrong fidelity number -- fix the two
constants below if needed.

Fidelity is only computed for recordings where a Systolic or Diastolic
timing value exists (i.e. mostly Murmur=Present recordings). Murmur =
Absent recordings have no timing value to check against, so they get
has_expected_phase=False / fidelity_match=None rather than being
silently scored as a pass or fail -- there is no "no phase should
dominate" check implemented here.

Outputs:
  - gmm_murmur_comparison_results.pkl
  - gmm_murmur_per_recording_comparison.csv
  - phase_fidelity_per_recording.tsv
  - Console summary: per-fold accuracy, pooled accuracy, per-phase
    reliance, GT-vs-predicted agreement rate, phase-attribution
    fidelity (all computed over pooled out-of-fold predictions)
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repo root: makes `paths` and the packages importable from any cwd
import paths


from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from segmenter_eval.evaluate_segmenter_all_datasets import CIRCOR_DIR

FEATURE_TABLE_PATH = paths.CIRCOR_FEATURE_TABLE
RESULTS_PATH = paths.GMM_RESULTS_PKL
PER_RECORDING_CSV_PATH = paths.GMM_PER_RECORDING_CSV
PHASE_FIDELITY_TSV_PATH = paths.GMM_PHASE_FIDELITY_TSV

# Same guessed path as features/build_circor_phase_features.py -- fix if it's
# somewhere else on your machine.
CIRCOR_METADATA_PATH = paths.CIRCOR_METADATA_PATH

LABEL_COLUMN = "murmur"
CLASSES = ("Present", "Absent")
PHASES = ("S1", "Systole", "S2", "Diastole")
NON_FEATURE_COLS = ("recording_id", "boundary_source", "outcome", "murmur", "patient_id", "fold")

# ASSUMED CirCor/PhysioNet-2022 standard column names -- NOT verified
# against your actual metadata CSV. See module docstring.
SYSTOLIC_TIMING_COLUMN = "Systolic murmur timing"
DIASTOLIC_TIMING_COLUMN = "Diastolic murmur timing"

N_COMPONENTS = 2
COVARIANCE_TYPE = "diag"
RANDOM_STATE = 42
N_FOLDS = 5  # NEW -- replaces TEST_SIZE / single train_test_split


# ---------------------------------------------------------------------
# Metadata: Murmur label + patient_id
# ---------------------------------------------------------------------

def load_murmur_lookup(metadata_path):
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"CirCor metadata CSV not found at {metadata_path}. "
            f"Fix CIRCOR_METADATA_PATH at the top of this script."
        )
    df = pd.read_csv(metadata_path)
    required = {"Patient ID", "Murmur"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"CirCor metadata CSV is missing expected column(s) {missing}. "
            f"Columns found: {list(df.columns)}."
        )

    primary, secondary = {}, {}
    for _, row in df.iterrows():
        pid = str(row["Patient ID"]).strip()
        murmur = row["Murmur"]
        if pd.isna(murmur):
            continue
        primary[pid] = murmur
        if "Additional ID" in df.columns and not pd.isna(row["Additional ID"]):
            aid = str(row["Additional ID"]).strip()
            secondary.setdefault(aid, murmur)

    def lookup(patient_id):
        if patient_id in primary:
            return primary[patient_id]
        if patient_id in secondary:
            return secondary[patient_id]
        return None

    return lookup


def load_murmur_timing_lookup(metadata_path):
    """Returns patient_id -> set of expected phases ("Systole" and/or
    "Diastole"), based on whether CirCor's Systolic/Diastolic murmur
    timing fields are non-null for that patient.

    ASSUMPTION FLAGGED: column names below are the standard published
    CirCor/PhysioNet-2022 names, not verified against your file. Raises
    loudly with your actual column list if they don't match, rather
    than silently returning wrong/empty results.
    """
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"CirCor metadata CSV not found at {metadata_path}. "
            f"Fix CIRCOR_METADATA_PATH at the top of this script."
        )
    df = pd.read_csv(metadata_path)
    required = {"Patient ID", SYSTOLIC_TIMING_COLUMN, DIASTOLIC_TIMING_COLUMN}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"CirCor metadata CSV is missing expected column(s) {missing} needed for phase "
            f"fidelity. Columns found: {list(df.columns)}. SYSTOLIC_TIMING_COLUMN / "
            f"DIASTOLIC_TIMING_COLUMN at the top of this script are ASSUMED standard "
            f"CirCor/PhysioNet-2022 names -- fix them to match your actual CSV headers."
        )

    lookup = {}
    for _, row in df.iterrows():
        pid = str(row["Patient ID"]).strip()
        expected = set()
        if not pd.isna(row[SYSTOLIC_TIMING_COLUMN]):
            expected.add("Systole")
        if not pd.isna(row[DIASTOLIC_TIMING_COLUMN]):
            expected.add("Diastole")
        lookup[pid] = expected
        if "Additional ID" in df.columns and not pd.isna(row["Additional ID"]):
            aid = str(row["Additional ID"]).strip()
            lookup.setdefault(aid, expected)

    return lookup


def get_patient_id(recording_id):
    return recording_id.split("_", 1)[0]


# ---------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------

def load_and_label_table():
    df = pd.read_csv(FEATURE_TABLE_PATH, sep="\t")
    print(f"Loaded {len(df)} rows ({df['recording_id'].nunique()} unique recordings) from feature table")

    lookup = load_murmur_lookup(CIRCOR_METADATA_PATH)
    df["patient_id"] = df["recording_id"].apply(get_patient_id)
    df["murmur"] = df["patient_id"].apply(lookup)

    # Sanity check: Murmur is a patient-level field, so every recording
    # for a given patient should agree. If it doesn't, something is off
    # in the ID-matching logic and is worth knowing about, not silently
    # trusting.
    per_patient_nunique = df.dropna(subset=["murmur"]).groupby("patient_id")["murmur"].nunique()
    inconsistent = per_patient_nunique[per_patient_nunique > 1]
    if len(inconsistent) > 0:
        print(f"WARNING: {len(inconsistent)} patient_id(s) have inconsistent Murmur labels "
              f"across their recordings: {list(inconsistent.index)[:10]} -- check ID matching.")

    n_no_label = df["murmur"].isna().sum()
    n_unknown = (df["murmur"] == "Unknown").sum()
    df = df[df["murmur"].isin(CLASSES)].copy()
    print(f"Dropped {n_no_label} row(s) with no Murmur label found, "
          f"{n_unknown} row(s) with Murmur=Unknown. "
          f"{len(df)} rows remain ({df['recording_id'].nunique()} recordings, "
          f"{df['patient_id'].nunique()} patients).")

    return df


def get_feature_columns(df):
    candidate_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    all_nan_cols = [c for c in candidate_cols if df[c].isna().all()]
    if all_nan_cols:
        print(f"Dropping {len(all_nan_cols)} all-NaN column(s): {all_nan_cols}")
    return [c for c in candidate_cols if c not in all_nan_cols]


def phase_columns(feature_cols, phase):
    prefix = f"{phase}_"
    return [c for c in feature_cols if c.startswith(prefix)]


def make_patient_folds(df, n_folds=N_FOLDS, random_state=RANDOM_STATE):
    """Patient-level stratified K-fold (NEW -- replaces split_patient_ids's
    single train_test_split).

    Dedups to one row per patient_id first (Murmur is a patient-level
    label), then runs StratifiedKFold on that deduped table so class
    balance (Present/Absent) is preserved per fold without needing
    GroupKFold -- there's already exactly one row per group by
    construction, so plain StratifiedKFold IS group-safe here.

    Returns a list of (train_pids, test_pids) tuples, one per fold. Every
    patient appears in exactly one fold's test_pids across the full list,
    so concatenating all folds' held-out predictions covers every patient
    exactly once with no leakage and no double-counting.
    """
    patient_label = (df[["patient_id", "murmur"]]
                      .drop_duplicates(subset="patient_id", keep="first")
                      .reset_index(drop=True))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    folds = []
    for train_idx, test_idx in skf.split(patient_label["patient_id"], patient_label["murmur"]):
        train_pids = set(patient_label.loc[train_idx, "patient_id"])
        test_pids = set(patient_label.loc[test_idx, "patient_id"])
        folds.append((train_pids, test_pids))
    return folds


def prepare_arm(df, boundary_source, feature_cols, train_pids, test_pids):
    arm_df = df[df["boundary_source"] == boundary_source].copy()

    train_df = arm_df[arm_df["patient_id"].isin(train_pids)].copy()
    test_df = arm_df[arm_df["patient_id"].isin(test_pids)].copy()

    n_train_before, n_test_before = len(train_df), len(test_df)
    train_df = train_df.dropna(subset=feature_cols)
    test_df = test_df.dropna(subset=feature_cols)
    if n_train_before != len(train_df) or n_test_before != len(test_df):
        print(f"  [{boundary_source}] dropped {n_train_before - len(train_df)} train / "
              f"{n_test_before - len(test_df)} test row(s) with missing-phase NaNs")

    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols])
    train_df[feature_cols] = scaler.transform(train_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])

    return train_df, test_df, scaler


# ---------------------------------------------------------------------
# Fitting / classification
# ---------------------------------------------------------------------

def fit_all_gmms(train_df, feature_cols):
    gmms = {}
    for cls in CLASSES:
        gmms[cls] = {}
        class_rows = train_df[train_df[LABEL_COLUMN] == cls]
        for phase in PHASES:
            cols = phase_columns(feature_cols, phase)
            X = class_rows[cols].values
            gmm = GaussianMixture(
                n_components=N_COMPONENTS,
                covariance_type=COVARIANCE_TYPE,
                random_state=RANDOM_STATE,
            )
            gmm.fit(X)
            gmms[cls][phase] = gmm
    return gmms


def classify_recording(row, gmms, feature_cols):
    per_class_per_phase = {cls: {} for cls in CLASSES}
    per_class_total = {}

    for cls in CLASSES:
        total_ll = 0.0
        for phase in PHASES:
            cols = phase_columns(feature_cols, phase)
            x = row[cols].values.astype(float).reshape(1, -1)
            ll = gmms[cls][phase].score_samples(x)[0]
            per_class_per_phase[cls][phase] = float(ll)
            total_ll += ll
        per_class_total[cls] = total_ll

    predicted_label = max(per_class_total, key=lambda k: per_class_total[k])

    # Discriminative margin per phase: Present minus Absent log-likelihood.
    # Sums exactly to the total decision margin
    # (per_class_total["Present"] - per_class_total["Absent"]), unlike raw
    # per-class log-likelihood, which conflates "phase fits this class
    # well" with "phase actually distinguishes the two classes".
    phase_margins = {
        phase: per_class_per_phase["Present"][phase] - per_class_per_phase["Absent"][phase]
        for phase in PHASES
    }
    dominant_phase = max(phase_margins, key=lambda p: abs(phase_margins[p]))

    return {
        "recording_id": row["recording_id"],
        "true_label": row[LABEL_COLUMN],
        "predicted_label": predicted_label,
        "correct": predicted_label == row[LABEL_COLUMN],
        "per_class_total_loglik": per_class_total,
        "per_class_per_phase_loglik": per_class_per_phase,
        "phase_margins": phase_margins,
        "dominant_phase": dominant_phase,
    }


def run_arm(df, boundary_source, feature_cols, train_pids, test_pids, verbose=True):
    if verbose:
        print(f"\n--- Arm: {boundary_source} ---")
    train_df, test_df, scaler = prepare_arm(df, boundary_source, feature_cols, train_pids, test_pids)
    if verbose:
        print(f"  train: {len(train_df)}  test: {len(test_df)}")

    gmms = fit_all_gmms(train_df, feature_cols)

    per_recording_results = [classify_recording(row, gmms, feature_cols) for _, row in test_df.iterrows()]
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
        "gmms": gmms,
        "scaler": scaler,
        "train_recording_ids": train_df["recording_id"].tolist(),
        "test_recording_ids": test_df["recording_id"].tolist(),
    }


def pool_fold_results(fold_results):
    """Concatenates per_recording_results across folds into one pooled
    out-of-fold result for a boundary_source, then recomputes accuracy /
    per-class accuracy / confusion over the pool.

    Safe to concatenate without double-counting: make_patient_folds
    guarantees each patient (and therefore every one of that patient's
    recordings) appears in exactly one fold's test set, so every
    recording contributes exactly once across the pooled list -- this is
    a genuine out-of-fold prediction set, not folds' metrics averaged.
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
        # test_recording_ids kept for parity with the old single-split
        # result shape -- union across folds, since each recording
        # appears in exactly one fold's test set.
        "test_recording_ids": [r["recording_id"] for r in all_results],
    }


# ---------------------------------------------------------------------
# Cross-arm comparison
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
            row[f"gt_{phase}_loglik"] = gt_r["per_class_per_phase_loglik"][gt_r["predicted_label"]][phase]
            row[f"pred_{phase}_loglik"] = pred_r["per_class_per_phase_loglik"][pred_r["predicted_label"]][phase]
        rows.append(row)
    return pd.DataFrame(rows)


def print_phase_reliance_summary(result, label):
    correct_results = [r for r in result["per_recording_results"] if r["correct"]]
    if not correct_results:
        print(f"  [{label}] no correctly-classified test recordings")
        return
    print(f"  [{label}] mean per-phase log-likelihood on {len(correct_results)} correctly-classified "
          f"test recordings (predicted class):")
    for phase in PHASES:
        vals = [r["per_class_per_phase_loglik"][r["predicted_label"]][phase] for r in correct_results]
        print(f"    {phase:>10}: {np.mean(vals):.3f}")


# ---------------------------------------------------------------------
# Phase-attribution fidelity
# ---------------------------------------------------------------------

def compute_weight_share(phase_margins):
    """Continuous complement to dominant_phase: each phase's share of total
    |margin| across all four phases (sums to ~1.0), instead of just which
    phase wins the argmax. Lets you say "Systole carries only 40% of total
    attribution weight on average" rather than only "Systole wins the vote
    40% of the time" -- the win-count throws away how close the other
    phases were, which matters when the model is only weakly favoring the
    winner.
    """
    abs_margins = {p: abs(phase_margins[p]) for p in PHASES}
    total = sum(abs_margins.values())
    if total == 0:
        return {p: 1.0 / len(PHASES) for p in PHASES}
    return {p: abs_margins[p] / total for p in PHASES}


def build_phase_fidelity_table(result, boundary_source, timing_lookup):
    rows = []
    for r in result["per_recording_results"]:
        pid = get_patient_id(r["recording_id"])
        expected_phases = timing_lookup.get(pid, set())
        share = compute_weight_share(r["phase_margins"])
        row = {
            "recording_id": r["recording_id"],
            "boundary_source": boundary_source,
            "true_label": r["true_label"],
            "predicted_label": r["predicted_label"],
            "correct": r["correct"],
            "dominant_phase": r["dominant_phase"],
            "expected_phases": ",".join(sorted(expected_phases)) if expected_phases else "",
            "has_expected_phase": len(expected_phases) > 0,
            "fidelity_match": (r["dominant_phase"] in expected_phases) if expected_phases else None,
        }
        for phase in PHASES:
            row[f"margin_{phase}"] = r["phase_margins"][phase]
            row[f"share_{phase}"] = share[phase]
        rows.append(row)
    return pd.DataFrame(rows)


def compute_majority_baseline(fidelity_df):
    """Finds the single phase most often present in expected_phases across the
    scoped subset, and the hit rate "always guess that phase" would get.

    This exists because raw fidelity is uninterpretable on its own: if one
    phase (e.g. Systole) dominates expected_phases, a model that always
    predicts that phase scores high without reading the recording at all.
    Every fidelity number below should be read against this baseline, not
    in isolation.
    """
    scoped = fidelity_df[fidelity_df["has_expected_phase"]]
    if len(scoped) == 0:
        return None, float("nan")
    phase_counts = {phase: 0 for phase in PHASES}
    for expected in scoped["expected_phases"]:
        for p in expected.split(","):
            if p in phase_counts:
                phase_counts[p] += 1
    majority_phase = max(phase_counts, key=lambda p: phase_counts[p])
    baseline_hit_rate = scoped["expected_phases"].apply(
        lambda expected: majority_phase in expected.split(",")
    ).mean()
    return majority_phase, baseline_hit_rate


def compute_hard_subset_fidelity(fidelity_df, majority_phase):
    """Fidelity restricted to recordings whose expected_phases is NOT just
    the majority phase alone -- i.e. the subset where "always guess
    <majority_phase>" cannot trivially win, because doing so would be wrong.
    A model actually reading per-recording signal should hold up better here
    than a majority-baseline artifact would.
    """
    scoped = fidelity_df[fidelity_df["has_expected_phase"]]
    if len(scoped) == 0 or majority_phase is None:
        return scoped, float("nan")
    hard = scoped[scoped["expected_phases"] != majority_phase]
    hard_rate = hard["fidelity_match"].mean() if len(hard) > 0 else float("nan")
    return hard, hard_rate


def compute_phase_precision_recall(fidelity_df):
    """Per-phase precision (of the recordings where this phase was predicted
    dominant, how many were actually expected) and recall (of the recordings
    where this phase was expected, how many were predicted dominant).
    Breaks the one blended hit-rate into a form that can't be inflated by a
    single dominant expected phase the way the raw fidelity rate can.
    """
    scoped = fidelity_df[fidelity_df["has_expected_phase"]]
    rows = []
    for phase in PHASES:
        predicted_this_phase = scoped[scoped["dominant_phase"] == phase]
        precision = predicted_this_phase["fidelity_match"].mean() if len(predicted_this_phase) > 0 else float("nan")
        expected_this_phase = scoped[scoped["expected_phases"].apply(lambda e: phase in e.split(","))]
        recall = ((expected_this_phase["dominant_phase"] == phase).mean()
                  if len(expected_this_phase) > 0 else float("nan"))
        rows.append({
            "phase": phase,
            "n_predicted_dominant": len(predicted_this_phase),
            "precision": precision,
            "n_expected": len(expected_this_phase),
            "recall": recall,
        })
    return pd.DataFrame(rows)


def print_weight_share_table(gt_fidelity_df, pred_fidelity_df):
    """Compact GT-vs-predicted x phase table of mean weight-share, e.g.:

              S1  Systole      S2  Diastole
    GT      25.1%    28.6%   22.4%     24.0%
    predicted 26.1%  28.5%   21.4%     24.0%

    Pulled from the share_<phase> columns build_phase_fidelity_table
    already computes (via compute_weight_share).
    """
    rows = {}
    for row_label, fdf in (("GT", gt_fidelity_df), ("predicted", pred_fidelity_df)):
        scoped = fdf[fdf["has_expected_phase"]]
        rows[row_label] = {phase: scoped[f"share_{phase}"].mean() for phase in PHASES}
    table = pd.DataFrame(rows).T[list(PHASES)]
    formatted = pd.DataFrame(
        {col: [f"{v * 100:.1f}%" for v in table[col]] for col in table.columns},
        index=table.index,
    )
    print(formatted.to_string())
    return table


def print_win_rate_table(gt_fidelity_df, pred_fidelity_df):
    """Same shape as print_weight_share_table but for the discrete
    dominant-phase win-rate."""
    rows = {}
    for row_label, fdf in (("GT", gt_fidelity_df), ("predicted", pred_fidelity_df)):
        scoped = fdf[fdf["has_expected_phase"]]
        counts = scoped["dominant_phase"].value_counts(normalize=True).reindex(PHASES, fill_value=0.0)
        rows[row_label] = {phase: counts[phase] for phase in PHASES}
    table = pd.DataFrame(rows).T[list(PHASES)]
    formatted = pd.DataFrame(
        {col: [f"{v * 100:.1f}%" for v in table[col]] for col in table.columns},
        index=table.index,
    )
    print(formatted.to_string())
    return table


def print_fidelity_summary(fidelity_df, label):
    scoped = fidelity_df[fidelity_df["has_expected_phase"]]
    print(f"\n  [{label}] phase-attribution fidelity "
          f"(recordings with a known Systolic/Diastolic murmur timing only):")
    if len(scoped) == 0:
        print("    no recordings with expected-phase ground truth found -- check "
              "SYSTOLIC_TIMING_COLUMN / DIASTOLIC_TIMING_COLUMN assumptions")
        return

    rate = scoped["fidelity_match"].mean()
    majority_phase, baseline_rate = compute_majority_baseline(fidelity_df)
    lift = rate - baseline_rate

    print(f"    n = {len(scoped)} (of {len(fidelity_df)} total pooled out-of-fold recordings in this arm)")
    print(f"    dominant phase matches expected phase: {rate * 100:.1f}% "
          f"({int(scoped['fidelity_match'].sum())}/{len(scoped)})")
    print(f"    majority-baseline ('always guess {majority_phase}'): {baseline_rate * 100:.1f}%")
    print(f"    lift over majority baseline: {lift * 100:+.1f} points "
          f"{'(below trivial baseline -- treat raw fidelity with caution)' if lift < 0 else ''}")

    hard_df, hard_rate = compute_hard_subset_fidelity(fidelity_df, majority_phase)
    if len(hard_df) > 0:
        print(f"    hard subset (expected_phases != '{majority_phase}' alone), "
              f"n={len(hard_df)}: fidelity {hard_rate * 100:.1f}% "
              f"(majority baseline scores 0% here by construction)")
    else:
        print(f"    hard subset (expected_phases != '{majority_phase}' alone): n=0 -- "
              f"every scoped recording's expected set is just '{majority_phase}'")

    pr_df = compute_phase_precision_recall(fidelity_df)
    print(f"    per-phase precision/recall against expected_phases:")
    print("    " + pr_df.to_string(index=False).replace("\n", "\n    "))

    print(f"    dominant-phase distribution: {scoped['dominant_phase'].value_counts().to_dict()}")

    mean_shares = {phase: scoped[f"share_{phase}"].mean() for phase in PHASES}
    print(f"    mean phase weight-share (continuous, |margin|-normalized, sums to ~100%): "
          + ", ".join(f"{p}={mean_shares[p] * 100:.1f}%" for p in PHASES))


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    print(f"Loading {FEATURE_TABLE_PATH}")
    df = load_and_label_table()

    feature_cols = get_feature_columns(df)
    print(f"Using {len(feature_cols)} feature columns")

    folds = make_patient_folds(df)
    print(f"{N_FOLDS}-fold patient-level split: "
          f"~{len(folds[0][0])} train / ~{len(folds[0][1])} test patients per fold")

    print("\n" + "=" * 60)
    print(f"FITTING BOTH ARMS ACROSS {N_FOLDS} FOLDS")
    print("=" * 60)
    gt_fold_results, pred_fold_results = [], []
    for i, (train_pids, test_pids) in enumerate(folds):
        print(f"\n--- Fold {i + 1}/{N_FOLDS} ---")
        gt_fold_results.append(
            run_arm(df, "ground_truth", feature_cols, train_pids, test_pids, verbose=False)
        )
        pred_fold_results.append(
            run_arm(df, "predicted", feature_cols, train_pids, test_pids, verbose=False)
        )

    # Pool out-of-fold predictions -- every recording contributes exactly
    # once, from whichever fold held its patient out. This is what all
    # downstream accuracy / fidelity / weight-share numbers are computed
    # over, not any single fold's numbers.
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

    print("\nPer-phase reliance (raw log-likelihood -- NOT a fidelity metric, see phase "
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

    # Baseline-corrected fidelity summary, saved (not just printed) so it can
    # be pulled into the write-up without re-deriving it from the TSV.
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

    output = {
        "ground_truth": gt_result,
        "predicted": pred_result,
        "gt_fold_results": gt_fold_results,
        "pred_fold_results": pred_fold_results,
        "feature_cols": feature_cols,
        # Kept for downstream scripts (e.g. experiments/arm_b_nn_shap.py)
        # that currently load a single train/test split out of this pickle --
        # see note in main() below / update that script to load `folds`
        # instead once it's converted to k-fold too.
        "folds": [(list(tr), list(te)) for tr, te in folds],
        "fidelity_summary": fidelity_summary,
        "win_rate_table": win_rate_table,
        "weight_share_table": weight_share_table,
        "config": {
            "label_column": LABEL_COLUMN,
            "n_components": N_COMPONENTS,
            "covariance_type": COVARIANCE_TYPE,
            "random_state": RANDOM_STATE,
            "n_folds": N_FOLDS,
            "classes": CLASSES,
            "phases": PHASES,
        },
    }
    with open(RESULTS_PATH, "wb") as f:
        pickle.dump(output, f)
    print(f"Saved full results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()