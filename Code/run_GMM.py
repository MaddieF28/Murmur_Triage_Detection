"""
run_GMMpy

Arm A: class-conditional generative GMM classifier.

For each (class, phase) cell, fits a Gaussian Mixture Model (diagonal
covariance, fixed n_components across all cells for a controlled
comparison -- see design discussion) to that cell's standardized
training features. Classifies a recording by summing per-phase
log-likelihoods across all 4 phases per class (equal weight) and
picking the highest-total class.

Saves a comprehensive results file for downstream plotting/analysis --
full per-class, per-phase log-likelihood breakdowns, not just final
predictions -- plus the fitted GMM parameters themselves.
"""

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture

from extract_features import standardize_features  # your extract.py

# --- config ---------------------------------------------------------------

THIS_DIR = Path(__file__).resolve().parent
FEATURE_TABLE_PATH = THIS_DIR / "pcg_feature_table.csv"
RESULTS_PATH = THIS_DIR / "GMM_results.pkl"

CLASSES = ("AS", "MR", "MS", "MVP", "N")
PHASES = ("S1", "Systole", "S2", "Diastole")  # matches segment.py's casing

N_COMPONENTS = 2          # fixed across all class x phase cells -- deliberate,
                           # see discussion: letting this vary per cell would
                           # confound model capacity with real data differences
COVARIANCE_TYPE = "diag"  # diagonal, given ~160 train points / ~40 dims per cell

RANDOM_STATE = 42
TEST_SIZE = 0.2


# --- data prep --------------------------------------------------------------

def load_and_split():
    df = pd.read_csv(FEATURE_TABLE_PATH)

    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, stratify=df["label"], random_state=RANDOM_STATE
    )

    train_scaled, test_scaled, scaler = standardize_features(train_df, test_df)
    return train_scaled, test_scaled, scaler


def phase_columns(df: pd.DataFrame, phase: str) -> list:
    """All feature columns belonging to one phase, e.g. 'S1_mfcc_1_mean'."""
    prefix = f"{phase}_"
    return [c for c in df.columns if c.startswith(prefix)]


# --- fitting ----------------------------------------------------------------

def fit_all_gmms(train_df: pd.DataFrame) -> dict:
    """
    Fits one GMM per (class, phase) cell.

    Returns
    -------
    dict: gmms[class][phase] -> fitted GaussianMixture
    """
    gmms = {}

    for cls in CLASSES:
        gmms[cls] = {}
        class_rows = train_df[train_df["label"] == cls]

        for phase in PHASES:
            cols = phase_columns(train_df, phase)
            X = class_rows[cols].values

            n_nan = np.isnan(X).sum()
            if n_nan > 0:
                print(
                    f"  WARNING: {cls}/{phase} training data has {n_nan} NaN "
                    f"values (missing-phase recordings) -- dropping those rows "
                    f"for fitting this cell."
                )
                mask = ~np.isnan(X).any(axis=1)
                X = X[mask]

            gmm = GaussianMixture(
                n_components=N_COMPONENTS,
                covariance_type=COVARIANCE_TYPE,
                random_state=RANDOM_STATE,
            )
            gmm.fit(X)
            gmms[cls][phase] = gmm

            print(f"  fitted {cls}/{phase} on {len(X)} recordings")

    return gmms


# --- classification -----------------------------------------------------

def classify_recording(row: pd.Series, gmms: dict) -> dict:
    """
    Computes per-class, per-phase log-likelihood for one test recording,
    sums across phases (equal weight, skipping any NaN/missing phase),
    and returns the full breakdown plus the predicted class.
    """
    per_class_per_phase = {cls: {} for cls in CLASSES}
    per_class_total = {}
    missing_phases = []

    for cls in CLASSES:
        total_ll = 0.0
        any_phase_used = False

        for phase in PHASES:
            cols = phase_columns_cache[phase]
            x = np.asarray(row[cols].values, dtype=float).reshape(1, -1)

            if np.isnan(x).any():
                per_class_per_phase[cls][phase] = None
                if phase not in missing_phases:
                    missing_phases.append(phase)
                continue

            ll = gmms[cls][phase].score_samples(x)[0]  # log-likelihood
            per_class_per_phase[cls][phase] = float(ll)
            total_ll += ll
            any_phase_used = True

        per_class_total[cls] = total_ll if any_phase_used else float("-inf")

    predicted_class = max(per_class_total, key=lambda k: per_class_total[k])

    return {
        "recording_id": row["recording_id"],
        "true_label": row["label"],
        "predicted_label": predicted_class,
        "per_class_total_loglik": per_class_total,
        "per_class_per_phase_loglik": per_class_per_phase,
        "missing_phases": missing_phases,
    }


# --- main ---------------------------------------------------------------

if __name__ == "__main__":
    print(f"Loading {FEATURE_TABLE_PATH}")
    train_df, test_df, scaler = load_and_split()
    print(f"Train: {len(train_df)}  Test: {len(test_df)}")

    print("\nFitting GMMs (fixed n_components="
          f"{N_COMPONENTS}, covariance_type={COVARIANCE_TYPE})...")
    gmms = fit_all_gmms(train_df)

    # Cache phase column lists once (same for train/test, same columns).
    phase_columns_cache = {phase: phase_columns(test_df, phase) for phase in PHASES}

    print("\nClassifying test recordings...")
    per_recording_results = []
    for _, row in test_df.iterrows():
        result = classify_recording(row, gmms)
        per_recording_results.append(result)

    # --- summary metrics ---
    y_true = [r["true_label"] for r in per_recording_results]
    y_pred = [r["predicted_label"] for r in per_recording_results]

    accuracy = np.mean([t == p for t, p in zip(y_true, y_pred)])

    per_class_accuracy = {}
    for cls in CLASSES:
        idx = [i for i, t in enumerate(y_true) if t == cls]
        correct = sum(1 for i in idx if y_pred[i] == y_true[i])
        per_class_accuracy[cls] = correct / len(idx) if idx else float("nan")

    confusion = pd.crosstab(
        pd.Series(y_true, name="true"), pd.Series(y_pred, name="predicted")
    )

    print("\n" + "=" * 60)
    print("ARM A (GMM) RESULTS")
    print("=" * 60)
    print(f"Overall accuracy: {accuracy * 100:.1f}%")
    print("\nPer-class accuracy:")
    for cls, acc in per_class_accuracy.items():
        print(f"  {cls}: {acc * 100:.1f}%")
    print("\nConfusion matrix:")
    print(confusion)

    # --- save everything for later plotting/analysis ---
    output = {
        "per_recording_results": per_recording_results,
        "accuracy": accuracy,
        "per_class_accuracy": per_class_accuracy,
        "confusion_matrix": confusion,
        "gmms": gmms,  # fitted model parameters (means_, covariances_, weights_)
        "config": {
            "n_components": N_COMPONENTS,
            "covariance_type": COVARIANCE_TYPE,
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "classes": CLASSES,
            "phases": PHASES,
        },
        "train_recording_ids": train_df["recording_id"].tolist(),
        "test_recording_ids": test_df["recording_id"].tolist(),
    }

    with open(RESULTS_PATH, "wb") as f:
        pickle.dump(output, f)

    print(f"\nFull results saved to {RESULTS_PATH}")