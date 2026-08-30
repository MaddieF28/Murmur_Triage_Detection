"""
arm_a_pruned.py

Same GMM classifier as arm_a.py, but using a pruned feature set based
on the feature diagnostics: drops MFCC features that showed strong
univariate separation (ANOVA) but near-zero or negative permutation
importance in the actual classifier -- likely redundant with the
amplitude/spectral/DWT features the model actually relies on.

Keeps: all temporal_amplitude, spectral, and dwt features (all showed
real permutation importance in at least one phase), plus a reduced
MFCC set (mfcc_1, mfcc_3, mfcc_7, mfcc_8 -- mean+std each) that
appeared in both the top ANOVA list and had nonzero permutation
importance somewhere. Drops the rest of the MFCCs (2,4,5,6,9,10,11,12,13).

Everything else -- split, standardization, GMM config, classification
logic, output format -- is identical to arm_a.py, so results are
directly comparable.
"""

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture

from extract_features import standardize_features

# --- config ---------------------------------------------------------------

THIS_DIR = Path(__file__).resolve().parent
FEATURE_TABLE_PATH = THIS_DIR / "pcg_feature_table.csv"
RESULTS_PATH = THIS_DIR / "pruned_GMM_results.pkl"

CLASSES = ("AS", "MR", "MS", "MVP", "N")
PHASES = ("S1", "Systole", "S2", "Diastole")

N_COMPONENTS = 2
COVARIANCE_TYPE = "diag"

RANDOM_STATE = 42  # same as arm_a.py -- keeps the split identical for comparison
TEST_SIZE = 0.2

# MFCC indices to KEEP (mean + std for each). Everything else in mfcc
# gets dropped. Non-mfcc categories (temporal_amplitude, spectral, dwt)
# are kept in full.
KEPT_MFCC_INDICES = (1, 3, 7, 8)


# --- feature pruning --------------------------------------------------------

def is_kept_column(col: str) -> bool:
    """
    col looks like 'Systole_mfcc_3_mean' or 'S1_amp_mean' or
    'Diastole_dwt_energy_detail_2'. Non-mfcc columns are always kept;
    mfcc columns are kept only if their index is in KEPT_MFCC_INDICES.
    """
    if "mfcc_" not in col:
        return True

    # extract the mfcc index, e.g. 'Systole_mfcc_3_mean' -> 3
    parts = col.split("mfcc_")[1]  # '3_mean'
    idx = int(parts.split("_")[0])
    return idx in KEPT_MFCC_INDICES


def prune_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [c for c in df.columns if c in ("recording_id", "label") or is_kept_column(c)]
    dropped = [c for c in df.columns if c not in keep_cols]
    print(f"Pruned {len(dropped)} columns, kept {len(keep_cols)} (including recording_id/label).")
    return df[keep_cols]


# --- data prep --------------------------------------------------------------

def load_and_split():
    df = pd.read_csv(FEATURE_TABLE_PATH)
    df = prune_feature_table(df)

    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, stratify=df["label"], random_state=RANDOM_STATE
    )

    train_scaled, test_scaled, scaler = standardize_features(train_df, test_df)
    return train_scaled, test_scaled, scaler


def phase_columns(df: pd.DataFrame, phase: str) -> list:
    prefix = f"{phase}_"
    return [c for c in df.columns if c.startswith(prefix)]


# --- fitting (identical logic to arm_a.py) ---------------------------------

def fit_all_gmms(train_df: pd.DataFrame) -> dict:
    gmms = {}
    for cls in CLASSES:
        gmms[cls] = {}
        class_rows = train_df[train_df["label"] == cls]

        for phase in PHASES:
            cols = phase_columns(train_df, phase)
            X = class_rows[cols].values

            n_nan = np.isnan(X).sum()
            if n_nan > 0:
                print(f"  WARNING: {cls}/{phase} has {n_nan} NaNs -- dropping those rows.")
                mask = ~np.isnan(X).any(axis=1)
                X = X[mask]

            gmm = GaussianMixture(
                n_components=N_COMPONENTS,
                covariance_type=COVARIANCE_TYPE,
                random_state=RANDOM_STATE,
            )
            gmm.fit(X)
            gmms[cls][phase] = gmm
            print(f"  fitted {cls}/{phase} on {len(X)} recordings, {len(cols)} features")

    return gmms


# --- classification (identical logic to arm_a.py) ---------------------------

def classify_recording(row: pd.Series, gmms: dict, phase_cols_cache: dict) -> dict:
    per_class_per_phase = {cls: {} for cls in CLASSES}
    per_class_total = {}
    missing_phases = []

    for cls in CLASSES:
        total_ll = 0.0
        any_phase_used = False

        for phase in PHASES:
            cols = phase_cols_cache[phase]
            x = np.asarray(row[cols].values, dtype=float).reshape(1, -1)

            if np.isnan(x).any():
                per_class_per_phase[cls][phase] = None
                if phase not in missing_phases:
                    missing_phases.append(phase)
                continue

            ll = gmms[cls][phase].score_samples(x)[0]
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


# --- main -------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Loading {FEATURE_TABLE_PATH}")
    train_df, test_df, scaler = load_and_split()
    print(f"Train: {len(train_df)}  Test: {len(test_df)}")

    print(f"\nFitting GMMs (pruned features, n_components={N_COMPONENTS}, "
          f"covariance_type={COVARIANCE_TYPE})...")
    gmms = fit_all_gmms(train_df)

    phase_cols_cache = {phase: phase_columns(test_df, phase) for phase in PHASES}

    print("\nClassifying test recordings...")
    per_recording_results = []
    for _, row in test_df.iterrows():
        result = classify_recording(row, gmms, phase_cols_cache)
        per_recording_results.append(result)

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
    print("ARM A -- PRUNED FEATURES -- RESULTS")
    print("=" * 60)
    print(f"Overall accuracy: {accuracy * 100:.1f}%  (full-feature baseline was 90.5%)")
    print("\nPer-class accuracy:")
    for cls, acc in per_class_accuracy.items():
        print(f"  {cls}: {acc * 100:.1f}%")
    print("\nConfusion matrix:")
    print(confusion)

    output = {
        "per_recording_results": per_recording_results,
        "accuracy": accuracy,
        "per_class_accuracy": per_class_accuracy,
        "confusion_matrix": confusion,
        "gmms": gmms,
        "config": {
            "n_components": N_COMPONENTS,
            "covariance_type": COVARIANCE_TYPE,
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "classes": CLASSES,
            "phases": PHASES,
            "kept_mfcc_indices": KEPT_MFCC_INDICES,
        },
        "train_recording_ids": train_df["recording_id"].tolist(),
        "test_recording_ids": test_df["recording_id"].tolist(),
    }

    with open(RESULTS_PATH, "wb") as f:
        pickle.dump(output, f)

    print(f"\nFull results saved to {RESULTS_PATH}")