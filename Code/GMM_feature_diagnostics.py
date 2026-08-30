"""
feature_diagnostics.py

Two complementary analyses to find out which features carry real
signal vs which are noise:

1. ANOVA F-statistic per feature per phase (train data) -- how well
   does this single feature separate the 5 classes, independent of
   the model.
2. Permutation importance per feature (test data, using the actual
   fitted GMMs from arm_a_results.pkl) -- shuffle one feature at a
   time and measure the resulting drop in the REAL classifier's
   accuracy, both overall and broken down per true class.

Everything gets saved to CSV with full per-feature detail, plus
aggregated summaries by feature category (mfcc/spectral/dwt/temporal),
so there's a lot to paste back for interpretation.
"""

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from scipy.stats import f_oneway
from sklearn.model_selection import train_test_split

from extract_features import standardize_features

# --- config -----------------------------------------------------------

THIS_DIR = Path(__file__).resolve().parent
ARM_A_RESULTS_PATH = THIS_DIR / "GMM_results.pkl"
FEATURE_TABLE_PATH = THIS_DIR / "pcg_feature_table.csv"
OUTPUT_DIR = THIS_DIR / "feature_diagnostics"
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42  # must match arm_a.py

# Feature category keyword map -- used to aggregate importance by type.
CATEGORY_KEYWORDS = {
    "mfcc": ["mfcc"],
    "spectral": ["spec_centroid", "spec_rolloff", "spec_flatness"],
    "dwt": ["dwt_energy"],
    "temporal_amplitude": ["duration_sec", "amp_mean", "amp_max", "amp_std"],
}


def categorize_feature(feature_name: str) -> str:
    """feature_name is like 'mfcc_3_mean' (phase prefix already stripped)."""
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in feature_name for kw in keywords):
            return category
    return "other"


# --- load / rebuild -----------------------------------------------------

def load_results():
    with open(ARM_A_RESULTS_PATH, "rb") as f:
        return pickle.load(f)


def rebuild_split(results):
    df = pd.read_csv(FEATURE_TABLE_PATH)
    cfg = results["config"]
    train_df, test_df = train_test_split(
        df, test_size=cfg["test_size"], stratify=df["label"],
        random_state=cfg["random_state"],
    )
    train_scaled, test_scaled, _ = standardize_features(train_df, test_df)

    assert set(train_scaled["recording_id"]) == set(results["train_recording_ids"]), (
        "Rebuilt split doesn't match arm_a_results.pkl -- check random_state/test_size."
    )
    return train_scaled, test_scaled


def phase_columns(df: pd.DataFrame, phase: str) -> list:
    prefix = f"{phase}_"
    return [c for c in df.columns if c.startswith(prefix)]


# --- 1. ANOVA F-stat per feature per phase --------------------------------

def compute_anova(train_df: pd.DataFrame, phases: tuple, classes: tuple) -> pd.DataFrame:
    rows = []
    for phase in phases:
        cols = phase_columns(train_df, phase)
        for col in cols:
            groups = []
            for cls in classes:
                vals = train_df.loc[train_df["label"] == cls, col].dropna().values
                if len(vals) > 0:
                    groups.append(vals)

            if len(groups) < 2:
                continue

            f_stat, p_val = f_oneway(*groups)
            feature_name = col[len(phase) + 1:]  # strip "Phase_" prefix

            rows.append({
                "phase": phase,
                "feature": feature_name,
                "full_column": col,
                "category": categorize_feature(feature_name),
                "f_statistic": f_stat,
                "p_value": p_val,
            })

    return pd.DataFrame(rows)


# --- 2. Permutation importance on the REAL classifier ---------------------

def classify_recording_local(row: pd.Series, gmms: dict, phase_cols_cache: dict,
                              classes: tuple, phases: tuple) -> str:
    """Self-contained reimplementation of arm_a.py's classify_recording,
    so this script doesn't depend on arm_a.py's fragile module-level
    global (phase_columns_cache) that only gets set inside its
    __main__ block."""
    per_class_total = {}

    for cls in classes:
        total_ll = 0.0
        any_phase_used = False

        for phase in phases:
            cols = phase_cols_cache[phase]
            x = np.asarray(row[cols].values, dtype=float).reshape(1, -1)

            if np.isnan(x).any():
                continue

            ll = gmms[cls][phase].score_samples(x)[0]
            total_ll += ll
            any_phase_used = True

        per_class_total[cls] = total_ll if any_phase_used else float("-inf")

    return max(per_class_total, key=lambda k: per_class_total[k])


def baseline_accuracy(test_df: pd.DataFrame, gmms: dict, phase_cols_cache: dict,
                       classes: tuple, phases: tuple) -> tuple:
    """Returns (overall_accuracy, per_class_accuracy dict)."""
    preds = []
    for _, row in test_df.iterrows():
        preds.append(classify_recording_local(row, gmms, phase_cols_cache, classes, phases))

    y_true = test_df["label"].values
    y_pred = np.array(preds)

    overall = np.mean(y_true == y_pred)

    per_class = {}
    for cls in classes:
        mask = y_true == cls
        per_class[cls] = np.mean(y_pred[mask] == cls) if mask.sum() > 0 else float("nan")

    return overall, per_class


def permutation_importance(test_df: pd.DataFrame, gmms: dict, phase_cols_cache: dict,
                            classes: tuple, phases: tuple,
                            baseline_overall: float, baseline_per_class: dict,
                            n_repeats: int = 5, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.RandomState(random_state)
    rows = []

    for phase in phases:
        cols = phase_cols_cache[phase]

        for col in cols:
            feature_name = col[len(phase) + 1:]
            overall_drops = []
            per_class_drops = {cls: [] for cls in classes}

            for repeat in range(n_repeats):
                shuffled_df = test_df.copy()
                shuffled_vals = shuffled_df[col].values.copy()
                rng.shuffle(shuffled_vals)
                shuffled_df[col] = shuffled_vals

                acc, per_class_acc = baseline_accuracy(
                    shuffled_df, gmms, phase_cols_cache, classes, phases
                )
                overall_drops.append(baseline_overall - acc)
                for cls in classes:
                    per_class_drops[cls].append(baseline_per_class[cls] - per_class_acc[cls])

            row = {
                "phase": phase,
                "feature": feature_name,
                "full_column": col,
                "category": categorize_feature(feature_name),
                "mean_overall_accuracy_drop": np.mean(overall_drops),
                "std_overall_accuracy_drop": np.std(overall_drops),
            }
            for cls in classes:
                row[f"mean_accuracy_drop_{cls}"] = np.mean(per_class_drops[cls])

            rows.append(row)
            print(f"  {phase}/{feature_name}: overall drop = {row['mean_overall_accuracy_drop']:.4f}")

    return pd.DataFrame(rows)


# --- category-level aggregation --------------------------------------------

def aggregate_by_category(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    return (
        df.groupby(["phase", "category"])[value_col]
        .agg(["mean", "std", "count"])
        .reset_index()
    )


# --- main -------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading results and rebuilding split...")
    results = load_results()
    train_df, test_df = rebuild_split(results)

    classes = results["config"]["classes"]
    phases = results["config"]["phases"]
    gmms = results["gmms"]
    phase_cols_cache = {phase: phase_columns(test_df, phase) for phase in phases}

    # --- Part 1: ANOVA ---
    print("\nComputing ANOVA F-statistics (train data)...")
    anova_df = compute_anova(train_df, phases, classes)
    anova_df = anova_df.sort_values("f_statistic", ascending=False)
    anova_df.to_csv(OUTPUT_DIR / "anova_full.csv", index=False)

    anova_category_summary = aggregate_by_category(anova_df, "f_statistic")
    anova_category_summary.to_csv(OUTPUT_DIR / "anova_by_category.csv", index=False)

    print("\nTop 10 features by ANOVA F-statistic (all phases):")
    print(anova_df.head(10).to_string(index=False))
    print("\nBottom 10 features by ANOVA F-statistic (all phases, likely noisiest):")
    print(anova_df.tail(10).to_string(index=False))
    print("\nANOVA F-statistic by category (mean, averaged across all features in that category/phase):")
    print(anova_category_summary.to_string(index=False))

    # --- Part 2: Permutation importance ---
    print("\nComputing baseline accuracy...")
    baseline_overall, baseline_per_class = baseline_accuracy(
        test_df, gmms, phase_cols_cache, classes, phases
    )
    print(f"Baseline overall accuracy: {baseline_overall*100:.1f}%")
    print(f"Baseline per-class accuracy: {baseline_per_class}")

    print("\nRunning permutation importance (this will take a while -- "
          f"{sum(len(v) for v in phase_cols_cache.values())} features x 5 repeats)...")
    perm_df = permutation_importance(
        test_df, gmms, phase_cols_cache, classes, phases,
        baseline_overall, baseline_per_class, n_repeats=5,
    )
    perm_df = perm_df.sort_values("mean_overall_accuracy_drop", ascending=False)
    perm_df.to_csv(OUTPUT_DIR / "permutation_importance_full.csv", index=False)

    perm_category_summary = aggregate_by_category(perm_df, "mean_overall_accuracy_drop")
    perm_category_summary.to_csv(OUTPUT_DIR / "permutation_importance_by_category.csv", index=False)

    print("\nTop 15 features by permutation importance (most helpful, all phases):")
    print(perm_df.head(15)[["phase", "feature", "category", "mean_overall_accuracy_drop"]].to_string(index=False))
    print("\nBottom 15 features by permutation importance (least helpful / possibly noise):")
    print(perm_df.tail(15)[["phase", "feature", "category", "mean_overall_accuracy_drop"]].to_string(index=False))
    print("\nPermutation importance by category (mean accuracy drop when shuffled):")
    print(perm_category_summary.to_string(index=False))

    # --- combined summary (ANOVA rank vs permutation rank, per feature) ---
    combined = anova_df.merge(
        perm_df[["phase", "feature", "mean_overall_accuracy_drop"] +
                [f"mean_accuracy_drop_{cls}" for cls in classes]],
        on=["phase", "feature"], how="outer",
    )
    combined.to_csv(OUTPUT_DIR / "combined_anova_and_permutation.csv", index=False)

    with open(OUTPUT_DIR / "summary.pkl", "wb") as f:
        pickle.dump({
            "anova_df": anova_df,
            "perm_df": perm_df,
            "anova_category_summary": anova_category_summary,
            "perm_category_summary": perm_category_summary,
            "baseline_overall": baseline_overall,
            "baseline_per_class": baseline_per_class,
        }, f)

    print(f"\nAll diagnostics saved to {OUTPUT_DIR}")