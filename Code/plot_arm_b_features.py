"""
plot_arm_b_results.py

Loads arm_b_results.pkl and generates a full set of figures and numeric
summaries for Arm B (MLP) interpretability analysis, mirroring
plot_gmm_results.py's structure so the two arms are directly comparable.

Two sections differ from Arm A because they're GMM-specific concepts
with no MLP equivalent:
  - "06 density overlay" -> replaced with per-phase SHAP importance bars
  - "07 component separation" -> replaced with overall phase-importance
    ranking (which phase the MLP leans on most, aggregated across SHAP)

Everything else (confusion matrix, per-class accuracy, phase
contribution, margin distribution, per-phase standalone accuracy) uses
the same logic as Arm A so the two are apples-to-apples.

Saves into ./arm_b_figures/ -- figures as .png, tables as .csv/.pkl.
"""

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from extract_features import standardize_features
from sklearn.model_selection import train_test_split

# --- config ---------------------------------------------------------------

THIS_DIR = Path(__file__).resolve().parent
ARM_B_RESULTS_PATH = THIS_DIR / "arm_b_results.pkl"
FEATURE_TABLE_PATH = THIS_DIR / "pcg_feature_table.csv"

OUTPUT_DIR = THIS_DIR / "arm_b_figures"
OUTPUT_DIR.mkdir(exist_ok=True)


# --- loading ----------------------------------------------------------------

def load_results():
    with open(ARM_B_RESULTS_PATH, "rb") as f:
        results = pickle.load(f)
    return results


def rebuild_train_test(results):
    """
    Arm B already saved the exact recording_ids it trained/tested on
    (reused from Arm A's pruned split), so rebuild the standardized
    dataframes by filtering on those ids directly -- no need to re-derive
    the split via train_test_split like arm_a's plotting script does.
    """
    df = pd.read_csv(FEATURE_TABLE_PATH)
    train_ids = set(results["train_recording_ids"])
    test_ids = set(results["test_recording_ids"])

    train_df = df[df["recording_id"].isin(train_ids)].reset_index(drop=True)
    test_df = df[df["recording_id"].isin(test_ids)].reset_index(drop=True)

    assert len(train_df) == len(train_ids), "Train split size mismatch -- check recording_id overlap."
    assert len(test_df) == len(test_ids), "Test split size mismatch -- check recording_id overlap."

    train_scaled, test_scaled, _ = standardize_features(train_df, test_df)
    return train_scaled, test_scaled


# --- 1. Confusion matrix heatmap ------------------------------------------

def plot_confusion_matrix(results):
    cm = results["confusion_matrix"]
    classes = results["config"]["classes"]
    cm = cm.reindex(index=classes, columns=classes, fill_value=0)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm.values, cmap="Oranges")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Arm B (MLP) Confusion Matrix — Accuracy: {results['accuracy']*100:.1f}%")

    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm.values[i, j], ha="center", va="center",
                     color="white" if cm.values[i, j] > cm.values.max() / 2 else "black")

    fig.colorbar(im, ax=ax, label="count")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_confusion_matrix.png", dpi=150)
    plt.close(fig)


# --- 2. Per-class accuracy bar chart ---------------------------------------

def plot_per_class_accuracy(results):
    acc = results["per_class_accuracy"]
    classes = list(acc.keys())
    values = [acc[c] * 100 for c in classes]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(classes, values, color="darkorange")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Arm B (MLP) — Per-Class Accuracy")
    ax.axhline(results["accuracy"] * 100, color="red", linestyle="--",
               label=f"Overall: {results['accuracy']*100:.1f}%")
    ax.legend()
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1, f"{v:.1f}%", ha="center")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_per_class_accuracy.png", dpi=150)
    plt.close(fig)


# --- 3 & 4. Per-phase log-prob contribution, correct vs incorrect ----------

def phase_contribution_table(results):
    """
    Same logic as Arm A's version: each phase's relative share of the
    predicted class's total. Arm B's per-phase values are already
    log-softmax outputs (true log-probabilities, not arbitrary-scale
    log-likelihoods), but the same softmax-style normalization is kept
    so the "relative contribution" numbers are computed identically
    across arms and stay comparable.
    """
    rows = []
    for rec in results["per_recording_results"]:
        pred = rec["predicted_label"]
        true = rec["true_label"]
        correct = pred == true
        phase_lps = rec["per_class_per_phase_logprob"][pred]

        valid = {p: v for p, v in phase_lps.items() if v is not None}
        if not valid:
            continue

        vals = np.array(list(valid.values()))
        weights = np.exp(vals - vals.max())
        weights = weights / weights.sum()

        for phase, w in zip(valid.keys(), weights):
            rows.append({
                "recording_id": rec["recording_id"],
                "true_label": true,
                "predicted_label": pred,
                "correct": correct,
                "phase": phase,
                "relative_contribution": w,
            })

    return pd.DataFrame(rows)


def plot_phase_contribution(contribution_df, correct: bool, filename: str, title: str):
    subset = contribution_df[contribution_df["correct"] == correct]
    if subset.empty:
        print(f"  (skipping {filename} -- no recordings in this subset)")
        return

    pivot = subset.groupby(["true_label", "phase"])["relative_contribution"].mean().unstack()

    fig, ax = plt.subplots(figsize=(7, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("Mean relative phase contribution")
    ax.set_xlabel("True class")
    ax.set_title(title)
    ax.legend(title="Phase", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=150)
    plt.close(fig)


# --- 5. Prediction margin (winner vs runner-up), correct vs incorrect ------

def margin_table(results):
    rows = []
    for rec in results["per_recording_results"]:
        totals = rec["per_class_total_logprob"]
        sorted_vals = sorted(totals.values(), reverse=True)
        if len(sorted_vals) < 2 or sorted_vals[0] == float("-inf"):
            continue
        margin = sorted_vals[0] - sorted_vals[1]
        rows.append({
            "recording_id": rec["recording_id"],
            "true_label": rec["true_label"],
            "predicted_label": rec["predicted_label"],
            "correct": rec["predicted_label"] == rec["true_label"],
            "margin": margin,
        })
    return pd.DataFrame(rows)


def plot_margin_distribution(margin_df):
    fig, ax = plt.subplots(figsize=(7, 5))
    correct = margin_df[margin_df["correct"]]["margin"]
    incorrect = margin_df[~margin_df["correct"]]["margin"]

    ax.hist(correct, bins=30, alpha=0.6, label=f"Correct (n={len(correct)})", color="green")
    ax.hist(incorrect, bins=30, alpha=0.6, label=f"Incorrect (n={len(incorrect)})", color="red")
    ax.set_xlabel("Log-prob margin (winner - runner-up)")
    ax.set_ylabel("Count")
    ax.set_title("Arm B — Prediction Margin: Correct vs Incorrect")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "05_margin_distribution.png", dpi=150)
    plt.close(fig)


# --- 6. SHAP-based per-phase feature importance (replaces density overlay) -

def plot_shap_importance_per_phase(results):
    """
    Arm A's density overlay showed where GMM component means sit
    relative to the training data for a phase's top 2 features. Arm B
    has no analogous "component" concept, so this shows the MLP
    equivalent: per-phase feature importance from SHAP, aggregated
    across classes (mean |SHAP value|).
    """
    shap_results = results["shap_results"]
    top_features_by_phase = {}

    for phase, phase_data in shap_results.items():
        shap_values = phase_data["shap_values"]  # (n_samples, n_features, n_classes)
        feature_names = phase_data["feature_names"]

        mean_abs_shap = np.abs(shap_values).mean(axis=(0, 2))
        order = np.argsort(mean_abs_shap)[::-1]
        sorted_features = [feature_names[i] for i in order]
        sorted_importance = mean_abs_shap[order]
        top_features_by_phase[phase] = list(zip(sorted_features, sorted_importance))

        fig, ax = plt.subplots(figsize=(6, max(4, len(feature_names) * 0.25)))
        ax.barh(sorted_features[::-1], sorted_importance[::-1], color="darkorange")
        ax.set_xlabel("Mean |SHAP value| (across classes)")
        ax.set_title(f"{phase} — SHAP feature importance")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"06_shap_importance_{phase}.png", dpi=150)
        plt.close(fig)

    return top_features_by_phase


# --- 7. Overall phase importance ranking (replaces component separation) ---

def phase_importance_table(results):
    """
    Arm A's component-separation plot showed which phase's GMM found
    the most distinct per-class structure. This is the MLP analog:
    total SHAP magnitude per phase, summed across all features and
    classes -- a rough proxy for "how much this phase drives the
    model's decisions overall", so you can state directly whether the
    two arms agree on which phase matters most (a convergent-evidence
    point for the paper, same as the earlier Diastole finding).
    """
    rows = []
    for phase, phase_data in results["shap_results"].items():
        shap_values = phase_data["shap_values"]
        total_importance = np.abs(shap_values).mean()
        rows.append({"phase": phase, "mean_abs_shap": total_importance})
    return pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False)


def plot_phase_importance(phase_importance_df):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(phase_importance_df["phase"], phase_importance_df["mean_abs_shap"], color="darkorange")
    ax.set_ylabel("Mean |SHAP value| (all features, all classes)")
    ax.set_title("Arm B — Overall Phase Importance")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "07_phase_importance.png", dpi=150)
    plt.close(fig)


# --- 8. Per-phase-only prediction accuracy ---------------------------------

def per_phase_accuracy_table(results):
    """
    Same logic as Arm A: classify using ONLY one phase's log-prob at a
    time and measure standalone accuracy, broken down per true class too.
    """
    phases = results["config"]["phases"]
    classes = results["config"]["classes"]

    rows = []
    for rec in results["per_recording_results"]:
        true = rec["true_label"]
        per_class_per_phase = rec["per_class_per_phase_logprob"]

        for phase in phases:
            phase_lps = {}
            for cls in classes:
                v = per_class_per_phase[cls][phase]
                if v is not None:
                    phase_lps[cls] = v

            if not phase_lps:
                continue

            phase_pred = max(phase_lps, key=lambda k: phase_lps[k])

            rows.append({
                "recording_id": rec["recording_id"],
                "true_label": true,
                "phase": phase,
                "phase_predicted_label": phase_pred,
                "correct": phase_pred == true,
            })

    detail_df = pd.DataFrame(rows)
    overall = detail_df.groupby("phase")["correct"].mean().rename("accuracy")
    per_class = (
        detail_df.groupby(["true_label", "phase"])["correct"]
        .mean()
        .unstack()
    )

    return detail_df, overall, per_class


def plot_per_phase_accuracy(overall_series, per_class_df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    overall_series.plot(kind="bar", ax=ax, color="darkorange")
    ax.set_ylabel("Accuracy (using this phase alone)")
    ax.set_ylim(0, 1)
    ax.set_title("Per-Phase Standalone Accuracy (all classes)")
    for i, v in enumerate(overall_series):
        ax.text(i, v + 0.02, f"{v*100:.1f}%", ha="center")

    ax2 = axes[1]
    per_class_df.plot(kind="bar", ax=ax2)
    ax2.set_ylabel("Accuracy (using this phase alone)")
    ax2.set_ylim(0, 1)
    ax2.set_title("Per-Phase Standalone Accuracy, by True Class")
    ax2.legend(title="Phase", bbox_to_anchor=(1.02, 1), loc="upper left")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "08_per_phase_accuracy.png", dpi=150)
    plt.close(fig)


# --- overall + per-class accuracy summary CSV ---

def save_accuracy_summary_csv(results, output_path):
    rows = [{"class": "OVERALL", "accuracy": results["accuracy"]}]
    for cls, acc in results["per_class_accuracy"].items():
        rows.append({"class": cls, "accuracy": acc})
    pd.DataFrame(rows).to_csv(output_path, index=False)


# --- main -------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading results...")
    results = load_results()

    save_accuracy_summary_csv(results, OUTPUT_DIR / "accuracy_summary.csv")

    print("Generating figures...")

    print("  01: confusion matrix")
    plot_confusion_matrix(results)

    print("  02: per-class accuracy")
    plot_per_class_accuracy(results)

    print("  03/04: phase contribution (correct vs incorrect)")
    contribution_df = phase_contribution_table(results)
    contribution_df.to_csv(OUTPUT_DIR / "phase_contribution_data.csv", index=False)
    plot_phase_contribution(
        contribution_df, correct=True,
        filename="03_phase_contribution_correct.png",
        title="Mean Phase Contribution — Correct Predictions",
    )
    plot_phase_contribution(
        contribution_df, correct=False,
        filename="04_phase_contribution_incorrect.png",
        title="Mean Phase Contribution — Incorrect Predictions",
    )

    print("  05: margin distribution")
    margin_df = margin_table(results)
    margin_df.to_csv(OUTPUT_DIR / "margin_data.csv", index=False)
    plot_margin_distribution(margin_df)

    print("  06: SHAP importance per phase")
    top_features_by_phase = plot_shap_importance_per_phase(results)
    with open(OUTPUT_DIR / "shap_top_features_by_phase.pkl", "wb") as f:
        pickle.dump(top_features_by_phase, f)

    print("  07: overall phase importance")
    phase_imp_df = phase_importance_table(results)
    phase_imp_df.to_csv(OUTPUT_DIR / "phase_importance_data.csv", index=False)
    plot_phase_importance(phase_imp_df)

    print("  08: per-phase standalone accuracy")
    phase_acc_detail, phase_acc_overall, phase_acc_per_class = per_phase_accuracy_table(results)
    phase_acc_detail.to_csv(OUTPUT_DIR / "per_phase_accuracy_detail.csv", index=False)
    phase_acc_overall.to_csv(OUTPUT_DIR / "per_phase_accuracy_overall.csv")
    phase_acc_per_class.to_csv(OUTPUT_DIR / "per_phase_accuracy_by_class.csv")
    plot_per_phase_accuracy(phase_acc_overall, phase_acc_per_class)

    print("\nPer-phase standalone accuracy:")
    print(phase_acc_overall)
    print("\nOverall phase importance (SHAP):")
    print(phase_imp_df)

    summary = {
        "contribution_df": contribution_df,
        "margin_df": margin_df,
        "phase_importance_df": phase_imp_df,
        "accuracy": results["accuracy"],
        "per_class_accuracy": results["per_class_accuracy"],
    }
    with open(OUTPUT_DIR / "summary.pkl", "wb") as f:
        pickle.dump(summary, f)

    print(f"\nAll figures and data saved to {OUTPUT_DIR}")