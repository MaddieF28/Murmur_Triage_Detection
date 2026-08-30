"""
plot_gmm_results.py

Loads arm_a_results.pkl (and the original feature table, to get training
data for the density overlay plots) and generates a full set of figures
and numeric summaries for Arm A (GMM) interpretability analysis.

Everything gets saved into ./gmm_results/ -- figures as .png, and the
underlying numeric tables as .csv/.pkl so plots can be rebuilt later
without recomputing.
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
ARM_A_RESULTS_PATH = THIS_DIR / "GMM_results.pkl"
FEATURE_TABLE_PATH = THIS_DIR / "pcg_feature_table.csv"

OUTPUT_DIR = THIS_DIR / "GMM_figures"
OUTPUT_DIR.mkdir(exist_ok=True)


# --- loading ----------------------------------------------------------------

def load_results():
    with open(ARM_A_RESULTS_PATH, "rb") as f:
        results = pickle.load(f)
    return results


def rebuild_train_data(results):
    """
    Recreate the exact same train/test split used in arm_a.py (same
    random_state + stratify), then standardize the same way, so we have
    the actual training feature values to overlay against the fitted
    GMM densities in the density plots.
    """
    df = pd.read_csv(FEATURE_TABLE_PATH)
    cfg = results["config"]

    train_df, test_df = train_test_split(
        df, test_size=cfg["test_size"], stratify=df["label"],
        random_state=cfg["random_state"],
    )
    train_scaled, test_scaled, _ = standardize_features(train_df, test_df)

    # Sanity check this actually matches the saved split.
    assert set(train_scaled["recording_id"]) == set(results["train_recording_ids"]), (
        "Rebuilt train split doesn't match arm_a_results.pkl's saved split -- "
        "check random_state/test_size match."
    )
    return train_scaled, test_scaled


# --- 1. Confusion matrix heatmap ------------------------------------------

def plot_confusion_matrix(results):
    cm = results["confusion_matrix"]
    classes = results["config"]["classes"]
    cm = cm.reindex(index=classes, columns=classes, fill_value=0)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm.values, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Arm A (GMM) Confusion Matrix — Accuracy: {results['accuracy']*100:.1f}%")

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
    bars = ax.bar(classes, values, color="steelblue")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Arm A (GMM) — Per-Class Accuracy")
    ax.axhline(results["accuracy"] * 100, color="red", linestyle="--",
               label=f"Overall: {results['accuracy']*100:.1f}%")
    ax.legend()
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1, f"{v:.1f}%", ha="center")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_per_class_accuracy.png", dpi=150)
    plt.close(fig)


# --- 3 & 4. Per-phase log-likelihood contribution, correct vs incorrect ----

def phase_contribution_table(results):
    """
    For every test recording, compute each phase's share of the
    *predicted* class's total log-likelihood (normalized so phases are
    comparable in relative terms, since raw log-likelihood scale can
    vary a lot recording to recording). Returns a long-form DataFrame:
    one row per (recording, phase), with columns for true/pred label,
    correctness, and that phase's relative contribution.
    """
    rows = []
    for rec in results["per_recording_results"]:
        pred = rec["predicted_label"]
        true = rec["true_label"]
        correct = pred == true
        phase_lls = rec["per_class_per_phase_loglik"][pred]

        # Only use phases that weren't missing for this recording.
        valid = {p: v for p, v in phase_lls.items() if v is not None}
        if not valid:
            continue

        # Relative contribution: shift so the least-negative phase is
        # comparable -- use softmax-style normalization over the phases'
        # log-likelihoods so contributions sum to 1 and are always
        # positive/comparable regardless of the absolute log-likelihood
        # scale for this particular recording.
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
        totals = rec["per_class_total_loglik"]
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
    ax.set_xlabel("Log-likelihood margin (winner - runner-up)")
    ax.set_ylabel("Count")
    ax.set_title("Arm A — Prediction Margin: Correct vs Incorrect")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "05_margin_distribution.png", dpi=150)
    plt.close(fig)


# --- 6. Fitted GMM density overlay on training data -------------------------

def most_informative_features(gmms_for_phase: dict, phase_cols: list) -> list:
    """
    Rank phase_cols by how much the class-wise GMM component means vary
    across classes for that feature -- a proxy for which features most
    distinguish classes within this phase. Returns top 3 feature names.
    """
    class_means = []
    for cls, gmm in gmms_for_phase.items():
        # average across components, weighted by component weight
        weighted_mean = (gmm.means_ * gmm.weights_[:, None]).sum(axis=0)
        class_means.append(weighted_mean)
    class_means = np.array(class_means)  # shape (n_classes, n_features)

    variance_across_classes = class_means.var(axis=0)
    top_idx = np.argsort(variance_across_classes)[::-1][:3]
    return [phase_cols[i] for i in top_idx]


def plot_density_overlay(results, train_df):
    gmms = results["gmms"]
    classes = results["config"]["classes"]
    phases = results["config"]["phases"]

    for phase in phases:
        cols = [c for c in train_df.columns if c.startswith(f"{phase}_")]
        gmms_for_phase = {cls: gmms[cls][phase] for cls in classes}
        top_features = most_informative_features(gmms_for_phase, cols)

        if len(top_features) < 2:
            continue
        f1, f2 = top_features[0], top_features[1]
        f1_idx = cols.index(f1)
        f2_idx = cols.index(f2)

        fig, ax = plt.subplots(figsize=(7, 6))
        colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(classes)))

        for cls, color in zip(classes, colors):
            class_rows = train_df[train_df["label"] == cls]
            X = class_rows[cols].values
            X = X[~np.isnan(X).any(axis=1)]  # drop missing-phase rows

            ax.scatter(X[:, f1_idx], X[:, f2_idx], s=10, alpha=0.4,
                       color=color, label=f"{cls} (data)")

            gmm = gmms_for_phase[cls]
            for mean in gmm.means_:
                ax.scatter(mean[f1_idx], mean[f2_idx], marker="x", s=150,
                           color=color, linewidths=3)

        ax.set_xlabel(f1)
        ax.set_ylabel(f2)
        ax.set_title(f"{phase} — GMM component means (x) over training data")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"06_density_overlay_{phase}.png", dpi=150)
        plt.close(fig)


# --- 7. Component weight/mean separation summary ---------------------------

def component_separation_table(results):
    """
    For each class x phase GMM, measures how separated its components
    are (Euclidean distance between the 2 component means in
    standardized feature space) and each component's weight -- tells
    you whether n_components=2 was actually capturing real structure
    or whether the two components collapsed near-identical.
    """
    rows = []
    gmms = results["gmms"]
    for cls, phase_dict in gmms.items():
        for phase, gmm in phase_dict.items():
            if gmm.n_components < 2:
                continue
            dist = np.linalg.norm(gmm.means_[0] - gmm.means_[1])
            rows.append({
                "class": cls,
                "phase": phase,
                "component_0_weight": gmm.weights_[0],
                "component_1_weight": gmm.weights_[1],
                "mean_separation_distance": dist,
            })
    return pd.DataFrame(rows)


def plot_component_separation(sep_df):
    fig, ax = plt.subplots(figsize=(9, 5))
    pivot = sep_df.pivot(index="class", columns="phase", values="mean_separation_distance")
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("Component mean separation (Euclidean dist, standardized space)")
    ax.set_title("GMM Component Separation — larger = more distinct subtypes found")
    ax.legend(title="Phase", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "07_component_separation.png", dpi=150)
    plt.close(fig)

# --- 8. Per-phase-only prediction accuracy ---------------------------------

def per_phase_accuracy_table(results):
    """
    For each phase in isolation, classify using ONLY that phase's
    log-likelihood (ignore the other 3 phases) and see how accurate
    that single phase is on its own. This is different from the
    contribution-share plots (03/04) -- those show how much each phase
    influenced the *combined* prediction; this shows how good each
    phase would be as a standalone classifier. Also breaks it down per
    true class, so you can see e.g. whether Diastole alone is enough to
    catch MS specifically (matching your clinically-expected-phase
    intuition), even if it's not the best phase overall.
    """
    phases = results["config"]["phases"]
    classes = results["config"]["classes"]

    rows = []
    for rec in results["per_recording_results"]:
        true = rec["true_label"]
        per_class_per_phase = rec["per_class_per_phase_loglik"]

        for phase in phases:
            # Gather this phase's log-likelihood under every class.
            phase_lls = {}
            for cls in classes:
                v = per_class_per_phase[cls][phase]
                if v is not None:
                    phase_lls[cls] = v

            if not phase_lls:
                continue  # this phase was missing for this recording entirely

            phase_pred = max(phase_lls, key=lambda k: phase_lls[k])

            rows.append({
                "recording_id": rec["recording_id"],
                "true_label": true,
                "phase": phase,
                "phase_predicted_label": phase_pred,
                "correct": phase_pred == true,
            })

    detail_df = pd.DataFrame(rows)

    # Overall accuracy per phase (across all classes).
    overall = detail_df.groupby("phase")["correct"].mean().rename("accuracy")

    # Accuracy per phase, per true class -- shows which phase is most
    # discriminative for which specific class.
    per_class = (
        detail_df.groupby(["true_label", "phase"])["correct"]
        .mean()
        .unstack()
    )

    return detail_df, overall, per_class


def plot_per_phase_accuracy(overall_series, per_class_df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Overall per-phase accuracy.
    ax = axes[0]
    overall_series.plot(kind="bar", ax=ax, color="steelblue")
    ax.set_ylabel("Accuracy (using this phase alone)")
    ax.set_ylim(0, 1)
    ax.set_title("Per-Phase Standalone Accuracy (all classes)")
    for i, v in enumerate(overall_series):
        ax.text(i, v + 0.02, f"{v*100:.1f}%", ha="center")

    # Per-class x per-phase accuracy.
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

    print("Rebuilding training data for density overlays...")
    train_df, test_df = rebuild_train_data(results)

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

    print("  06: density overlays (per phase)")
    plot_density_overlay(results, train_df)

    print("  07: component separation")
    sep_df = component_separation_table(results)
    sep_df.to_csv(OUTPUT_DIR / "component_separation_data.csv", index=False)
    plot_component_separation(sep_df)

    print("  08: per-phase standalone accuracy")
    phase_acc_detail, phase_acc_overall, phase_acc_per_class = per_phase_accuracy_table(results)
    phase_acc_detail.to_csv(OUTPUT_DIR / "per_phase_accuracy_detail.csv", index=False)
    phase_acc_overall.to_csv(OUTPUT_DIR / "per_phase_accuracy_overall.csv")
    phase_acc_per_class.to_csv(OUTPUT_DIR / "per_phase_accuracy_by_class.csv")
    plot_per_phase_accuracy(phase_acc_overall, phase_acc_per_class)

    print("\nPer-phase standalone accuracy:")
    print(phase_acc_overall)

    # Save a combined summary pickle too, in case you want to rebuild
    # anything later without re-running arm_a.py.
    summary = {
        "contribution_df": contribution_df,
        "margin_df": margin_df,
        "component_separation_df": sep_df,
        "accuracy": results["accuracy"],
        "per_class_accuracy": results["per_class_accuracy"],
    }
    with open(OUTPUT_DIR / "summary.pkl", "wb") as f:
        pickle.dump(summary, f)

    print(f"\nAll figures and data saved to {OUTPUT_DIR}")