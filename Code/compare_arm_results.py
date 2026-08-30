"""
compare_arm_results.py

Loads the summary CSVs/pickles saved by plot_gmm_results.py (Arm A)
and plot_arm_b_results.py (Arm B) and produces side-by-side comparison
figures for the results section -- overall/per-class accuracy, per-phase
standalone accuracy, prediction margin/confidence, and phase-level
"where does the model look" comparison (Arm A's phase contribution vs
Arm B's SHAP-derived phase importance).

Both arms were evaluated on the SAME held-out test_recording_ids
(Arm B reused Arm A's pruned split), so these are directly comparable,
not just visually similar.

Saves into ./comparison_figures/.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

THIS_DIR = Path(__file__).resolve().parent
ARM_A_DIR = THIS_DIR / "GMM_figures"
ARM_B_DIR = THIS_DIR / "arm_b_figures"
OUTPUT_DIR = THIS_DIR / "comparison_figures"
OUTPUT_DIR.mkdir(exist_ok=True)

ARM_A_COLOR = "steelblue"
ARM_B_COLOR = "darkorange"


# --- loading ----------------------------------------------------------------

def load_all():
    data = {}
    data["a_accuracy"] = pd.read_csv(ARM_A_DIR / "accuracy_summary.csv")
    data["b_accuracy"] = pd.read_csv(ARM_B_DIR / "accuracy_summary.csv")

    data["a_phase_acc"] = pd.read_csv(ARM_A_DIR / "per_phase_accuracy_overall.csv", index_col=0)
    data["b_phase_acc"] = pd.read_csv(ARM_B_DIR / "per_phase_accuracy_overall.csv", index_col=0)

    data["a_margin"] = pd.read_csv(ARM_A_DIR / "margin_data.csv")
    data["b_margin"] = pd.read_csv(ARM_B_DIR / "margin_data.csv")

    data["a_contribution"] = pd.read_csv(ARM_A_DIR / "phase_contribution_data.csv")
    data["b_contribution"] = pd.read_csv(ARM_B_DIR / "phase_contribution_data.csv")

    data["b_phase_importance"] = pd.read_csv(ARM_B_DIR / "phase_importance_data.csv")

    return data


# --- 1. Overall + per-class accuracy, side by side --------------------------

def plot_accuracy_comparison(data):
    a = data["a_accuracy"].set_index("class")["accuracy"]
    b = data["b_accuracy"].set_index("class")["accuracy"]

    classes = [c for c in a.index if c != "OVERALL"]
    order = ["OVERALL"] + classes

    a_vals = [a.get(c, np.nan) * 100 for c in order]
    b_vals = [b.get(c, np.nan) * 100 for c in order]

    x = np.arange(len(order))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars_a = ax.bar(x - width / 2, a_vals, width, label="Arm A (GMM)", color=ARM_A_COLOR)
    bars_b = ax.bar(x + width / 2, b_vals, width, label="Arm B (MLP)", color=ARM_B_COLOR)

    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Overall and Per-Class Accuracy — Arm A vs Arm B")
    ax.legend()

    for bars in (bars_a, bars_b):
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(bar.get_x() + bar.get_width() / 2, h + 1, f"{h:.1f}", ha="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_accuracy_comparison.png", dpi=150)
    plt.close(fig)


# --- 2. Per-phase standalone accuracy, side by side --------------------------

def plot_per_phase_accuracy_comparison(data):
    a = data["a_phase_acc"]["accuracy"]
    b = data["b_phase_acc"]["accuracy"]

    phases = list(a.index)  # both arms use the same PHASES tuple/order
    x = np.arange(len(phases))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 5))
    bars_a = ax.bar(x - width / 2, a.values * 100, width, label="Arm A (GMM)", color=ARM_A_COLOR)
    bars_b = ax.bar(x + width / 2, b.reindex(phases).values * 100, width, label="Arm B (MLP)", color=ARM_B_COLOR)

    ax.set_xticks(x)
    ax.set_xticklabels(phases)
    ax.set_ylabel("Standalone accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Per-Phase Standalone Accuracy — Arm A vs Arm B")
    ax.legend()

    for bars in (bars_a, bars_b):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 1, f"{h:.1f}", ha="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_per_phase_accuracy_comparison.png", dpi=150)
    plt.close(fig)

    # Also worth flagging explicitly: do both arms agree on the hardest phase?
    a_worst = a.idxmin()
    b_worst = b.idxmin()
    print(f"  Arm A's hardest phase (standalone): {a_worst} ({a.min()*100:.1f}%)")
    print(f"  Arm B's hardest phase (standalone): {b_worst} ({b.min()*100:.1f}%)")
    if a_worst == b_worst:
        print(f"  -> Both arms agree: {a_worst} is hardest -- convergent evidence, worth stating in results.")
    else:
        print("  -> Arms disagree on hardest phase -- worth digging into why in discussion.")


# --- 3. Prediction confidence/margin distributions, overlaid ----------------

def plot_margin_comparison(data):
    """
    Arm A's margin is a log-likelihood gap (arbitrary scale, can be
    large). Arm B's margin is a log-softmax-probability gap (bounded,
    since probabilities sum to 1). These are NOT on the same numeric
    scale, so this plot shows shape/separation of correct-vs-incorrect
    within each arm side by side -- not a claim that the raw numbers
    are comparable across arms.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)

    for ax, (label, df, color_correct, color_incorrect) in zip(
        axes,
        [
            ("Arm A (GMM) — log-likelihood margin", data["a_margin"], "green", "red"),
            ("Arm B (MLP) — log-prob margin", data["b_margin"], "green", "red"),
        ],
    ):
        correct = df[df["correct"]]["margin"]
        incorrect = df[~df["correct"]]["margin"]
        ax.hist(correct, bins=30, alpha=0.6, label=f"Correct (n={len(correct)})", color=color_correct)
        ax.hist(incorrect, bins=30, alpha=0.6, label=f"Incorrect (n={len(incorrect)})", color=color_incorrect)
        ax.set_xlabel("Margin (winner - runner-up)")
        ax.set_ylabel("Count")
        ax.set_title(label)
        ax.legend()

    fig.suptitle("Prediction Margin Separation — Arm A vs Arm B (not on the same numeric scale)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "03_margin_comparison.png", dpi=150)
    plt.close(fig)


# --- 4. Where each arm's attention concentrates, by phase --------------------

def plot_phase_attention_comparison(data):
    """
    Arm A: mean relative phase contribution to the predicted class's
    total log-likelihood, restricted to CORRECT predictions (i.e. when
    the model got it right, which phases mattered most).
    Arm B: overall SHAP-based phase importance (already
    class-and-correctness-agnostic, since SHAP importance was computed
    over all test recordings).

    These answer slightly different questions (A: "which phase drove
    correct decisions" vs B: "which phase does the model attend to in
    general") -- caption this distinction in the paper rather than
    treating the two bars as the same measurement.
    """
    a_contrib = data["a_contribution"]
    a_correct = a_contrib[a_contrib["correct"]]
    a_by_phase = a_correct.groupby("phase")["relative_contribution"].mean()

    b_importance = data["b_phase_importance"].set_index("phase")["mean_abs_shap"]
    b_normalized = b_importance / b_importance.sum()  # put on a 0-1 relative scale like Arm A

    phases = list(a_by_phase.index)
    x = np.arange(len(phases))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 5))
    bars_a = ax.bar(x - width / 2, a_by_phase.values, width,
                     label="Arm A: mean contribution (correct preds)", color=ARM_A_COLOR)
    bars_b = ax.bar(x + width / 2, b_normalized.reindex(phases).values, width,
                     label="Arm B: normalized SHAP importance", color=ARM_B_COLOR)

    ax.set_xticks(x)
    ax.set_xticklabels(phases)
    ax.set_ylabel("Relative importance (each arm's own measure, both normalized to sum to 1)")
    ax.set_title("Phase-Level Attention — Arm A vs Arm B")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "04_phase_attention_comparison.png", dpi=150)
    plt.close(fig)


# --- CSV/table exports (add alongside each plot call) -----------------------

def save_accuracy_comparison_csv(data):
    a = data["a_accuracy"].set_index("class")["accuracy"]
    b = data["b_accuracy"].set_index("class")["accuracy"]
    classes = [c for c in a.index if c != "OVERALL"]
    order = ["OVERALL"] + classes

    df = pd.DataFrame({
        "class": order,
        "arm_a_accuracy": [a.get(c, np.nan) for c in order],
        "arm_b_accuracy": [b.get(c, np.nan) for c in order],
    })
    df["gap_b_minus_a"] = df["arm_b_accuracy"] - df["arm_a_accuracy"]
    df.to_csv(OUTPUT_DIR / "01_accuracy_comparison.csv", index=False)
    return df


def save_per_phase_accuracy_csv(data):
    a = data["a_phase_acc"]["accuracy"]
    b = data["b_phase_acc"]["accuracy"]
    phases = list(a.index)

    df = pd.DataFrame({
        "phase": phases,
        "arm_a_accuracy": a.values,
        "arm_b_accuracy": b.reindex(phases).values,
    })
    df["gap_b_minus_a"] = df["arm_b_accuracy"] - df["arm_a_accuracy"]
    df.to_csv(OUTPUT_DIR / "02_per_phase_accuracy_comparison.csv", index=False)
    return df


def save_margin_summary_csv(data):
    """Not the raw margin values (those are already in each arm's own
    margin_data.csv) -- this is the summary stats a reader actually
    wants in a table: mean/median margin split by correct/incorrect,
    per arm."""
    rows = []
    for arm_label, df in [("Arm A (GMM)", data["a_margin"]), ("Arm B (MLP)", data["b_margin"])]:
        for correct_flag in (True, False):
            subset = df[df["correct"] == correct_flag]["margin"]
            rows.append({
                "arm": arm_label,
                "correct": correct_flag,
                "n": len(subset),
                "mean_margin": subset.mean(),
                "median_margin": subset.median(),
                "std_margin": subset.std(),
            })
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "03_margin_summary.csv", index=False)
    return out


def save_phase_attention_csv(data):
    a_contrib = data["a_contribution"]
    a_correct = a_contrib[a_contrib["correct"]]
    a_by_phase = a_correct.groupby("phase")["relative_contribution"].mean()

    b_importance = data["b_phase_importance"].set_index("phase")["mean_abs_shap"]
    b_normalized = b_importance / b_importance.sum()

    phases = list(a_by_phase.index)
    df = pd.DataFrame({
        "phase": phases,
        "arm_a_mean_contribution_correct_preds": a_by_phase.values,
        "arm_b_normalized_shap_importance": b_normalized.reindex(phases).values,
    })
    df.to_csv(OUTPUT_DIR / "04_phase_attention_comparison.csv", index=False)
    return df


def save_headline_summary_csv(data):
    """One-row summary of the numbers you'll actually type into the
    paper's abstract/results paragraph."""
    a_overall = data["a_accuracy"].set_index("class").loc["OVERALL", "accuracy"]
    b_overall = data["b_accuracy"].set_index("class").loc["OVERALL", "accuracy"]

    a_phase_acc = data["a_phase_acc"]["accuracy"]
    b_phase_acc = data["b_phase_acc"]["accuracy"]
    a_worst = a_phase_acc.idxmin()
    b_worst = b_phase_acc.idxmin()

    row = {
        "arm_a_overall_accuracy": a_overall,
        "arm_b_overall_accuracy": b_overall,
        "gap_b_minus_a": b_overall - a_overall,
        "arm_a_hardest_phase": a_worst,
        "arm_a_hardest_phase_accuracy": a_phase_acc.min(),
        "arm_b_hardest_phase": b_worst,
        "arm_b_hardest_phase_accuracy": b_phase_acc.min(),
        "arms_agree_on_hardest_phase": a_worst == b_worst,
    }
    pd.DataFrame([row]).to_csv(OUTPUT_DIR / "00_headline_summary.csv", index=False)
    return row

if __name__ == "__main__":
    print("Loading Arm A and Arm B summary data...")
    data = load_all()

    print("  01: accuracy comparison")
    plot_accuracy_comparison(data)
    save_accuracy_comparison_csv(data)

    print("  02: per-phase standalone accuracy comparison")
    plot_per_phase_accuracy_comparison(data)
    save_per_phase_accuracy_csv(data)

    print("  03: margin/confidence comparison")
    plot_margin_comparison(data)
    save_margin_summary_csv(data)

    print("  04: phase-level attention comparison")
    plot_phase_attention_comparison(data)
    save_phase_attention_csv(data)

    headline = save_headline_summary_csv(data)
    print(f"\nOverall accuracy: Arm A = {headline['arm_a_overall_accuracy']*100:.1f}%, "
          f"Arm B = {headline['arm_b_overall_accuracy']*100:.1f}%")
    print(f"Gap: {headline['gap_b_minus_a']*100:+.1f} percentage points")

    print(f"\nAll comparison figures and tables saved to {OUTPUT_DIR}")