"""
phase_fidelity_analysis.py

Implements the core contribution of the proposal (Section 5.5):
explanation-fidelity evaluation. For each correctly classified test
recording, determines which phase each arm's model attributed the
winning-class decision to, then measures agreement against a
PREREGISTERED table of clinically-expected dominant phases per class.

This is deliberately class-level (not per-recording IoU against expert
annotation) because Yaseen has no ground-truth phase/region labels --
stated as a limitation in the proposal (Section 4, Section 8), not
glossed over here.

Requires arm_a_results.pkl and arm_b_results.pkl directly (not the
figures-folder CSVs, since those don't retain per-phase-per-class
values needed to find each recording's top phase).

Saves into ./fidelity_results/.
"""

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

THIS_DIR = Path(__file__).resolve().parent
ARM_A_RESULTS_PATH = THIS_DIR / "GMM_results.pkl"
ARM_B_RESULTS_PATH = THIS_DIR / "arm_b_results.pkl"
OUTPUT_DIR = THIS_DIR / "fidelity_results"
OUTPUT_DIR.mkdir(exist_ok=True)

ARM_A_COLOR = "steelblue"
ARM_B_COLOR = "darkorange"


# --- 0. Preregistered expected phase per class (Section 5.5 table) ---------
# Written down BEFORE computing any agreement numbers, per the proposal's
# explicit instruction to preregister this from clinical literature
# rather than deriving it from the data after the fact.

EXPECTED_PHASE = {
    "AS": "Systole",
    "MR": "Systole",
    "MVP": "Systole",   # proposal says "late systole"; Springer segmentation
                         # doesn't distinguish early/late systole, so this
                         # maps to the closest available phase label -- note
                         # this simplification explicitly in the writeup.
    "MS": "Diastole",
    "N": None,           # no dominant phase expected -- excluded from
                          # agreement-rate computation, reported separately.
}


# --- loading ------------------------------------------------------------

def load_arm_results():
    with open(ARM_A_RESULTS_PATH, "rb") as f:
        arm_a = pickle.load(f)
    with open(ARM_B_RESULTS_PATH, "rb") as f:
        arm_b = pickle.load(f)
    return arm_a, arm_b


# --- 1. Per-recording top-phase attribution ---------------------------------

def top_phase_for_recording(rec: dict, loglik_key: str) -> str | None:
    """
    Given one recording's result dict (from either arm), find which
    phase gave the STRONGEST evidence for the model's own predicted
    class -- i.e. the phase this decision is most attributable to.
    Returns None if the predicted class had no valid (non-missing)
    phases at all (shouldn't normally happen for a made prediction).
    """
    pred = rec["predicted_label"]
    phase_vals = rec[loglik_key][pred]  # {phase: value or None}
    valid = {p: v for p, v in phase_vals.items() if v is not None}
    if not valid:
        return None
    return max(valid, key=lambda p: valid[p])


def build_attribution_table(arm_a_results: dict, arm_b_results: dict) -> pd.DataFrame:
    """
    One row per test recording that BOTH arms classified correctly
    (fidelity is only evaluated on correct predictions -- an
    explanation for a wrong answer isn't being validated against the
    clinical prior, per the proposal's framing in 5.5).
    """
    a_by_id = {r["recording_id"]: r for r in arm_a_results["per_recording_results"]}
    b_by_id = {r["recording_id"]: r for r in arm_b_results["per_recording_results"]}

    common_ids = set(a_by_id) & set(b_by_id)
    if len(common_ids) != len(a_by_id) or len(common_ids) != len(b_by_id):
        print(f"  NOTE: {len(a_by_id)} Arm A / {len(b_by_id)} Arm B recordings, "
              f"{len(common_ids)} in common -- using the intersection.")

    rows = []
    for rid in common_ids:
        a_rec = a_by_id[rid]
        b_rec = b_by_id[rid]
        true = a_rec["true_label"]
        assert true == b_rec["true_label"], f"Label mismatch for {rid}"

        a_correct = a_rec["predicted_label"] == true
        b_correct = b_rec["predicted_label"] == true

        rows.append({
            "recording_id": rid,
            "true_label": true,
            "arm_a_correct": a_correct,
            "arm_a_predicted": a_rec["predicted_label"],
            "arm_a_top_phase": top_phase_for_recording(a_rec, "per_class_per_phase_loglik") if a_correct else None,
            "arm_b_correct": b_correct,
            "arm_b_predicted": b_rec["predicted_label"],
            "arm_b_top_phase": top_phase_for_recording(b_rec, "per_class_per_phase_logprob") if b_correct else None,
        })

    return pd.DataFrame(rows)


# --- 2. Agreement rate vs. preregistered expectation, with Wilson CI -------

def wilson_ci(successes: int, n: int, z: float = 1.96):
    """
    Wilson score interval -- more reliable than a normal approximation
    for small n / extreme proportions, which is exactly the regime
    here (proposal explicitly flags small-N per class and asks for
    intervals, not just point estimates -- Section 5.5, 6.2).
    """
    if n == 0:
        return (np.nan, np.nan)
    phat = successes / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    half_width = (z * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, center - half_width), min(1.0, center + half_width))


def agreement_table(attribution_df: pd.DataFrame, arm: str) -> pd.DataFrame:
    """
    arm: 'a' or 'b'. Computes, per class (excluding N, which has no
    expected phase), the fraction of correctly-classified recordings
    whose top-attributed phase matches EXPECTED_PHASE[class].
    """
    correct_col = f"arm_{arm}_correct"
    phase_col = f"arm_{arm}_top_phase"

    rows = []
    for cls, expected in EXPECTED_PHASE.items():
        if expected is None:
            continue  # N: no dominant phase to check agreement against
        subset = attribution_df[
            (attribution_df["true_label"] == cls) & (attribution_df[correct_col])
        ]
        n = len(subset)
        if n == 0:
            rows.append({"class": cls, "expected_phase": expected, "n_correct": 0,
                         "n_agree": 0, "agreement_rate": np.nan, "ci_low": np.nan, "ci_high": np.nan})
            continue

        n_agree = (subset[phase_col] == expected).sum()
        rate = n_agree / n
        ci_low, ci_high = wilson_ci(n_agree, n)

        rows.append({
            "class": cls,
            "expected_phase": expected,
            "n_correct": n,
            "n_agree": n_agree,
            "agreement_rate": rate,
            "ci_low": ci_low,
            "ci_high": ci_high,
        })

    return pd.DataFrame(rows)


# --- 3. Plots ---------------------------------------------------------------

def plot_agreement_comparison(a_df: pd.DataFrame, b_df: pd.DataFrame):
    """
    Side-by-side agreement rate per class, Arm A vs Arm B, with
    Wilson CI error bars -- the headline figure for Section 5.5.
    """
    classes = a_df["class"].tolist()
    x = np.arange(len(classes))
    width = 0.35

    a_rate = a_df["agreement_rate"].values
    a_err = np.array([a_rate - a_df["ci_low"].values, a_df["ci_high"].values - a_rate])
    b_rate = b_df["agreement_rate"].values
    b_err = np.array([b_rate - b_df["ci_low"].values, b_df["ci_high"].values - b_rate])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, a_rate * 100, width, yerr=a_err * 100, capsize=4,
           label="Arm A (GMM)", color=ARM_A_COLOR)
    ax.bar(x + width / 2, b_rate * 100, width, yerr=b_err * 100, capsize=4,
           label="Arm B (MLP + SHAP)", color=ARM_B_COLOR)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n(expect {EXPECTED_PHASE[c]})" for c in classes])
    ax.set_ylabel("Agreement with preregistered expected phase (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Explanation Fidelity: Agreement with Clinically-Expected Phase\n(correctly classified recordings only, 95% Wilson CI)")
    ax.axhline(50, color="gray", linestyle=":", linewidth=1, label="Chance-ish reference (not exact -- 4 phases)")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_fidelity_agreement.png", dpi=150)
    plt.close(fig)


def plot_attributed_phase_distribution(attribution_df: pd.DataFrame):
    """
    For each class, what phase did each arm actually attribute
    correct predictions to (full distribution, not just match/no-match
    against expectation) -- useful for seeing WHERE disagreement goes
    when it isn't the expected phase.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, arm, label, color in [
        (axes[0], "a", "Arm A (GMM)", ARM_A_COLOR),
        (axes[1], "b", "Arm B (MLP)", ARM_B_COLOR),
    ]:
        correct_col = f"arm_{arm}_correct"
        phase_col = f"arm_{arm}_top_phase"
        subset = attribution_df[attribution_df[correct_col]]

        pivot = pd.crosstab(subset["true_label"], subset[phase_col], normalize="index")
        pivot.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
        ax.set_title(label)
        ax.set_ylabel("Share of correct predictions attributed to phase")
        ax.set_xlabel("True class")
        ax.legend(title="Attributed phase", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)

    fig.suptitle("Full Distribution of Attributed Phase per Class (correct predictions)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_attributed_phase_distribution.png", dpi=150)
    plt.close(fig)


# --- main -------------------------------------------------------------------

if __name__ == "__main__":
    print("Preregistered expected phase per class:")
    for cls, phase in EXPECTED_PHASE.items():
        print(f"  {cls}: {phase}")

    print("\nLoading Arm A and Arm B results...")
    arm_a_results, arm_b_results = load_arm_results()

    print("Building per-recording attribution table...")
    attribution_df = build_attribution_table(arm_a_results, arm_b_results)
    attribution_df.to_csv(OUTPUT_DIR / "attribution_table.csv", index=False)
    print(f"  {len(attribution_df)} recordings in common test set")

    print("\nComputing agreement rates (Arm A)...")
    a_agreement = agreement_table(attribution_df, "a")
    a_agreement.to_csv(OUTPUT_DIR / "arm_a_fidelity_agreement.csv", index=False)
    print(a_agreement.to_string(index=False))

    print("\nComputing agreement rates (Arm B)...")
    b_agreement = agreement_table(attribution_df, "b")
    b_agreement.to_csv(OUTPUT_DIR / "arm_b_fidelity_agreement.csv", index=False)
    print(b_agreement.to_string(index=False))

    print("\nPlotting...")
    plot_agreement_comparison(a_agreement, b_agreement)
    plot_attributed_phase_distribution(attribution_df)

    # Combined table for easy side-by-side reading / paper table
    combined = a_agreement.merge(
        b_agreement, on=["class", "expected_phase"], suffixes=("_arm_a", "_arm_b")
    )
    combined.to_csv(OUTPUT_DIR / "00_fidelity_comparison_combined.csv", index=False)

    print(f"\nAll fidelity analysis results saved to {OUTPUT_DIR}")