"""
experiments/compare_arms_fidelity.py

Final comparison: prints the full phase-attribution-fidelity breakdown
for BOTH models (GMM = Arm A, NN = Arm B), each split by boundary_source
(ground_truth / predicted), then the discrete win-rate table and the
continuous weight-share table for each model, then one combined
weight-share table with all four rows side by side so GMM and NN can be
read off the same table.

Why weight-share is printed as the primary table, not just win-rate
----------------------------------------------------------------------
The discrete dominant_phase argmax (win-rate table) can look decisive
even when the underlying margins are close -- whichever phase happens
to be marginally largest "wins" the vote even if the model isn't
meaningfully concentrating on it. The continuous weight-share table
(mean |margin|-normalized share per phase, from compute_weight_share)
is the more honest number: it shows how concentrated vs. diffuse the
attribution actually is, not just which phase edged out the others.
Both tables are printed, but weight-share is what should be trusted
for claims like "the model is/isn't concentrating attribution like a
clinician would" -- see print_fidelity_summary output and the combined
table at the bottom for the actual GMM-vs-NN comparison.

Paired bootstrap CI + sign-flip permutation test (new)
----------------------------------------------------------------------
The combined weight-share table shows a gap (e.g. GMM's Systole share
sitting well above NN's), but a gap alone doesn't say whether it's a
stable model-level effect or noise from recording-to-recording
variation. GMM and NN are evaluated on the identical (paired) set of
recordings via k-fold, so paired_diffs/bootstrap_ci/sign_flip_permutation_test
below test that directly: a percentile bootstrap CI on the mean
GMM-minus-NN difference, plus a sign-flip permutation p-value, for (a)
the Systole weight-share specifically -- the phase the headline claim
rests on -- and (b) a Herfindahl-style concentration index (sum of
squared phase shares; 0.25=uniform, 1.0=fully concentrated) that
doesn't assume Systole is the phase that matters. Both are run
separately for ground_truth and predicted boundary_source.

Inputs (must already exist -- run experiments/arm_a_gmm.py and
experiments/arm_b_nn_shap.py first):
  - phase_fidelity_per_recording.tsv     (GMM, from run_GMM's
    PHASE_FIDELITY_TSV_PATH)
  - phase_fidelity_per_recording_nn.tsv  (NN, from the NN script's
    PHASE_FIDELITY_TSV_PATH)

These TSVs (not the .pkl files) are used deliberately: they already
have one row per test recording per boundary_source, with the
dominant_phase / margin_<phase> / share_<phase> / expected_phases /
fidelity_match columns print_fidelity_summary, print_win_rate_table,
and print_weight_share_table need -- no need to reload sklearn model
objects out of the pickles just to print a comparison table.

Booleans round-trip through CSV as the strings "True"/"False" (and
empty-string for None), not real bools/NaN -- _load_fidelity_df fixes
that up on read so .mean() etc. behave the way they do inside the
original run scripts.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repo root: makes `paths` and the packages importable from any cwd
import paths


from pathlib import Path

import numpy as np
import pandas as pd

from experiments.arm_a_gmm import (
    PHASES,
    PHASE_FIDELITY_TSV_PATH as GMM_FIDELITY_TSV_PATH,
    print_fidelity_summary,
    print_win_rate_table,
    print_weight_share_table,
)
from experiments.arm_b_nn_shap import (
    PHASE_FIDELITY_TSV_PATH as NN_FIDELITY_TSV_PATH,
)

WHOLECLIP_FIDELITY_TSV_PATH = paths.WHOLECLIP_PHASE_FIDELITY_TSV  # Arm C, optional
WHOLECLIP_LABEL = "NN-wholeclip+SHAP"



# ---------------------------------------------------------------------
# Load + fix up the saved per-recording fidelity TSVs
# ---------------------------------------------------------------------

def _load_fidelity_df(path):
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run the corresponding comparison script first "
            f"(experiments/arm_a_gmm.py or experiments/arm_b_nn_shap.py) "
            f"so this file gets written."
        )
    df = pd.read_csv(path, sep="\t")

    # bool columns round-trip through CSV as "True"/"False"/"" -- but pandas'
    # C parser auto-detects an all-True/False/empty column and silently
    # converts it to REAL bool objects (not the strings "True"/"False") on
    # read, unless there happen to be other value types mixed in. Which
    # form you get is not guaranteed, so .map({"True": True, ...}) against
    # string keys can silently miss every row and return all-NaN (this bit
    # fidelity_match in the first run: precision was NaN everywhere while
    # recall, which doesn't touch this column, was fine). Stringifying
    # first sidesteps the ambiguity regardless of which form pandas chose.
    has_expected_str = df["has_expected_phase"].astype(str)
    df["has_expected_phase"] = has_expected_str == "True"

    correct_str = df["correct"].astype(str)
    df["correct"] = correct_str == "True"

    fidelity_str = df["fidelity_match"].astype(str)
    df["fidelity_match"] = np.where(
        fidelity_str == "True", True,
        np.where(fidelity_str == "False", False, np.nan),
    )

    # expected_phases: real empty set was saved as "" -- read back as
    # NaN by pandas on an all-string column sometimes, normalize to "".
    df["expected_phases"] = df["expected_phases"].fillna("")

    return df


def load_model_fidelity(path):
    """Splits a model's saved fidelity TSV (which has both boundary_source
    arms concatenated) back into (gt_fidelity_df, pred_fidelity_df)."""
    df = _load_fidelity_df(path)
    gt_df = df[df["boundary_source"] == "ground_truth"].reset_index(drop=True)
    pred_df = df[df["boundary_source"] == "predicted"].reset_index(drop=True)
    return gt_df, pred_df


# ---------------------------------------------------------------------
# Combined GMM-vs-NN weight-share table
# ---------------------------------------------------------------------

def print_combined_weight_share_table(model_fidelity):
    """model_fidelity: dict of model_label -> (gt_df, pred_df).
    Prints one table with a row per (model, boundary_source) combo,
    all sharing the same phase columns, so GMM and NN read off the
    same table instead of two separate ones you have to eyeball
    side by side.
    """
    rows = {}
    for model_label, (gt_df, pred_df) in model_fidelity.items():
        for arm_label, fdf in (("GT", gt_df), ("predicted", pred_df)):
            scoped = fdf[fdf["has_expected_phase"]]
            row_label = f"{model_label} - {arm_label}"
            rows[row_label] = {phase: scoped[f"share_{phase}"].mean() for phase in PHASES}

    table = pd.DataFrame(rows).T[list(PHASES)]
    formatted = pd.DataFrame(
        {col: [f"{v * 100:.1f}%" for v in table[col]] for col in table.columns},
        index=table.index,
    )
    print(formatted.to_string())

    uniform = 100.0 / len(PHASES)
    print(f"\n  (uniform / no-concentration baseline: {uniform:.1f}% per phase -- rows close to "
          f"this across the board indicate diffuse, not concentrated, attribution)")
    return table


def add_concentration_index(df):
    """Herfindahl-style concentration index: sum of squared phase shares.
    0.25 (=4 x 0.25^2) under perfect uniformity across the 4 phases, rising
    toward 1.0 as attribution concentrates entirely on one phase. Unlike
    share_Systole, this doesn't assume Systole is the phase that matters --
    it answers "how peaked is the attribution at all", model-agnostic.
    """
    df = df.copy()
    df["concentration_index"] = sum(df[f"share_{p}"] ** 2 for p in PHASES)
    return df


def paired_diffs(gmm_df, nn_df, value_col):
    """Pairs GMM and NN rows by recording_id (both already restricted to
    has_expected_phase==True upstream) and returns the array of
    (gmm - nn) differences in value_col, plus how many pairs were found.
    Paired, not independent-sample, because both models were evaluated on
    the identical reused test-patient split -- most recording_ids should
    match, but this checks rather than assumes.
    """
    merged = gmm_df[["recording_id", value_col]].merge(
        nn_df[["recording_id", value_col]], on="recording_id", suffixes=("_gmm", "_nn")
    )
    diffs = (merged[f"{value_col}_gmm"] - merged[f"{value_col}_nn"]).to_numpy()
    return diffs, len(merged), len(gmm_df), len(nn_df)


def bootstrap_ci(diffs, n_boot=10000, alpha=0.05, seed=42):
    """Percentile bootstrap CI on the mean of paired differences."""
    rng = np.random.RandomState(seed)
    n = len(diffs)
    idx = rng.randint(0, n, size=(n_boot, n))
    boot_means = diffs[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return diffs.mean(), lo, hi


def sign_flip_permutation_test(diffs, n_perm=20000, seed=42):
    """Paired permutation (sign-flip) test: under the null that GMM and NN
    are exchangeable per recording, randomly flipping the sign of each
    pair's difference should look like the observed difference about as
    often as not. p-value = fraction of permuted |mean diff| >= observed
    |mean diff| (two-sided).
    """
    rng = np.random.RandomState(seed)
    n = len(diffs)
    signs = rng.choice([-1, 1], size=(n_perm, n))
    perm_means = (signs * diffs).mean(axis=1)
    observed = diffs.mean()
    p_value = (np.abs(perm_means) >= np.abs(observed)).mean()
    return p_value


def print_stat_comparison(label, gmm_df, nn_df, value_col, as_percent=True,
                          a_label="GMM", b_label="NN"):
    """Paired a - b comparison. Parameter names keep the original GMM/NN
    wording; a_label / b_label only change the printed names so the same
    function serves the three-way comparison once Arm C exists."""
    diffs, n_paired, n_gmm, n_nn = paired_diffs(gmm_df, nn_df, value_col)
    if n_paired == 0:
        print(f"  [{label}] no overlapping recording_ids between {a_label} (n={n_gmm}) and "
              f"{b_label} (n={n_nn}) scoped sets -- can't pair, skipping.")
        return
    if n_paired < n_gmm or n_paired < n_nn:
        print(f"  NOTE: [{label}] paired on {n_paired} recording_id(s) common to both "
              f"({a_label} scoped n={n_gmm}, {b_label} scoped n={n_nn}) -- unmatched rows dropped.")

    mean_diff, lo, hi = bootstrap_ci(diffs)
    p_value = sign_flip_permutation_test(diffs)

    fmt = (lambda v: f"{v * 100:+.1f} pts") if as_percent else (lambda v: f"{v:+.4f}")
    sig_note = "significant at alpha=0.05" if hi < 0 or lo > 0 else "NOT significant at alpha=0.05"
    print(f"  [{label}] {a_label} - {b_label} mean difference, n={n_paired} paired recordings:")
    print(f"    mean diff: {fmt(mean_diff)}   95% bootstrap CI: [{fmt(lo)}, {fmt(hi)}]   "
          f"permutation p={p_value:.4f}  ({sig_note})")


def print_accuracy_comparison(model_fidelity):
    print("\n  accuracy by model x boundary_source:")
    for model_label, (gt_df, pred_df) in model_fidelity.items():
        for arm_label, fdf in (("ground_truth", gt_df), ("predicted", pred_df)):
            acc = fdf["correct"].mean()
            print(f"    {model_label:>4} / {arm_label:>12}: {acc * 100:.1f}%  (n={len(fdf)})")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    gmm_gt_df, gmm_pred_df = load_model_fidelity(GMM_FIDELITY_TSV_PATH)
    nn_gt_df, nn_pred_df = load_model_fidelity(NN_FIDELITY_TSV_PATH)

    model_fidelity = {
        "GMM": (gmm_gt_df, gmm_pred_df),
        "NN": (nn_gt_df, nn_pred_df),
    }

    # Arm C (whole-clip NN + post-hoc KernelExplainer SHAP, signed per-phase
    # SHAP sums as margins) joins the table when its TSV exists. Optional so
    # the two-arm comparison still runs before / without Arm C.
    if WHOLECLIP_FIDELITY_TSV_PATH.exists():
        wc_gt_df, wc_pred_df = load_model_fidelity(WHOLECLIP_FIDELITY_TSV_PATH)
        model_fidelity[WHOLECLIP_LABEL] = (wc_gt_df, wc_pred_df)
    else:
        print(f"NOTE: {WHOLECLIP_FIDELITY_TSV_PATH} not found -- run "
              f"experiments/arm_c_wholeclip_nn_shap.py to add the whole-clip NN + SHAP "
              f"baseline to this comparison. Continuing with GMM vs NN only.")

    print_accuracy_comparison(model_fidelity)

    for model_label, (gt_df, pred_df) in model_fidelity.items():
        print("\n" + "=" * 70)
        print(f"{model_label} -- DETAILED PHASE-ATTRIBUTION FIDELITY BREAKDOWN")
        print("=" * 70)
        print_fidelity_summary(gt_df, f"{model_label} - ground_truth")
        print_fidelity_summary(pred_df, f"{model_label} - predicted")

        print(f"\n--- {model_label}: discrete dominant-phase win-rate (GT vs. predicted x phase) ---")
        print("  (argmax-based -- can look decisive even when margins are close; see weight-share below)")
        print_win_rate_table(gt_df, pred_df)

        print(f"\n--- {model_label}: continuous mean weight-share (GT vs. predicted x phase) ---")
        print("  (|margin|-normalized -- the more trustworthy concentration metric)")
        print_weight_share_table(gt_df, pred_df)

    print("\n" + "=" * 70)
    print("COMBINED: GMM vs. NN weight-share, all four rows together")
    print("=" * 70)
    print_combined_weight_share_table(model_fidelity)

    # -------------------------------------------------------------
    # Paired bootstrap CI + sign-flip permutation test
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PAIRED STATS: is the between-model weight-share gap real, or noise?")
    print("=" * 70)
    print("  (paired on recording_id -- every model sees the identical k-fold-pooled")
    print("   recordings, so a paired bootstrap CI + sign-flip permutation test is")
    print("   the right way to ask whether the combined-table gap above is stable.)")

    model_labels = list(model_fidelity.keys())
    pairs = [(a, b) for i, a in enumerate(model_labels) for b in model_labels[i + 1:]]

    for boundary_source, arm_label in [("ground_truth", "GT boundaries"),
                                        ("predicted", "Predicted boundaries")]:
        print(f"\n--- {arm_label} ---")
        idx = 0 if boundary_source == "ground_truth" else 1
        scoped = {}
        for label in model_labels:
            fdf = model_fidelity[label][idx]
            scoped[label] = add_concentration_index(fdf[fdf["has_expected_phase"]])

        for a, b in pairs:
            # (a) Systole weight-share -- the phase the headline claim rests on
            print_stat_comparison(
                f"Systole weight-share ({arm_label})",
                scoped[a], scoped[b], "share_Systole", as_percent=True, a_label=a, b_label=b,
            )
            # (b) Herfindahl-style concentration index -- model-agnostic, doesn't
            # assume Systole specifically is the phase that matters
            print_stat_comparison(
                f"Concentration index ({arm_label})",
                scoped[a], scoped[b], "concentration_index", as_percent=False, a_label=a, b_label=b,
            )

    # Arm C only: the absolute-mass aggregation (sum |phi| per phase) as a
    # robustness view next to the signed sums used in every table above.
    if WHOLECLIP_LABEL in model_fidelity:
        print("\n" + "=" * 70)
        print(f"{WHOLECLIP_LABEL}: absolute-mass SHAP aggregation (robustness, not additive)")
        print("=" * 70)
        for arm_label, fdf in (("GT", model_fidelity[WHOLECLIP_LABEL][0]),
                               ("predicted", model_fidelity[WHOLECLIP_LABEL][1])):
            sc = fdf[fdf["has_expected_phase"]]
            if not all(f"share_abs_{p}" in sc.columns for p in PHASES):
                print(f"  [{arm_label}] share_abs_* columns missing -- re-run Arm C.")
                continue
            shares = "  ".join(f"{p}={sc[f'share_abs_{p}'].mean() * 100:.1f}%" for p in PHASES)
            match = (sc["fidelity_match_abs"].astype(str) == "True").mean()
            print(f"  [{arm_label}] n={len(sc)}  dominant matches expected {match * 100:.1f}%  mean share: {shares}")


if __name__ == "__main__":
    main()