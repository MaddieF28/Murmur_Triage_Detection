"""
analysis/patient_clustered_paired_tests.py

Re-runs every paired comparison the paper reports, with the PATIENT
rather than the recording as the resampling unit.

Why
---
Camera-ready, reviewer 2, limitation 3: "Statistical comparisons appear
to operate at the recording level even though recordings are clustered
within patients." They do. experiments/compare_arms_fidelity.py
(bootstrap_ci, sign_flip_permutation_test) and
segmenter_eval/compare_upstream_vs_local.py (paired) both resample and
sign-flip individual recordings. CirCor has up to four valve-site
recordings per patient (the 616 Present recordings of the fidelity
subset come from 179 patients). Recordings of one patient share a heart,
a murmur and a label, so their paired differences are not independent,
and treating them as independent makes the intervals too narrow.

What
----
For each comparison:
  1. recording level, with the ORIGINAL functions and seeds. This must
     reproduce the paper's numbers, which checks that the inputs and the
     pairing are identical to what the paper used;
  2. patient-cluster bootstrap: draw patients with replacement; the
     statistic is the mean paired difference over all recordings of the
     drawn patients, so the point estimate is the same recording-level
     mean the paper reports and only the uncertainty changes;
  3. patient-level sign-flip permutation: one random sign per patient,
     applied to all of that patient's recordings.

Comparisons (the ones with a CI or p-value in the paper):
  - systole weight-share, GMM - per-phase MLP, MLP+SHAP - GMM,
    MLP+SHAP - per-phase MLP, under ground-truth and predicted
    boundaries (Present recordings with an annotated phase, n=616);
  - segmenter frame accuracy, modified - upstream decoder (Sec. IV-A,
    631 test recordings of the CirCor 80/20 split).

Nothing is retrained; inputs are the per-recording outputs already on
disk. Same resample counts as the paper: 10,000 bootstrap, 20,000
permutations, so the smallest reportable p is 0.00005.

Output: paths.PATIENT_CLUSTERED_TESTS_CSV, plus a console table.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repo root: makes `paths` and the packages importable from any cwd
import paths

import numpy as np
import pandas as pd

from experiments.compare_arms_fidelity import (
    load_model_fidelity,
    paired_diffs,
    bootstrap_ci,
    sign_flip_permutation_test,
)
from segmenter_eval.compare_upstream_vs_local import (
    paired as segmenter_paired,
    SEED as SEGMENTER_SEED,
)

N_BOOT = 10000
N_PERM = 20000
SEED = 42

CLASSIFIER_TSVS = {
    "GMM": paths.GMM_PHASE_FIDELITY_TSV,
    "per-phase MLP": paths.NN_PHASE_FIDELITY_TSV,
    "MLP+SHAP": paths.WHOLECLIP_PHASE_FIDELITY_TSV,
}
PAIRS = (("GMM", "per-phase MLP"), ("MLP+SHAP", "GMM"), ("MLP+SHAP", "per-phase MLP"))


def patient_ids(recording_ids):
    """CirCor recording ids are <patient>_<site>[_<n>]."""
    return pd.Series(recording_ids).astype(str).str.split("_").str[0].to_numpy()


def patient_clustered(diffs, patients, n_boot=N_BOOT, n_perm=N_PERM, seed=SEED):
    """Cluster bootstrap CI and cluster sign-flip p for the mean paired
    difference over recordings, resampling / flipping whole patients."""
    per_patient = pd.DataFrame({"d": diffs, "patient": patients}).groupby("patient")["d"]
    sums = per_patient.sum().to_numpy()
    counts = per_patient.count().to_numpy()
    k, n = len(sums), counts.sum()
    observed = sums.sum() / n

    rng = np.random.RandomState(seed)
    idx = rng.randint(0, k, size=(n_boot, k))
    boot_means = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    lo, hi = np.percentile(boot_means, [2.5, 97.5])

    rng = np.random.RandomState(seed)
    signs = rng.choice([-1, 1], size=(n_perm, k))
    perm_means = (signs * sums).sum(axis=1) / n
    p = (np.abs(perm_means) >= abs(observed)).mean()
    return observed, lo, hi, p, k


def classifier_rows():
    data = {name: load_model_fidelity(path) for name, path in CLASSIFIER_TSVS.items()}
    rows = []
    for b_i, boundary in enumerate(("ground_truth", "predicted")):
        scoped = {name: dfs[b_i][dfs[b_i]["has_expected_phase"]] for name, dfs in data.items()}
        for a, b in PAIRS:
            merged = scoped[a][["recording_id", "share_Systole"]].merge(
                scoped[b][["recording_id", "share_Systole"]], on="recording_id", suffixes=("_a", "_b"))
            diffs = (merged["share_Systole_a"] - merged["share_Systole_b"]).to_numpy()
            reference, _, _, _ = paired_diffs(scoped[a], scoped[b], "share_Systole")
            assert np.array_equal(diffs, reference), "pairing differs from compare_arms_fidelity"

            mean, rec_lo, rec_hi = bootstrap_ci(diffs)
            rec_p = sign_flip_permutation_test(diffs)
            _, pat_lo, pat_hi, pat_p, k = patient_clustered(diffs, patient_ids(merged["recording_id"]))
            rows.append(dict(comparison=f"systole share: {a} - {b}", boundary=boundary,
                             n_recordings=len(diffs), n_patients=k, mean_diff=mean,
                             rec_ci_lo=rec_lo, rec_ci_hi=rec_hi, rec_p=rec_p,
                             patient_ci_lo=pat_lo, patient_ci_hi=pat_hi, patient_p=pat_p))
    return rows


def segmenter_row():
    seg = pd.read_csv(paths.UPSTREAM_VS_LOCAL_PER_RECORDING_CSV)
    delta = seg["delta"].to_numpy()  # acc_local - acc_upstream
    (rec_lo, rec_hi), rec_p_str = segmenter_paired(delta, np.random.default_rng(SEGMENTER_SEED))
    rec_p = 0.0 if rec_p_str.startswith("p <") else float(rec_p_str.split("=")[1])
    _, pat_lo, pat_hi, pat_p, k = patient_clustered(delta, patient_ids(seg["recording"]))
    return dict(comparison="segmenter frame accuracy: modified - upstream", boundary="80/20 test split",
                n_recordings=len(delta), n_patients=k, mean_diff=delta.mean(),
                rec_ci_lo=rec_lo, rec_ci_hi=rec_hi, rec_p=rec_p,
                patient_ci_lo=pat_lo, patient_ci_hi=pat_hi, patient_p=pat_p)


def fmt_p(p):
    return f"< {1 / N_PERM:.5f}" if p == 0 else f"= {p:.4f}"


def main():
    rows = classifier_rows() + [segmenter_row()]
    out = pd.DataFrame(rows)
    out.to_csv(paths.PATIENT_CLUSTERED_TESTS_CSV, index=False)

    print("Paired differences in points (x100). Recording-level columns must match the paper.\n")
    for r in rows:
        print(f"{r['comparison']}  [{r['boundary']}]  n={r['n_recordings']} recordings, "
              f"{r['n_patients']} patients")
        print(f"    mean diff {100 * r['mean_diff']:+.1f}")
        print(f"    recording-level 95% CI [{100 * r['rec_ci_lo']:+.1f}, {100 * r['rec_ci_hi']:+.1f}]  "
              f"p {fmt_p(r['rec_p'])}")
        print(f"    patient-level   95% CI [{100 * r['patient_ci_lo']:+.1f}, {100 * r['patient_ci_hi']:+.1f}]  "
              f"p {fmt_p(r['patient_p'])}")
    print(f"\nWrote {paths.PATIENT_CLUSTERED_TESTS_CSV}")


if __name__ == "__main__":
    main()
