"""
experiments/yaseen_three_arms.py

Phase 2: the three CirCor arms re-run 5-class (AS / MR / MS / MVP / N) on
Yaseen 2018, with CLASS-LEVEL phase-attribution fidelity against the
preregistered clinical table. One script rather than three because the
arms share every non-model step here (folds, scaling, margin definition,
fidelity report) and Yaseen has no second boundary source to loop over.

Confidence tier (design decision 7): Yaseen fidelity is class-level only.
Yaseen has disease labels but no phase annotations and no per-recording
murmur timing, so nothing here is a per-recording fidelity number and
nothing here may be merged with the CirCor per-recording results.

Preregistered expected phase (design decision 3, written before any Yaseen
number was computed):
    AS, MR, MVP -> Systole      (MVP is "late systole" clinically; the
                                 segmenter has no early/late split)
    MS          -> Diastole
    N           -> no dominant phase expected: reported as mean
                   concentration index and dominant-phase spread, not as
                   a fidelity rate.

Models (identical to the CirCor arms, 5-class):
  Arm A  per (class, phase) 2-component diagonal GMM; score_p(c) = log p(x_p | c);
         total(c) = sum_p score_p(c).
  Arm B  per phase MLPClassifier((12,)) over 5 classes; score_p(c) =
         log P(c | x_p) - log pi_c (prior removed per silo, the fix carried
         over from CirCor Arm B); total(c) = log pi_c + sum_p score_p(c).
  Arm C  whole-clip MLPClassifier((48,)); total(c) = log P(c | x);
         KernelExplainer on the 5 log-probability outputs.

Per-phase margin, multi-class (additive by construction)
--------------------------------------------------------
With c* the predicted class and c2 the runner-up on total score,
    m_p = score_p(c*) - score_p(c2)      (Arms A, B)
    m_p = sum_{j in phase p} [ phi_j(c*) - phi_j(c2) ]   (Arm C, SHAP)
so that sum_p m_p = total(c*) - total(c2) exactly (for B the prior terms
cancel; for C up to SHAP's sampling error, which is checked and printed).
dominant_phase = argmax_p |m_p|; weight-share s_p = |m_p| / sum_q |m_q|;
concentration C = sum_p s_p^2. This is the same definition as the CirCor
arms with "Absent" replaced by "the strongest rival class".

The sibling repo's earlier Yaseen fidelity used a different, weaker
statistic: the phase with the highest RAW log-likelihood for the predicted
class (which conflates "fits this class well" with "separates this class
from the rest"). It is stored as top_raw_phase for Arms A and B so the old
numbers can be reconciled, but it is not the fidelity metric here.

Fidelity conventions: reported for ALL recordings of a class and for
CORRECTLY CLASSIFIED ones (the sibling convention), with Wilson 95% CIs on
argmax rates and percentile-bootstrap CIs on mean expected-phase share;
then again restricted to is_clean_segmentation recordings (sibling's
definition: all phases present and mean Systole >= 0.10 s) with the
retention fraction next to every number. Every cell carries n.

Splits (two modes; run both and report both)
--------------------------------------------
  py experiments\\yaseen_three_arms.py            recording-level StratifiedKFold
  py experiments\\yaseen_three_arms.py group      StratifiedGroupKFold over the
                                                  near-duplicate source groups in
                                                  paths.YASEEN_DUPLICATE_GROUPS_CSV
                                                  (analysis/yaseen_duplicate_groups.py)

Yaseen has no patient IDs, and it turns out it does not have 1000
independent recordings either: most files are overlapping crops of a much
smaller number of source recordings (N: ~10 sources for 200 files at
xcorr > 0.95). A recording-level split therefore leaks near-identical
audio across train and test, exactly the leak CirCor's patient-level split
exists to prevent. The recording-level mode is kept because it is what the
sibling repo and the published Yaseen literature use, so its numbers are
the comparable ones; the group mode is the honest one. Output files get a
"_grouped" suffix in group mode.

Recordings with any missing phase are dropped before splitting and the
drop count per class is printed (the segmentation retention number).

Outputs (suffix "_grouped" in group mode):
  results/arms/yaseen_three_arms_results.pkl
  results/arms/yaseen_phase_attribution_per_recording.tsv
  results/arms/yaseen_class_level_fidelity.csv
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repo root: makes `paths` and the packages importable from any cwd
import paths


import pickle
import time

import numpy as np
import pandas as pd
import shap
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

CLASSES = tuple(paths.YASEEN_CLASSES)
PHASES = ("S1", "Systole", "S2", "Diastole")
EXPECTED_PHASE = {"AS": "Systole", "MR": "Systole", "MVP": "Systole", "MS": "Diastole", "N": None}
ARMS = ("GMM", "NN-silo", "NN-wholeclip+SHAP")

RANDOM_STATE = 42
N_FOLDS = 5
N_COMPONENTS = 2
COVARIANCE_TYPE = "diag"
SILO_HIDDEN = (12,)
WHOLECLIP_HIDDEN = (48,)
MAX_ITER = 2000
EPS = 1e-9
SHAP_BACKGROUND_SIZE = 50
SHAP_NSAMPLES = "auto"
N_BOOT = 2000

NON_FEATURE_COLS = {"recording_id", "label", "boundary_source", "heart_rate_bpm", "systolic_time_sec",
                    "total_duration_sec", "n_clips"}

FEATURE_TABLE = paths.YASEEN_FEATURE_TABLE
QUALITY_CSV = paths.YASEEN_SEGMENTATION_QUALITY_CSV
GROUPS_CSV = paths.YASEEN_DUPLICATE_GROUPS_CSV

SPLIT_MODE = "group" if (len(_sys.argv) > 1 and _sys.argv[1].lower() == "group") else "recording"
_SUFFIX = "_grouped" if SPLIT_MODE == "group" else ""
RESULTS_PKL = paths.YASEEN_ARMS_RESULTS_PKL.with_name(paths.YASEEN_ARMS_RESULTS_PKL.stem + _SUFFIX + ".pkl")
PER_RECORDING_TSV = paths.YASEEN_PER_RECORDING_TSV.with_name(paths.YASEEN_PER_RECORDING_TSV.stem + _SUFFIX + ".tsv")
CLASS_FIDELITY_CSV = paths.YASEEN_CLASS_FIDELITY_CSV.with_name(paths.YASEEN_CLASS_FIDELITY_CSV.stem + _SUFFIX + ".csv")


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------

def phase_columns(feature_cols, phase):
    return [c for c in feature_cols if c.startswith(f"{phase}_")]


def load_table():
    df = pd.read_csv(FEATURE_TABLE, sep="\t")
    candidate = [c for c in df.columns if c not in NON_FEATURE_COLS]
    all_nan = [c for c in candidate if df[c].isna().all()]
    if all_nan:
        print(f"Dropping {len(all_nan)} all-NaN column(s): {all_nan}")
    feature_cols = [c for c in candidate if c not in all_nan]

    n_before = df.groupby("label").size()
    kept = df.dropna(subset=feature_cols).copy()
    n_after = kept.groupby("label").size()
    print("\nSegmentation retention (recordings with all four phases present, usable feature row):")
    for cls in CLASSES:
        b, a = int(n_before.get(cls, 0)), int(n_after.get(cls, 0))
        print(f"  {cls:>4}: {a}/{b} kept ({a / b * 100:.1f}%)" if b else f"  {cls:>4}: none")
    print(f"  total: {len(kept)}/{len(df)}")

    quality = pd.read_csv(QUALITY_CSV)[["recording_id", "is_clean_segmentation", "n_complete_cycles", "mean_dur_Systole"]]
    kept = kept.merge(quality, on="recording_id", how="left")
    print("\nClean-segmentation retention (all phases + mean Systole >= 0.10 s), of the kept rows:")
    for cls in CLASSES:
        sub = kept[kept["label"] == cls]
        print(f"  {cls:>4}: {int(sub['is_clean_segmentation'].sum())}/{len(sub)}")
    return kept.reset_index(drop=True), feature_cols


# ---------------------------------------------------------------------
# Scoring helpers: every arm returns, per test row, a (n_classes,) total
# score and a dict phase -> (n_classes,) per-phase score.
# ---------------------------------------------------------------------

def class_prior_log(train_labels):
    counts = np.array([(train_labels == c).sum() for c in CLASSES], dtype=float)
    return np.log(np.clip(counts / counts.sum(), EPS, 1.0))


def fit_arm_a(train_df, feature_cols):
    gmms = {}
    for c in CLASSES:
        rows = train_df[train_df["label"] == c]
        for p in PHASES:
            gmm = GaussianMixture(n_components=N_COMPONENTS, covariance_type=COVARIANCE_TYPE,
                                  random_state=RANDOM_STATE)
            gmm.fit(rows[phase_columns(feature_cols, p)].values)
            gmms[(c, p)] = gmm
    return gmms


def score_arm_a(gmms, test_df, feature_cols):
    per_phase = {}
    for p in PHASES:
        X = test_df[phase_columns(feature_cols, p)].values
        per_phase[p] = np.column_stack([gmms[(c, p)].score_samples(X) for c in CLASSES])
    total = sum(per_phase.values())
    return total, per_phase


def fit_arm_b(train_df, feature_cols):
    y = train_df["label"].map({c: i for i, c in enumerate(CLASSES)}).values
    nns = {}
    for p in PHASES:
        clf = MLPClassifier(hidden_layer_sizes=SILO_HIDDEN, max_iter=MAX_ITER, random_state=RANDOM_STATE,
                            early_stopping=True)
        clf.fit(train_df[phase_columns(feature_cols, p)].values, y)
        nns[p] = clf
    return nns, class_prior_log(train_df["label"].values)


def score_arm_b(nns, log_prior, test_df, feature_cols):
    per_phase = {}
    for p in PHASES:
        proba = nns[p].predict_proba(test_df[phase_columns(feature_cols, p)].values)
        full = np.full((len(test_df), len(CLASSES)), np.log(EPS))
        for j, cls_idx in enumerate(nns[p].classes_):
            full[:, int(cls_idx)] = np.log(np.clip(proba[:, j], EPS, 1.0))
        per_phase[p] = full - log_prior  # evidence term: prior removed per silo
    total = log_prior + sum(per_phase.values())
    return total, per_phase


def fit_arm_c(train_df, feature_cols):
    y = train_df["label"].map({c: i for i, c in enumerate(CLASSES)}).values
    clf = MLPClassifier(hidden_layer_sizes=WHOLECLIP_HIDDEN, max_iter=MAX_ITER, random_state=RANDOM_STATE,
                        early_stopping=True)
    clf.fit(train_df[feature_cols].values, y)
    return clf


def make_logproba_fn(clf):
    def f(X):
        proba = clf.predict_proba(np.asarray(X))
        full = np.full((len(proba), len(CLASSES)), np.log(EPS))
        for j, cls_idx in enumerate(clf.classes_):
            full[:, int(cls_idx)] = np.log(np.clip(proba[:, j], EPS, 1.0))
        return full
    return f


def score_arm_c(clf, train_df, test_df, feature_cols):
    f = make_logproba_fn(clf)
    X_test = test_df[feature_cols].values
    total = f(X_test)
    background = shap.sample(train_df[feature_cols], min(SHAP_BACKGROUND_SIZE, len(train_df)),
                             random_state=RANDOM_STATE)
    explainer = shap.KernelExplainer(f, background)
    t0 = time.time()
    raw = explainer.shap_values(X_test, nsamples=SHAP_NSAMPLES, silent=True)
    elapsed = time.time() - t0
    # normalise to (n, n_features, n_classes)
    if isinstance(raw, list):
        phi = np.stack([np.asarray(a) for a in raw], axis=-1)
    else:
        phi = np.asarray(raw)
    expected = np.asarray(explainer.expected_value).reshape(-1)
    return total, phi, expected, elapsed / len(X_test)


# ---------------------------------------------------------------------
# Per-recording attribution rows
# ---------------------------------------------------------------------

def attribution_rows(arm, test_df, total, per_phase=None, phi=None, expected=None, feature_cols=None, fold=None):
    rows = []
    phase_idx = {p: [i for i, c in enumerate(feature_cols) if c.startswith(f"{p}_")] for p in PHASES}
    for i, (_, r) in enumerate(test_df.iterrows()):
        order = np.argsort(-total[i])
        c_star, c2 = int(order[0]), int(order[1])
        if per_phase is not None:
            margins = {p: float(per_phase[p][i, c_star] - per_phase[p][i, c2]) for p in PHASES}
            top_raw = max(PHASES, key=lambda p: per_phase[p][i, c_star])
            recon = float(sum(margins.values()) - (total[i, c_star] - total[i, c2]))
        else:
            diff = phi[i, :, c_star] - phi[i, :, c2]
            margins = {p: float(diff[phase_idx[p]].sum()) for p in PHASES}
            top_raw = None
            recon = float(diff.sum() + (expected[c_star] - expected[c2]) - (total[i, c_star] - total[i, c2]))
        abs_m = {p: abs(v) for p, v in margins.items()}
        s = sum(abs_m.values())
        share = {p: (abs_m[p] / s if s > 0 else 0.25) for p in PHASES}
        row = {"arm": arm, "fold": fold, "recording_id": r["recording_id"], "true_label": r["label"],
               "predicted_label": CLASSES[c_star], "runner_up": CLASSES[c2],
               "correct": CLASSES[c_star] == r["label"],
               "is_clean_segmentation": bool(r["is_clean_segmentation"]),
               "decision_margin": float(total[i, c_star] - total[i, c2]),
               "reconstruction_error": recon,
               "dominant_phase": max(margins, key=lambda p: abs(margins[p])),
               "top_raw_phase": top_raw,
               "concentration": float(sum(v ** 2 for v in share.values()))}
        for p in PHASES:
            row[f"margin_{p}"] = margins[p]
            row[f"share_{p}"] = share[p]
        rows.append(row)
    return rows


# ---------------------------------------------------------------------
# Class-level fidelity
# ---------------------------------------------------------------------

def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return centre - half, centre + half


def boot_mean_ci(values, seed=RANDOM_STATE):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(values), size=(N_BOOT, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def class_fidelity(df, arm, subset_label):
    out = []
    for cls in CLASSES:
        sub = df[(df["arm"] == arm) & (df["true_label"] == cls)]
        for scope, s in (("all", sub), ("correct_only", sub[sub["correct"]])):
            exp = EXPECTED_PHASE[cls]
            row = {"arm": arm, "subset": subset_label, "class": cls, "scope": scope, "n": len(s),
                   "expected_phase": exp or "none",
                   "class_recall": sub["correct"].mean() if len(sub) else float("nan"),
                   "mean_concentration": s["concentration"].mean() if len(s) else float("nan")}
            for p in PHASES:
                row[f"share_{p}"] = s[f"share_{p}"].mean() if len(s) else float("nan")
                row[f"dominant_{p}"] = int((s["dominant_phase"] == p).sum())
            if exp is not None and len(s):
                k = int((s["dominant_phase"] == exp).sum())
                lo, hi = wilson(k, len(s))
                row.update({"argmax_fidelity": k / len(s), "argmax_ci_lo": lo, "argmax_ci_hi": hi,
                            "expected_share": s[f"share_{exp}"].mean()})
                blo, bhi = boot_mean_ci(s[f"share_{exp}"].values)
                row.update({"expected_share_ci_lo": blo, "expected_share_ci_hi": bhi})
                # majority baseline: the most common dominant phase across ALL classes in this arm/subset
                if "top_raw_phase" in s and s["top_raw_phase"].notna().any():
                    row["top_raw_fidelity_sibling_def"] = float((s["top_raw_phase"] == exp).mean())
            else:
                row.update({"argmax_fidelity": float("nan"), "argmax_ci_lo": float("nan"), "argmax_ci_hi": float("nan"),
                            "expected_share": float("nan"), "expected_share_ci_lo": float("nan"),
                            "expected_share_ci_hi": float("nan")})
            out.append(row)
    return out


def print_fidelity_table(fid_df, arm, subset_label, scope):
    sub = fid_df[(fid_df["arm"] == arm) & (fid_df["subset"] == subset_label) & (fid_df["scope"] == scope)]
    print(f"\n  [{arm}] {subset_label}, {scope}:")
    print(f"    {'class':>4} {'n':>4} {'recall':>7} {'expected':>9} {'argmax fid [95% CI]':>24} "
          f"{'exp. share [95% CI]':>24} {'conc':>6}   share S1/Sys/S2/Dia    dominant S1/Sys/S2/Dia")
    for _, r in sub.iterrows():
        if r["expected_phase"] == "none":
            fid = "     (none expected)    "
            sh = "                        "
        else:
            fid = f"{r['argmax_fidelity'] * 100:5.1f}% [{r['argmax_ci_lo'] * 100:4.1f},{r['argmax_ci_hi'] * 100:5.1f}]"
            sh = f"{r['expected_share'] * 100:5.1f}% [{r['expected_share_ci_lo'] * 100:4.1f},{r['expected_share_ci_hi'] * 100:5.1f}]"
        shares = "/".join(f"{r[f'share_{p}'] * 100:4.1f}" for p in PHASES)
        doms = "/".join(f"{int(r[f'dominant_{p}']):3d}" for p in PHASES)
        print(f"    {r['class']:>4} {int(r['n']):>4} {r['class_recall'] * 100:6.1f}% {r['expected_phase']:>9} "
              f"{fid:>24} {sh:>24} {r['mean_concentration']:6.3f}   {shares}   {doms}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    print(f"Loading {FEATURE_TABLE}")
    df, feature_cols = load_table()
    print(f"\nUsing {len(feature_cols)} feature columns "
          f"({', '.join(f'{p}: {len(phase_columns(feature_cols, p))}' for p in PHASES)})")

    if SPLIT_MODE == "group":
        if not GROUPS_CSV.exists():
            raise FileNotFoundError(f"{GROUPS_CSV} not found -- run analysis/yaseen_duplicate_groups.py first")
        groups_df = pd.read_csv(GROUPS_CSV)[["recording_id", "group_id", "group_size"]]
        df = df.merge(groups_df, on="recording_id", how="left")
        if df["group_id"].isna().any():
            raise ValueError("some recordings have no duplicate-group id -- regenerate the groups CSV")
        print(f"\nGroup-level split: {df['group_id'].nunique()} near-duplicate source groups for {len(df)} recordings "
              f"(per class: " + ", ".join(f"{c} {df[df['label'] == c]['group_id'].nunique()}" for c in CLASSES) + ")")
        splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        split_iter = splitter.split(df, df["label"], groups=df["group_id"])
        split_desc = "STRATIFIED GROUP K-FOLD over near-duplicate source groups (no crop of a test source is in training)"
    else:
        splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        split_iter = splitter.split(df, df["label"])
        split_desc = "STRATIFIED K-FOLD, recording level (sibling-repo / literature convention; leaks near-duplicate crops)"

    all_rows, shap_cost, fold_models = [], [], []
    print("\n" + "=" * 70)
    print(f"{N_FOLDS}-FOLD {split_desc}")
    print("=" * 70)
    for fold, (tr, te) in enumerate(split_iter):
        train_df, test_df = df.iloc[tr].copy(), df.iloc[te].copy()
        scaler = StandardScaler().fit(train_df[feature_cols])
        train_df[feature_cols] = scaler.transform(train_df[feature_cols])
        test_df[feature_cols] = scaler.transform(test_df[feature_cols])

        gmms = fit_arm_a(train_df, feature_cols)
        tot_a, pp_a = score_arm_a(gmms, test_df, feature_cols)
        rows_a = attribution_rows("GMM", test_df, tot_a, per_phase=pp_a, feature_cols=feature_cols, fold=fold)

        nns, log_prior = fit_arm_b(train_df, feature_cols)
        tot_b, pp_b = score_arm_b(nns, log_prior, test_df, feature_cols)
        rows_b = attribution_rows("NN-silo", test_df, tot_b, per_phase=pp_b, feature_cols=feature_cols, fold=fold)

        clf = fit_arm_c(train_df, feature_cols)
        tot_c, phi, expected, sec = score_arm_c(clf, train_df, test_df, feature_cols)
        rows_c = attribution_rows("NN-wholeclip+SHAP", test_df, tot_c, phi=phi, expected=expected,
                                  feature_cols=feature_cols, fold=fold)
        shap_cost.append(sec)

        for name, rows in (("GMM", rows_a), ("NN-silo", rows_b), ("NN-wholeclip+SHAP", rows_c)):
            acc = np.mean([r["correct"] for r in rows])
            print(f"  fold {fold + 1}: {name:>18} accuracy {acc * 100:5.1f}%  (n={len(rows)})"
                  + (f"  SHAP {sec:.2f} s/rec, max |recon err| {max(abs(r['reconstruction_error']) for r in rows):.4f}"
                     if name.startswith("NN-wholeclip") else ""))
        all_rows.extend(rows_a + rows_b + rows_c)
        fold_models.append({"fold": fold, "scaler": scaler, "gmms": gmms, "nns": nns, "log_prior": log_prior,
                            "wholeclip": clf, "test_recording_ids": test_df["recording_id"].tolist()})

    per_rec = pd.DataFrame(all_rows)
    per_rec.to_csv(PER_RECORDING_TSV, sep="\t", index=False)

    print("\n" + "=" * 70)
    print("POOLED OUT-OF-FOLD CLASSIFICATION")
    print("=" * 70)
    for arm in ARMS:
        a = per_rec[per_rec["arm"] == arm]
        y_true = a["true_label"].map({c: i for i, c in enumerate(CLASSES)})
        y_pred = a["predicted_label"].map({c: i for i, c in enumerate(CLASSES)})
        cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASSES)))
        recalls = {c: cm[i, i] / cm[i].sum() for i, c in enumerate(CLASSES)}
        print(f"\n[{arm}] accuracy {a['correct'].mean() * 100:.1f}%  macro-F1 {f1_score(y_true, y_pred, average='macro'):.3f}  "
              f"per-class recall " + ", ".join(f"{c} {v * 100:.0f}%" for c, v in recalls.items()))
        print(pd.DataFrame(cm, index=[f"true {c}" for c in CLASSES], columns=list(CLASSES)).to_string())
    print(f"\nSHAP cost: {np.mean(shap_cost):.2f} s per recording; max |reconstruction error| "
          f"{per_rec[per_rec['arm'] == 'NN-wholeclip+SHAP']['reconstruction_error'].abs().max():.4f}")

    print("\n" + "=" * 70)
    print("CLASS-LEVEL PHASE-ATTRIBUTION FIDELITY vs PREREGISTERED TABLE")
    print("  expected: AS, MR, MVP -> Systole; MS -> Diastole; N -> none (concentration reported)")
    print("  margin_p = score_p(predicted) - score_p(runner-up); share = |margin| normalised")
    print("=" * 70)
    fid_rows = []
    clean = per_rec[per_rec["is_clean_segmentation"]]
    for arm in ARMS:
        fid_rows += class_fidelity(per_rec, arm, "all_segmentations")
        fid_rows += class_fidelity(clean, arm, "clean_segmentation_only")
    fid_df = pd.DataFrame(fid_rows)
    fid_df.to_csv(CLASS_FIDELITY_CSV, index=False)

    # majority baseline per arm: most common dominant phase over all recordings
    for arm in ARMS:
        a = per_rec[per_rec["arm"] == arm]
        dom = a["dominant_phase"].value_counts(normalize=True)
        print(f"\n[{arm}] dominant-phase distribution over all {len(a)} recordings: "
              + ", ".join(f"{p} {dom.get(p, 0) * 100:.1f}%" for p in PHASES)
              + f"  (majority baseline for a Systole-expected class = {dom.get('Systole', 0) * 100:.1f}%, "
                f"for MS = {dom.get('Diastole', 0) * 100:.1f}%)")
        for subset_label in ("all_segmentations", "clean_segmentation_only"):
            for scope in ("all", "correct_only"):
                print_fidelity_table(fid_df, arm, subset_label, scope)

    print("\n[sibling-repo definition, for reconciliation only: top RAW per-phase score of the predicted class, correct-only]")
    for arm in ("GMM", "NN-silo"):
        a = per_rec[(per_rec["arm"] == arm) & per_rec["correct"]]
        parts = []
        for cls in CLASSES:
            exp = EXPECTED_PHASE[cls]
            s = a[a["true_label"] == cls]
            if exp is None or len(s) == 0:
                continue
            parts.append(f"{cls} {(s['top_raw_phase'] == exp).mean() * 100:.1f}% (n={len(s)})")
        print(f"  {arm}: " + "; ".join(parts))

    with open(RESULTS_PKL, "wb") as f:
        pickle.dump({"per_recording": per_rec, "class_fidelity": fid_df, "fold_models": fold_models,
                     "feature_cols": feature_cols, "expected_phase": EXPECTED_PHASE,
                     "config": {"split_mode": SPLIT_MODE, "classes": CLASSES, "phases": PHASES, "n_folds": N_FOLDS,
                                "random_state": RANDOM_STATE, "n_components": N_COMPONENTS,
                                "covariance_type": COVARIANCE_TYPE, "silo_hidden": SILO_HIDDEN,
                                "wholeclip_hidden": WHOLECLIP_HIDDEN, "shap_background_size": SHAP_BACKGROUND_SIZE,
                                "shap_nsamples": SHAP_NSAMPLES, "shap_seconds_per_recording": float(np.mean(shap_cost))}},
                    f)
    print(f"\nSaved {PER_RECORDING_TSV}, {CLASS_FIDELITY_CSV}, {RESULTS_PKL}")


if __name__ == "__main__":
    main()
