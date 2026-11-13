"""
analysis/signed_phase_contributions.py

Signed per-phase contributions alongside the magnitude-based summary.

Why
---
Camera-ready, reviewer 2, limitation 2: "The use of |m_p| treats strong
negative evidence as phase agreement. This can obscure whether a phase
supports or contradicts the predicted class. Please report signed phase
contributions, preferably stratified by predicted label and correctness,
alongside the current magnitude-based summary."

That is an accurate description of the metric. The dominant phase is
argmax_p |m_p|, so in a Present recording a phase pushing hard toward
Absent can be "dominant" and count as agreement. Magnitude is the
intended reading of "where the decision rested", but it does not say
which way that phase voted. This script reports the direction.

What
----
For each classifier (GMM, per-phase MLP, MLP+SHAP) and boundary source,
on the Present recordings with an annotated phase (n=616, the paper's
fidelity subset), stratified by predicted label. On this subset
correctness and predicted label coincide: predicted Present means
correct, predicted Absent means missed.
  - supports_decision: % of recordings whose dominant phase
    (argmax |m_p|) has the sign of the predicted label (+ for Present,
    - for Absent);
  - signed systole share: m_Sys / sum_q |m_q|, in [-1, 1]. Its absolute
    value is the paper's systole weight-share; the sign says whether
    systole voted Present (+) or Absent (-);
  - agreement_magnitude: the paper's agreement (dominant = argmax |m_p|);
  - agreement_toward_decision: agreement when the dominant phase is
    instead the phase contributing most TOWARD the decision,
    argmax_p m_p * sign(decision).
Also reports supports_decision on all 3001 recordings, split by the
four (true, predicted) cells.

The decision sign is the predicted label. For the per-phase MLP and
MLP+SHAP the decision also includes a term that belongs to no phase
(logit pi; the SHAP base value), so a phase can oppose the decision
without the decision changing.

Inputs: the per-recording fidelity TSVs of the three arms (signed
margin_<phase> columns). Nothing is retrained.
Output: paths.SIGNED_PHASE_CONTRIBUTIONS_CSV, plus a console table.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repo root: makes `paths` and the packages importable from any cwd
import paths

import numpy as np
import pandas as pd

from experiments.compare_arms_fidelity import load_model_fidelity

PHASES = ("S1", "Systole", "S2", "Diastole")
CLASSIFIER_TSVS = {
    "GMM": paths.GMM_PHASE_FIDELITY_TSV,
    "per-phase MLP": paths.NN_PHASE_FIDELITY_TSV,
    "MLP+SHAP": paths.WHOLECLIP_PHASE_FIDELITY_TSV,
}


def add_signed_columns(df):
    df = df.copy()
    margins = df[[f"margin_{p}" for p in PHASES]].to_numpy(dtype=float)
    decision_sign = np.where(df["predicted_label"] == "Present", 1.0, -1.0)
    magnitude_idx = np.abs(margins).argmax(axis=1)
    toward_idx = (margins * decision_sign[:, None]).argmax(axis=1)
    phases = np.array(PHASES)

    assert (phases[magnitude_idx] == df["dominant_phase"].to_numpy()).all(), \
        "dominant_phase is not argmax |margin| -- check the inputs"
    dominant_margin = margins[np.arange(len(df)), magnitude_idx]
    df["supports_decision"] = np.sign(dominant_margin) == decision_sign
    df["signed_share_Systole"] = margins[:, PHASES.index("Systole")] / np.abs(margins).sum(axis=1)
    df["dominant_toward_decision"] = phases[toward_idx]

    expected = df["expected_phases"].fillna("").astype(str).str.split(",")
    df["agreement_magnitude"] = [d in e for d, e in zip(df["dominant_phase"], expected)]
    df["agreement_toward_decision"] = [d in e for d, e in zip(df["dominant_toward_decision"], expected)]
    return df


def summarize(df, label_cols):
    g = df.groupby(label_cols)
    return pd.DataFrame({
        "n": g.size(),
        "supports_decision_pct": 100 * g["supports_decision"].mean(),
        "mean_signed_share_Systole_pct": 100 * g["signed_share_Systole"].mean(),
        "agreement_magnitude_pct": 100 * g["agreement_magnitude"].mean(),
        "agreement_toward_decision_pct": 100 * g["agreement_toward_decision"].mean(),
    })


def main():
    rows = []
    for name, path in CLASSIFIER_TSVS.items():
        for boundary, df in zip(("ground_truth", "predicted"), load_model_fidelity(path)):
            df = add_signed_columns(df)
            present = df[df["has_expected_phase"]]

            overall = summarize(present.assign(predicted="all"), ["predicted"])
            by_pred = summarize(present.rename(columns={"predicted_label": "predicted"}), ["predicted"])
            for pred, r in pd.concat([overall, by_pred]).iterrows():
                rows.append(dict(classifier=name, boundary=boundary, subset="Present with annotated phase",
                                 true_label="Present", predicted=pred, **r.to_dict()))

            cells = summarize(df.rename(columns={"predicted_label": "predicted"}), ["true_label", "predicted"])
            for (true, pred), r in cells.iterrows():
                rows.append(dict(classifier=name, boundary=boundary, subset="all recordings",
                                 true_label=true, predicted=pred, n=r["n"],
                                 supports_decision_pct=r["supports_decision_pct"]))

    out = pd.DataFrame(rows)
    out.to_csv(paths.SIGNED_PHASE_CONTRIBUTIONS_CSV, index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    show = out[out["subset"] == "Present with annotated phase"].drop(columns=["subset", "true_label"])
    print("Present recordings with an annotated phase (n=616 per classifier x boundary source)")
    print("predicted Present = correctly classified; predicted Absent = missed\n")
    print(show.round(1).to_string(index=False))
    print("\nAll recordings: % whose dominant phase pushes toward the predicted label")
    allrec = out[out["subset"] == "all recordings"][["classifier", "boundary", "true_label", "predicted",
                                                     "n", "supports_decision_pct"]]
    print(allrec.round(1).to_string(index=False))
    print(f"\nWrote {paths.SIGNED_PHASE_CONTRIBUTIONS_CSV}")


if __name__ == "__main__":
    main()
