"""
analysis/make_paper_figures.py

Builds the figures referenced by paper/main.tex as PNGs in
paths.PAPER_FIGURES_DIR, from result files already on disk. Nothing is
recomputed; the script reads the arms' per-recording TSVs, the grouped
Yaseen class-level CSV and the CirCor 80/20 confusion counts, and draws them
with default matplotlib styling (plt.rcdefaults(); default colour cycle,
fonts and layout), so the figures look like ordinary matplotlib output.

Why a script and not hand-made figures
--------------------------------------
Every number in the paper is meant to be regenerable from the repo. Figures
are numbers too. Re-running this after any arm re-run keeps the PNGs in step
with the tables.

Figures (file name -> paper figure)
-----------------------------------
fig_segmenter.png             (a) upstream Springer code vs the modified
                              decoder on the same CirCor 80/20 split (mean /
                              median / frame-weighted accuracy); (b) per-phase
                              frame recall and precision of the modified
                              decoder from the 80/20 confusion counts
fig_accuracy_vs_share.png     two bar charts: (a) classification accuracy,
                              (b) mean systole weight-share, per classifier,
                              ground-truth vs predicted boundaries
fig_yaseen_share.png          grouped bars: expected-phase share per class
                              and classifier on Yaseen (group split), with
                              95% bootstrap CI and the 25% uniform line
fig_cohens_d.png              feature-level class separation per phase
                              (mean |Cohen's d|, Present vs Absent). Optional.

The confusion counts are the CirCor block of
results/logs/rerun_evaluate_segmenter_all_datasets.txt (80/20 split, seed 42).
They are typed in here because that log is the only place they exist; if the
segmenter evaluation is re-run, update CONFUSION from the new log.

The SHAP background-stability result (21.9% dominant-phase flips) is reported
in the text only; its figure was cut from the paper.

Camera-ready colours (2026-10-24)
---------------------------------
The two figures placed in the paper are the figures exactly as submitted,
except for the bar colours. The author chose those colours:
  - fig_segmenter keeps matplotlib's default blue/orange in both panels,
    so it is identical to the submitted image;
  - fig_accuracy_vs_share uses silver/green in panel (a) and light
    blue/wine in panel (b), each in the order ground-truth / predicted
    boundaries.

Two checks back this up:
  - With colours=None, the script reproduces both images embedded in the
    submitted PDF pixel for pixel (0 differing pixels).
  - In the recoloured figure, no grey, black or white pixel (text, axes,
    dashed line, background) differs from the submitted image.

fig_accuracy_vs_share is drawn in the submitted layout: 6 x 7.2 in, y-axis
to 100, one-column legends. After submission this script had drifted to a
6 x 5.2 in variant (commit 1e588f23); that variant was never in the paper.

`py analysis\\make_paper_figures.py --camera-ready` writes only those two
figures, to paths.CAMERA_READY_FIGURES_DIR, so the submitted PNGs in
paths.PAPER_FIGURES_DIR stay as they were.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repo root: makes `paths` and the packages importable from any cwd
import paths

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcdefaults()
plt.rcParams["font.size"] = 11  # figures are shrunk to a 3.5 in column; keep text legible
DPI = 200
OUT_DIR = paths.PAPER_FIGURES_DIR
# Bar colours per panel, as chosen by the author (camera-ready, 2026-10-24).
# None means matplotlib's default colour cycle (blue, orange), as in the
# submitted figures.
SEGMENTER_COLORS_A = None                    # upstream / modified: default blue, orange
SEGMENTER_COLORS_B = None                    # recall / precision: default blue, orange
ACCURACY_COLORS_A = ("#CCCCCC", "#117733")   # silver, green: ground-truth / predicted boundaries
ACCURACY_COLORS_B = ("#88CCEE", "#882255")   # light blue, wine: ground-truth / predicted boundaries

CLASSIFIERS =["GMM", "Per-phase MLP", "MLP+SHAP"]
PHASES = ("S1", "Systole", "S2", "Diastole")

# CirCor 80/20 test split, rows = true state, cols = predicted state
# (Unclassified, S1, Systole, S2, Diastole). From the segmenter log; see docstring.
CONFUSION = np.array([
    [346, 61728,  2503,  1560,   6146],   # S1
    [ 25,  5884, 68087,  4535,   3521],   # Systole
    [ 45,  1926,  5225, 51785,   4782],   # S2
    [241,  3846,  2573,  7026, 127571],   # Diastole
], dtype=float)

# Feature-level mean |Cohen's d| over 44 features, Present vs Absent
# (results/paper_results_and_methods.md Sec. 3.5)
COHENS_D = {"ground_truth": [0.335, 0.487, 0.206, 0.224],
            "predicted":    [0.385, 0.460, 0.229, 0.252]}


def _save(fig, name):
    out = OUT_DIR / name
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"  wrote {out.relative_to(paths.ROOT) if out.is_relative_to(paths.ROOT) else out}")


# ----------------------------------------------------------------------------
# segmenter: (a) upstream vs modified decoder, (b) per-phase recall/precision
# ----------------------------------------------------------------------------
def fig_segmenter(colors_a=SEGMENTER_COLORS_A, colors_b=SEGMENTER_COLORS_B):
    d = pd.read_csv(paths.UPSTREAM_VS_LOCAL_PER_RECORDING_CSV)
    arms = [("acc_upstream", "upstream Springer code"), ("acc_local", "modified decoder (this work)")]
    stats = {}
    for col, _ in arms:
        a = d[col]
        stats[col] = dict(mean=100 * a.mean(), median=100 * a.median(),
                          weighted=100 * np.average(a, weights=d["n_frames"]),
                          below5=int((a < 0.05).sum()))
    print(f"  upstream vs local: n={len(d)}, mean {stats['acc_upstream']['mean']:.1f} -> "
          f"{stats['acc_local']['mean']:.1f}, paired mean delta {100 * d['delta'].mean():+.2f} pts, "
          f"<5%: {stats['acc_upstream']['below5']} -> {stats['acc_local']['below5']}")

    recall = 100 * np.diag(CONFUSION[:, 1:]) / CONFUSION.sum(axis=1)
    precision = 100 * np.diag(CONFUSION[:, 1:]) / CONFUSION[:, 1:].sum(axis=0)
    print(f"  per-phase recall {np.round(recall, 1).tolist()}, precision {np.round(precision, 1).tolist()}")

    fig, (a, b) = plt.subplots(2, 1, figsize=(6, 5.2))
    x = np.arange(3); w = 0.38
    for k, (col, lab) in enumerate(arms):
        vals = [stats[col]["mean"], stats[col]["median"], stats[col]["weighted"]]
        bars = a.bar(x + (k - 0.5) * w, vals, w, label=lab, color=None if colors_a is None else colors_a[k])
        a.bar_label(bars, fmt="%.1f", padding=2, fontsize=9)
    a.set_xticks(x); a.set_xticklabels(["mean per recording", "median per recording", "frame-weighted"])
    a.set_ylabel("Frame accuracy (%)"); a.set_ylim(0, 120); a.set_yticks([0, 25, 50, 75, 100])
    a.set_title(f"(a) Upstream vs modified decoder ({len(d)} test recordings)")
    a.legend(loc="upper left", fontsize=9, ncol=2)

    x4 = np.arange(4)
    for k, (vals, lab) in enumerate([(recall, "recall"), (precision, "precision")]):
        bars = b.bar(x4 + (k - 0.5) * w, vals, w, label=lab, color=None if colors_b is None else colors_b[k])
        b.bar_label(bars, fmt="%.1f", padding=2, fontsize=9)
    b.set_xticks(x4); b.set_xticklabels(PHASES)
    b.set_ylabel("Frame accuracy (%)"); b.set_ylim(0, 120); b.set_yticks([0, 25, 50, 75, 100])
    b.set_title("(b) Modified decoder, per-phase frame recall and precision")
    b.legend(loc="upper left", fontsize=9, ncol=2)
    _save(fig, "fig_segmenter.png")


# ----------------------------------------------------------------------------
# accuracy and systole share, two bar charts
# ----------------------------------------------------------------------------
def _load_summary():
    files = {"GMM": paths.GMM_PHASE_FIDELITY_TSV,
             "Per-phase MLP": paths.NN_PHASE_FIDELITY_TSV,
             "MLP+SHAP": paths.WHOLECLIP_PHASE_FIDELITY_TSV}
    rows = []
    for name, f in files.items():
        d = pd.read_csv(f, sep="\t")
        for src in ("ground_truth", "predicted"):
            s = d[d["boundary_source"] == src]
            fid = s[s["has_expected_phase"].astype(bool)]
            rows.append(dict(classifier=name, source=src,
                             accuracy=100.0 * s["correct"].astype(bool).mean(),
                             systole_share=100.0 * fid["share_Systole"].mean(),
                             n=len(s), n_fid=len(fid)))
    return pd.DataFrame(rows)


def _grouped_bars(ax, summary, col, ylabel, title, ylim, colors):
    x = np.arange(len(CLASSIFIERS)); w = 0.38
    for k, (src, lab) in enumerate([("ground_truth", "ground-truth boundaries"),
                                    ("predicted", "predicted boundaries")]):
        vals = [summary[(summary.classifier == c) & (summary.source == src)][col].iloc[0]
                for c in CLASSIFIERS]
        bars = ax.bar(x + (k - 0.5) * w, vals, w, label=lab, color=None if colors is None else colors[k])
        ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(CLASSIFIERS)
    ax.set_ylabel(ylabel); ax.set_title(title); ax.set_ylim(*ylim)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.legend(loc="upper left", fontsize=8)


def fig_accuracy_vs_share(summary, colors_a=ACCURACY_COLORS_A, colors_b=ACCURACY_COLORS_B):
    # Layout of the figure in the submitted paper: 6 x 7.2 in, y-axis to 100,
    # one-column legends in matplotlib's default order (the uniform line first).
    fig, (a, b) = plt.subplots(2, 1, figsize=(6, 7.2))
    _grouped_bars(a, summary, "accuracy", "Accuracy (%)",
                  "(a) Classification accuracy, n = 3001", (0, 100), colors_a)
    _grouped_bars(b, summary, "systole_share", "Mean systole weight-share (%)",
                  "(b) Weight on systole, 616 Present recordings", (0, 100), colors_b)
    b.axhline(25, color="gray", ls="--", lw=1, label="uniform share (25%)")
    b.legend(loc="upper left", fontsize=8)
    _save(fig, "fig_accuracy_vs_share.png")


# ----------------------------------------------------------------------------
# Yaseen expected-phase share, grouped bars with CI
# ----------------------------------------------------------------------------
def _arm_label(raw):
    r = str(raw).lower()
    if "gmm" in r:
        return "GMM"
    if "whole" in r or "shap" in r:
        return "MLP+SHAP"
    return "Per-phase MLP"


def fig_yaseen():
    d = pd.read_csv(paths.YASEEN_CLASS_FIDELITY_GROUPED_CSV)
    d["label"] = d["arm"].map(_arm_label)
    d = d[d["scope"] == "all"]
    groups = [("AS", "all_segmentations", "AS\n(systole)"),
              ("AS", "clean_segmentation_only", "AS, clean subset\n(systole, n=56)"),
              ("MR", "all_segmentations", "MR\n(systole)"),
              ("MVP", "all_segmentations", "MVP\n(systole)"),
              ("MS", "all_segmentations", "MS\n(diastole)")]
    x = np.arange(len(groups)); w = 0.26
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for k, arm in enumerate(CLASSIFIERS):
        vals, lo, hi = [], [], []
        for cls, subset, _ in groups:
            r = d[(d.label == arm) & (d["class"] == cls) & (d.subset == subset)].iloc[0]
            vals.append(100 * r.expected_share)
            lo.append(100 * (r.expected_share - r.expected_share_ci_lo))
            hi.append(100 * (r.expected_share_ci_hi - r.expected_share))
        ax.bar(x + (k - 1) * w, vals, w, yerr=[lo, hi], capsize=3, label=arm)
    ax.axhline(25, color="gray", ls="--", lw=1, label="uniform share (25%)")
    ax.set_xticks(x); ax.set_xticklabels([g[2] for g in groups], fontsize=9)
    ax.set_ylabel("Weight on expected phase (%)")
    ax.set_title("Yaseen 2018: weight on the clinically expected phase\n(group split, n = 200 per class, 95% bootstrap CI)")
    ax.set_ylim(0, 88)
    ax.legend(loc="upper right", ncol=2, fontsize=9)
    ms = d[(d["class"] == "MS") & (d.subset == "all_segmentations")].set_index("label")["expected_share"]
    print("  Yaseen MS diastole share:", {k: round(100 * v, 1) for k, v in ms.items()})
    _save(fig, "fig_yaseen_share.png")


# ----------------------------------------------------------------------------
# Cohen's d per phase (optional)
# ----------------------------------------------------------------------------
def fig_cohens_d():
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    x = np.arange(4); w = 0.38
    for k, (src, lab) in enumerate([("ground_truth", "ground-truth boundaries"),
                                    ("predicted", "predicted boundaries")]):
        bars = ax.bar(x + (k - 0.5) * w, COHENS_D[src], w, label=lab)
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(PHASES)
    ax.set_ylabel("Mean |Cohen's d| over 44 features")
    ax.set_title("Class separation of each phase's features\n(Present vs Absent; larger = more separable)")
    ax.set_ylim(0, 0.6)
    ax.legend()
    _save(fig, "fig_cohens_d.png")


def main():
    global OUT_DIR
    camera_ready = "--camera-ready" in _sys.argv[1:]
    if camera_ready:
        OUT_DIR = paths.CAMERA_READY_FIGURES_DIR
    print(f"Writing figures to {OUT_DIR}")
    fig_segmenter()
    summary = _load_summary()
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    fig_accuracy_vs_share(summary)
    if not camera_ready:  # the camera-ready run writes only the two figures placed in the paper
        fig_yaseen()
        fig_cohens_d()
    print("done")


if __name__ == "__main__":
    main()
