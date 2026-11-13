"""
Head-to-head: the unmodified upstream Springer segmenter vs this repo's
`segmenter/` package, trained and tested on the identical CirCor split.

Why this exists
---------------
`segmenter/` contains changes relative to
EchoStatements/Springer-Segmentation-Python (commit 276b9cc). Every
segmenter change should be reported before/after, but the reference
numbers were all produced *after* the changes; there was never a
same-split run of the original code. This script closes that gap.

What it does
------------
1. Loads CirCor (BROKEN_RECORDINGS excluded, malformed tsvs skipped) with the
   repo's loader and makes the same 80/20 recording-level split as
   `evaluate_segmenter_all_datasets.py` (seed 42). The loader is shared on
   purpose: the upstream `utils.py` crashes on single-row tsvs and data
   loading is not part of the algorithm under test.
2. Trains two models on the identical training clips:
     upstream -- `third_party/springer_upstream/` imported flat, as shipped
                 (own feature extractor, own heart-rate defaults 60-200 bpm,
                 own Viterbi with hard duration cutoff etc.)
     local    -- `segmenter/` as it is now.
   Each model uses its own feature extractor for training too, since the
   feature-length fix in extract_features.py changes training features.
3. Predicts every test clip with both, scores per recording, and reports:
   mean / median / 25th / 75th / frame-weighted accuracy, catastrophic
   (<5%) and poor (5-50%) counts, confusion matrices, and a paired
   comparison (wins/ties/losses, mean delta with 10,000-sample paired
   bootstrap CI, 20,000-permutation sign-flip p-value).

Outputs
-------
  paths.UPSTREAM_VS_LOCAL_PER_RECORDING_CSV   one row per test recording
  paths.UPSTREAM_VS_LOCAL_SUMMARY_MD          the printed report, as markdown

Usage
-----
  py segmenter_eval/compare_upstream_vs_local.py               # full CirCor
  py segmenter_eval/compare_upstream_vs_local.py --dataset tiny_test
  py segmenter_eval/compare_upstream_vs_local.py --limit 50    # first 50 test recordings

Caveat: this is upstream-as-shipped vs local-as-is. It does not attribute the
difference to individual changes; that needs a per-change ablation.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repo root: makes `paths` and the packages importable from any cwd
import paths

import argparse
import csv
import importlib
import sys
import time

import numpy as np

from segmenter.duration_distributions import DataDistribution as LocalDataDistribution
from segmenter.segmentation_model import SegmentationModel as LocalSegmentationModel
from segmenter.data_loading import create_segmentation_array
from segmenter_eval.evaluate_segmenter_all_datasets import (
    load_dataset, split_80_20, BROKEN_RECORDINGS, STATE_LABELS, STATE_ORDER,
    BOUNDARY_WINDOW, RECORDING_FREQUENCY, FEATURE_FREQUENCY,
)

N_BOOT = 10_000
N_PERM = 20_000
SEED = 42
CATASTROPHIC = 0.05
POOR = 0.50


def import_upstream():
    """Import the vendored upstream modules under their original flat names."""
    up = str(paths.UPSTREAM_SPRINGER_DIR)
    if not paths.UPSTREAM_SPRINGER_DIR.is_dir():
        sys.exit(f"upstream snapshot not found at {up}")
    sys.path.insert(0, up)
    try:
        for name in ("get_dwt", "extract_features", "heart_rate",
                     "duration_distributions", "viterbi", "segmentation_model"):
            mod = importlib.import_module(name)
            assert mod.__file__.startswith(up), f"{name} resolved to {mod.__file__}, not the upstream snapshot"
        seg_mod = importlib.import_module("segmentation_model")
        dist_mod = importlib.import_module("duration_distributions")
    finally:
        sys.path.remove(up)
    return seg_mod.SegmentationModel, dist_mod.DataDistribution


def build_clips(recordings, segmentations, names):
    clips, annotations, clip_names = [], [], []
    for rec, seg, name in zip(recordings, segmentations, names):
        clipped, truth = create_segmentation_array(
            rec, seg, recording_frequency=RECORDING_FREQUENCY, feature_frequency=FEATURE_FREQUENCY)
        clips.extend(clipped)
        annotations.extend(truth)
        clip_names.extend([name] * len(clipped))
    return clips, annotations, clip_names


def score(annotations, predictions, clip_names):
    """Per-recording frame accuracy plus pooled confusion / boundary stats."""
    n_states = len(STATE_ORDER)
    state_to_idx = {s: i for i, s in enumerate(STATE_ORDER)}
    confusion = np.zeros((n_states, n_states), dtype=np.int64)
    per_rec = {}  # name -> [correct, total]
    b_correct = b_total = i_correct = i_total = 0

    for seg, pred, name in zip(annotations, predictions, clip_names):
        seg = np.asarray(seg)
        pred = np.asarray(pred)
        n = min(len(seg), len(pred))
        seg, pred = seg[:n], pred[:n]
        correct = seg == pred

        t = np.array([state_to_idx[s] for s in seg])
        p = np.array([state_to_idx.get(s, 0) for s in pred])
        confusion += np.bincount(t * n_states + p, minlength=n_states ** 2).reshape(n_states, n_states)

        boundary = np.zeros(n, dtype=bool)
        for cp in np.nonzero(np.diff(seg))[0]:
            boundary[max(0, cp - BOUNDARY_WINDOW + 1):min(n, cp + BOUNDARY_WINDOW + 1)] = True
        b_correct += correct[boundary].sum()
        b_total += boundary.sum()
        i_correct += correct[~boundary].sum()
        i_total += (~boundary).sum()

        per_rec.setdefault(name, [0, 0])
        per_rec[name][0] += int(correct.sum())
        per_rec[name][1] += n

    acc = {k: c / t for k, (c, t) in per_rec.items()}
    frames = {k: t for k, (c, t) in per_rec.items()}
    return {
        "acc": acc, "frames": frames, "confusion": confusion,
        "boundary_acc": b_correct / b_total if b_total else float("nan"),
        "interior_acc": i_correct / i_total if i_total else float("nan"),
    }


def summarise(label, res, lines):
    a = np.array(list(res["acc"].values()))
    w = np.array([res["frames"][k] for k in res["acc"]])
    lines.append(f"\n### {label}\n")
    lines.append("| n recordings | mean | median | p25 | p75 | frame-weighted | <5% | 5-50% | boundary | interior |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    lines.append(f"| {len(a)} | {a.mean():.4f} | {np.median(a):.4f} | {np.percentile(a, 25):.4f} | "
                 f"{np.percentile(a, 75):.4f} | {np.average(a, weights=w):.4f} | "
                 f"{(a < CATASTROPHIC).sum()} | {((a >= CATASTROPHIC) & (a < POOR)).sum()} | "
                 f"{res['boundary_acc']:.4f} | {res['interior_acc']:.4f} |")
    lines.append("\nConfusion (rows = true, cols = predicted):\n")
    lines.append("| | " + " | ".join(STATE_LABELS[s] for s in STATE_ORDER) + " |")
    lines.append("|---|" + "---|" * len(STATE_ORDER))
    for i, s in enumerate(STATE_ORDER):
        lines.append(f"| {STATE_LABELS[s]} | " + " | ".join(str(v) for v in res["confusion"][i]) + " |")


def paired(delta, rng):
    delta = np.asarray(delta)
    n = len(delta)
    boots = np.array([delta[rng.integers(0, n, n)].mean() for _ in range(N_BOOT)])
    ci = np.percentile(boots, [2.5, 97.5])
    obs = abs(delta.mean())
    signs = rng.choice([-1, 1], size=(N_PERM, n))
    perm = np.abs((signs * delta).mean(axis=1))
    p = (perm >= obs).sum() / N_PERM
    p_str = f"p < {1 / N_PERM:.5f}" if p == 0 else f"p = {p:.5f}"
    return ci, p_str


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="circor", choices=["circor", "tiny_test"])
    ap.add_argument("--limit", type=int, default=None, help="cap the number of test recordings")
    ap.add_argument("--train-limit", type=int, default=None, help="cap the number of training recordings")
    args = ap.parse_args()

    UpSegModel, UpDataDist = import_upstream()

    if args.dataset == "circor":
        recs, segs, names = load_dataset(paths.CIRCOR_DIR, "CirCor", exclude=BROKEN_RECORDINGS)
    else:
        recs, segs, names = load_dataset(paths.TINY_TEST_DIR, "tiny_test")

    train_set, test_set = split_80_20(names, seed=SEED)
    train_idx = [i for i, n in enumerate(names) if n in train_set]
    test_idx = [i for i, n in enumerate(names) if n in test_set]
    if args.train_limit:
        train_idx = train_idx[:args.train_limit]
    if args.limit:
        test_idx = test_idx[:args.limit]
    print(f"{args.dataset}: {len(train_idx)} train / {len(test_idx)} test recordings")

    train_segs = [segs[i] for i in train_idx]
    train_clips, train_ann, _ = build_clips([recs[i] for i in train_idx], train_segs, [names[i] for i in train_idx])
    test_clips, test_ann, test_names = build_clips([recs[i] for i in test_idx], [segs[i] for i in test_idx],
                                                   [names[i] for i in test_idx])
    print(f"{len(train_clips)} training clips, {len(test_clips)} test clips")

    # The upstream feature extractor raises ValueError on clips whose envelope,
    # wavelet and PSD columns resample to different lengths (observed only on
    # 1.02 s clips, 5 of 3296 on CirCor). The local extract_features.py fixes
    # this. So that both arms train on identical data, such clips are dropped
    # from BOTH arms, and the count/names are reported.
    up_probe = UpSegModel()
    dropped = {"train": [], "test": []}

    def usable(clips, ann, cnames, part):
        keep_c, keep_a, keep_n = [], [], []
        for c, a, n in zip(clips, ann, cnames):
            try:
                up_probe.feature_extractor(c, up_probe.sampling_frequency)
            except ValueError:
                dropped[part].append((n, len(c)))
                continue
            keep_c.append(c); keep_a.append(a); keep_n.append(n)
        return keep_c, keep_a, keep_n

    train_clips, train_ann, _ = usable(train_clips, train_ann, [""] * len(train_clips), "train")
    test_clips, test_ann, test_names = usable(test_clips, test_ann, test_names, "test")
    print(f"dropped {len(dropped['train'])} train / {len(dropped['test'])} test clips upstream cannot featurise: "
          f"{[n for n, _ in dropped['test']]}")

    arms = {
        "upstream": (UpSegModel, UpDataDist),
        "local": (LocalSegmentationModel, LocalDataDistribution),
    }
    results, timings = {}, {}
    for label, (Model, Dist) in arms.items():
        np.random.seed(SEED)
        t0 = time.time()
        model = Model()
        model.fit(train_clips, train_ann, data_distribution=Dist(train_segs))
        t_fit = time.time() - t0
        print(f"[{label}] trained in {t_fit:.0f} s; predicting {len(test_clips)} clips...", flush=True)
        t0 = time.time()
        preds = []
        for k, clip in enumerate(test_clips):
            preds.append(model.predict(clip))
            if (k + 1) % 50 == 0:
                print(f"  [{label}] {k + 1}/{len(test_clips)}  ({time.time() - t0:.0f} s)", flush=True)
        t_pred = time.time() - t0
        timings[label] = (t_fit, t_pred)
        results[label] = score(test_ann, preds, test_names)
        print(f"[{label}] predicted in {t_pred:.0f} s; mean per-recording accuracy "
              f"{np.mean(list(results[label]['acc'].values())):.4f}", flush=True)

    # ---- paired comparison ----
    common = [n for n in results["upstream"]["acc"] if n in results["local"]["acc"]]
    up = np.array([results["upstream"]["acc"][n] for n in common])
    lo = np.array([results["local"]["acc"][n] for n in common])
    delta = lo - up
    rng = np.random.default_rng(SEED)
    ci, p_str = paired(delta, rng)

    lines = [f"# Upstream vs local segmenter, {args.dataset}, split seed {SEED}\n",
             f"Train {len(train_idx)} recordings / {len(train_clips)} clips; "
             f"test {len(test_idx)} recordings / {len(test_clips)} clips. "
             f"Upstream snapshot: {paths.UPSTREAM_SPRINGER_DIR.relative_to(paths.ROOT)} (commit 276b9cc).",
             f"Dropped from both arms because the upstream feature extractor raises ValueError "
             f"(np.stack length mismatch): {len(dropped['train'])} train clips, {len(dropped['test'])} test clips"
             + (f" ({', '.join(f'{n} {L / RECORDING_FREQUENCY:.2f} s' for n, L in dropped['test'])})." if dropped['test'] else "."),
             f"Timing: upstream fit {timings['upstream'][0]:.0f} s / predict {timings['upstream'][1]:.0f} s; "
             f"local fit {timings['local'][0]:.0f} s / predict {timings['local'][1]:.0f} s."]
    summarise("Upstream (as shipped)", results["upstream"], lines)
    summarise("Local (segmenter/)", results["local"], lines)
    lines.append("\n### Paired per-recording comparison (local - upstream)\n")
    lines.append("| n | local wins | ties | upstream wins | mean delta (pts) | 95% bootstrap CI | sign-flip |")
    lines.append("|---|---|---|---|---|---|---|")
    lines.append(f"| {len(delta)} | {(delta > 0).sum()} | {(delta == 0).sum()} | {(delta < 0).sum()} | "
                 f"{100 * delta.mean():+.2f} | [{100 * ci[0]:+.2f}, {100 * ci[1]:+.2f}] | {p_str} |")
    lines.append(f"\nDelta percentiles (pts): p10 {100 * np.percentile(delta, 10):+.2f}, "
                 f"p25 {100 * np.percentile(delta, 25):+.2f}, median {100 * np.median(delta):+.2f}, "
                 f"p75 {100 * np.percentile(delta, 75):+.2f}, p90 {100 * np.percentile(delta, 90):+.2f}")
    order = np.argsort(delta)
    lines.append("\nLargest upstream advantages: " + ", ".join(
        f"{common[i]} ({100 * up[i]:.1f} -> {100 * lo[i]:.1f})" for i in order[:5]))
    lines.append("Largest local advantages: " + ", ".join(
        f"{common[i]} ({100 * up[i]:.1f} -> {100 * lo[i]:.1f})" for i in order[::-1][:5]))

    report = "\n".join(lines)
    print("\n" + report)

    with open(paths.UPSTREAM_VS_LOCAL_PER_RECORDING_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["recording", "n_frames", "acc_upstream", "acc_local", "delta"])
        for n in common:
            w.writerow([n, results["local"]["frames"][n], results["upstream"]["acc"][n],
                        results["local"]["acc"][n], results["local"]["acc"][n] - results["upstream"]["acc"][n]])
    with open(paths.UPSTREAM_VS_LOCAL_SUMMARY_MD, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"\nWrote {paths.UPSTREAM_VS_LOCAL_PER_RECORDING_CSV}\nWrote {paths.UPSTREAM_VS_LOCAL_SUMMARY_MD}")


if __name__ == "__main__":
    main()
