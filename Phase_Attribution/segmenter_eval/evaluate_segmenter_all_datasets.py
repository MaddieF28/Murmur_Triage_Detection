"""
80/20 split for every dataset (CirCor, tiny_test, training-a..f), trains on
them all at the same time and compares performance.

Generalized from the training-a-only version: the training-a constant is
replaced with a dict covering all six PhysioNet 2016 challenge subsets, each
loaded (and skipped with a warning, not a crash, if its converted_tsvs
directory isn't there yet -- e.g. you haven't run data_prep/convert_physionet2016_annotations_to_tsv.py /
data_prep/resample_physionet2016_to_4000hz.py for it). Everything downstream -- the 80/20
split, training, and reporting -- already iterated generically over a
`datasets` dict, so none of that logic needed to change.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repo root: makes `paths` and the packages importable from any cwd
import paths


import csv
import os
import random

import numpy as np

from segmenter.duration_distributions import DataDistribution
from segmenter.segmentation_model import SegmentationModel
from segmenter.data_loading import get_wavs_and_tsvs, create_segmentation_array

STATE_LABELS = {0: "Unclassified", 1: "S1", 2: "Systole", 3: "S2", 4: "Diastole"}
STATE_ORDER = [0, 1, 2, 3, 4]
BOUNDARY_WINDOW = 5
TEST_FRACTION = 0.20
RANDOM_SEED = 42  # fixed for reproducible splits -- change if you want a different split each run
RECORDING_FREQUENCY = 4000  # all datasets are now at true 4000 Hz
FEATURE_FREQUENCY = 50

CIRCOR_DIR = paths.CIRCOR_DIR
TINY_TEST_DIR = paths.TINY_TEST_DIR

TRAINING_LETTERS = "abcdef"
# each of these must contain BOTH .tsv and 4000 Hz .wav files -- produced by
# data_prep/convert_physionet2016_annotations_to_tsv.py followed by data_prep/resample_physionet2016_to_4000hz.py
TRAINING_DIRS = paths.TRAINING_DIRS

BROKEN_RECORDINGS = {"49983_PV", "50074_MV", "50303_PV", "50693_AV", "85161_PV"}

# any recording scoring below this gets a true-vs-predicted label distribution
# dump -- exact-zero or near-zero accuracy with 5 possible states usually means
# a total mismatch (e.g. model output stuck on one state) rather than "just hard"
ZERO_ACCURACY_DIAGNOSTIC_THRESHOLD = 0.05


def split_80_20(names, seed=RANDOM_SEED):
    names = list(names)
    rng = random.Random(seed)
    rng.shuffle(names)
    n_test = max(1, round(len(names) * TEST_FRACTION)) if names else 0
    test_names = set(names[:n_test])
    train_names = set(names[n_test:])
    return train_names, test_names


def load_dataset(directory, dataset_label, exclude=None):
    recordings, segmentations, names = get_wavs_and_tsvs(directory, return_names=True)
    names = [n.rsplit(".", 1)[0] for n in names]

    if exclude:
        keep = [(r, s, n) for r, s, n in zip(recordings, segmentations, names) if n not in exclude]
        excluded_count = len(names) - len(keep)
        recordings, segmentations, names = map(list, zip(*keep)) if keep else ([], [], [])
        print(f"{dataset_label}: loaded {len(names)} recordings (excluded {excluded_count} known-broken)")
    else:
        print(f"{dataset_label}: loaded {len(names)} recordings")

    return recordings, segmentations, names


def build_clips(recordings, segmentations, names, dataset_label):
    clips, annotations, clip_names, clip_datasets = [], [], [], []
    for rec, seg, name in zip(recordings, segmentations, names):
        clipped_recording, ground_truth = create_segmentation_array(
            rec, seg, recording_frequency=RECORDING_FREQUENCY, feature_frequency=FEATURE_FREQUENCY
        )
        clips.extend(clipped_recording)
        annotations.extend(ground_truth)
        clip_names.extend([name] * len(clipped_recording))
        clip_datasets.extend([dataset_label] * len(clipped_recording))
    return clips, annotations, clip_names, clip_datasets


def compute_predictions(model, clips):
    return [model.predict(clip) for clip in clips]


def rle_string(seq, state_labels=STATE_LABELS):
    """Compresses a sequence of state codes into 'Label×count,Label×count,...' form."""
    seq = list(seq)
    if not seq:
        return ""
    parts = []
    current = seq[0]
    count = 1
    for s in seq[1:]:
        if s == current:
            count += 1
        else:
            parts.append(f"{state_labels[current]}×{count}")
            current = s
            count = 1
    parts.append(f"{state_labels[current]}×{count}")
    return ",".join(parts)


def print_zero_accuracy_diagnostics(recording_accuracy, recording_label_stats, label,
                                     threshold=ZERO_ACCURACY_DIAGNOSTIC_THRESHOLD):
    flagged = {name: acc for name, acc in recording_accuracy.items() if acc < threshold}
    if not flagged:
        return

    print(f"\n--- {label}: {len(flagged)} recording(s) below {threshold:.0%} accuracy "
          f"(true vs predicted label distribution + raw sequence) ---")
    for name, acc in sorted(flagged.items(), key=lambda kv: kv[1]):
        stats = recording_label_stats[name]
        true_counts = stats["true_counts"]
        pred_counts = stats["pred_counts"]
        total = true_counts.sum()
        duration_sec = total / FEATURE_FREQUENCY
        print(f"\n  {name}  (accuracy={acc:.4f}, {stats['n_clips']} clips, "
              f"{total} frames, ~{duration_sec:.2f}s)")
        header = "    " + "".join(f"{STATE_LABELS[s]:>14}" for s in STATE_ORDER)
        print(header)
        true_row = "    " + "".join(f"{true_counts[i]:>14}" for i in range(len(STATE_ORDER)))
        pred_row = "    " + "".join(f"{pred_counts[i]:>14}" for i in range(len(STATE_ORDER)))
        print(f"    true:{true_row}")
        print(f"    pred:{pred_row}")

        full_true_seq = np.concatenate(stats["seg_sequence"])
        full_pred_seq = np.concatenate(stats["pred_sequence"])
        print(f"    true sequence: {rle_string(full_true_seq)}")
        print(f"    pred sequence: {rle_string(full_pred_seq)}")


def report_metrics(annotations, predictions, clip_names, label):
    n_states = len(STATE_ORDER)
    state_to_idx = {state: i for i, state in enumerate(STATE_ORDER)}
    confusion = np.zeros((n_states, n_states), dtype=np.int64)
    accuracies = np.zeros(len(annotations))
    weights = np.zeros(len(annotations))
    boundary_correct = boundary_total = interior_correct = interior_total = 0
    recording_stats = {}
    recording_label_stats = {}  # name -> {"true_counts": arr, "pred_counts": arr, "n_clips": int}

    for idx, (seg, pred, name) in enumerate(zip(annotations, predictions, clip_names)):
        correct_mask = (seg == pred)
        accuracies[idx] = correct_mask.mean()
        weights[idx] = seg.shape[0]

        true_idx = np.array([state_to_idx[s] for s in seg])
        pred_idx = np.array([state_to_idx.get(s, state_to_idx[0]) for s in pred])
        flat_idx = true_idx * n_states + pred_idx
        confusion += np.bincount(flat_idx, minlength=n_states * n_states).reshape(n_states, n_states)

        if name not in recording_label_stats:
            recording_label_stats[name] = {
                "true_counts": np.zeros(n_states, dtype=np.int64),
                "pred_counts": np.zeros(n_states, dtype=np.int64),
                "n_clips": 0,
                "seg_sequence": [],
                "pred_sequence": [],
            }
        recording_label_stats[name]["true_counts"] += np.bincount(true_idx, minlength=n_states)
        recording_label_stats[name]["pred_counts"] += np.bincount(pred_idx, minlength=n_states)
        recording_label_stats[name]["n_clips"] += 1
        recording_label_stats[name]["seg_sequence"].append(seg)
        recording_label_stats[name]["pred_sequence"].append(pred)

        boundary_mask = np.zeros(seg.shape[0], dtype=bool)
        change_points = np.nonzero(np.diff(seg))[0]
        for cp in change_points:
            lo = max(0, cp - BOUNDARY_WINDOW + 1)
            hi = min(seg.shape[0], cp + BOUNDARY_WINDOW + 1)
            boundary_mask[lo:hi] = True
        interior_mask = ~boundary_mask

        boundary_correct += correct_mask[boundary_mask].sum()
        boundary_total += boundary_mask.sum()
        interior_correct += correct_mask[interior_mask].sum()
        interior_total += interior_mask.sum()

        if name not in recording_stats:
            recording_stats[name] = [0, 0]
        recording_stats[name][0] += correct_mask.sum()
        recording_stats[name][1] += seg.shape[0]

    print(f"\n=== {label} ===")
    print(f"clips: {len(annotations)}")
    if len(annotations) == 0:
        print("(no clips -- nothing to report)")
        return {}

    print(f"average accuracy: {accuracies.mean():.4f}")
    print(f"average weight-corrected accuracy: {np.average(accuracies, weights=weights):.4f}")

    print("Confusion matrix (rows = true state, cols = predicted state):")
    header = "        " + "".join(f"{STATE_LABELS[s]:>10}" for s in STATE_ORDER)
    print(header)
    for i, true_state in enumerate(STATE_ORDER):
        row = "".join(f"{confusion[i, j]:>10}" for j in range(n_states))
        print(f"{STATE_LABELS[true_state]:>8}{row}")

    boundary_acc = boundary_correct / boundary_total if boundary_total > 0 else float("nan")
    interior_acc = interior_correct / interior_total if interior_total > 0 else float("nan")
    print(f"Boundary-region accuracy: {boundary_acc:.4f} ({int(boundary_total)} frames)")
    print(f"Interior accuracy: {interior_acc:.4f} ({int(interior_total)} frames)")

    recording_accuracy = {
        name: (correct / total if total > 0 else float("nan"))
        for name, (correct, total) in recording_stats.items()
    }
    print_zero_accuracy_diagnostics(recording_accuracy, recording_label_stats, label)
    return recording_accuracy


def main():
    circor_rec, circor_seg, circor_names = load_dataset(CIRCOR_DIR, "CirCor", exclude=BROKEN_RECORDINGS)
    tiny_rec, tiny_seg, tiny_names = load_dataset(TINY_TEST_DIR, "tiny_test")

    datasets = {
        "CirCor": (circor_rec, circor_seg, circor_names),
        "tiny_test": (tiny_rec, tiny_seg, tiny_names),
    }

    for label, directory in TRAINING_DIRS.items():
        if not os.path.isdir(directory):
            print(f"{label}: SKIPPED -- directory not found: {directory} "
                  f"(run data_prep/convert_physionet2016_annotations_to_tsv.py and data_prep/resample_physionet2016_to_4000hz.py first)")
            continue
        rec, seg, names = load_dataset(directory, label)
        datasets[label] = (rec, seg, names)

    train_clips, train_annotations = [], []
    train_segmentations_for_distribution = []
    test_by_dataset = {}

    for label, (recs, segs, names) in datasets.items():
        train_set, test_set = split_80_20(names)
        train_idx = [i for i, n in enumerate(names) if n in train_set]
        test_idx = [i for i, n in enumerate(names) if n in test_set]

        train_recs_d = [recs[i] for i in train_idx]
        train_segs_d = [segs[i] for i in train_idx]
        train_segmentations_for_distribution.extend(train_segs_d)

        clips_d, ann_d, _, _ = build_clips(train_recs_d, train_segs_d, [names[i] for i in train_idx], label)
        train_clips.extend(clips_d)
        train_annotations.extend(ann_d)

        test_by_dataset[label] = (
            [recs[i] for i in test_idx],
            [segs[i] for i in test_idx],
            [names[i] for i in test_idx],
        )
        print(f"{label}: {len(train_idx)} train / {len(test_idx)} test ({len(names)} total)")

    print(f"\nTraining on {len(train_clips)} clips "
          f"(combined across {', '.join(datasets.keys())}, all at {RECORDING_FREQUENCY} Hz)...")
    model = SegmentationModel()
    data_distribution = DataDistribution(train_segmentations_for_distribution)
    model.fit(train_clips, train_annotations, data_distribution=data_distribution)

    all_annotations, all_predictions, all_clip_names = [], [], []
    per_recording_all = {}

    for label, (test_recs, test_segs, test_names_list) in test_by_dataset.items():
        clips_d, ann_d, names_d, _ = build_clips(test_recs, test_segs, test_names_list, label)
        preds_d = compute_predictions(model, clips_d)
        per_recording_accuracy = report_metrics(ann_d, preds_d, names_d, label)

        all_annotations.extend(ann_d)
        all_predictions.extend(preds_d)
        all_clip_names.extend(names_d)

        for name, acc in per_recording_accuracy.items():
            per_recording_all[f"{label}/{name}"] = acc

    report_metrics(all_annotations, all_predictions, all_clip_names, "ALL DATASETS COMBINED")

    csv_path = paths.TEST_ACCURACY_BY_RECORDING_CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset/recording", "accuracy"])
        for name, acc in sorted(per_recording_all.items(), key=lambda kv: kv[1]):
            writer.writerow([name, acc])
    print(f"\nWrote {csv_path}")


if __name__ == "__main__":
    main()