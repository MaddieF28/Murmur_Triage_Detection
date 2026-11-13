import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repo root: makes `paths` and the packages importable from any cwd
import paths

import sys
import random
import csv

from segmenter.duration_distributions import DataDistribution
from segmenter.segmentation_model import SegmentationModel
from segmenter.data_loading import get_wavs_and_tsvs, get_heart_rate_from_tsv, create_segmentation_array, create_train_test_split
import numpy as np

STATE_LABELS = {0: "Unclassified", 1: "S1", 2: "Systole", 3: "S2", 4: "Diastole"}
STATE_ORDER = [0, 1, 2, 3, 4]
BOUNDARY_WINDOW = 5  # feature-frames on each side of a true transition counted as "boundary" (~100ms at 50Hz)


def main(data_dir, train_frac=0.8, heartrates_from_tsv=False):
    # Get training recordings and segmentations (patient-level split, no size cap)
    train_recordings, train_segmentations, \
        test_recordings, test_segmentations, \
        train_names, test_names = create_train_test_split(directory=data_dir,
                                                          frac_train=train_frac,
                                                          return_names=True)

    # Preprocess into clips with annotations of the same length
    clips = []
    annotations = []
    for rec, seg in zip(train_recordings, train_segmentations):
        clipped_recording, ground_truth = create_segmentation_array(rec,
                                                                    seg,
                                                                    recording_frequency=4000,
                                                                    feature_frequency=50)
        clips.extend(clipped_recording)
        annotations.extend(ground_truth)

    # Train the feature_prob_model
    model = SegmentationModel()
    data_distribution = DataDistribution(train_segmentations)
    model.fit(clips, annotations, data_distribution=data_distribution)

    # Process test set in to clips and annotations, tracking which recording each clip came from
    clips = []
    annotations = []
    heartrates = []
    clip_recording_names = []
    for rec, seg, name in zip(test_recordings, test_segmentations, test_names):
        clipped_recording, ground_truth = create_segmentation_array(rec,
                                                                    seg,
                                                                    recording_frequency=4000,
                                                                    feature_frequency=50)
        if heartrates_from_tsv:
            heartrate = get_heart_rate_from_tsv(seg)
            for _ in range(len(clipped_recording)):
                heartrates.append(heartrate)
        clips.extend(clipped_recording)
        annotations.extend(ground_truth)
        clip_recording_names.extend([name] * len(clipped_recording))

    # Evaluate performance on clips
    idx = 0
    accuracies = np.zeros(len(clips))
    weights = np.zeros(len(clips))

    n_states = len(STATE_ORDER)
    confusion = np.zeros((n_states, n_states), dtype=np.int64)  # rows = true state, cols = predicted state
    state_to_idx = {state: i for i, state in enumerate(STATE_ORDER)}

    boundary_correct = 0
    boundary_total = 0
    interior_correct = 0
    interior_total = 0

    recording_stats = {}  # name -> [correct_frames, total_frames]

    for rec, seg in zip(clips, annotations):
        if heartrates_from_tsv:
            annotation = model.predict(rec, heart_rate=heartrates[idx])
        else:
            annotation = model.predict(rec)

        correct_mask = (seg == annotation)
        accuracies[idx] = correct_mask.mean()
        weights[idx] = seg.shape[0]

        # confusion matrix
        true_idx = np.array([state_to_idx[s] for s in seg])
        pred_idx = np.array([state_to_idx.get(s, state_to_idx[0]) for s in annotation])
        flat_idx = true_idx * n_states + pred_idx
        confusion += np.bincount(flat_idx, minlength=n_states * n_states).reshape(n_states, n_states)

        # boundary vs interior
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

        # per-recording accuracy
        name = clip_recording_names[idx]
        if name not in recording_stats:
            recording_stats[name] = [0, 0]
        recording_stats[name][0] += correct_mask.sum()
        recording_stats[name][1] += seg.shape[0]

        idx += 1

    print(f"average accuracy: {accuracies.mean()}")
    print(f"average weight-corrected accuracy: {np.average(accuracies, weights=weights)}")

    # --- Confusion matrix ---
    print("\nConfusion matrix (rows = true state, cols = predicted state):")
    header = "        " + "".join(f"{STATE_LABELS[s]:>10}" for s in STATE_ORDER)
    print(header)
    for i, true_state in enumerate(STATE_ORDER):
        row = "".join(f"{confusion[i, j]:>10}" for j in range(n_states))
        print(f"{STATE_LABELS[true_state]:>8}{row}")

    # --- Boundary vs interior accuracy ---
    boundary_acc = boundary_correct / boundary_total if boundary_total > 0 else float("nan")
    interior_acc = interior_correct / interior_total if interior_total > 0 else float("nan")
    print(f"\nBoundary-region accuracy (within {BOUNDARY_WINDOW} frames of a transition): "
          f"{boundary_acc:.4f} ({int(boundary_total)} frames)")
    print(f"Interior accuracy (away from transitions): {interior_acc:.4f} ({int(interior_total)} frames)")

    # --- Per-recording accuracy ---
    per_recording_accuracy = {
        name: (correct / total if total > 0 else float("nan"))
        for name, (correct, total) in recording_stats.items()
    }
    acc_values = np.array(list(per_recording_accuracy.values()))
    print(f"\nPer-recording accuracy across {len(acc_values)} test recordings:")
    print(f"  mean: {acc_values.mean():.4f}  std: {acc_values.std():.4f}  "
          f"min: {acc_values.min():.4f}  max: {acc_values.max():.4f}")

    worst = sorted(per_recording_accuracy.items(), key=lambda kv: kv[1])[:10]
    print("  worst 10 recordings:")
    for name, acc in worst:
        print(f"    {name}: {acc:.4f}")

    csv_path = paths.PER_RECORDING_ACCURACY_CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["recording", "accuracy", "n_frames"])
        for name, (correct, total) in recording_stats.items():
            writer.writerow([name, per_recording_accuracy[name], total])
    print(f"\nWrote per-recording accuracy to {csv_path}")


if __name__ == "__main__":
    random.seed(0)
    data_dir = sys.argv[1]
    train_frac = float(sys.argv[2]) if len(sys.argv) > 2 else 0.8
    main(data_dir, train_frac=train_frac, heartrates_from_tsv=False)