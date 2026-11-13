import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repo root: makes `paths` and the packages importable from any cwd
import paths

import csv
import numpy as np

from segmenter.duration_distributions import DataDistribution
from segmenter.segmentation_model import SegmentationModel
from segmenter.data_loading import get_wavs_and_tsvs, create_segmentation_array

STATE_LABELS = {0: "Unclassified", 1: "S1", 2: "Systole", 3: "S2", 4: "Diastole"}
STATE_ORDER = [0, 1, 2, 3, 4]
BOUNDARY_WINDOW = 5

CIRCOR_DIR = paths.CIRCOR_DIR
TINY_TEST_DIR = paths.TINY_TEST_DIR

BROKEN_RECORDINGS = {"49983_PV", "50074_MV", "50303_PV", "50693_AV", "85161_PV"}


def main():
    # --- Load 100% of CirCor as training data ---
    train_recordings, train_segmentations, train_names = get_wavs_and_tsvs(CIRCOR_DIR, return_names=True)

    filtered_recordings, filtered_segmentations, filtered_names = [], [], []
    excluded_count = 0
    for rec, seg, name in zip(train_recordings, train_segmentations, train_names):
        root_name = name.rsplit(".", 1)[0]  # strip .wav if get_wavs_and_tsvs returns full filenames
        if root_name in BROKEN_RECORDINGS:
            excluded_count += 1
            continue
        filtered_recordings.append(rec)
        filtered_segmentations.append(seg)
        filtered_names.append(root_name)

    print(f"Loaded {len(train_names)} CirCor recordings, excluded {excluded_count} known-broken ones "
          f"(expected 5: {BROKEN_RECORDINGS})")

    train_recordings = filtered_recordings
    train_segmentations = filtered_segmentations

    clips, annotations = [], []
    for rec, seg in zip(train_recordings, train_segmentations):
        clipped_recording, ground_truth = create_segmentation_array(rec, seg,
                                                                    recording_frequency=4000,
                                                                    feature_frequency=50)
        clips.extend(clipped_recording)
        annotations.extend(ground_truth)

    print(f"Training on {len(clips)} clips from {len(train_recordings)} recordings...")
    model = SegmentationModel()
    data_distribution = DataDistribution(train_segmentations)
    model.fit(clips, annotations, data_distribution=data_distribution)

    # --- Load tiny_test as the held-out test set ---
    test_recordings, test_segmentations, test_names = get_wavs_and_tsvs(TINY_TEST_DIR, return_names=True)
    test_names = [n.rsplit(".", 1)[0] for n in test_names]
    print(f"Testing on {len(test_names)} recordings from {TINY_TEST_DIR}")

    clips, annotations, clip_recording_names = [], [], []
    for rec, seg, name in zip(test_recordings, test_segmentations, test_names):
        clipped_recording, ground_truth = create_segmentation_array(rec, seg,
                                                                    recording_frequency=4000,
                                                                    feature_frequency=50)
        clips.extend(clipped_recording)
        annotations.extend(ground_truth)
        clip_recording_names.extend([name] * len(clipped_recording))

    idx = 0
    accuracies = np.zeros(len(clips))
    weights = np.zeros(len(clips))
    n_states = len(STATE_ORDER)
    confusion = np.zeros((n_states, n_states), dtype=np.int64)
    state_to_idx = {state: i for i, state in enumerate(STATE_ORDER)}
    boundary_correct = boundary_total = interior_correct = interior_total = 0
    recording_stats = {}

    for rec, seg in zip(clips, annotations):
        annotation = model.predict(rec)

        correct_mask = (seg == annotation)
        accuracies[idx] = correct_mask.mean()
        weights[idx] = seg.shape[0]

        true_idx = np.array([state_to_idx[s] for s in seg])
        pred_idx = np.array([state_to_idx.get(s, state_to_idx[0]) for s in annotation])
        flat_idx = true_idx * n_states + pred_idx
        confusion += np.bincount(flat_idx, minlength=n_states * n_states).reshape(n_states, n_states)

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

        name = clip_recording_names[idx]
        if name not in recording_stats:
            recording_stats[name] = [0, 0]
        recording_stats[name][0] += correct_mask.sum()
        recording_stats[name][1] += seg.shape[0]

        idx += 1

    print(f"\naverage accuracy: {accuracies.mean()}")
    print(f"average weight-corrected accuracy: {np.average(accuracies, weights=weights)}")

    print("\nConfusion matrix (rows = true state, cols = predicted state):")
    header = "        " + "".join(f"{STATE_LABELS[s]:>10}" for s in STATE_ORDER)
    print(header)
    for i, true_state in enumerate(STATE_ORDER):
        row = "".join(f"{confusion[i, j]:>10}" for j in range(n_states))
        print(f"{STATE_LABELS[true_state]:>8}{row}")

    boundary_acc = boundary_correct / boundary_total if boundary_total > 0 else float("nan")
    interior_acc = interior_correct / interior_total if interior_total > 0 else float("nan")
    print(f"\nBoundary-region accuracy: {boundary_acc:.4f} ({int(boundary_total)} frames)")
    print(f"Interior accuracy: {interior_acc:.4f} ({int(interior_total)} frames)")

    per_recording_accuracy = {
        name: (correct / total if total > 0 else float("nan"))
        for name, (correct, total) in recording_stats.items()
    }
    print(f"\nPer-recording accuracy across {len(per_recording_accuracy)} tiny_test recordings:")
    for name, acc in sorted(per_recording_accuracy.items(), key=lambda kv: kv[1]):
        print(f"    {name}: {acc:.4f}")

    csv_path = paths.SEGMENTER_RESULTS_DIR / "tiny_test_accuracy_circor_trained.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["recording", "accuracy", "n_frames"])
        for name, (correct, total) in recording_stats.items():
            writer.writerow([name, per_recording_accuracy[name], total])
    print(f"\nWrote {csv_path}")


if __name__ == "__main__":
    main()