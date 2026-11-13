"""
Trains ONE model on ONE train/test split, then evaluates it TWICE on the
SAME test set - once with autodetected heart rate, once with TSV-derived
true heart rate - so the comparison isn't confounded by split randomness.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repo root: makes `paths` and the packages importable from any cwd
import paths

import random
import numpy as np

from segmenter.duration_distributions import DataDistribution
from segmenter.segmentation_model import SegmentationModel
from segmenter.data_loading import get_heart_rate_from_tsv, create_segmentation_array, create_train_test_split

DATA_DIR = paths.TRAINING_DIRS["training-a"]  # adjust if needed
RECORDING_FREQUENCY = 4000  # adjust to your actual sample rate if different
FEATURE_FREQUENCY = 50

random.seed(0)
train_recordings, train_segmentations, test_recordings, test_segmentations, \
    train_names, test_names = create_train_test_split(directory=DATA_DIR, frac_train=0.8, return_names=True)

# --- train once ---
train_clips, train_annotations = [], []
for rec, seg in zip(train_recordings, train_segmentations):
    c, g = create_segmentation_array(rec, seg, recording_frequency=RECORDING_FREQUENCY, feature_frequency=FEATURE_FREQUENCY)
    train_clips.extend(c)
    train_annotations.extend(g)

model = SegmentationModel()
data_distribution = DataDistribution(train_segmentations)
model.fit(train_clips, train_annotations, data_distribution=data_distribution)

# --- build test clips ONCE, keep the true heart rate per clip alongside ---
test_items = []  # (clip, true_states, true_heart_rate, recording_name)
for rec, seg, name in zip(test_recordings, test_segmentations, test_names):
    clipped_recording, ground_truth = create_segmentation_array(rec, seg, recording_frequency=RECORDING_FREQUENCY, feature_frequency=FEATURE_FREQUENCY)
    try:
        true_hr = get_heart_rate_from_tsv(seg)
    except Exception:
        true_hr = None
    for clip, states in zip(clipped_recording, ground_truth):
        test_items.append((clip, states, true_hr, name))

print(f"{len(test_items)} test clips across {len(test_names)} recordings\n")

# --- evaluate with AUTODETECTED heart rate ---
accs_auto = []
for clip, true_states, true_hr, name in test_items:
    pred = model.predict(clip)  # heart_rate=None -> autodetected
    n = min(len(true_states), len(pred))
    accs_auto.append((np.array(true_states[:n]) == np.array(pred[:n])).mean())

# --- evaluate with TRUE (TSV-derived) heart rate ---
accs_true = []
for clip, true_states, true_hr, name in test_items:
    if true_hr is None or np.isnan(true_hr):
        continue
    pred = model.predict(clip, heart_rate=true_hr)
    n = min(len(true_states), len(pred))
    accs_true.append((np.array(true_states[:n]) == np.array(pred[:n])).mean())

print(f"Mean accuracy, AUTODETECTED heart rate: {np.mean(accs_auto):.4f}  (n={len(accs_auto)})")
print(f"Mean accuracy, TRUE (TSV) heart rate:    {np.mean(accs_true):.4f}  (n={len(accs_true)})")
print(f"\nDifference: {np.mean(accs_true) - np.mean(accs_auto):+.4f}")