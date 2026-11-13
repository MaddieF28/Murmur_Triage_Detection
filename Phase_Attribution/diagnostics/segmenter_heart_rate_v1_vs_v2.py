"""
Trains ONE model, then evaluates it twice on the SAME test clips:
once using heart rate estimates from the OLD get_heart_rate (fastest-candidate-first),
once using the PATCHED version (strongest-peak-first).
Requires diagnostics/heart_rate_v2_patch.py to be in the same folder (or import from wherever you saved it).
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repo root: makes `paths` and the packages importable from any cwd
import paths

import random
import numpy as np

from segmenter.duration_distributions import DataDistribution
from segmenter.segmentation_model import SegmentationModel
from segmenter.data_loading import create_segmentation_array, create_train_test_split
from segmenter.heart_rate import get_heart_rate as get_heart_rate_OLD
from diagnostics.heart_rate_v2_patch import get_heart_rate as get_heart_rate_NEW

DATA_DIR = paths.TRAINING_DIRS["training-a"]  # adjust if needed
RECORDING_FREQUENCY = 4000
FEATURE_FREQUENCY = 50

random.seed(0)
train_recordings, train_segmentations, test_recordings, test_segmentations, \
    train_names, test_names = create_train_test_split(directory=DATA_DIR, frac_train=0.8, return_names=True)

train_clips, train_annotations = [], []
for rec, seg in zip(train_recordings, train_segmentations):
    c, g = create_segmentation_array(rec, seg, recording_frequency=RECORDING_FREQUENCY, feature_frequency=FEATURE_FREQUENCY)
    train_clips.extend(c)
    train_annotations.extend(g)

model = SegmentationModel()
data_distribution = DataDistribution(train_segmentations)
model.fit(train_clips, train_annotations, data_distribution=data_distribution)

test_items = []
for rec, seg, name in zip(test_recordings, test_segmentations, test_names):
    clipped, ground_truth = create_segmentation_array(rec, seg, recording_frequency=RECORDING_FREQUENCY, feature_frequency=FEATURE_FREQUENCY)
    for clip, states in zip(clipped, ground_truth):
        test_items.append((clip, states))

print(f"{len(test_items)} test clips\n")

def eval_with(hr_func, min_hr=40, max_hr=200):
    accs = []
    for clip, true_states in test_items:
        heart_rates, systolic_intervals = hr_func(clip, model.sampling_frequency,
                                                    min_heart_rate=min_hr, max_heart_rate=max_hr, multiple_rates=True)
        pred = model.predict(clip, heart_rate=heart_rates[0])
        n = min(len(true_states), len(pred))
        accs.append((np.array(true_states[:n]) == np.array(pred[:n])).mean())
    return np.mean(accs)

acc_old = eval_with(get_heart_rate_OLD)
acc_new = eval_with(get_heart_rate_NEW)

print(f"OLD (fastest-candidate-first): {acc_old:.4f}")
print(f"NEW (strongest-peak-first):    {acc_new:.4f}")
print(f"Difference: {acc_new - acc_old:+.4f}")