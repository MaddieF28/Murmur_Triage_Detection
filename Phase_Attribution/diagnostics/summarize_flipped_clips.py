"""
diagnostics/summarize_flipped_clips.py

Trains the model exactly as segmenter_eval/evaluate_segmenter_all_datasets.py's main() does (imports its
loading/splitting functions directly rather than reimplementing them, so
this can't silently drift out of sync with what produced your accuracy
report), then automatically:

  1. Finds the 14 known near-zero-accuracy CirCor clips from your report.
  2. Samples a handful of random clips that scored well, for comparison.
  3. Runs diagnose_flips.summarize_clip on all of them and prints/writes
     ONE COMPACT ROW PER CLIP -- accuracy, which rotation the model's own
     math scores highest, the verify_against_real check, and the
     tail-clamp-distortion / backtrack-mismatch counts -- instead of the
     full per-frame internals dump. See DEEP_DIVE_CLIPS below if you want
     the full dump (delta grid, backtrack trace, etc.) for one or two
     specific clips once the summary table points you at them.
  4. Writes the same table to diagnostic_output.txt (and still shows it on
     screen as it runs), so you can send the file back.

Run this from the same directory you run segmenter_eval/evaluate_segmenter_all_datasets.py from.

Two things worth knowing, not blocking:
  - build_clips()'s output IS raw audio (create_segmentation_array slices
    the original recording directly), so passing it straight into
    diagnose_clip_deep's feature_extractor(...)/get_heart_rate(...) calls
    is correct -- no double-processing.
  - model.fit() is always called here with an explicit data_distribution
    instance, which sidesteps a latent bug in SegmentationModel.fit():
    when data_distribution is None it sets self.data_distribution to the
    DataDistribution *class* itself (`self.data_distribution =
    DataDistribution`, no instantiation), not an instance. That would break
    at predict() time if ever hit -- it isn't hit here, but worth knowing
    if you ever call model.fit(...) without passing data_distribution.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repo root: makes `paths` and the packages importable from any cwd
import paths

import os
import random
import sys

import numpy as np

from segmenter.duration_distributions import DataDistribution
from segmenter.segmentation_model import SegmentationModel
from segmenter_eval.evaluate_segmenter_all_datasets import (
    load_dataset, build_clips, split_80_20,
    CIRCOR_DIR, TINY_TEST_DIR, TRAINING_DIRS, BROKEN_RECORDINGS,
    RANDOM_SEED,
)
from diagnostics.diagnose_flips import (
    diagnose_clip_deep, summarize_clip, print_clip_summary_table,
    DEFAULT_MIN_HEART_RATE, DEFAULT_MAX_HEART_RATE,
)

# the 14 clips that came back at (near-)zero accuracy in your CirCor report
KNOWN_BAD_CLIPS = {
    "49638_MV", "50007_AV", "50296_AV", "50304_PV", "68484_TV", "69060_TV",
    "74417_MV", "78582_AV", "84738_PV", "85145_MV", "85202_AV", "85213_TV",
    "85282_PV", "50146_MV",
}

# Clip names (a subset of KNOWN_BAD_CLIPS, or good-clip names) to also run
# through the FULL diagnose_clip_deep dump, in addition to the summary
# table -- use this once the summary table points at specific clips worth
# looking at frame-by-frame. Empty by default: the summary table alone is
# the point of this run.
#
# Set to these two because they're the anomaly in the first summary run:
# best-scoring candidate was "ground_truth" (not actual_output, not a
# flip) yet real-decode accuracy was still ~0%. Either the real Viterbi
# search missed its own global optimum, or score_path's re-scoring doesn't
# exactly reproduce a boundary/clamping detail of the real DP -- worth
# resolving before trusting the other 12 clips' best-scoring labels.
DEEP_DIVE_CLIPS = {"85213_TV", "69060_TV"}

# Also print each segment's emission/duration/transition breakdown (not
# just the total log-likelihood) for the deep-dive clips above -- needed
# here to see WHERE ground truth's score comes from, segment by segment.
DEEP_DIVE_VERBOSE_SEGMENTS = True

N_RANDOM_GOOD_CLIPS = 10
GOOD_CLIP_ACCURACY_THRESHOLD = 0.85  # only sample from clips that scored at least this well
OUTPUT_FILE = paths.DIAGNOSTICS_RESULTS_DIR / "diagnostic_output.txt"


class Tee:
    """Writes to two streams at once, so print() calls show up both on the
    console (progress) and in the output file (to send back)."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


def train_model():
    """Mirrors run_all_datasets.main()'s training block exactly, so the
    model here matches the one that produced your accuracy report."""
    circor_rec, circor_seg, circor_names = load_dataset(CIRCOR_DIR, "CirCor", exclude=BROKEN_RECORDINGS)
    tiny_rec, tiny_seg, tiny_names = load_dataset(TINY_TEST_DIR, "tiny_test")

    datasets = {
        "CirCor": (circor_rec, circor_seg, circor_names),
        "tiny_test": (tiny_rec, tiny_seg, tiny_names),
    }
    for label, directory in TRAINING_DIRS.items():
        if not os.path.isdir(directory):
            print(f"{label}: SKIPPED -- directory not found: {directory}")
            continue
        rec, seg, names = load_dataset(directory, label)
        datasets[label] = (rec, seg, names)

    train_clips, train_annotations = [], []
    train_segmentations_for_distribution = []
    test_by_dataset = {}

    for label, (recs, segs, names) in datasets.items():
        train_set, test_set = split_80_20(names, seed=RANDOM_SEED)
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

    print(f"Training on {len(train_clips)} clips...")
    model = SegmentationModel()
    data_distribution = DataDistribution(train_segmentations_for_distribution)
    model.fit(train_clips, train_annotations, data_distribution=data_distribution)
    return model, test_by_dataset


def build_named_clip_index(test_by_dataset, dataset_label):
    """Returns (clips, annotations, names) for every clip in a dataset's
    test split, plus a name -> [(clip, ann), ...] lookup (a name can map to
    more than one clip if create_segmentation_array split that recording
    into multiple pieces)."""
    recs, segs, names = test_by_dataset[dataset_label]
    clips, ann, clip_names, _ = build_clips(recs, segs, names, dataset_label)
    by_name = {}
    for clip, a, name in zip(clips, ann, clip_names):
        by_name.setdefault(name, []).append((clip, a))
    return clips, ann, clip_names, by_name


def pick_random_good_clips(model, clips, annotations, clip_names, n, min_accuracy, seed=RANDOM_SEED):
    """Samples up to n clip names whose per-clip accuracy under this model
    is at least min_accuracy, for comparison against the known-bad ones."""
    rng = random.Random(seed)
    candidates = []
    seen_names = set()
    for clip, ann, name in zip(clips, annotations, clip_names):
        if name in seen_names:
            continue
        # must use the same heart-rate bounds diagnose_clip_deep/summarize_clip
        # use below, or "good" here is measured under a different search
        # window than what the diagnostic actually reports for these clips
        pred = model.predict(clip, min_heart_rate=DEFAULT_MIN_HEART_RATE, max_heart_rate=DEFAULT_MAX_HEART_RATE)
        min_len = min(len(pred), len(ann))
        acc = float(np.mean(np.asarray(pred[:min_len]) == np.asarray(ann[:min_len])))
        if acc >= min_accuracy:
            candidates.append(name)
            seen_names.add(name)
    rng.shuffle(candidates)
    return candidates[:n]


def main():
    model, test_by_dataset = train_model()

    # all 14 known-bad clips are from CirCor per your report
    circor_clips, circor_ann, circor_names, by_name = build_named_clip_index(test_by_dataset, "CirCor")

    missing = KNOWN_BAD_CLIPS - set(by_name.keys())
    if missing:
        print(f"WARNING: {len(missing)} known-bad clip name(s) not found in this run's "
              f"CirCor test split (the split is seeded, so this shouldn't happen unless "
              f"the dataset on disk changed): {sorted(missing)}")

    good_names = pick_random_good_clips(model, circor_clips, circor_ann, circor_names,
                                        n=N_RANDOM_GOOD_CLIPS, min_accuracy=GOOD_CLIP_ACCURACY_THRESHOLD)

    found_bad = {name: by_name[name] for name in KNOWN_BAD_CLIPS if name in by_name}

    with open(OUTPUT_FILE, "w") as f:
        original_stdout = sys.stdout
        sys.stdout = Tee(original_stdout, f)
        try:
            bad_summaries = []
            print(f"=== {len(found_bad)} KNOWN-BAD CLIPS: summary ===")
            for name, clip_list in found_bad.items():
                for i, (clip, ann) in enumerate(clip_list):
                    label = name if len(clip_list) == 1 else f"{name} (segment {i})"
                    bad_summaries.append(summarize_clip(label, clip, ann, model))
            print_clip_summary_table(bad_summaries)

            good_summaries = []
            print(f"\n=== {len(good_names)} RANDOM GOOD CLIPS "
                  f"(accuracy >= {GOOD_CLIP_ACCURACY_THRESHOLD:.0%}): summary ===")
            for name in good_names:
                clip_list = by_name[name]
                for i, (clip, ann) in enumerate(clip_list):
                    label = name if len(clip_list) == 1 else f"{name} (segment {i})"
                    good_summaries.append(summarize_clip(label, clip, ann, model))
            print_clip_summary_table(good_summaries)

            if DEEP_DIVE_CLIPS:
                print(f"\n\n=== FULL INTERNALS DUMP for DEEP_DIVE_CLIPS: {sorted(DEEP_DIVE_CLIPS)} ===")
                all_named = dict(found_bad)
                for name in good_names:
                    all_named.setdefault(name, by_name[name])
                for name in DEEP_DIVE_CLIPS:
                    if name not in all_named:
                        print(f"    (skipping '{name}': not found among this run's bad/good clips)")
                        continue
                    for i, (clip, ann) in enumerate(all_named[name]):
                        label = name if len(all_named[name]) == 1 else f"{name} (segment {i})"
                        diagnose_clip_deep(label, clip, ann, model,
                                          verbose_segments=DEEP_DIVE_VERBOSE_SEGMENTS)
        finally:
            sys.stdout = original_stdout

    print(f"\nWrote summary table to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()