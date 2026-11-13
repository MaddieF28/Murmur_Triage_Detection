"""
features/build_circor_phase_features.py

Preprocessing step for the "does the classifier rely on segments the way
we'd expect" experiment. Produces ONE feature table (TSV) with one row per
(CirCor recording, boundary_source) pair, boundary_source in
{"ground_truth", "predicted"}, ready to feed a GMM (Arm A) / NN (Arm B)
binary Normal-vs-Abnormal classifier. Does not try to match the column
format of any earlier feature-extraction script -- this is a fresh,
self-contained feature set built for the current (4000 Hz) segmenter repo.

Why 5-fold cross-fitting instead of one train/predict pass
------------------------------------------------------------
We need predicted boundaries for EVERY CirCor recording, but a recording's
own segmenter-training data can't be used to predict its own boundaries
(that's leakage -- the segmenter would partly be "recognizing" data it
trained on, not generalizing). Standard fix: split CirCor into 5 folds;
for each fold, train the segmenter on (100% of every non-CirCor dataset)
+ (the other 4/5 of CirCor), then predict only on the held-out 1/5. Every
CirCor recording ends up held-out exactly once, so every prediction is
leak-free, and each fold's segmenter still trains on ~80% of CirCor (same
scale as your normal segmenter_eval/evaluate_segmenter_all_datasets.py split) rather than a weaker
sub-sample. Non-CirCor datasets have no diagnosis labels, so they're never
predicted on -- they're only ever training material, always at 100%.

Cost: trains the segmenter 5 times (one per fold). That's the accepted
tradeoff for leak-free predictions on 100% of CirCor without ever training
and predicting on the same recording.

REQUIRES YOU TO CHECK ONE THING BEFORE RUNNING
------------------------------------------------
CIRCOR_METADATA_PATH below is a guess at where the patient metadata CSV
(the one with columns "Patient ID", "Outcome", "Additional ID") lives,
based on the standard CirCor release layout (a sibling file to the
training_data/ folder CIRCOR_DIR already points at). If it's somewhere
else, fix the constant -- the script fails loudly with a clear message
if it can't find/parse it, rather than guessing silently.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repo root: makes `paths` and the packages importable from any cwd
import paths


import random
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import librosa

try:
    import pywt
    HAVE_PYWT = True
except ImportError:
    HAVE_PYWT = False
    warnings.warn("pywt not installed -- DWT energy features will be skipped. "
                   "pip install PyWavelets if you want them.")

from segmenter.duration_distributions import DataDistribution
from segmenter.segmentation_model import SegmentationModel
from segmenter.heart_rate import get_heart_rate
from segmenter_eval.evaluate_segmenter_all_datasets import (
    load_dataset, build_clips,
    CIRCOR_DIR, TINY_TEST_DIR, TRAINING_DIRS, BROKEN_RECORDINGS,
    RANDOM_SEED, RECORDING_FREQUENCY, FEATURE_FREQUENCY, STATE_LABELS,
)

# ---------------------------------------------------------------------
# CHECK / EDIT THIS -- see module docstring above.
# ---------------------------------------------------------------------
CIRCOR_METADATA_PATH = paths.CIRCOR_METADATA_PATH

N_FOLDS = 5
SAMPLES_PER_FRAME = RECORDING_FREQUENCY // FEATURE_FREQUENCY  # 4000/50 = 80
PHASES = ("S1", "Systole", "S2", "Diastole")  # matches STATE_LABELS[1..4] exactly

N_MFCC = 13
WAVELET = "db6"
WAVELET_LEVEL = 5
MIN_SEGMENT_SAMPLES = 512  # pad shorter phase segments before frame-based features

OUTPUT_FEATURE_TABLE = paths.CIRCOR_FEATURE_TABLE
OUTPUT_LOG = paths.LOGS_DIR / "build_circor_binary_features_log.txt"
CHECKPOINT_PATH = paths.CIRCOR_PREDICTED_FRAMES_CHECKPOINT  # safety net between fold-training and feature extraction


# =======================================================================
# Metadata / label lookup
# =======================================================================

def load_outcome_lookup(metadata_path):
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"CirCor metadata CSV not found at {metadata_path}. "
            f"Fix CIRCOR_METADATA_PATH at the top of this script -- "
            f"this is a guessed path, not a confirmed one."
        )
    df = pd.read_csv(metadata_path)
    required = {"Patient ID", "Outcome"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"CirCor metadata CSV at {metadata_path} is missing expected "
            f"column(s) {missing}. Columns actually found: {list(df.columns)}. "
            f"Fix CIRCOR_METADATA_PATH or check the file."
        )

    primary = {}
    secondary = {}
    for _, row in df.iterrows():
        pid = str(row["Patient ID"]).strip()
        outcome = row["Outcome"]
        if pd.isna(outcome):
            continue
        primary[pid] = outcome
        if "Additional ID" in df.columns and not pd.isna(row["Additional ID"]):
            aid = str(row["Additional ID"]).strip()
            # Don't overwrite a primary-ID mapping with a secondary one.
            secondary.setdefault(aid, outcome)

    def lookup(recording_name):
        patient_id = recording_name.split("_", 1)[0]
        if patient_id in primary:
            return primary[patient_id]
        if patient_id in secondary:
            return secondary[patient_id]
        return None

    return lookup


# =======================================================================
# Frame-array (0-4, 50 Hz) -> (phase_label, start_sample, end_sample)
# =======================================================================

def frames_to_segments(frame_states, sample_offset):
    """Run-length-encodes a per-frame state array into raw-sample-indexed
    (phase_label, start_sample, end_sample) tuples, skipping Unclassified
    (state 0) runs. sample_offset is where this clip starts within the
    combined (multi-clip) recording audio."""
    frame_states = np.asarray(frame_states)
    segments = []
    if len(frame_states) == 0:
        return segments

    run_start = 0
    current = frame_states[0]
    for i in range(1, len(frame_states) + 1):
        if i == len(frame_states) or frame_states[i] != current:
            if current in (1, 2, 3, 4):
                start_sample = sample_offset + run_start * SAMPLES_PER_FRAME
                end_sample = sample_offset + i * SAMPLES_PER_FRAME
                segments.append((STATE_LABELS[current], start_sample, end_sample))
            if i < len(frame_states):
                current = frame_states[i]
                run_start = i
    return segments


# =======================================================================
# Feature extraction (fresh, fixed to this repo's actual 4000 Hz audio --
# NOT reusing the old repo's extract_features.py, which hardcoded 8000 Hz
# and expected a different segment/pipeline format entirely)
# =======================================================================

def _pad_if_short(y, min_len=MIN_SEGMENT_SAMPLES):
    if len(y) >= min_len:
        return y
    return np.pad(y, (0, min_len - len(y)), mode="constant")


def _n_fft_for(y, max_n_fft=2048):
    n = min(max_n_fft, len(y))
    return int(2 ** np.floor(np.log2(max(n, 32))))


def _temporal_features(y, sr, duration_samples):
    return {
        "duration_sec": duration_samples / sr,
        "amp_mean": float(np.mean(np.abs(y))),
        "amp_max": float(np.max(np.abs(y))),
        "amp_std": float(np.std(y)),
        "rms_energy": float(np.sqrt(np.mean(y.astype(np.float64) ** 2))),
        "zcr_mean": float(np.mean(librosa.feature.zero_crossing_rate(y)[0])),
    }


def _mfcc_features(y, sr):
    n_fft = _n_fft_for(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=n_fft, hop_length=n_fft // 4)
    feats = {}
    for i in range(N_MFCC):
        feats[f"mfcc_{i+1}_mean"] = float(np.mean(mfcc[i]))
        feats[f"mfcc_{i+1}_std"] = float(np.std(mfcc[i]))
    return feats


def _spectral_features(y, sr):
    n_fft = _n_fft_for(y)
    hop = n_fft // 4
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft, hop_length=hop)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft, hop_length=hop)[0]
    flatness = librosa.feature.spectral_flatness(y=y, n_fft=n_fft, hop_length=hop)[0]
    try:
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft, hop_length=hop)
        contrast_mean = float(np.mean(contrast))
    except Exception:
        # spectral_contrast can fail on very short/quiet segments -- not
        # critical enough to abort the whole recording over.
        contrast_mean = float("nan")
    return {
        "spec_centroid_mean": float(np.mean(centroid)),
        "spec_centroid_std": float(np.std(centroid)),
        "spec_bandwidth_mean": float(np.mean(bandwidth)),
        "spec_rolloff_mean": float(np.mean(rolloff)),
        "spec_flatness_mean": float(np.mean(flatness)),
        "spec_contrast_mean": contrast_mean,
    }


def _dwt_features(y):
    expected = ["dwt_energy_approx"] + [f"dwt_energy_detail_{lvl}" for lvl in range(1, WAVELET_LEVEL + 1)]
    if not HAVE_PYWT:
        return {name: float("nan") for name in expected}
    max_level = pywt.dwt_max_level(len(y), 12)
    level = min(WAVELET_LEVEL, max_level) if max_level > 0 else 1
    coeffs = pywt.wavedec(y, WAVELET, level=level)
    feats = {}
    for i, c in enumerate(coeffs):
        name = "approx" if i == 0 else f"detail_{len(coeffs) - i}"
        feats[f"dwt_energy_{name}"] = float(np.sum(c ** 2))
    for name in expected:
        feats.setdefault(name, 0.0)
    return {k: feats[k] for k in expected}


def _phase_cycle_features(y, sr):
    duration_samples = len(y)
    y_proc = _pad_if_short(y)
    feats = {}
    feats.update(_temporal_features(y, sr, duration_samples))
    feats.update(_mfcc_features(y_proc, sr))
    feats.update(_spectral_features(y_proc, sr))
    feats.update(_dwt_features(y_proc))
    return feats


def extract_recording_features(y, sr, segments):
    """segments: list of (phase_label, start_sample, end_sample). Returns
    dict phase -> {feature: value}, or phase -> None if that phase never
    occurred (segmentation produced zero instances of it)."""
    by_phase = {p: [] for p in PHASES}
    for phase_label, start, end in segments:
        if phase_label not in PHASES:
            raise ValueError(f"Unexpected phase label {phase_label!r}")
        seg_audio = y[start:end]
        if len(seg_audio) == 0:
            continue
        by_phase[phase_label].append(_phase_cycle_features(seg_audio, sr))

    result = {}
    for phase in PHASES:
        cycles = by_phase[phase]
        if not cycles:
            result[phase] = None
            continue
        keys = cycles[0].keys()
        result[phase] = {k: float(np.mean([c[k] for c in cycles])) for k in keys}
        result[phase]["n_cycles"] = len(cycles)
    return result


def flatten_recording_features(phase_features):
    template = None
    for v in phase_features.values():
        if v is not None:
            template = list(v.keys())
            break
    if template is None:
        return None  # every phase missing -- caller decides how to handle

    flat = {}
    for phase in PHASES:
        vals = phase_features.get(phase)
        for name in template:
            flat[f"{phase}_{name}"] = vals[name] if vals is not None else float("nan")
    return flat


# =======================================================================
# Tee for logging to console + file
# =======================================================================

class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


# =======================================================================
# Segmenter training helpers
# =======================================================================

def load_static_non_circor_material():
    """Loads every non-CirCor dataset at 100% -- used as training material
    in every fold, never predicted on (no diagnosis labels to compare
    against)."""
    tiny_rec, tiny_seg, tiny_names = load_dataset(TINY_TEST_DIR, "tiny_test")
    datasets = {"tiny_test": (tiny_rec, tiny_seg, tiny_names)}

    import os
    for label, directory in TRAINING_DIRS.items():
        if not os.path.isdir(directory):
            print(f"{label}: SKIPPED -- directory not found: {directory}")
            continue
        rec, seg, names = load_dataset(directory, label)
        datasets[label] = (rec, seg, names)

    static_clips, static_ann, static_segs = [], [], []
    for label, (recs, segs, names) in datasets.items():
        clips_d, ann_d, _, _ = build_clips(recs, segs, names, label)
        static_clips.extend(clips_d)
        static_ann.extend(ann_d)
        static_segs.extend(segs)
        print(f"{label}: {len(names)} recordings loaded (100%, static training material)")

    return static_clips, static_ann, static_segs


def make_folds(names, n_folds, seed):
    names = list(names)
    rng = random.Random(seed)
    rng.shuffle(names)
    folds = [[] for _ in range(n_folds)]
    for i, name in enumerate(names):
        folds[i % n_folds].append(name)
    return [set(f) for f in folds]


# =======================================================================
# Main
# =======================================================================

def main():
    log_f = open(OUTPUT_LOG, "w")
    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, log_f)

    try:
        print("=" * 70)
        print("CirCor binary-task feature preprocessing (ground truth + predicted)")
        print("=" * 70)

        outcome_lookup = load_outcome_lookup(CIRCOR_METADATA_PATH)

        print("\nLoading non-CirCor datasets (100%, static training material)...")
        static_clips, static_ann, static_segs = load_static_non_circor_material()
        print(f"Static non-CirCor training material: {len(static_clips)} clips")

        print("\nLoading CirCor...")
        circor_recs, circor_segs, circor_names = load_dataset(CIRCOR_DIR, "CirCor", exclude=BROKEN_RECORDINGS)
        print(f"CirCor: {len(circor_names)} recordings")

        print("\nBuilding CirCor clips (raw audio + ground-truth frame arrays) once...")
        circor_clips_flat, circor_ann_flat, circor_clipnames_flat, _ = build_clips(
            circor_recs, circor_segs, circor_names, "CirCor"
        )
        circor_by_name = {}
        for clip, ann, name in zip(circor_clips_flat, circor_ann_flat, circor_clipnames_flat):
            circor_by_name.setdefault(name, []).append((clip, ann))
        name_to_seg = dict(zip(circor_names, circor_segs))
        print(f"CirCor split into {len(circor_clips_flat)} clips across {len(circor_by_name)} recordings")

        print(f"\nBuilding {N_FOLDS} folds over 100% of CirCor (seed={RANDOM_SEED})...")
        folds = make_folds(list(circor_by_name.keys()), N_FOLDS, RANDOM_SEED)
        for i, f in enumerate(folds):
            print(f"  fold {i}: {len(f)} recordings held out")

        predicted_frames_by_name = {}
        fold_of_name = {}

        for fold_idx, held_out_names in enumerate(folds):
            print(f"\n--- Fold {fold_idx+1}/{N_FOLDS}: {len(held_out_names)} held out ---")
            train_names_circor = set(circor_by_name.keys()) - held_out_names

            train_clips_circor, train_ann_circor = [], []
            for name in train_names_circor:
                for clip, ann in circor_by_name[name]:
                    train_clips_circor.append(clip)
                    train_ann_circor.append(ann)

            train_segs_circor = [name_to_seg[name] for name in train_names_circor]

            combined_clips = static_clips + train_clips_circor
            combined_ann = static_ann + train_ann_circor
            combined_segs = static_segs + train_segs_circor

            print(f"  training on {len(combined_clips)} clips "
                  f"({len(static_clips)} static + {len(train_clips_circor)} CirCor)...")
            model = SegmentationModel()
            data_distribution = DataDistribution(combined_segs)
            model.fit(combined_clips, combined_ann, data_distribution=data_distribution)

            print(f"  predicting on {len(held_out_names)} held-out CirCor recordings...")
            for name in held_out_names:
                fold_of_name[name] = fold_idx
                preds = []
                for clip, _ in circor_by_name[name]:
                    preds.append(model.predict(clip))
                predicted_frames_by_name[name] = preds

        # Safety checkpoint -- feature extraction below is independent of
        # the (expensive) training above; if it fails, don't lose the
        # 5-fold training work.
        try:
            import pickle
            with open(CHECKPOINT_PATH, "wb") as f:
                pickle.dump({"predicted_frames_by_name": predicted_frames_by_name,
                             "fold_of_name": fold_of_name}, f)
            print(f"\nCheckpoint saved to {CHECKPOINT_PATH} "
                  f"(predicted frames for all {len(predicted_frames_by_name)} recordings)")
        except Exception as e:
            print(f"\nWARNING: could not save checkpoint ({e}) -- continuing anyway.")

        # -----------------------------------------------------------------
        # Feature extraction: ground truth + predicted, per recording
        # -----------------------------------------------------------------
        print("\n" + "=" * 70)
        print("Feature extraction")
        print("=" * 70)

        rows = []
        no_label_names = []
        no_segments_names = []  # (name, boundary_source) with zero usable phase segments

        for name in sorted(circor_by_name.keys()):
            outcome = outcome_lookup(name)
            if outcome is None:
                no_label_names.append(name)
                continue

            clip_list = circor_by_name[name]

            # Concatenate this recording's clips into one combined waveform,
            # tracking sample offsets so frame->sample math stays correct
            # across clip boundaries.
            combined_y_parts = []
            gt_segments = []
            pred_segments = []
            offset = 0
            for clip_idx, (raw_clip, gt_frames) in enumerate(clip_list):
                combined_y_parts.append(raw_clip)
                gt_segments.extend(frames_to_segments(gt_frames, offset))
                pred_frames = predicted_frames_by_name[name][clip_idx]
                pred_segments.extend(frames_to_segments(pred_frames, offset))
                offset += len(raw_clip)
            combined_y = np.concatenate(combined_y_parts).astype(float)

            # Recording-level (non-phase-specific) features.
            try:
                hr_list, sys_time_list = get_heart_rate(combined_y, RECORDING_FREQUENCY, multiple_rates=False)
                heart_rate_bpm = float(hr_list[0])
                systolic_time_sec = float(sys_time_list[0])
            except Exception as e:
                heart_rate_bpm = float("nan")
                systolic_time_sec = float("nan")
                print(f"  WARNING: get_heart_rate failed for {name}: {e}")

            recording_level = {
                "heart_rate_bpm": heart_rate_bpm,
                "systolic_time_sec": systolic_time_sec,
                "total_duration_sec": len(combined_y) / RECORDING_FREQUENCY,
                "n_clips": len(clip_list),
            }

            for boundary_source, segs in (("ground_truth", gt_segments), ("predicted", pred_segments)):
                if not segs:
                    no_segments_names.append((name, boundary_source))
                    continue
                phase_feats = extract_recording_features(combined_y, RECORDING_FREQUENCY, segs)
                flat = flatten_recording_features(phase_feats)
                if flat is None:
                    no_segments_names.append((name, boundary_source))
                    continue

                row = {
                    "recording_id": name,
                    "boundary_source": boundary_source,
                    "outcome": outcome,
                    "fold": fold_of_name[name],
                }
                row.update(recording_level)
                row.update(flat)
                rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(OUTPUT_FEATURE_TABLE, sep="\t", index=False)

        # -----------------------------------------------------------------
        # Summary
        # -----------------------------------------------------------------
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Total CirCor recordings: {len(circor_by_name)}")
        print(f"Recordings with no Outcome label found (skipped): {len(no_label_names)}")
        if no_label_names:
            print(f"  {no_label_names[:20]}{' ...' if len(no_label_names) > 20 else ''}")
        print(f"Recordings/boundary-source pairs with zero usable phase segments (skipped): "
              f"{len(no_segments_names)}")
        if no_segments_names:
            print(f"  {no_segments_names[:20]}{' ...' if len(no_segments_names) > 20 else ''}")

        print(f"\nFinal feature table: {df.shape[0]} rows x {df.shape[1]} columns")
        print(f"Saved to: {OUTPUT_FEATURE_TABLE}")

        if not df.empty:
            print("\nRows per boundary_source:")
            print(df["boundary_source"].value_counts())
            print("\nRows per outcome (combined across boundary sources):")
            print(df["outcome"].value_counts())
            print("\nRows per (boundary_source, outcome):")
            print(df.groupby(["boundary_source", "outcome"]).size())
            print("\nRows per fold:")
            print(df.groupby("fold")["recording_id"].nunique())

            nan_counts = df.isna().sum()
            nan_counts = nan_counts[nan_counts > 0]
            if len(nan_counts) > 0:
                print(f"\nColumns with NaN values ({len(nan_counts)} columns affected, "
                      f"usually missing-phase recordings):")
                print(nan_counts.sort_values(ascending=False).head(20))

        print(f"\nDone. Log written to {OUTPUT_LOG}")

    finally:
        sys.stdout = original_stdout
        log_f.close()


if __name__ == "__main__":
    main()