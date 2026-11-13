"""
features/build_yaseen_phase_features.py

Phase 2 feature build: segment every Yaseen 2018 recording with the
Springer HSMM and extract the SAME 45-per-phase feature set the CirCor
table uses, so the three CirCor arms can be re-run 5-class on Yaseen
without changing a feature definition.

Why one segmenter, trained on everything annotated
--------------------------------------------------
Yaseen has no phase annotations, so (a) there is no ground_truth boundary
source here, only "predicted", and (b) there is no leakage to guard
against: no Yaseen recording is ever in the segmenter's training set. The
segmenter is therefore trained ONCE on all annotated material (CirCor
minus the known-broken recordings, PhysioNet 2016 training-a..f,
tiny_test) and applied to all 1000 Yaseen wavs. This is the "segmenter
trusted on Yaseen only after being checked against CirCor" convention:
the CirCor cross-fit accuracy (83.3% mean, 77 catastrophic of 3156) is
the evidence for trusting it; Yaseen gets no accuracy number of its own.

Segmentation quality bookkeeping (required, not optional)
---------------------------------------------------------
The sibling repo's earlier Yaseen runs found AS segmentation retention of
~2% (mean Systole 0.066 s vs 0.155 s overall). This script records, per
recording: cycles per phase, mean phase durations, number of complete
S1->Systole->S2->Diastole cycles, estimated heart rate, and two flags:
  - has_all_phases: every phase has >= 1 segment (needed for a feature row
    with no NaNs; the arms drop rows with missing phases)
  - is_clean_segmentation: has_all_phases AND mean Systole duration
    >= MIN_SYSTOLE_DURATION_SEC (0.10 s), the sibling repo's definition,
    kept so the Yaseen numbers are comparable across repos.
Written to paths.YASEEN_SEGMENTATION_QUALITY_CSV. Retention per class is
printed and must go in the paper next to any Yaseen fidelity number.

Audio scale: CirCor clips reach the segmenter and the feature extractor as
int16 sample values cast to float (no normalisation). Yaseen 4000 Hz copies
(data_prep/resample_yaseen_to_4000hz.py) are int16 too and are handled
identically, so amplitude features are on the same scale as CirCor's.

Outputs:
  results/features/yaseen_phase_features.tsv       one row per recording
  results/segmenter/yaseen_predicted_frames_checkpoint.pkl
  results/segmenter/yaseen_segmentation_quality.csv
  results/logs/build_yaseen_phase_features_log.txt
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repo root: makes `paths` and the packages importable from any cwd
import paths


import pickle
import sys

import numpy as np
import pandas as pd
import scipy.io.wavfile

from segmenter.duration_distributions import DataDistribution
from segmenter.segmentation_model import SegmentationModel
from segmenter.heart_rate import get_heart_rate
from segmenter_eval.evaluate_segmenter_all_datasets import (
    load_dataset, build_clips, CIRCOR_DIR, BROKEN_RECORDINGS, RECORDING_FREQUENCY, STATE_LABELS,
)
from features.build_circor_phase_features import (
    PHASES, SAMPLES_PER_FRAME, Tee,
    extract_recording_features, flatten_recording_features, frames_to_segments,
    load_static_non_circor_material,
)

MIN_SYSTOLE_DURATION_SEC = 0.10  # sibling repo's "clean segmentation" threshold

OUTPUT_FEATURE_TABLE = paths.YASEEN_FEATURE_TABLE
OUTPUT_QUALITY_CSV = paths.YASEEN_SEGMENTATION_QUALITY_CSV
CHECKPOINT_PATH = paths.YASEEN_PREDICTED_FRAMES_CHECKPOINT
OUTPUT_LOG = paths.LOGS_DIR / "build_yaseen_phase_features_log.txt"


def count_complete_cycles(segments):
    """Number of S1 -> Systole -> S2 -> Diastole runs in label order."""
    labels = [s[0] for s in segments]
    n = 0
    for i in range(len(labels) - 3):
        if labels[i:i + 4] == ["S1", "Systole", "S2", "Diastole"]:
            n += 1
    return n


def segmentation_quality(name, label, segments, y, sr, n_frames):
    by_phase = {p: [] for p in PHASES}
    for phase_label, start, end in segments:
        by_phase[phase_label].append((end - start) / sr)
    row = {"recording_id": name, "label": label,
           "duration_sec": len(y) / sr, "n_frames": n_frames,
           "n_segments": len(segments), "n_complete_cycles": count_complete_cycles(segments)}
    for p in PHASES:
        d = by_phase[p]
        row[f"n_{p}"] = len(d)
        row[f"mean_dur_{p}"] = float(np.mean(d)) if d else float("nan")
    row["has_all_phases"] = all(row[f"n_{p}"] > 0 for p in PHASES)
    row["is_clean_segmentation"] = bool(row["has_all_phases"]
                                        and row["mean_dur_Systole"] >= MIN_SYSTOLE_DURATION_SEC)
    try:
        hr, st = get_heart_rate(y, sr, multiple_rates=False)
        row["heart_rate_bpm"], row["systolic_time_sec"] = float(hr[0]), float(st[0])
    except Exception:
        row["heart_rate_bpm"], row["systolic_time_sec"] = float("nan"), float("nan")
    return row


def main():
    log_f = open(OUTPUT_LOG, "w")
    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, log_f)
    try:
        print("=" * 70)
        print("Yaseen 2018 phase-feature build (predicted boundaries only)")
        print("=" * 70)
        if not paths.YASEEN_DIR.exists():
            raise FileNotFoundError(f"{paths.YASEEN_DIR} not found -- run data_prep/resample_yaseen_to_4000hz.py first")

        print("\nLoading annotated training material (100%, no Yaseen recording is annotated)...")
        static_clips, static_ann, static_segs = load_static_non_circor_material()
        circor_recs, circor_segs, circor_names = load_dataset(CIRCOR_DIR, "CirCor", exclude=BROKEN_RECORDINGS)
        circor_clips, circor_ann, _, _ = build_clips(circor_recs, circor_segs, circor_names, "CirCor")
        train_clips = static_clips + circor_clips
        train_ann = static_ann + circor_ann
        train_segs = static_segs + circor_segs
        print(f"Training segmenter once on {len(train_clips)} clips "
              f"({len(static_clips)} PhysioNet/tiny_test + {len(circor_clips)} CirCor)...")
        model = SegmentationModel()
        model.fit(train_clips, train_ann, data_distribution=DataDistribution(train_segs))

        print("\nSegmenting Yaseen...")
        predicted_frames = {}
        quality_rows, feature_rows = [], []
        for cls in paths.YASEEN_CLASSES:
            wavs = sorted((paths.YASEEN_DIR / cls).glob("*.wav"))
            for wav in wavs:
                sr, rec = scipy.io.wavfile.read(wav)
                if sr != RECORDING_FREQUENCY:
                    raise ValueError(f"{wav} is {sr} Hz, expected {RECORDING_FREQUENCY}")
                if rec.ndim > 1:
                    rec = rec[:, 0]
                name = wav.stem
                frames = np.asarray(model.predict(rec))
                predicted_frames[name] = frames
                y = rec.astype(float)
                segments = frames_to_segments(frames, 0)
                quality_rows.append(segmentation_quality(name, cls, segments, y, sr, len(frames)))

                row = {"recording_id": name, "label": cls, "boundary_source": "predicted",
                       "heart_rate_bpm": quality_rows[-1]["heart_rate_bpm"],
                       "systolic_time_sec": quality_rows[-1]["systolic_time_sec"],
                       "total_duration_sec": len(y) / sr, "n_clips": 1}
                flat = flatten_recording_features(extract_recording_features(y, sr, segments)) if segments else None
                if flat is not None:
                    row.update(flat)
                feature_rows.append(row)
            print(f"  {cls}: {len(wavs)} recordings segmented")

        with open(CHECKPOINT_PATH, "wb") as f:
            pickle.dump({"predicted_frames_by_name": predicted_frames}, f)
        print(f"\nCheckpoint saved to {CHECKPOINT_PATH}")

        quality_df = pd.DataFrame(quality_rows)
        quality_df.to_csv(OUTPUT_QUALITY_CSV, index=False)
        feat_df = pd.DataFrame(feature_rows)
        feat_df.to_csv(OUTPUT_FEATURE_TABLE, sep="\t", index=False)

        print("\n" + "=" * 70)
        print("SEGMENTATION QUALITY BY CLASS")
        print("=" * 70)
        summary = quality_df.groupby("label").agg(
            n=("recording_id", "size"),
            has_all_phases=("has_all_phases", "mean"),
            clean=("is_clean_segmentation", "mean"),
            complete_cycles_median=("n_complete_cycles", "median"),
            systole_dur_mean=("mean_dur_Systole", "mean"),
            diastole_dur_mean=("mean_dur_Diastole", "mean"),
            hr_median=("heart_rate_bpm", "median"),
        )
        print(summary.round(3).to_string())
        print(f"\nOverall: has_all_phases {quality_df['has_all_phases'].mean():.1%}, "
              f"clean (mean Systole >= {MIN_SYSTOLE_DURATION_SEC}s) {quality_df['is_clean_segmentation'].mean():.1%}")
        print("\nMean Systole duration percentiles (25/50/75) by class:")
        print(quality_df.groupby("label")["mean_dur_Systole"].quantile([0.25, 0.5, 0.75]).unstack().round(3).to_string())
        print(f"\nFeature table: {feat_df.shape[0]} rows x {feat_df.shape[1]} columns -> {OUTPUT_FEATURE_TABLE}")
        nan_cols = feat_df.isna().sum()
        nan_cols = nan_cols[nan_cols > 0]
        print(f"Columns with NaN: {len(nan_cols)} (missing-phase recordings and all-NaN spectral contrast)")
        print(f"Quality table -> {OUTPUT_QUALITY_CSV}")
    finally:
        sys.stdout = original_stdout
        log_f.close()


if __name__ == "__main__":
    main()
