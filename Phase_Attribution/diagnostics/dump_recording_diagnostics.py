"""
Pulls a wide set of per-recording features/diagnostics and writes them all
to one CSV. v2: computes autocorrelation only ONCE per recording (was
duplicated in v1, causing the slowdown) and prints progress.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repo root: makes `paths` and the packages importable from any cwd
import paths

import os
import glob
import time
import numpy as np
import pandas as pd
import scipy.io.wavfile as wavfile
from scipy.signal import correlate, find_peaks

from segmenter.data_loading import get_heart_rate_from_tsv, create_segmentation_array
from segmenter.extract_features import (
    get_hilbert_envelope,
    get_power_spectral_density,
    get_butterworth_low_pass_filter,
    get_butterworth_high_pass_filter,
    schmidt_spike_removal,
)

DATA_DIR = paths.TRAINING_DIRS["training-a"]
ACCURACY_CSV = paths.TEST_ACCURACY_BY_RECORDING_CSV
REFERENCE_CSV = None
OUT_CSV = paths.DIAGNOSTICS_RESULTS_DIR / "full_diagnostic_dump.csv"
FEATURE_FREQUENCY = 50


def safe(fn, default=np.nan):
    try:
        return fn()
    except Exception:
        return default


def heart_rate_full_diagnostics(audio, sr, min_hr=60, max_hr=200):
    """
    Single-pass replacement for get_heart_rate() + margin calc combined -
    computes autocorrelation ONCE and extracts everything from it.
    """
    x = get_butterworth_low_pass_filter(audio, 2, 100, sr)
    x = get_butterworth_high_pass_filter(x, 2, 20, sr)
    x = schmidt_spike_removal(x, sr)
    x = np.abs(x - x.mean())

    c = correlate(x, x, method="fft")
    c = c / c[int(c.shape[0] / 2)]
    autocorr = c[x.shape[0]:]

    min_idx = 60 * sr / max_hr
    max_idx = 60 * sr / min_hr
    window = autocorr[int(min_idx):int(max_idx)]
    peaks, props = find_peaks(window, width=4, prominence=0.05)

    if len(peaks) == 0:
        return {
            "estimated_hr": np.nan, "n_hr_candidates": 0, "hr_candidates_all": "",
            "autocorr_top_peak_height": np.nan, "autocorr_peak_margin": np.nan,
            "estimated_systolic_time": np.nan,
        }

    heights = window[peaks]
    order = np.argsort(heights)[::-1]
    sorted_heights = heights[order]
    true_indices = peaks[order] + min_idx
    candidate_hrs = list(60. / (true_indices / sr))

    top_height = sorted_heights[0]
    second_height = sorted_heights[1] if len(sorted_heights) > 1 else 0.0
    margin = (top_height - second_height) / (top_height + 1e-9)

    min_sys_duration = np.round(0.1 * sr)
    max_sys_duration = np.round(((60. / candidate_hrs[0]) * sr) / 2)
    pos = np.argmax(autocorr[int(min_sys_duration): int(max_sys_duration)]) if max_sys_duration > min_sys_duration else 0
    systolic_time = (min_sys_duration + pos) / sr

    return {
        "estimated_hr": candidate_hrs[0],
        "n_hr_candidates": len(candidate_hrs),
        "hr_candidates_all": ";".join(f"{h:.1f}" for h in candidate_hrs),
        "autocorr_top_peak_height": top_height,
        "autocorr_peak_margin": margin,
        "estimated_systolic_time": systolic_time,
    }


def true_rhythm_stats(tsv):
    out = {}
    for state_val, label in [(1, "s1"), (2, "systole"), (3, "s2"), (4, "diastole")]:
        rows = tsv[tsv[:, 2] == state_val]
        durations = rows[:, 1] - rows[:, 0]
        out[f"{label}_dur_mean"] = durations.mean() if len(durations) else np.nan
        out[f"{label}_dur_std"] = durations.std() if len(durations) else np.nan
        out[f"n_{label}_segments"] = len(durations)

    s1_rows = tsv[tsv[:, 2] == 1]
    s1_starts = np.sort(s1_rows[:, 0])
    if len(s1_starts) > 1:
        cycle_lengths = np.diff(s1_starts)
        out["cycle_length_mean"] = cycle_lengths.mean()
        out["cycle_length_std"] = cycle_lengths.std()
        out["cycle_length_cv"] = cycle_lengths.std() / (cycle_lengths.mean() + 1e-9)
        out["n_beats"] = len(s1_starts)
    else:
        out["cycle_length_mean"] = np.nan
        out["cycle_length_std"] = np.nan
        out["cycle_length_cv"] = np.nan
        out["n_beats"] = len(s1_starts)

    recording_end = tsv[:, 1].max()
    annotated_time = (tsv[:, 1] - tsv[:, 0]).sum()
    out["fraction_unclassified"] = 1 - (annotated_time / recording_end) if recording_end > 0 else np.nan
    return out


rows = []
wav_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.wav")))
print(f"Found {len(wav_files)} wav files in {DATA_DIR}")

start_time = time.time()
skipped_empty = 0

for i, wav_path in enumerate(wav_files):
    name = os.path.splitext(os.path.basename(wav_path))[0]
    tsv_path = os.path.join(DATA_DIR, name + ".tsv")
    if not os.path.exists(tsv_path):
        continue

    if os.path.getsize(tsv_path) == 0:
        skipped_empty += 1
        continue

    tsv = np.loadtxt(tsv_path, delimiter="\t")
    if tsv.ndim != 2:
        skipped_empty += 1
        continue

    sr, audio_raw = wavfile.read(wav_path)
    audio = audio_raw.astype(float)

    row = {"recording": name, "sample_rate": sr, "duration_sec": len(audio) / sr}

    row.update(safe(lambda: heart_rate_full_diagnostics(audio, sr), {}))

    true_hr = safe(lambda: get_heart_rate_from_tsv(tsv))
    row["true_hr"] = true_hr
    est = row.get("estimated_hr", np.nan)
    row["hr_ratio_est_over_true"] = (est / true_hr) if (true_hr and not np.isnan(true_hr) and true_hr != 0 and not np.isnan(est)) else np.nan
    cand_str = row.get("hr_candidates_all", "")
    cands = [float(c) for c in cand_str.split(";")] if cand_str else []
    row["true_hr_in_candidates_within_10pct"] = safe(lambda: any(abs(h - true_hr) / true_hr < 0.10 for h in cands), False)

    window = int(sr * 0.5)
    n_windows = len(audio) // window if window > 0 else 0
    if n_windows >= 4:
        rms = np.array([np.sqrt(np.mean(audio[j*window:(j+1)*window] ** 2)) for j in range(n_windows)])
        row["loudness_cv"] = rms.std() / (rms.mean() + 1e-9)
        row["dropout_fraction"] = (rms < 0.2 * rms.max()).mean()
        row["loudness_dynamic_range"] = rms.max() / (rms.min() + 1e-9)
    else:
        row["loudness_cv"] = np.nan
        row["dropout_fraction"] = np.nan
        row["loudness_dynamic_range"] = np.nan

    max_possible = np.iinfo(audio_raw.dtype).max if np.issubdtype(audio_raw.dtype, np.integer) else np.abs(audio_raw).max()
    row["clipping_fraction"] = safe(lambda: (np.abs(audio_raw) >= 0.99 * max_possible).mean())
    row["zero_crossing_rate"] = safe(lambda: np.mean(np.abs(np.diff(np.sign(audio))) > 0))
    row["rms_overall"] = np.sqrt(np.mean(audio ** 2))
    row["peak_amplitude"] = np.abs(audio).max()
    row["crest_factor"] = row["peak_amplitude"] / (row["rms_overall"] + 1e-9)

    def band_energy_ratio():
        low = get_butterworth_low_pass_filter(audio, 2, 150, sr)
        low = get_butterworth_high_pass_filter(low, 2, 20, sr)
        return np.sum(low ** 2) / (np.sum(audio ** 2) + 1e-9)
    row["low_band_energy_ratio"] = safe(band_energy_ratio)

    row.update(safe(lambda: true_rhythm_stats(tsv), {}))

    def clip_stats():
        clips, annots = create_segmentation_array(audio_raw, tsv, recording_frequency=sr, feature_frequency=FEATURE_FREQUENCY)
        lengths = [len(c) / sr for c in clips]
        return len(clips), (np.mean(lengths) if lengths else np.nan)
    n_clips, mean_clip_len = safe(clip_stats, (np.nan, np.nan))
    row["n_clips_extracted"] = n_clips
    row["mean_clip_length_sec"] = mean_clip_len

    rows.append(row)

    if (i + 1) % 25 == 0:
        elapsed = time.time() - start_time
        print(f"  processed {i+1}/{len(wav_files)} files  ({elapsed:.1f}s elapsed)")

print(f"\nDone. {len(rows)} recordings processed, {skipped_empty} skipped (empty/malformed tsv).")

df = pd.DataFrame(rows)

if os.path.exists(ACCURACY_CSV):
    acc_df = pd.read_csv(ACCURACY_CSV)[["recording", "accuracy"]]
    df = df.merge(acc_df, on="recording", how="left")
    print(f"Merged accuracy for {df['accuracy'].notna().sum()} / {len(df)} recordings")

if REFERENCE_CSV and os.path.exists(REFERENCE_CSV):
    ref_df = pd.read_csv(REFERENCE_CSV, header=None, names=["recording", "label"])
    ref_df["recording"] = ref_df["recording"].astype(str)
    df = df.merge(ref_df, on="recording", how="left")
    print(f"Merged reference label for {df['label'].notna().sum()} / {len(df)} recordings")

df.to_csv(OUT_CSV, index=False)
print(f"\nWrote {OUT_CSV} with columns:\n{list(df.columns)}")