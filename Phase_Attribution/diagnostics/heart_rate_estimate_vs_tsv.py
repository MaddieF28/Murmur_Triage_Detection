"""
Compares get_heart_rate()'s autocorrelation-based estimate against the TRUE
heart rate derived from ground-truth .tsv annotations, for every recording
in per_recording_accuracy.csv. Tests the "estimator is finding double the
real rate" hypothesis directly.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repo root: makes `paths` and the packages importable from any cwd
import paths

import os
import numpy as np
import pandas as pd
import scipy.io.wavfile as wavfile

from segmenter.heart_rate import get_heart_rate
from segmenter.data_loading import get_heart_rate_from_tsv

DATA_DIR = paths.TRAINING_DIRS["training-a"]
ACCURACY_CSV = paths.PER_RECORDING_ACCURACY_CSV

acc_df = pd.read_csv(ACCURACY_CSV)
print(f"Loaded {len(acc_df)} recordings from {ACCURACY_CSV}")

rows = []
for _, row in acc_df.iterrows():
    name = row["recording"]
    wav_path = os.path.join(DATA_DIR, name + ".wav")
    tsv_path = os.path.join(DATA_DIR, name + ".tsv")
    if not os.path.exists(wav_path) or not os.path.exists(tsv_path):
        print(f"  (skipping {name}, file not found)")
        continue

    sr, audio = wavfile.read(wav_path)
    audio = audio.astype(float)

    try:
        heart_rates, _ = get_heart_rate(audio, sr, multiple_rates=True)
        estimated_hr = heart_rates[0]
    except Exception:
        estimated_hr = np.nan

    try:
        tsv = np.loadtxt(tsv_path, delimiter="\t")
        true_hr = get_heart_rate_from_tsv(tsv)
    except Exception as e:
        true_hr = np.nan

    ratio = estimated_hr / true_hr if (true_hr and not np.isnan(true_hr) and true_hr != 0) else np.nan

    rows.append({
        "recording": name,
        "accuracy": row["accuracy"],
        "estimated_hr": estimated_hr,
        "true_hr": true_hr,
        "ratio_est_over_true": ratio,
    })

df = pd.DataFrame(rows).sort_values("accuracy")

print(f"\nMatched {len(df)} recordings\n")
print("=== Worst 10 by accuracy ===")
print(df.head(10).to_string(index=False))

print("\n=== Best 10 by accuracy ===")
print(df.tail(10).to_string(index=False))

print("\n=== Correlation of accuracy with ratio (estimated/true heart rate) ===")
print(f"  corr={df['accuracy'].corr(df['ratio_est_over_true']):.3f}")

# flag recordings where estimate is roughly double (or half) the true rate
close_to_double = df[(df["ratio_est_over_true"] > 1.7) & (df["ratio_est_over_true"] < 2.3)]
close_to_half = df[(df["ratio_est_over_true"] > 0.4) & (df["ratio_est_over_true"] < 0.6)]
print(f"\nRecordings where estimate ~= 2x true rate: {len(close_to_double)} / {len(df)}")
print(f"Recordings where estimate ~= 0.5x true rate: {len(close_to_half)} / {len(df)}")

print(f"\nMean accuracy when ratio is close to 1.0 (+/-15%): "
      f"{df[(df['ratio_est_over_true'] > 0.85) & (df['ratio_est_over_true'] < 1.15)]['accuracy'].mean():.4f}")
print(f"Mean accuracy when ratio is NOT close to 1.0: "
      f"{df[~((df['ratio_est_over_true'] > 0.85) & (df['ratio_est_over_true'] < 1.15))]['accuracy'].mean():.4f}")

df.to_csv(paths.DIAGNOSTICS_RESULTS_DIR / "heart_rate_check.csv", index=False)
print("\nWrote full table to heart_rate_check.csv")