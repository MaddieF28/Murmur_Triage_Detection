"""
Correlates recording "patchiness" (variable loudness / intermittent audibility)
against per-recording accuracy already computed in per_recording_accuracy.csv.
Also captures each recording's estimated heart rate for a plausibility check.

Run this AFTER segmenter_eval/train_test_single_directory.py has produced per_recording_accuracy.csv.
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

DATA_DIR = paths.TRAINING_DIRS["training-a"]  # <-- set this to whatever folder you used for THIS run (the a0xxx recordings)
ACCURACY_CSV = paths.PER_RECORDING_ACCURACY_CSV

acc_df = pd.read_csv(ACCURACY_CSV)
print(f"Loaded {len(acc_df)} recordings from {ACCURACY_CSV}")

rows = []
for _, row in acc_df.iterrows():
    name = row["recording"]
    wav_path = os.path.join(DATA_DIR, name + ".wav")
    if not os.path.exists(wav_path):
        print(f"  (skipping {name}, wav not found at {wav_path})")
        continue

    sr, audio = wavfile.read(wav_path)
    audio = audio.astype(float)

    # rolling RMS envelope, ~0.5s windows, to measure loudness variability over time
    window = int(sr * 0.5)
    n_windows = len(audio) // window
    if n_windows < 4:
        continue
    rms = np.array([
        np.sqrt(np.mean(audio[i*window:(i+1)*window] ** 2))
        for i in range(n_windows)
    ])

    cv = rms.std() / (rms.mean() + 1e-9)  # coefficient of variation - higher = more "patchy"
    # fraction of windows that are "quiet" relative to the loudest parts of this same recording
    quiet_threshold = 0.2 * rms.max()
    dropout_fraction = (rms < quiet_threshold).mean()

    # heart rate estimate + how many candidate rates were found (more candidates = more ambiguous)
    try:
        heart_rates, _ = get_heart_rate(audio, sr, multiple_rates=True)
        hr_estimate = heart_rates[0]
        n_hr_candidates = len(heart_rates)
    except Exception as e:
        hr_estimate = np.nan
        n_hr_candidates = np.nan

    rows.append({
        "recording": name,
        "accuracy": row["accuracy"],
        "loudness_cv": cv,
        "dropout_fraction": dropout_fraction,
        "heart_rate_estimate": hr_estimate,
        "n_heart_rate_candidates": n_hr_candidates,
    })

df = pd.DataFrame(rows)
df = df.sort_values("accuracy")

print(f"\nMatched {len(df)} recordings with audio\n")
print("=== Worst 10 by accuracy ===")
print(df.head(10).to_string(index=False))

print("\n=== Best 10 by accuracy ===")
print(df.tail(10).to_string(index=False))

print("\n=== Correlations with accuracy ===")
for col in ["loudness_cv", "dropout_fraction", "n_heart_rate_candidates"]:
    corr = df["accuracy"].corr(df[col])
    print(f"  accuracy vs {col}: corr={corr:.3f}")

df.to_csv(paths.DIAGNOSTICS_RESULTS_DIR / "loudness_vs_accuracy.csv", index=False)
print("\nWrote full table to loudness_vs_accuracy.csv")