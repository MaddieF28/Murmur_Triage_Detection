
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import welch

# ============================================================
# SETTINGS
# ============================================================

# This script is inside the "code" folder.
# The five dataset folders are one level above it.
DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "filtered_dataset")

CLASSES = ["N", "AS", "MS", "MR", "MVP"]

# Frequency band from the paper
LOW_FREQ = 25
HIGH_FREQ = 400

# Where to save plots/results
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "analysis_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_audio(filepath):
    """Load a WAV file and convert it to floating point."""

    sample_rate, audio = wavfile.read(filepath)

    # Convert integer PCM to approximately [-1, 1]
    if np.issubdtype(audio.dtype, np.integer):
        info = np.iinfo(audio.dtype)
        audio = audio.astype(np.float64)

        # Handle signed PCM
        if info.min < 0:
            audio /= max(abs(info.min), info.max)

        # Handle unsigned PCM
        else:
            audio = (audio - info.max / 2) / (info.max / 2)

    else:
        audio = audio.astype(np.float64)

    # If stereo, average channels
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    return sample_rate, audio


def calculate_psd(audio, sample_rate):
    """Calculate PSD using Welch's method."""

    frequencies, psd = welch(
        audio,
        fs=sample_rate,
        nperseg=min(1024, len(audio))
    )

    return frequencies, psd


def calculate_energy_percentages(frequencies, psd):
    """
    Calculate:
      - total energy
      - energy below 25 Hz
      - energy within 25-400 Hz
      - energy above 400 Hz
    """

    total_energy = np.trapezoid(psd, frequencies)

    below_band = frequencies < LOW_FREQ
    inside_band = (frequencies >= LOW_FREQ) & (frequencies <= HIGH_FREQ)
    above_band = frequencies > HIGH_FREQ

    low_energy = np.trapezoid(
        psd[below_band],
        frequencies[below_band]
    ) if np.any(below_band) else 0

    band_energy = np.trapezoid(
        psd[inside_band],
        frequencies[inside_band]
    ) if np.any(inside_band) else 0

    high_energy = np.trapezoid(
        psd[above_band],
        frequencies[above_band]
    ) if np.any(above_band) else 0

    if total_energy > 0:
        low_percent = 100 * low_energy / total_energy
        band_percent = 100 * band_energy / total_energy
        high_percent = 100 * high_energy / total_energy
    else:
        low_percent = 0
        band_percent = 0
        high_percent = 0

    return low_percent, band_percent, high_percent


# ============================================================
# MAIN ANALYSIS
# ============================================================

all_results = []
class_psds = {}

print("=" * 70)
print("FREQUENCY ANALYSIS")
print("=" * 70)
print(f"Dataset directory: {os.path.abspath(DATASET_DIR)}")
print()

for class_name in CLASSES:

    class_dir = os.path.join(DATASET_DIR, class_name)

    if not os.path.isdir(class_dir):
        print(f"WARNING: Folder not found: {class_dir}")
        continue

    wav_files = [
        f for f in os.listdir(class_dir)
        if f.lower().endswith(".wav")
    ]

    print(f"Processing {class_name}: {len(wav_files)} WAV files")

    class_psd_list = []

    for filename in wav_files:

        filepath = os.path.join(class_dir, filename)

        try:
            sample_rate, audio = load_audio(filepath)

            # Calculate PSD
            frequencies, psd = calculate_psd(audio, sample_rate)

            # Calculate energy distribution
            low_pct, band_pct, high_pct = calculate_energy_percentages(
                frequencies,
                psd
            )

            # Save result
            all_results.append({
                "class": class_name,
                "filename": filename,
                "sample_rate": sample_rate,
                "energy_below_25Hz_percent": low_pct,
                "energy_25_400Hz_percent": band_pct,
                "energy_above_400Hz_percent": high_pct
            })

            class_psd_list.append(psd)

        except Exception as e:
            print(f"  ERROR: {filename}: {e}")

    # ========================================================
    # Average PSD for this class
    # ========================================================

    if class_psd_list:

        # PSDs can have different frequency resolutions if
        # sample rates differ. Only average when frequency
        # arrays are compatible.
        min_length = min(len(p) for p in class_psd_list)

        trimmed_psds = [
            p[:min_length]
            for p in class_psd_list
        ]

        class_psds[class_name] = np.mean(
            np.array(trimmed_psds),
            axis=0
        )

        print(f"  Successfully processed: {len(class_psd_list)}")

print()


# ============================================================
# SUMMARY
# ============================================================

print("=" * 70)
print("ENERGY ABOVE 400 Hz")
print("=" * 70)

for class_name in CLASSES:

    class_results = [
        r for r in all_results
        if r["class"] == class_name
    ]

    if not class_results:
        continue

    values = np.array([
        r["energy_above_400Hz_percent"]
        for r in class_results
    ])

    print(
        f"{class_name:>4}: "
        f"mean = {np.mean(values):7.3f}% | "
        f"median = {np.median(values):7.3f}% | "
        f"max = {np.max(values):7.3f}% | "
        f"min = {np.min(values):7.3f}%"
    )


# ============================================================
# PLOT 1: CLASS-AVERAGED PSD
# ============================================================

plt.figure(figsize=(12, 7))

for class_name in CLASSES:

    if class_name not in class_psds:
        continue

    psd = class_psds[class_name]

    # Use the frequency array from the last processed file.
    # For consistent datasets this will be the same across files.
    plt.semilogy(
        frequencies[:len(psd)],
        psd,
        label=class_name
    )

# Mark proposed filter boundaries
plt.axvline(
    LOW_FREQ,
    linestyle="--",
    label="25 Hz"
)

plt.axvline(
    HIGH_FREQ,
    linestyle="--",
    label="400 Hz"
)

plt.xlabel("Frequency (Hz)")
plt.ylabel("Power Spectral Density")
plt.title("Average Power Spectral Density by Class")

plt.xlim(0, min(2000, max(frequencies)))

plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()

psd_plot_path = os.path.join(
    OUTPUT_DIR,
    "class_average_psd.png"
)

plt.savefig(psd_plot_path, dpi=300)
plt.show()


# ============================================================
# PLOT 2: ENERGY ABOVE 400 Hz
# ============================================================

plt.figure(figsize=(10, 6))

for class_name in CLASSES:

    values = [
        r["energy_above_400Hz_percent"]
        for r in all_results
        if r["class"] == class_name
    ]

    if values:
        plt.hist(
            values,
            bins=30,
            alpha=0.5,
            label=class_name
        )

plt.xlabel("Energy Above 400 Hz (%)")
plt.ylabel("Number of Files")
plt.title("Energy Above 400 Hz Across WAV Files")

plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

histogram_path = os.path.join(
    OUTPUT_DIR,
    "energy_above_400Hz_histogram.png"
)

plt.savefig(histogram_path, dpi=300)
plt.show()


# ============================================================
# SAVE CSV
# ============================================================

import csv

csv_path = os.path.join(
    OUTPUT_DIR,
    "frequency_analysis_results.csv"
)

with open(csv_path, "w", newline="") as f:

    if all_results:

        fieldnames = all_results[0].keys()

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(all_results)


print("=" * 70)
print("DONE")
print("=" * 70)
print(f"PSD plot:       {psd_plot_path}")
print(f"Histogram:      {histogram_path}")
print(f"CSV results:    {csv_path}")
print()
print(f"Files analyzed: {len(all_results)}")

