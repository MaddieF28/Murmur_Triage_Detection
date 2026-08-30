import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# ============================================================
# SETTINGS
# ============================================================

CODE_DIR = os.path.dirname(__file__)

DATASET_DIR = os.path.join(CODE_DIR, "..", "filtered_dataset")

THRESHOLD = 5.0

CLASSES = ["N", "AS", "MS", "MR", "MVP"]


# ============================================================
# FIND WAV FILES
# ============================================================

outliers = []

for class_name in CLASSES:

    class_dir = os.path.join(DATASET_DIR, class_name)

    if not os.path.exists(class_dir):
        print(f"WARNING: Could not find {class_dir}")
        continue

    for filename in os.listdir(class_dir):

        if not filename.lower().endswith(".wav"):
            continue

        filepath = os.path.join(class_dir, filename)

        try:
            sample_rate, audio = wavfile.read(filepath)

            # Convert stereo to mono
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)

            # Convert to float
            audio = audio.astype(float)

            # Remove DC offset
            audio = audio - np.mean(audio)

            # ------------------------------------------------
            # FFT
            # ------------------------------------------------

            n = len(audio)

            frequencies = np.fft.rfftfreq(
                n,
                d=1 / sample_rate
            )

            fft = np.fft.rfft(audio)

            power = np.abs(fft) ** 2

            # ------------------------------------------------
            # Energy above 400 Hz
            # ------------------------------------------------

            total_energy = np.sum(power)

            high_freq_energy = np.sum(
                power[frequencies > 400]
            )

            percent_above_400 = (
                high_freq_energy / total_energy * 100
            )

            if percent_above_400 > THRESHOLD:

                outliers.append({
                    "class": class_name,
                    "filename": filename,
                    "filepath": filepath,
                    "sample_rate": sample_rate,
                    "audio": audio,
                    "frequencies": frequencies,
                    "power": power,
                    "percent": percent_above_400
                })

        except Exception as e:

            print(
                f"Could not process {filename}: {e}"
            )


# ============================================================
# SORT OUTLIERS
# ============================================================

outliers.sort(
    key=lambda x: x["percent"],
    reverse=True
)


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 80)
print(f"HIGH-FREQUENCY OUTLIERS (> {THRESHOLD}%)")
print("=" * 80)

for i, result in enumerate(outliers):

    print(
        f"{i + 1:2d}. "
        f"{result['class']:3s}  "
        f"{result['percent']:8.3f}%  "
        f"{result['filename']}"
    )

print()
print(f"Total outliers: {len(outliers)}")


# ============================================================
# PLOT EACH OUTLIER
# ============================================================

for i, result in enumerate(outliers):

    audio = result["audio"]
    frequencies = result["frequencies"]
    power = result["power"]

    sample_rate = result["sample_rate"]

    # --------------------------------------------------------
    # Time axis
    # --------------------------------------------------------

    time = np.arange(len(audio)) / sample_rate

    # --------------------------------------------------------
    # Limit frequency plot to 2000 Hz
    # --------------------------------------------------------

    max_frequency = 2000

    freq_mask = frequencies <= max_frequency

    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------

    plt.figure(figsize=(12, 8))

    # ========================================================
    # WAVEFORM
    # ========================================================

    plt.subplot(2, 1, 1)

    plt.plot(time, audio)

    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")

    plt.title(
        f"{result['class']} - {result['filename']}\n"
        f"Waveform"
    )

    plt.grid(True)

    # ========================================================
    # FREQUENCY SPECTRUM
    # ========================================================

    plt.subplot(2, 1, 2)

    plt.plot(
        frequencies[freq_mask],
        power[freq_mask]
    )

    # Mark 400 Hz
    plt.axvline(
        400,
        linestyle="--",
        label="400 Hz"
    )

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power")

    plt.title(
        f"Frequency Spectrum "
        f"({result['percent']:.3f}% above 400 Hz)"
    )

    plt.xlim(0, max_frequency)

    plt.grid(True)

    plt.legend()

    # ========================================================
    # SHOW
    # ========================================================

    plt.tight_layout()

    print()
    print(
        f"Showing {i + 1}/{len(outliers)}: "
        f"{result['class']}/{result['filename']}"
    )

    plt.show()