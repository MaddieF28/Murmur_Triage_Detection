import os
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt

# ============================================================
# SETTINGS
# ============================================================

CODE_DIR = os.path.dirname(__file__)

INPUT_DIR = os.path.join(CODE_DIR, "..")

OUTPUT_DIR = os.path.join(
    CODE_DIR,
    "..",
    "filtered_dataset"
)

LOW_CUTOFF = 25
HIGH_CUTOFF = 400

FILTER_ORDER = 4

CLASSES = ["N", "AS", "MS", "MR", "MVP"]


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

for class_name in CLASSES:

    output_class_dir = os.path.join(
        OUTPUT_DIR,
        class_name
    )

    os.makedirs(
        output_class_dir,
        exist_ok=True
    )


# ============================================================
# PROCESS FILES
# ============================================================

for class_name in CLASSES:

    input_class_dir = os.path.join(
        INPUT_DIR,
        class_name
    )

    output_class_dir = os.path.join(
        OUTPUT_DIR,
        class_name
    )

    if not os.path.exists(input_class_dir):

        print(
            f"WARNING: {input_class_dir} does not exist"
        )

        continue

    wav_files = [
        f for f in os.listdir(input_class_dir)
        if f.lower().endswith(".wav")
    ]

    print()
    print(
        f"Processing {class_name}: "
        f"{len(wav_files)} WAV files"
    )

    processed = 0

    for filename in wav_files:

        input_path = os.path.join(
            input_class_dir,
            filename
        )

        output_path = os.path.join(
            output_class_dir,
            filename
        )

        try:

            # ------------------------------------------------
            # Load audio
            # ------------------------------------------------

            sample_rate, audio = wavfile.read(
                input_path
            )

            # ------------------------------------------------
            # Convert stereo to mono
            # ------------------------------------------------

            if audio.ndim > 1:

                audio = np.mean(
                    audio,
                    axis=1
                )

            # ------------------------------------------------
            # Convert to float
            # ------------------------------------------------

            original_dtype = audio.dtype

            audio = audio.astype(
                np.float64
            )

            # ------------------------------------------------
            # Normalize before filtering
            # ------------------------------------------------

            max_value = np.max(
                np.abs(audio)
            )

            if max_value > 0:

                audio_normalized = (
                    audio / max_value
                )

            else:

                audio_normalized = audio

            # ------------------------------------------------
            # Create Butterworth bandpass filter
            # ------------------------------------------------

            nyquist = sample_rate / 2

            low = LOW_CUTOFF / nyquist
            high = HIGH_CUTOFF / nyquist

            sos = butter(
                FILTER_ORDER,
                [low, high],
                btype="bandpass",
                output="sos"
            )

            # ------------------------------------------------
            # Apply zero-phase filter
            # ------------------------------------------------

            filtered = sosfiltfilt(
                sos,
                audio_normalized
            )

            # ------------------------------------------------
            # Scale back to original amplitude
            # ------------------------------------------------

            filtered_max = np.max(
                np.abs(filtered)
            )

            if filtered_max > 0:

                filtered = (
                    filtered / filtered_max
                )

            # ------------------------------------------------
            # Convert back to WAV format
            # ------------------------------------------------

            if original_dtype == np.int16:

                filtered = (
                    filtered * 32767
                ).astype(np.int16)

            elif original_dtype == np.int32:

                filtered = (
                    filtered * 2147483647
                ).astype(np.int32)

            else:

                filtered = filtered.astype(
                    np.float32
                )

            # ------------------------------------------------
            # Save
            # ------------------------------------------------

            wavfile.write(
                output_path,
                sample_rate,
                filtered
            )

            processed += 1

        except Exception as e:

            print(
                f"ERROR: {filename}: {e}"
            )

    print(
        f"Successfully processed: {processed}"
    )


# ============================================================
# DONE
# ============================================================

print()
print("=" * 70)
print("FILTERING COMPLETE")
print("=" * 70)

print()
print(f"Filter: {LOW_CUTOFF}–{HIGH_CUTOFF} Hz")
print(f"Order: {FILTER_ORDER}")
print()
print(f"Output directory:")
print(OUTPUT_DIR)