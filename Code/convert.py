"""
convert.py

Reads real Yaseen .wav recordings from per-class folders, resamples them from
their native 8000 Hz down to the Springer segmentation algorithm's expected
audio_Fs (1000 Hz, per default_Springer_HSMM_options.py), and saves the
result to a pickle file that segment.py can load and loop over.

Expected folder layout (wav folders are siblings of this file's folder):

    <parent>/
        code/
            convert.py      <- this file
        AS/
            *.wav
        MR/
            *.wav
        MS/
            *.wav
        MVP/
            *.wav
        N/
            *.wav

No filtering or normalization is applied here — only resampling. Springer's
algorithm expects a single fixed audio_Fs for every recording it processes,
so every file is resampled from its native rate down to that target.
"""

from pathlib import Path
import pickle

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

# --- config -------------------------------------------------------------

# Target rate the Springer algorithm expects (audio_Fs from
# default_Springer_HSMM_options.py). Change this if that default ever changes.
TARGET_FS = 1000

# Class folder names -> these become the labels. Edit if your folder names
# differ from this.
CLASS_LABELS = ["AS", "MR", "MS", "MVP", "N"]

# This file's folder is "code/"; wav class folders are siblings of "code/"'s
# parent, i.e. one level up from this file, then into each class folder.
THIS_DIR = Path(__file__).resolve().parent
DATA_ROOT = THIS_DIR.parent

OUTPUT_PATH = THIS_DIR / "converted_data.pkl"


# --- conversion -----------------------------------------------------------

def load_and_resample_wav(wav_path: Path, target_fs: int) -> np.ndarray:
    """Read a wav file and resample it to target_fs. Assumes mono input."""
    native_fs, signal = wavfile.read(wav_path)

    if signal.ndim != 1:
        raise ValueError(
            f"{wav_path} is not mono (got shape {signal.shape}); "
            "this script assumes mono input and doesn't know how you want "
            "multi-channel audio handled."
        )

    # wavfile.read returns integer PCM data for most wav files; convert to
    # float64 so resample_poly and downstream processing behave predictably.
    signal = signal.astype(np.float64)

    if native_fs == target_fs:
        return signal

    # resample_poly wants the up/down ratio in lowest terms. For the
    # documented 8000 -> 1000 case this reduces to up=1, down=8.
    from math import gcd
    g = gcd(native_fs, target_fs)
    up = target_fs // g
    down = native_fs // g

    return resample_poly(signal, up, down)


def convert_all(data_root: Path, class_labels: list, target_fs: int) -> list:
    """Walk each class folder, convert every wav, return a list of records."""
    records = []

    for label in class_labels:
        class_dir = data_root / label
        if not class_dir.is_dir():
            print(f"WARNING: expected folder not found, skipping: {class_dir}")
            continue

        wav_files = sorted(class_dir.glob("*.wav"))
        print(f"{label}: {len(wav_files)} files")

        for wav_path in wav_files:
            try:
                signal = load_and_resample_wav(wav_path, target_fs)
            except Exception as e:
                print(f"  FAILED on {wav_path.name}: {e}")
                continue

            records.append({
                "filename": wav_path.name,
                "label": label,
                "fs": target_fs,
                "signal": signal,
            })

    return records


if __name__ == "__main__":
    print(f"Data root: {DATA_ROOT}")
    print(f"Target Fs: {TARGET_FS}")
    print()

    records = convert_all(DATA_ROOT, CLASS_LABELS, TARGET_FS)

    print()
    print(f"Converted {len(records)} recordings total.")

    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(records, f)

    print(f"Saved to {OUTPUT_PATH}")