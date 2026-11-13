"""
Resamples every PhysioNet 2016 challenge training set (training-a through
training-f) to true 4000 Hz and writes the result straight into
data/physionet2016_converted/training-{letter}/, alongside the .tsv files produced by
data_prep/convert_physionet2016_annotations_to_tsv.py -- same end location segmenter_eval/evaluate_segmenter_all_datasets.py already reads
from, so no separate copy step is needed.

Run data_prep/convert_physionet2016_annotations_to_tsv.py first so data/physionet2016_converted/training-{letter}/ exists.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repo root: makes `paths` and the packages importable from any cwd
import paths


import os
import glob
import scipy.io.wavfile as wavfile
from scipy.signal import resample_poly

BASE_DIR = str(paths.PHYSIONET2016_RAW_DIR)
TRAINING_LETTERS = "abcdef"
TARGET_SR = 4000


def resample_dataset(letter):
    src_dir = os.path.join(BASE_DIR, f"training-{letter}")
    converted_dir = str(paths.PHYSIONET2016_CONVERTED_DIR / f"training-{letter}")

    if not os.path.isdir(src_dir):
        print(f"[training-{letter}] SKIPPED -- source wav dir not found: {src_dir}")
        return
    if not os.path.isdir(converted_dir):
        print(
            f"[training-{letter}] SKIPPED -- {converted_dir} does not exist yet "
            f"(run data_prep/convert_physionet2016_annotations_to_tsv.py first)"
        )
        return

    # clear any stale wavs already sitting in data/physionet2016_converted/training-{letter}
    stale = glob.glob(os.path.join(converted_dir, "*.wav"))
    for f in stale:
        os.remove(f)
    print(f"training-{letter}: cleared {len(stale)} stale .wav file(s) from {converted_dir}")

    wav_files = sorted(glob.glob(os.path.join(src_dir, "*.wav")))
    print(f"training-{letter}: found {len(wav_files)} .wav files to resample")

    converted = 0
    skipped_already_correct = 0
    for wav_path in wav_files:
        sr, audio = wavfile.read(wav_path)
        stem = os.path.basename(wav_path)
        out_path = os.path.join(converted_dir, stem)

        if sr == TARGET_SR:
            skipped_already_correct += 1
            wavfile.write(out_path, sr, audio)
            continue

        if TARGET_SR % sr != 0:
            raise ValueError(
                f"training-{letter}/{stem}: source rate {sr} does not evenly divide into "
                f"{TARGET_SR} -- resample_poly needs an integer up/down ratio for a clean conversion"
            )

        up = TARGET_SR // sr
        resampled = resample_poly(audio, up=up, down=1)
        resampled = resampled.astype(audio.dtype)  # preserve original dtype (e.g. int16)

        wavfile.write(out_path, TARGET_SR, resampled)
        converted += 1

    print(
        f"training-{letter}: resampled {converted}, already-correct {skipped_already_correct}, "
        f"wrote directly into {converted_dir}"
    )


def main():
    for letter in TRAINING_LETTERS:
        resample_dataset(letter)


if __name__ == "__main__":
    main()