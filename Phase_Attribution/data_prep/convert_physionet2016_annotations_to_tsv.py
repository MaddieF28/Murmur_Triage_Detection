import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repo root: makes `paths` and the packages importable from any cwd
import paths

import os
import glob
import scipy.io as sio
import scipy.io.wavfile as wavfile
import numpy as np

STATE_TO_CODE = {"S1": 1, "systole": 2, "S2": 3, "diastole": 4}
NOISE_LABELS = {"(N", "N)"}
ANNOTATION_SAMPLE_RATE = 2000  # Hz, per the 2016 dataset README

ANNOTATION_DIR = os.path.join(paths.PHYSIONET2016_RAW_DIR, "annotations", "annotations", "hand_corrected", "training-a_StateAns")
WAV_DIR = os.path.join(paths.PHYSIONET2016_RAW_DIR, "training-a")
OUTPUT_DIR = str(paths.TRAINING_DIRS["training-a"])

os.makedirs(OUTPUT_DIR, exist_ok=True)
"""
Converts hand-corrected StateAns .mat annotations into .tsv files for every
PhysioNet 2016 challenge training set (training-a through training-f).

Generalized from the original training-a-only conversion script -- same
per-file logic, just looped over all six letters with per-dataset directory
skipping (so a missing subset doesn't crash the whole run) and an overall
summary at the end.
"""

import os
import glob
import scipy.io as sio
import scipy.io.wavfile as wavfile
import numpy as np

STATE_TO_CODE = {"S1": 1, "systole": 2, "S2": 3, "diastole": 4}
NOISE_LABELS = {"(N", "N)"}
ANNOTATION_SAMPLE_RATE = 2000  # Hz, per the 2016 dataset README -- assumed same for a-f

BASE_DIR = str(paths.PHYSIONET2016_RAW_DIR)
TRAINING_LETTERS = "abcdef"


def unwrap_scalar(x):
    while hasattr(x, "item"):
        x = x.item()
    return x


def get_raw_labels(mat_path):
    """Reads state_ans and returns (starts_samples, labels) with no filtering."""
    mat = sio.loadmat(mat_path)
    state_ans = mat["state_ans"]

    starts_samples = []
    labels = []
    for row in state_ans:
        starts_samples.append(int(unwrap_scalar(row[0])))
        labels.append(str(unwrap_scalar(row[1])))
    return starts_samples, labels


def convert_to_tsv_array(starts_samples, labels, audio_duration_sec):
    """Assumes labels contains ONLY valid states -- call only after the noise check."""
    rows = []
    for i in range(len(starts_samples)):
        start_sec = starts_samples[i] / ANNOTATION_SAMPLE_RATE
        if i + 1 < len(starts_samples):
            end_sec = starts_samples[i + 1] / ANNOTATION_SAMPLE_RATE
        else:
            end_sec = audio_duration_sec
        rows.append([start_sec, end_sec, STATE_TO_CODE[labels[i]]])
    return np.array(rows).reshape(-1, 3)


def find_matching_wav(stem, wav_dir):
    candidate = os.path.join(wav_dir, f"{stem}.wav")
    if os.path.exists(candidate):
        return candidate
    matches = glob.glob(os.path.join(wav_dir, "*.wav"))
    for m in matches:
        if os.path.splitext(os.path.basename(m))[0].lower() == stem.lower():
            return m
    return None


def convert_one_dataset(letter):
    annotation_dir = os.path.join(
        BASE_DIR, "annotations", "annotations", "hand_corrected", f"training-{letter}_StateAns"
    )
    wav_dir = os.path.join(BASE_DIR, f"training-{letter}")
    output_dir = str(paths.PHYSIONET2016_CONVERTED_DIR / f"training-{letter}")

    if not os.path.isdir(annotation_dir):
        print(f"[training-{letter}] SKIPPED ENTIRE DATASET -- annotation dir not found: {annotation_dir}")
        return None
    if not os.path.isdir(wav_dir):
        print(f"[training-{letter}] SKIPPED ENTIRE DATASET -- wav dir not found: {wav_dir}")
        return None

    os.makedirs(output_dir, exist_ok=True)

    mat_files = sorted(glob.glob(os.path.join(annotation_dir, "*_StateAns.mat")))
    print(f"\n=== training-{letter}: found {len(mat_files)} annotation files in {annotation_dir} ===")

    converted = 0
    excluded_noisy = []
    failed = []
    other_unrecognized = []

    for mat_path in mat_files:
        stem = os.path.basename(mat_path).replace("_StateAns.mat", "")
        wav_path = find_matching_wav(stem, wav_dir)

        if wav_path is None:
            print(f"[training-{letter}/{stem}] SKIPPED -- no matching .wav found in {wav_dir}")
            failed.append((stem, "no matching wav"))
            continue

        try:
            starts_samples, labels = get_raw_labels(mat_path)
            label_set = set(labels)

            if label_set & NOISE_LABELS:
                excluded_noisy.append(stem)
                continue

            unexpected = label_set - set(STATE_TO_CODE.keys())
            if unexpected:
                print(f"[training-{letter}/{stem}] SKIPPED -- unrecognized label(s): {unexpected}")
                other_unrecognized.append((stem, unexpected))
                continue

            sr, audio = wavfile.read(wav_path)
            duration_sec = len(audio) / sr

            tsv_array = convert_to_tsv_array(starts_samples, labels, duration_sec)

            out_path = os.path.join(output_dir, f"{stem}.tsv")
            np.savetxt(out_path, tsv_array, delimiter="\t", fmt=["%.6f", "%.6f", "%d"])
            converted += 1

        except Exception as e:
            print(f"[training-{letter}/{stem}] FAILED -- {type(e).__name__}: {e}")
            failed.append((stem, str(e)))

    print(f"--- training-{letter} summary ---")
    print(f"Converted: {converted}/{len(mat_files)}")
    print(f"Excluded (contains noise marker): {len(excluded_noisy)}")
    print(f"Failed: {len(failed)}")
    print(f"Other unrecognized labels: {len(other_unrecognized)}")
    if excluded_noisy:
        print(f"Excluded files: {excluded_noisy}")
    if failed:
        print("Failed files:")
        for stem, reason in failed:
            print(f"  {stem}: {reason}")
    if other_unrecognized:
        print("Files with unexpected labels (not noise, not a valid state -- investigate):")
        for stem, labs in other_unrecognized:
            print(f"  {stem}: {labs}")

    return {
        "letter": letter,
        "converted": converted,
        "total": len(mat_files),
        "excluded_noisy": excluded_noisy,
        "failed": failed,
        "other_unrecognized": other_unrecognized,
    }


def main():
    results = []
    for letter in TRAINING_LETTERS:
        r = convert_one_dataset(letter)
        if r is not None:
            results.append(r)

    print("\n\n=== OVERALL SUMMARY ===")
    for r in results:
        print(
            f"training-{r['letter']}: {r['converted']}/{r['total']} converted, "
            f"{len(r['excluded_noisy'])} excluded (noise), {len(r['failed'])} failed, "
            f"{len(r['other_unrecognized'])} unrecognized-label files"
        )
    missing = set(TRAINING_LETTERS) - {r["letter"] for r in results}
    if missing:
        print(f"Datasets skipped entirely (dirs not found): {sorted(missing)}")


if __name__ == "__main__":
    main()

def unwrap_scalar(x):
    while hasattr(x, "item"):
        x = x.item()
    return x


def get_raw_labels(mat_path):
    """Reads state_ans and returns (starts_samples, labels) with no filtering."""
    mat = sio.loadmat(mat_path)
    state_ans = mat["state_ans"]

    starts_samples = []
    labels = []
    for row in state_ans:
        starts_samples.append(int(unwrap_scalar(row[0])))
        labels.append(str(unwrap_scalar(row[1])))
    return starts_samples, labels


def convert_to_tsv_array(starts_samples, labels, audio_duration_sec):
    """Assumes labels contains ONLY valid states -- call only after the noise check."""
    rows = []
    for i in range(len(starts_samples)):
        start_sec = starts_samples[i] / ANNOTATION_SAMPLE_RATE
        if i + 1 < len(starts_samples):
            end_sec = starts_samples[i + 1] / ANNOTATION_SAMPLE_RATE
        else:
            end_sec = audio_duration_sec
        rows.append([start_sec, end_sec, STATE_TO_CODE[labels[i]]])
    return np.array(rows).reshape(-1, 3)


def find_matching_wav(stem, wav_dir):
    candidate = os.path.join(wav_dir, f"{stem}.wav")
    if os.path.exists(candidate):
        return candidate
    matches = glob.glob(os.path.join(wav_dir, "*.wav"))
    for m in matches:
        if os.path.splitext(os.path.basename(m))[0].lower() == stem.lower():
            return m
    return None


def main():
    mat_files = sorted(glob.glob(os.path.join(ANNOTATION_DIR, "*_StateAns.mat")))
    print(f"Found {len(mat_files)} annotation files in {ANNOTATION_DIR}")

    converted = 0
    excluded_noisy = []
    failed = []
    other_unrecognized = []  # labels seen that are neither a valid state nor a known noise marker

    for mat_path in mat_files:
        stem = os.path.basename(mat_path).replace("_StateAns.mat", "")
        wav_path = find_matching_wav(stem, WAV_DIR)

        if wav_path is None:
            print(f"[{stem}] SKIPPED -- no matching .wav found in {WAV_DIR}")
            failed.append((stem, "no matching wav"))
            continue

        try:
            starts_samples, labels = get_raw_labels(mat_path)
            label_set = set(labels)

            if label_set & NOISE_LABELS:
                excluded_noisy.append(stem)
                continue

            unexpected = label_set - set(STATE_TO_CODE.keys())
            if unexpected:
                print(f"[{stem}] SKIPPED -- unrecognized label(s) not in STATE_TO_CODE or NOISE_LABELS: {unexpected}")
                other_unrecognized.append((stem, unexpected))
                continue

            sr, audio = wavfile.read(wav_path)
            duration_sec = len(audio) / sr

            tsv_array = convert_to_tsv_array(starts_samples, labels, duration_sec)

            out_path = os.path.join(OUTPUT_DIR, f"{stem}.tsv")
            np.savetxt(out_path, tsv_array, delimiter="\t", fmt=["%.6f", "%.6f", "%d"])
            converted += 1

        except Exception as e:
            print(f"[{stem}] FAILED -- {type(e).__name__}: {e}")
            failed.append((stem, str(e)))

    print("\n--- Summary ---")
    print(f"Converted: {converted}/{len(mat_files)}")
    print(f"Excluded (contains noise marker): {len(excluded_noisy)}")
    print(f"Failed: {len(failed)}")
    print(f"Other unrecognized labels: {len(other_unrecognized)}")

    if excluded_noisy:
        print(f"\nExcluded files: {excluded_noisy}")
    if failed:
        print(f"\nFailed files:")
        for stem, reason in failed:
            print(f"  {stem}: {reason}")
    if other_unrecognized:
        print(f"\nFiles with unexpected labels (not noise, not a valid state -- investigate):")
        for stem, labs in other_unrecognized:
            print(f"  {stem}: {labs}")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()