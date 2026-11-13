"""
data_prep/resample_yaseen_to_4000hz.py

Yaseen 2018 ships at 8000 Hz mono; every model in this repo (segmenter
features at 50 Hz = 80 samples per frame, duration priors, the CirCor
feature extractor) assumes 4000 Hz. This writes a 4000 Hz copy of every
Yaseen wav into paths.YASEEN_DIR/<class>/, preserving file names, using a
polyphase anti-aliased decimation (scipy resample_poly, up=1, down=2 for
the documented 8000 -> 4000 case; the general ratio is computed so a
mixed-rate folder would still be handled).

The raw 8000 Hz files under paths.YASEEN_RAW_DIR are left untouched.
Output is 16-bit PCM so segmenter/data_loading.py reads it exactly like
CirCor and the PhysioNet 2016 conversions.

Run once: py data_prep\\resample_yaseen_to_4000hz.py
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repo root: makes `paths` and the packages importable from any cwd
import paths


from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

TARGET_FS = 4000


def resample_file(src, dst):
    y, fs = sf.read(src, dtype="float64", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if fs != TARGET_FS:
        g = gcd(fs, TARGET_FS)
        y = resample_poly(y, TARGET_FS // g, fs // g)
    y = np.clip(y, -1.0, 1.0)
    sf.write(dst, (y * 32767).astype(np.int16), TARGET_FS, subtype="PCM_16")
    return fs, len(y) / TARGET_FS


def main():
    if not paths.YASEEN_RAW_DIR.exists():
        raise FileNotFoundError(f"Yaseen raw folder not found at {paths.YASEEN_RAW_DIR}")
    total = 0
    for cls in paths.YASEEN_CLASSES:
        src_dir = paths.YASEEN_RAW_DIR / cls
        dst_dir = paths.YASEEN_DIR / cls
        dst_dir.mkdir(parents=True, exist_ok=True)
        wavs = sorted(src_dir.glob("*.wav"))
        if not wavs:
            print(f"{cls}: no wavs found in {src_dir} -- skipped")
            continue
        rates, durations = set(), []
        for src in wavs:
            fs, dur = resample_file(src, dst_dir / src.name)
            rates.add(fs)
            durations.append(dur)
        total += len(wavs)
        print(f"{cls}: {len(wavs)} files, source rate(s) {sorted(rates)} -> {TARGET_FS} Hz, "
              f"duration s min {min(durations):.2f} / median {np.median(durations):.2f} / max {max(durations):.2f}")
    print(f"\nWrote {total} files under {paths.YASEEN_DIR}")


if __name__ == "__main__":
    main()
