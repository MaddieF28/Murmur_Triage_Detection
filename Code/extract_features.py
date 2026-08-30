"""
Phase-aware PCG feature extraction.

Shared feature pipeline for Arm A (generative Gaussian/GMM) and Arm B
(discriminative NN + SHAP). Both arms consume exactly the vectors this
module produces -- do not fork this per arm, that breaks the fairness
anchor the whole project design depends on.

Expected segmentation input format
-----------------------------------
For each recording, `segments` is a list of tuples:

    (phase_label, start_sample, end_sample)

where phase_label in {"S1", "systole", "S2", "diastole"} and
start_sample/end_sample are integer sample indices into the recording's
waveform (as produced by your Springer segmentation). A recording may
contain multiple cardiac cycles, so the same phase_label can appear
more than once -- e.g. two "S1" entries for a 2-cycle clip. Per-cycle
features are computed separately and then averaged into one vector per
phase per recording, so cycle count doesn't change the output shape.

If your segmentation function returns something else (e.g. a boundary
array + a separate label array, or times in seconds instead of sample
indices), convert to this tuple format before calling
`extract_recording_features` -- everything downstream assumes it.
"""

from __future__ import annotations

import numpy as np
import librosa
import pywt 
import pandas

# ---------------------------------------------------------------------
# Config -- keep these fixed across all phases and both arms.
# ---------------------------------------------------------------------

SR = 8000                # Yaseen dataset native sample rate
N_MFCC = 13               # static MFCCs only, no deltas
WAVELET = "db6"
WAVELET_LEVEL = 5
PHASES = ("S1", "Systole", "S2", "Diastole")

# Minimum samples needed to run librosa's default frame/hop settings.
# Very short segments (e.g. a clipped S1) get padded up to this length
# rather than crashing or silently returning NaNs.
MIN_SEGMENT_SAMPLES = 512


# ---------------------------------------------------------------------
# Per-cycle, per-phase feature extraction
# ---------------------------------------------------------------------

def _pad_if_short(y: np.ndarray, min_len: int = MIN_SEGMENT_SAMPLES) -> np.ndarray:
    """Zero-pad very short phase segments so frame-based features don't break."""
    if len(y) >= min_len:
        return y
    pad = min_len - len(y)
    return np.pad(y, (0, pad), mode="constant")


def _n_fft_for(y: np.ndarray, max_n_fft: int = 2048) -> int:
    """Largest power-of-2 FFT window that fits the segment, capped at max_n_fft."""
    n = min(max_n_fft, len(y))
    return int(2 ** np.floor(np.log2(max(n, 32))))


def _mfcc_features(y: np.ndarray, sr: int) -> dict:
    n_fft = _n_fft_for(y)
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=N_MFCC, n_fft=n_fft, hop_length=n_fft // 4
    )
    feats = {}
    for i in range(N_MFCC):
        feats[f"mfcc_{i+1}_mean"] = float(np.mean(mfcc[i]))
        feats[f"mfcc_{i+1}_std"] = float(np.std(mfcc[i]))
    return feats


def _spectral_features(y: np.ndarray, sr: int) -> dict:
    n_fft = _n_fft_for(y)
    hop = n_fft // 4
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft, hop_length=hop)[0]
    flatness = librosa.feature.spectral_flatness(y=y, n_fft=n_fft, hop_length=hop)[0]
    return {
        "spec_centroid_mean": float(np.mean(centroid)),
        "spec_centroid_std": float(np.std(centroid)),
        "spec_rolloff_mean": float(np.mean(rolloff)),
        "spec_flatness_mean": float(np.mean(flatness)),
    }


def _temporal_features(y: np.ndarray, sr: int, duration_samples: int) -> dict:
    return {
        "duration_sec": duration_samples / sr,
        "amp_mean": float(np.mean(np.abs(y))),
        "amp_max": float(np.max(np.abs(y))),
        "amp_std": float(np.std(y)),
    }


def _dwt_features(y: np.ndarray) -> dict:
    # Cap decomposition level so pywt doesn't error on short segments
    # (e.g. a brief S1) that can't support the full requested depth.
    max_level = pywt.dwt_max_level(len(y), 12)
    level = min(WAVELET_LEVEL, max_level) if max_level > 0 else 1
    coeffs = pywt.wavedec(y, WAVELET, level=level)
    feats = {}
    # coeffs[0] = final approximation, coeffs[1:] = detail coeffs, coarse->fine
    for i, c in enumerate(coeffs):
        name = "approx" if i == 0 else f"detail_{len(coeffs) - i}"
        feats[f"dwt_energy_{name}"] = float(np.sum(c ** 2))
    # Pad with zeros if this segment produced fewer levels than the
    # nominal WAVELET_LEVEL, so every recording has the same column set.
    expected = ["dwt_energy_approx"] + [
        f"dwt_energy_detail_{lvl}" for lvl in range(1, WAVELET_LEVEL + 1)
    ]
    for name in expected:
        feats.setdefault(name, 0.0)
    return {k: feats[k] for k in expected}


def _phase_cycle_features(y: np.ndarray, sr: int) -> dict:
    """All features for a single phase instance in a single cycle."""
    duration_samples = len(y)
    y_proc = _pad_if_short(y)
    feats = {}
    feats.update(_temporal_features(y, sr, duration_samples))
    feats.update(_mfcc_features(y_proc, sr))
    feats.update(_spectral_features(y_proc, sr))
    feats.update(_dwt_features(y_proc))
    return feats


# ---------------------------------------------------------------------
# Per-recording aggregation
# ---------------------------------------------------------------------

def extract_recording_features(
    y: np.ndarray, sr: int, segments: list[tuple[str, int, int]]
) -> dict:
    """
    Extract one feature vector per cardiac phase for a single recording.

    Parameters
    ----------
    y : 1D waveform for the whole recording (already resampled/filtered/
        denoised per the preprocessing step).
    sr : sample rate.
    segments : list of (phase_label, start_sample, end_sample), one entry
        per phase instance per cycle. Multiple cycles -> repeated labels.

    Returns
    -------
    dict mapping phase_label -> {feature_name: value}. If a phase never
    occurs in this recording (e.g. a segmentation failure), that phase's
    dict is filled with NaNs of the correct shape so downstream code can
    decide how to handle missing phases explicitly rather than silently
    dropping a column.
    """
    by_phase: dict[str, list[dict]] = {p: [] for p in PHASES}

    for phase_label, start, end in segments:
        if phase_label not in PHASES:
            raise ValueError(
                f"Unexpected phase label {phase_label!r}; expected one of {PHASES}"
            )
        segment_audio = y[start:end]
        if len(segment_audio) == 0:
            continue
        by_phase[phase_label].append(_phase_cycle_features(segment_audio, sr))

    result = {}
    for phase in PHASES:
        cycle_feats = by_phase[phase]
        if not cycle_feats:
            result[phase] = None  # caller decides how to handle missing phase
            continue
        # Average per-cycle vectors into one vector for this phase.
        keys = cycle_feats[0].keys()
        result[phase] = {
            k: float(np.mean([f[k] for f in cycle_feats])) for k in keys
        }
    return result


def flatten_recording_features(phase_features: dict) -> dict:
    """
    Flatten the per-phase dict into a single flat dict with
    'phase_featurename' keys, e.g. 'S1_mfcc_1_mean'. Missing phases
    (None) are filled with NaN using the feature-name set from whichever
    phase did extract successfully, so every recording's row has the
    identical column set for a DataFrame.
    """
    # Find a template of feature names from any non-missing phase.
    template = None
    for v in phase_features.values():
        if v is not None:
            template = list(v.keys())
            break
    if template is None:
        raise ValueError("All phases missing for this recording -- check segmentation.")

    flat = {}
    for phase in PHASES:
        vals = phase_features.get(phase)
        for name in template:
            flat[f"{phase}_{name}"] = (
                vals[name] if vals is not None else float("nan")
            )
    return flat


# ---------------------------------------------------------------------
# Dataset-level pipeline
# ---------------------------------------------------------------------

def build_feature_table(recordings: list[dict]):
    """
    recordings: list of dicts, each with keys:
        'y': waveform, 'sr': sample rate, 'segments': segment list
             (see extract_recording_features), 'label': class label,
             'recording_id': identifier string
    Returns a pandas DataFrame, one row per recording, with columns
    'recording_id', 'label', and one column per phase-feature combo.
    """
    import pandas as pd

    rows = []
    for rec in recordings:
        phase_feats = extract_recording_features(rec["y"], rec["sr"], rec["segments"])
        flat = flatten_recording_features(phase_feats)
        flat["recording_id"] = rec["recording_id"]
        flat["label"] = rec["label"]
        rows.append(flat)
    return pd.DataFrame(rows)

# ---------------------------------------------------------------------
# Load segmented_data.pkl and build feature table
# ---------------------------------------------------------------------

from pathlib import Path
import pickle


def load_segmented_data(pkl_path):
    """Load the output produced by run_Springer.py."""

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    # run_Springer.py saves:
    # {"results": results, "failures": failures}
    results = data["results"]

    return results


def main():

    # extract.py and segmented_data.pkl are assumed to be in
    # the same folder.
    this_dir = Path(__file__).resolve().parent

    input_path = this_dir / "segmented_data.pkl"
    output_path = this_dir / "pcg_feature_table.csv"

    print("=" * 60)
    print("PHASE-AWARE PCG FEATURE EXTRACTION")
    print("=" * 60)

    # -------------------------------------------------------------
    # Load segmentation results
    # -------------------------------------------------------------

    print(f"\nLoading: {input_path}")

    results = load_segmented_data(input_path)

    print(f"Loaded {len(results)} recordings.")

    # -------------------------------------------------------------
    # Convert segmented results into the format expected by
    # build_feature_table()
    # -------------------------------------------------------------

    recordings = []

    for record in results:

        recordings.append({
            "y": np.asarray(record["signal"], dtype=float),
            "sr": int(record["fs"]),
            "segments": record["phase_segments"],
            "label": record["label"],
            "recording_id": record["filename"],
        })

    print("Converted segmentation data.")

    # -------------------------------------------------------------
    # Extract features
    # -------------------------------------------------------------

    print("\nExtracting features...")

    df = build_feature_table(recordings)

    # -------------------------------------------------------------
    # Report
    # -------------------------------------------------------------

    print("\n" + "=" * 60)
    print("FEATURE EXTRACTION COMPLETE")
    print("=" * 60)

    print(f"Number of recordings : {len(df)}")
    print(f"Number of columns    : {len(df.columns)}")
    print(f"Number of features   : {len(df.columns) - 2}")

    print("\nClass distribution:")
    print(df["label"].value_counts())

    print("\nFeature table preview:")
    print(df.head())

    # -------------------------------------------------------------
    # Save
    # -------------------------------------------------------------

    df.to_csv(output_path, index=False)

    print(f"\nSaved feature table to:")
    print(output_path)

    print("\nDone.")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------
# Standardization (fit-on-train, apply-to-both — call this from train
# files AFTER splitting, never before, to avoid leaking test-set
# statistics into the fitted mean/std)
# ---------------------------------------------------------------------

def standardize_features(
    train_df: "pandas.DataFrame",
    test_df: "pandas.DataFrame",
    exclude_cols: tuple = ("recording_id", "label"),
):
    """
    Fit a StandardScaler (mean=0, std=1) on train_df's feature columns
    only, then apply it to both train_df and test_df. Both Arm A and
    Arm B should call this identically on the same train/test split so
    they see the same standardized numbers -- do not fit separately per
    arm, that would break the shared-feature fairness anchor.

    Parameters
    ----------
    train_df, test_df : DataFrames with the same columns as produced by
        build_feature_table (feature columns plus 'recording_id' and
        'label').
    exclude_cols : columns to leave untouched (identifiers/labels, not
        features).

    Returns
    -------
    (train_scaled, test_scaled, scaler)
        train_scaled, test_scaled : copies of the input DataFrames with
            feature columns replaced by standardized values.
        scaler : the fitted StandardScaler, so it can be reused later on
            genuinely new/unseen recordings at inference time (never
            refit on new data -- always .transform() with this same
            scaler).
    """
    from sklearn.preprocessing import StandardScaler

    feature_cols = [c for c in train_df.columns if c not in exclude_cols]

    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols])

    train_scaled = train_df.copy()
    test_scaled = test_df.copy()

    train_scaled[feature_cols] = scaler.transform(train_df[feature_cols])
    test_scaled[feature_cols] = scaler.transform(test_df[feature_cols])

    return train_scaled, test_scaled, scaler