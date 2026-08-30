"""
run_Springer.py

Trains the Springer HSMM segmentation model on the example .mat data
(which has ground-truth S1/S2 annotations), then runs the trained model
on real recordings loaded from converted_data.pkl (produced by convert.py).

For each real recording, converts the per-sample assigned_states array
(1=S1, 2=Systole, 3=S2, 4=Diastole) into a list of
(phase_label, start_sample, end_sample) tuples, and saves everything
needed downstream (signal, fs, label, phase segments) to
segmented_data.pkl.

Files that fail during segmentation are logged and skipped rather than
stopping the run.
"""

from pathlib import Path
import pickle

import numpy as np
from scipy.io import loadmat

from Springer_Segmentation.default_Springer_HSMM_options import default_springer_hsmm_options
from Springer_Segmentation.train_Springer_segmentation_algorithm import train_springer_segmentation_algorithm
from Springer_Segmentation.run_Springer_segmentation_algorithm import run_springer_segmentation_algorithm

# --- config -----------------------------------------------------------

THIS_DIR = Path(__file__).resolve().parent

EXAMPLE_MAT_PATH = THIS_DIR / "Springer_Segmentation" / "example_data.mat"
CONVERTED_DATA_PATH = THIS_DIR / "converted_data.pkl"
OUTPUT_PATH = THIS_DIR / "segmented_data.pkl"

# Maps assigned_states integer values to phase labels, same mapping used
# in example_springer_script.py's plotting section.
STATE_LABELS = {1: "S1", 2: "Systole", 3: "S2", 4: "Diastole"}


# --- training -----------------------------------------------------------

def train_on_example_data(springer_options):
    """Train the HMM on all 6 example recordings (all have annotations, so
    no need to hold any out as 'test' the way the original demo script did
    — the real held-out test set is the actual Yaseen data now)."""
    mat = loadmat(str(EXAMPLE_MAT_PATH), struct_as_record=False, squeeze_me=True)
    example_data = mat["example_data"]

    train_recordings = example_data.example_audio_data
    train_annotations = example_data.example_annotations

    print(f"Training on {len(train_recordings)} example recordings...")
    b_matrix, pi_vector, total_obs_distribution = train_springer_segmentation_algorithm(
        train_recordings, train_annotations, springer_options.audio_Fs, False
    )
    print("Training complete.")

    return b_matrix, pi_vector, total_obs_distribution


# --- state sequence -> phase segments -----------------------------------

def states_to_segments(assigned_states: np.ndarray) -> list:
    """Convert a per-sample state array into a list of
    (phase_label, start_sample, end_sample) tuples for contiguous runs."""
    if len(assigned_states) == 0:
        return []

    change_points = np.where(np.diff(assigned_states) != 0)[0]
    boundaries = np.concatenate([[0], change_points + 1, [len(assigned_states)]])

    segments = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        state_val = int(assigned_states[start])
        label = STATE_LABELS.get(state_val)
        if label is None:
            # Shouldn't happen given the algorithm only emits 1-4, but don't
            # silently mislabel if it does.
            print(f"  WARNING: unexpected state value {state_val}, skipping segment")
            continue
        segments.append((label, int(start), int(end)))

    return segments


# --- main run -------------------------------------------------------------

def run_on_real_data(records, springer_options, b_matrix, pi_vector, total_obs_distribution):
    results = []
    failures = []

    for i, record in enumerate(records):
        filename = record["filename"]
        try:
            assigned_states = run_springer_segmentation_algorithm(
                record["signal"], record["fs"], b_matrix, pi_vector,
                total_obs_distribution, False
            )
            segments = states_to_segments(assigned_states)

            if not segments:
                raise ValueError("segmentation produced no phase segments")

            results.append({
                "filename": filename,
                "label": record["label"],
                "fs": record["fs"],
                "signal": record["signal"],
                "phase_segments": segments,
            })

        except Exception as e:
            print(f"  FAILED on {filename}: {e}")
            failures.append({"filename": filename, "label": record["label"], "error": str(e)})

        if (i + 1) % 50 == 0:
            print(f"  ...{i + 1}/{len(records)} processed")

    return results, failures


if __name__ == "__main__":
    springer_options = default_springer_hsmm_options()

    b_matrix, pi_vector, total_obs_distribution = train_on_example_data(springer_options)

    print(f"\nLoading converted real data from {CONVERTED_DATA_PATH}")
    with open(CONVERTED_DATA_PATH, "rb") as f:
        records = pickle.load(f)
    print(f"Loaded {len(records)} recordings.")

    print("\nRunning segmentation on real data...")
    results, failures = run_on_real_data(
        records, springer_options, b_matrix, pi_vector, total_obs_distribution
    )

    print(f"\nSucceeded: {len(results)}")
    print(f"Failed: {len(failures)}")

    if failures:
        print("\nFailed files:")
        for f in failures:
            print(f"  {f['label']}/{f['filename']}: {f['error']}")

    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump({"results": results, "failures": failures}, f)

    print(f"\nSaved to {OUTPUT_PATH}")