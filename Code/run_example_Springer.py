# example_springer_script.py
from scipy.io import loadmat
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')

from Springer_Segmentation.default_Springer_HSMM_options import default_springer_hsmm_options
from Springer_Segmentation.train_Springer_segmentation_algorithm import train_springer_segmentation_algorithm
from Springer_Segmentation.run_Springer_segmentation_algorithm import run_springer_segmentation_algorithm
from Springer_Segmentation.get_heartrate_Schmidt import get_heart_rate_schmidt


springer_options = default_springer_hsmm_options()

# Load the audio data and the annotations. struct_as_record=False + squeeze_me=True
# gives dot-attribute access identical to MATLAB's example_data.example_audio_data /
# example_data.example_annotations.
mat = loadmat('Springer_Segmentation/example_data.mat', struct_as_record=False, squeeze_me=True)
example_data = mat['example_data']

print("=== Heart rate check across all 5 training recordings ===")
for i in range(5):
    s1 = np.asarray(example_data.example_annotations[i, 0], dtype=float).flatten()
    intervals = np.diff(s1) / springer_options.audio_segmentation_Fs
    gt_hr = 60 / np.mean(intervals)
    est_hr, _ = get_heart_rate_schmidt(example_data.example_audio_data[i], springer_options.audio_Fs)
    print(f"recording {i}: annotation={gt_hr:.1f} bpm, estimated={est_hr:.1f} bpm, diff={abs(gt_hr - est_hr):.1f}")
print()

# Split into train/test sets. MATLAB's [1:5] -> Python [0:5], MATLAB's (6) -> Python [5].
train_recordings = example_data.example_audio_data[0:5]
train_annotations = example_data.example_annotations[0:5, :]

test_recordings = [example_data.example_audio_data[5]]
test_annotations = example_data.example_annotations[5, :]

# Train the HMM.
b_matrix, pi_vector, total_obs_distribution = train_springer_segmentation_algorithm(
    train_recordings, train_annotations, springer_options.audio_Fs, False
)

# Run the HMM on the unseen test recording(s) and display the resulting segmentation.
num_pcgs = len(test_recordings)
for pcg_i in range(num_pcgs):
    assigned_states = run_springer_segmentation_algorithm(
        test_recordings[pcg_i], springer_options.audio_Fs, b_matrix, pi_vector,
        total_obs_distribution, True
    )
    
    print(f"assigned_states shape={assigned_states.shape}, unique values={np.unique(assigned_states)}, first 20={assigned_states[:20]}")
    import matplotlib.pyplot as plt

    print("=== Annotation structure check ===")
    print(f"type: {type(test_annotations)}")
    if hasattr(test_annotations, 'shape'):
        print(f"shape: {test_annotations.shape}")
    else:
        print(f"len: {len(test_annotations)}")
    print(f"contents: {test_annotations}")

    print("\n=== Heart rate comparison ===")
    try:
        s1_locations = test_annotations[0]
        s2_locations = test_annotations[1]

        s1_locations = np.asarray(s1_locations, dtype=float).flatten()
        print(f"s1_locations (first 10): {s1_locations[:10]}")
        print(f"number of annotated S1 events: {len(s1_locations)}")

        intervals_sec = np.diff(s1_locations) / springer_options.audio_segmentation_Fs
        annotation_heart_rate = 60 / np.mean(intervals_sec)

        estimated_heart_rate, estimated_systolic_time = get_heart_rate_schmidt(
            test_recordings[pcg_i], springer_options.audio_Fs
        )

        print(f"\nAnnotation-derived heart rate:      {annotation_heart_rate:.1f} bpm")
        print(f"Autocorrelation-estimated heart rate: {estimated_heart_rate:.1f} bpm")
        print(f"Difference: {abs(annotation_heart_rate - estimated_heart_rate):.1f} bpm")

    except Exception as e:
        print(f"Extraction failed with: {e}")
        print("Paste the '=== Annotation structure check ===' output above so we can fix the indexing.")

    from Springer_Segmentation.get_Springer_PCG_features import get_springer_pcg_features
    from Springer_Segmentation.label_PCG_states import label_pcg_states
    from Springer_Segmentation.expand_qt import expand_qt

    print("=== Segmentation agreement check ===")

    # Rebuild the same features used during segmentation, so the "ground truth"
    # state labels are derived on identical footing.
    pcg_features, features_fs = get_springer_pcg_features(test_recordings[pcg_i], springer_options.audio_Fs)

    # Ground-truth states from the annotations, at feature rate.
    ground_truth_states_feature_rate = label_pcg_states(
        pcg_features[:, 0], s1_locations, s2_locations, features_fs
    )

    # Expand to full audio rate, same way assigned_states was built, so the two
    # sequences are directly comparable sample-for-sample.
    ground_truth_states = expand_qt(
        ground_truth_states_feature_rate, features_fs, springer_options.audio_Fs, len(test_recordings[pcg_i])
    )

    exact_match = np.mean(ground_truth_states == assigned_states)
    print(f"Exact sample-level agreement: {exact_match*100:.1f}%")

    # Tolerance-window agreement: count a sample as "correct" if the assigned
    # state appears anywhere within +/- tolerance_ms of that sample in the
    # ground truth. Boundary timing is inherently fuzzy (both the annotations
    # and the HMM's duration model are approximations), so exact-sample
    # agreement understates how good the segmentation really is.
    tolerance_ms = 40
    tolerance_samples = round(tolerance_ms / 1000 * springer_options.audio_Fs)

    correct_with_tolerance = 0
    for idx in range(len(assigned_states)):
        lo = max(0, idx - tolerance_samples)
        hi = min(len(ground_truth_states), idx + tolerance_samples + 1)
        if assigned_states[idx] in ground_truth_states[lo:hi]:
            correct_with_tolerance += 1

    tolerant_match = correct_with_tolerance / len(assigned_states)
    print(f"Agreement within +/-{tolerance_ms}ms tolerance: {tolerant_match*100:.1f}%")

    t = np.arange(len(test_recordings[pcg_i])) / springer_options.audio_Fs

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(14, 6))

    ax1.plot(t, test_recordings[pcg_i], linewidth=0.5, color='black')
    ax1.set_ylabel("Amplitude")
    ax1.set_title("Test recording waveform")

    state_colors = {1: 'tab:red', 2: 'tab:blue', 3: 'tab:orange', 4: 'tab:green'}
    state_labels = {1: 'S1', 2: 'Systole', 3: 'S2', 4: 'Diastole'}

    change_points = np.where(np.diff(assigned_states) != 0)[0]
    boundaries = np.concatenate([[0], change_points + 1, [len(assigned_states)]])

    for start, end in zip(boundaries[:-1], boundaries[1:]):
        state_val = int(assigned_states[start])
        ax1.axvspan(t[start], t[min(end, len(t) - 1)], color=state_colors[state_val], alpha=0.2)

    ax2.plot(t, assigned_states, linewidth=1, color='tab:purple', drawstyle='steps-post')
    ax2.set_yticks([1, 2, 3, 4])
    ax2.set_yticklabels([state_labels[s] for s in [1, 2, 3, 4]])
    ax2.set_xlabel("Time (s)")
    ax2.set_title("Decoded state sequence")

    fig.tight_layout()
    plt.show()   