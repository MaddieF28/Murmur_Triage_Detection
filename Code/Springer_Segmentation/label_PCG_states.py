import numpy as np

def label_pcg_states(envelope, s1_positions, s2_positions, sampling_frequency):
    """Python port of labelPCGStates.m
    States: 1=S1, 2=systole, 3=S2, 4=diastole
    s1_positions / s2_positions are 0-indexed sample positions (adjust if your
    annotations come in as 1-indexed MATLAB positions)."""
    n = len(envelope)
    states = np.zeros(n, dtype=int)

    mean_S1 = 0.122 * sampling_frequency
    std_S1 = 0.022 * sampling_frequency
    mean_S2 = 0.092 * sampling_frequency
    std_S2 = 0.022 * sampling_frequency

    # S1 windows: from each R-peak to R-peak + mean_S1
    for s1 in s1_positions:
        start = max(0, int(s1))
        upper = min(n, int(round(s1 + mean_S1)))
        states[start:upper] = 1

    # S2 windows: search for envelope peak near each end-T-wave
    for i, s2 in enumerate(s2_positions):
        lower_bound = max(int(s2 - np.floor(mean_S2 + std_S2)), 0)
        upper_bound = min(n, int(np.ceil(s2 + np.floor(mean_S2 + std_S2))))
        window = envelope[lower_bound:upper_bound] * (states[lower_bound:upper_bound] != 1)
        if len(window) == 0:
            continue
        s2_index_rel = np.argmax(window)
        s2_index = min(n - 1, lower_bound + s2_index_rel)

        upper = min(n, int(np.ceil(s2_index + mean_S2 / 2)))
        lower = max(0, int(np.ceil(s2_index - mean_S2 / 2)))
        states[lower:upper] = 3

        # Diastole: from end of this S2 to the next R-peak
        diffs = np.array(s1_positions, dtype=float) - s2
        diffs[diffs < 0] = np.inf
        if np.all(np.isinf(diffs)):
            end_pos = n
        else:
            idx = np.argmin(diffs)
            end_pos = int(s1_positions[idx]) - 1
        diastole_start = int(np.ceil(s2_index + mean_S2 / 2))
        states[diastole_start:max(diastole_start, end_pos)] = 4

    # Fill in the lead-in before the first definite state
    definite = np.where(states != 0)[0]
    if len(definite) > 0:
        first_idx = definite[0]
        if first_idx > 0:
            if states[first_idx] == 1:
                states[:first_idx] = 4
            elif states[first_idx] == 3:
                states[:first_idx] = 2

        last_idx = definite[-1]
        if last_idx < n - 1:
            if states[last_idx] == 1:
                states[last_idx:] = 2
            elif states[last_idx] == 3:
                states[last_idx:] = 4

    # Anything still unlabeled defaults to systole (state 2)
    states[states == 0] = 2
    return states