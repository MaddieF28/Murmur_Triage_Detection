import numpy as np

def expand_qt(original_qt, old_fs, new_fs, new_length):
    """Python port of expand_qt.m — upsamples frame-level HMM states
    back to the original audio sampling rate."""
    original_qt = np.asarray(original_qt).flatten()
    expanded_qt = np.zeros(new_length, dtype=int)

    change_indices = np.where(np.diff(original_qt) != 0)[0]
    change_indices = np.append(change_indices, len(original_qt) - 1)

    start_index = 0
    value_at_mid = original_qt[0]
    for end_index in change_indices:
        mid_point = round((end_index - start_index) / 2) + start_index
        value_at_mid = original_qt[mid_point]

        expanded_start = round((start_index / old_fs) * new_fs)
        expanded_end = round((end_index / old_fs) * new_fs)
        expanded_end = min(expanded_end, new_length)

        expanded_qt[expanded_start:expanded_end] = value_at_mid
        start_index = end_index

    # The last segment's boundary maps to slightly before new_length
    # (since end_index is old_length - 1, not old_length). Fill the
    # remaining tail with the final segment's state value.
    if expanded_end < new_length:
        expanded_qt[expanded_end:] = value_at_mid

    return expanded_qt