import numpy as np
def schmidt_spike_removal(original_signal, fs):
    original_signal = np.asarray(original_signal).flatten()
    windowsize = round(fs / 2)  # 500ms windows

    trailing_samples = len(original_signal) % windowsize
    n_full = len(original_signal) - trailing_samples
    n_windows = n_full // windowsize

    sampleframes = original_signal[:n_full].reshape(windowsize, n_windows, order='F').copy()
    MAAs = np.max(np.abs(sampleframes), axis=0)

    while np.any(MAAs > np.median(MAAs) * 3):
        window_num = np.argmax(MAAs)
        column = sampleframes[:, window_num]
        spike_position = np.argmax(np.abs(column))

        signs = np.sign(column)
        zero_crossings = np.append(np.abs(np.diff(signs)) > 1, False)

        before = np.where(zero_crossings[:spike_position + 1])[0]
        spike_start = before[-1] if len(before) > 0 else 0

        zero_crossings_after = zero_crossings.copy()
        zero_crossings_after[:spike_position + 1] = False
        after = np.where(zero_crossings_after)[0]
        spike_end = after[0] if len(after) > 0 else windowsize - 1

        sampleframes[spike_start:spike_end + 1, window_num] = 0.0001  # matches original (not literal 0)
        MAAs = np.max(np.abs(sampleframes), axis=0)

    despiked_signal = sampleframes.reshape(-1, order='F')
    return np.concatenate([despiked_signal, original_signal[len(despiked_signal):]])